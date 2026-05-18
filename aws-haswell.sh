#!/usr/bin/env bash
# aws-haswell.sh — orquestra uma instância EC2 Haswell pra profiling do papagaio
# contra a microarch real do alvo (Mac mini Late 2014). Acesso via AWS SSM Session
# Manager (SSH-over-SSM) — sem SG inbound, sem rastrear IP público.
#
# Subcomandos:
#   doctor      checa pré-requisitos locais (aws cli, ssm plugin, creds)
#   provision   one-time: cria IAM role + instance profile + SG egress + lança a
#               instância + bootstrap. SEM portas inbound.
#   start       liga instância parada
#   stop        para instância (preserva EBS, libera compute)
#   status      mostra estado + status do agente SSM
#   ssh [cmd]   shell via SSM Session (ou roda comando remoto)
#   profile     rsync source → build remoto → k6+perf → puxa artefatos
#   destroy     termina instância + remove IAM role/profile + SG (irreversível)
#
# Local precisa: aws-cli + session-manager-plugin. Use 'doctor' pra checar.
# IAM creds precisam de perms em: ec2:*, iam:*, ssm:*.

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REGION=${AWS_REGION:-us-east-1}
INSTANCE_TYPE=${INSTANCE_TYPE:-m4.xlarge}   # Haswell-EP 2.4GHz, 2C/4T
VOLUME_SIZE=${VOLUME_SIZE:-30}
STATE_DIR=${STATE_DIR:-$HOME/.config/aws-haswell-papagaio}
STATE_FILE=$STATE_DIR/state
SSH_KEY=$STATE_DIR/sshkey
KNOWN_HOSTS=$STATE_DIR/known_hosts
SSH_CONFIG=$STATE_DIR/ssh_config
SG_NAME=papagaio-haswell-sg
ROLE_NAME=papagaio-haswell-ssm-role
PROFILE_NAME=papagaio-haswell-ssm-profile
NAME_TAG=papagaio-haswell

REMOTE_USER=ubuntu
REMOTE_HOME=/home/$REMOTE_USER
REMOTE_REPO=$REMOTE_HOME/papagaio
REMOTE_RINHA=$REMOTE_HOME/rinha-de-backend-2026

K6_RATE=${K6_RATE:-4000}
K6_DURATION=${K6_DURATION:-60s}
PERF_SECONDS=${PERF_SECONDS:-40}

mkdir -p "$STATE_DIR"
chmod 700 "$STATE_DIR"
[ -f "$STATE_FILE" ] && source "$STATE_FILE"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

aw() { aws --region "$REGION" "$@"; }
err() { echo "ERR: $*" >&2; exit 1; }
log() { echo "[$(date +%H:%M:%S)] $*"; }
require() { command -v "$1" >/dev/null || err "missing tool: $1"; }

save_state() {
    {
        echo "INSTANCE_ID=$INSTANCE_ID"
        echo "SG_ID=${SG_ID:-}"
    } > "$STATE_FILE"
}

require_provisioned() {
    [ -n "${INSTANCE_ID:-}" ] || err "não provisionado — rode '$0 provision' primeiro"
}

# SSH config file — único lugar com a config (ProxyCommand etc). ssh, scp e
# rsync todos honram `-F config_file`, evitando o inferno de quoting de array
# em bash + word-splitting do `-e "..."` do rsync.
write_ssh_config() {
    [ -n "${INSTANCE_ID:-}" ] || return 0
    cat > "$SSH_CONFIG" <<EOF
Host $INSTANCE_ID
    User $REMOTE_USER
    IdentityFile $SSH_KEY
    StrictHostKeyChecking accept-new
    UserKnownHostsFile $KNOWN_HOSTS
    ServerAliveInterval 30
    ProxyCommand sh -c "aws --region $REGION ssm start-session --target %h --document-name AWS-StartSSHSession --parameters 'portNumber=%p'"
EOF
    chmod 600 "$SSH_CONFIG"
}

ssh_remote() {
    write_ssh_config
    ssh -F "$SSH_CONFIG" "$INSTANCE_ID" "$@"
}

scp_to_remote() {
    write_ssh_config
    scp -F "$SSH_CONFIG" "$1" "$INSTANCE_ID:$2"
}

scp_from_remote() {
    write_ssh_config
    scp -r -F "$SSH_CONFIG" "$INSTANCE_ID:$1" "$2"
}

rsync_to_remote() {
    write_ssh_config
    local dest="${!#}"
    local args=( "${@:1:$#-1}" )
    rsync -e "ssh -F $SSH_CONFIG" "${args[@]}" "$INSTANCE_ID:$dest"
}

wait_ssm_online() {
    log "aguardando agente ssm ficar Online (até ~3min após start)..."
    for i in $(seq 1 60); do
        status=$(aw ssm describe-instance-information \
            --filters "Key=InstanceIds,Values=$INSTANCE_ID" \
            --query 'InstanceInformationList[0].PingStatus' \
            --output text 2>/dev/null)
        [ "$status" = "Online" ] && { log "ssm: Online"; return 0; }
        sleep 5
        printf '.'
    done
    echo
    err "ssm timeout — instância subiu mas agente não reportou"
}

wait_ssh_via_ssm() {
    log "aguardando ssh-via-ssm responder..."
    for i in $(seq 1 30); do
        if ssh_remote true 2>/dev/null; then
            log "ssh ok"
            return 0
        fi
        sleep 4
    done
    err "ssh-via-ssm timeout"
}

# ---------------------------------------------------------------------------
# User-data bootstrap (cloud-init, 1× no provision)
# ---------------------------------------------------------------------------

userdata_script() {
    local pubkey="$1"
    cat <<BOOTSTRAP
#!/bin/bash
set -e
exec > >(tee -a /var/log/papagaio-bootstrap.log) 2>&1
echo "[bootstrap] start \$(date)"

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y curl gnupg ca-certificates rsync build-essential pkg-config libssl-dev snapd \
                   linux-tools-common linux-tools-aws

# Tenta a versão exata do kernel (mais recente que linux-tools-aws meta pode estar atrasado).
# `linux-tools-aws` já cobre o caso geral; este apt-get é só best-effort.
KVER=\$(uname -r)
apt-get install -y "linux-tools-\$KVER" 2>/dev/null || true

# snapd precisa estar "seeded" antes de aceitar snap install.
# Sem isso: "error: too early for operation, device not yet seeded".
snap wait system seed.loaded

# SSM agent (snap é o caminho oficial recomendado pela AWS pra Ubuntu).
# `snap install` já habilita+inicia o service.
snap install amazon-ssm-agent --classic

# SSH pubkey do operador (pra que SSH-over-SSM autentique)
mkdir -p /home/ubuntu/.ssh
echo "$pubkey" >> /home/ubuntu/.ssh/authorized_keys
chmod 700 /home/ubuntu/.ssh
chmod 600 /home/ubuntu/.ssh/authorized_keys
chown -R ubuntu:ubuntu /home/ubuntu/.ssh

# Docker oficial
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg
. /etc/os-release
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \$VERSION_CODENAME stable" > /etc/apt/sources.list.d/docker.list
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
usermod -aG docker ubuntu

# k6
curl -fsSL https://dl.k6.io/key.gpg | gpg --dearmor -o /usr/share/keyrings/k6-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" > /etc/apt/sources.list.d/k6.list
apt-get update
apt-get install -y k6

# Rust + inferno (cargo cache fica em /home/ubuntu/.cargo, reusado entre builds)
sudo -u ubuntu bash -c 'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --no-modify-path'
sudo -u ubuntu bash -c '. ~/.cargo/env && cargo install inferno'
ln -sf /home/ubuntu/.cargo/bin/inferno-flamegraph /usr/local/bin/inferno-flamegraph
ln -sf /home/ubuntu/.cargo/bin/inferno-collapse-perf /usr/local/bin/inferno-collapse-perf

# perf knobs persistentes
cat > /etc/sysctl.d/99-papagaio-perf.conf <<E2E
kernel.perf_event_paranoid = -1
kernel.kptr_restrict = 0
kernel.perf_event_max_sample_rate = 1000000
kernel.perf_cpu_time_max_percent = 95
kernel.perf_event_max_stack = 1024
kernel.yama.ptrace_scope = 0
kernel.nmi_watchdog = 0
E2E
sysctl -p /etc/sysctl.d/99-papagaio-perf.conf || true
chmod -R o+rX /sys/kernel/tracing 2>/dev/null || true
chmod 755 /sys/kernel/tracing 2>/dev/null || true

# Workspace
mkdir -p /home/ubuntu/papagaio /home/ubuntu/rinha-de-backend-2026/test
chown -R ubuntu:ubuntu /home/ubuntu

# Runner que o script local invoca
cat > /usr/local/bin/papagaio-profile <<'RUNNER'
#!/bin/bash
set -e
K6_RATE=\$1; K6_DURATION=\$2; PERF_SECONDS=\$3
cd /home/ubuntu/papagaio
sudo chmod -R o+rX /sys/kernel/tracing 2>/dev/null || true
sudo chmod 755 /sys/kernel/tracing 2>/dev/null || true

docker compose -f docker-compose.yml -f docker-compose.profiling.yml build api1 2>&1 | tail -5
docker compose -f docker-compose.yml -f docker-compose.profiling.yml up -d 2>&1 | tail -3
sleep 3
curl -fsS http://localhost:9999/ready >/dev/null || { echo "stack não subiu"; exit 1; }

rm -rf profile-out
mkdir -p profile-out

API1=\$(docker inspect -f '{{.State.Pid}}' papagaio-api1)
API2=\$(docker inspect -f '{{.State.Pid}}' papagaio-api2)

k6 run -e RATE=\$K6_RATE -e DURATION=\$K6_DURATION test/profile.js > profile-out/k6.log 2>&1 &
K6PID=\$!
sleep 5
perf record -p \$API1,\$API2 -F 4999 --call-graph fp -e cycles \
    -o profile-out/perf.data -- sleep \$PERF_SECONDS 2>&1 | tail -3
wait \$K6PID 2>/dev/null

k6 run -e RATE=\$K6_RATE -e DURATION=20s test/profile.js > profile-out/k6-stat.log 2>&1 &
K6PID=\$!
sleep 3
perf stat -p \$API1,\$API2 \
    -e cycles,instructions,branches,branch-misses,cache-references,cache-misses,L1-dcache-load-misses,dTLB-load-misses \
    -o profile-out/perf-stat.txt -- sleep 12 2>&1 | tail -3
wait \$K6PID 2>/dev/null

perf script -i profile-out/perf.data 2>/dev/null \
    | inferno-collapse-perf \
    | inferno-flamegraph --hash --title "papagaio Haswell EC2 \$K6_RATE RPS" \
    > profile-out/flame.svg
perf report -i profile-out/perf.data --stdio --sort dso --no-children -g none \
    --percent-limit 0.3 > profile-out/perf-dso.txt 2>&1 || true
perf report -i profile-out/perf.data --stdio --sort symbol --no-children -g none \
    --percent-limit 0.2 > profile-out/perf-symbols.txt 2>&1 || true
grep -E "http_req_duration|http_req_failed|http_reqs" profile-out/k6.log | tail -10 > profile-out/k6-summary.txt

docker compose -f docker-compose.yml -f docker-compose.profiling.yml down >/dev/null 2>&1
echo "profile done"
RUNNER
chmod +x /usr/local/bin/papagaio-profile

touch /var/lib/papagaio-bootstrapped
echo "[bootstrap] done \$(date)"
BOOTSTRAP
}

# ---------------------------------------------------------------------------
# doctor: pré-requisitos locais
# ---------------------------------------------------------------------------

cmd_doctor() {
    local ok=0
    echo "=== aws-cli ==="
    if command -v aws >/dev/null; then
        aws --version 2>&1 | head -1
    else
        echo "  ✗ aws cli não instalado"
        echo "    instale: sudo apt install awscli"
        ok=1
    fi

    echo "=== session-manager-plugin ==="
    if command -v session-manager-plugin >/dev/null; then
        session-manager-plugin --version 2>&1 | head -1
    else
        echo "  ✗ session-manager-plugin não instalado"
        echo "    Ubuntu/Debian:"
        echo "      curl 'https://s3.amazonaws.com/session-manager-downloads/plugin/latest/ubuntu_64bit/session-manager-plugin.deb' -o /tmp/smp.deb"
        echo "      sudo dpkg -i /tmp/smp.deb"
        ok=1
    fi

    echo "=== aws credentials ==="
    if aw sts get-caller-identity >/dev/null 2>&1; then
        identity=$(aw sts get-caller-identity --query 'Arn' --output text)
        echo "  ✓ $identity (região: $REGION)"
    else
        echo "  ✗ aws creds não funcionam"
        echo "    configure: aws configure"
        ok=1
    fi

    echo "=== SSM endpoint reachable ==="
    if aw ssm describe-instance-information --max-results 1 >/dev/null 2>&1; then
        echo "  ✓ ssm acessível"
    else
        echo "  ✗ ssm não acessível (creds faltam permissão? região errada?)"
        ok=1
    fi

    return $ok
}

# ---------------------------------------------------------------------------
# provision (one-time)
# ---------------------------------------------------------------------------

cmd_provision() {
    require aws
    require ssh-keygen
    cmd_doctor >/dev/null || err "pré-requisitos faltando — rode '$0 doctor' pra detalhes"

    [ -n "${INSTANCE_ID:-}" ] && err "já provisionado ($INSTANCE_ID). 'destroy' antes pra refazer."

    log "checando default VPC na região $REGION..."
    DEFAULT_VPC=$(aw ec2 describe-vpcs --filters "Name=is-default,Values=true" \
        --query 'Vpcs[0].VpcId' --output text 2>/dev/null)
    [ -z "$DEFAULT_VPC" ] || [ "$DEFAULT_VPC" = "None" ] && \
        err "região $REGION sem default VPC. crie uma (ec2 create-default-vpc) ou mude AWS_REGION"
    log "vpc: $DEFAULT_VPC"

    # Subnet com auto-assign public IP (precisamos disso pra SSM agent alcançar
    # endpoints AWS sem VPC endpoint nem NAT). Default VPC subnets sempre têm
    # MapPublicIpOnLaunch=true, mas confirmamos pra falhar com mensagem clara.
    PUBLIC_SUBNET=$(aw ec2 describe-subnets \
        --filters "Name=vpc-id,Values=$DEFAULT_VPC" "Name=default-for-az,Values=true" \
        --query 'Subnets[0].SubnetId' --output text 2>/dev/null)
    [ -z "$PUBLIC_SUBNET" ] || [ "$PUBLIC_SUBNET" = "None" ] && \
        err "default VPC $DEFAULT_VPC sem default subnet"
    log "subnet: $PUBLIC_SUBNET"

    log "criando ssh key local (autenticação SSH-over-SSM)..."
    if [ ! -f "$SSH_KEY" ]; then
        ssh-keygen -t ed25519 -f "$SSH_KEY" -N "" -C "papagaio-haswell" >/dev/null
        chmod 400 "$SSH_KEY"
    fi
    PUBKEY=$(cat "$SSH_KEY.pub")

    log "criando IAM role + instance profile pra SSM..."
    TRUST='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
    aw iam create-role --role-name "$ROLE_NAME" \
        --assume-role-policy-document "$TRUST" >/dev/null 2>&1 || \
        log "  role já existia, reusando"
    aw iam attach-role-policy --role-name "$ROLE_NAME" \
        --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore >/dev/null 2>&1 || true
    aw iam create-instance-profile --instance-profile-name "$PROFILE_NAME" >/dev/null 2>&1 || \
        log "  instance profile já existia, reusando"
    aw iam add-role-to-instance-profile --instance-profile-name "$PROFILE_NAME" \
        --role-name "$ROLE_NAME" 2>/dev/null || true

    log "esperando propagação IAM (10s)..."
    sleep 10

    log "criando security group (egress-only, sem inbound)..."
    SG_ID=$(aw ec2 create-security-group \
        --group-name "$SG_NAME" \
        --description "papagaio Haswell — SSM only, no inbound" \
        --vpc-id "$DEFAULT_VPC" \
        --query 'GroupId' --output text 2>/dev/null) || \
    SG_ID=$(aw ec2 describe-security-groups \
        --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$DEFAULT_VPC" \
        --query 'SecurityGroups[0].GroupId' --output text)
    log "sg: $SG_ID (sem ingress; egress default = all)"

    log "buscando ami Ubuntu 24.04 amd64..."
    AMI_ID=$(aw ec2 describe-images --owners 099720109477 \
        --filters \
            "Name=name,Values=ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*" \
            "Name=architecture,Values=x86_64" \
            "Name=state,Values=available" \
        --query 'sort_by(Images, &CreationDate) | [-1].ImageId' --output text)
    [ -z "$AMI_ID" ] || [ "$AMI_ID" = "None" ] && err "ami não encontrada"
    log "ami: $AMI_ID"

    log "lançando $INSTANCE_TYPE..."
    UD=$(mktemp)
    userdata_script "$PUBKEY" > "$UD"
    INSTANCE_ID=$(aw ec2 run-instances \
        --image-id "$AMI_ID" \
        --instance-type "$INSTANCE_TYPE" \
        --iam-instance-profile "Name=$PROFILE_NAME" \
        --subnet-id "$PUBLIC_SUBNET" \
        --security-group-ids "$SG_ID" \
        --associate-public-ip-address \
        --user-data "file://$UD" \
        --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=$VOLUME_SIZE,VolumeType=gp3,DeleteOnTermination=true}" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME_TAG}]" \
        --metadata-options "HttpTokens=required,HttpEndpoint=enabled" \
        --query 'Instances[0].InstanceId' --output text)
    rm -f "$UD"
    save_state
    log "instance: $INSTANCE_ID"

    log "aguardando running..."
    aw ec2 wait instance-running --instance-ids "$INSTANCE_ID"
    wait_ssm_online

    log "aguardando bootstrap (~5-7min instalando docker+k6+rust)..."
    for i in $(seq 1 150); do
        if ssh_remote 'test -f /var/lib/papagaio-bootstrapped' 2>/dev/null; then
            echo
            log "✓ bootstrap done. teste: $0 ssh"
            return 0
        fi
        sleep 5
        printf '.'
    done
    echo
    err "bootstrap não completou em 12min. checa com: $0 ssh 'sudo tail /var/log/papagaio-bootstrap.log'"
}

# ---------------------------------------------------------------------------
# start / stop / status
# ---------------------------------------------------------------------------

cmd_start() {
    require_provisioned
    state=$(aw ec2 describe-instances --instance-ids "$INSTANCE_ID" \
        --query 'Reservations[0].Instances[0].State.Name' --output text)
    case "$state" in
        running) log "já running" ;;
        stopped|stopping)
            log "starting..."
            aw ec2 start-instances --instance-ids "$INSTANCE_ID" >/dev/null
            aw ec2 wait instance-running --instance-ids "$INSTANCE_ID"
            ;;
        *) err "estado inesperado: $state" ;;
    esac
    wait_ssm_online
    wait_ssh_via_ssm
}

cmd_stop() {
    require_provisioned
    log "stopping..."
    aw ec2 stop-instances --instance-ids "$INSTANCE_ID" >/dev/null
    aw ec2 wait instance-stopped --instance-ids "$INSTANCE_ID"
    log "stopped (EBS preservado, ~\$0.08/GB·mês)"
}

cmd_status() {
    if [ -z "${INSTANCE_ID:-}" ]; then
        echo "não provisionado"
        return 0
    fi
    state=$(aw ec2 describe-instances --instance-ids "$INSTANCE_ID" \
        --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null) || \
        { echo "instância $INSTANCE_ID não existe (state local stale)"; return 1; }
    ssm_status=$(aw ssm describe-instance-information \
        --filters "Key=InstanceIds,Values=$INSTANCE_ID" \
        --query 'InstanceInformationList[0].PingStatus' \
        --output text 2>/dev/null)
    [ "$ssm_status" = "None" ] && ssm_status="(offline)"
    cat <<S
instance: $INSTANCE_ID
type:     $INSTANCE_TYPE ($REGION)
state:    $state
ssm:      ${ssm_status:-(unknown)}
sg:       ${SG_ID:-?} (egress-only)
ssh:      $0 ssh
S
}

cmd_ssh() {
    require_provisioned
    if [ "$#" -eq 0 ]; then
        # shellcheck disable=SC2046
        exec ssh $(ssh_opts) $REMOTE_USER@$INSTANCE_ID
    else
        ssh_remote "$@"
    fi
}

# ---------------------------------------------------------------------------
# profile (main loop)
# ---------------------------------------------------------------------------

cmd_profile() {
    require_provisioned
    state=$(aw ec2 describe-instances --instance-ids "$INSTANCE_ID" \
        --query 'Reservations[0].Instances[0].State.Name' --output text)
    [ "$state" = "running" ] || cmd_start

    REPO=$(cd "$(dirname "$0")" && pwd)
    RINHA=$(cd "$REPO/../rinha-de-backend-2026" 2>/dev/null && pwd) || \
        err "rinha-de-backend-2026 não encontrado em $REPO/.."

    log "[1/4] sync sources..."
    rsync_to_remote -az --delete \
        --exclude='target' --exclude='.git' --exclude='profile-out' \
        --exclude='.venv' --exclude='__pycache__' \
        --exclude='profile-out-haswell' --exclude='log/' \
        "$REPO/" "$REMOTE_REPO/"

    log "[2/4] sync test data..."
    rsync_to_remote -az \
        "$RINHA/test/test.js" "$RINHA/test/test-data.json" \
        "$REMOTE_RINHA/test/"

    log "[3/4] build + profile (rate=$K6_RATE duration=$K6_DURATION perf=${PERF_SECONDS}s)..."
    ssh_remote "papagaio-profile $K6_RATE $K6_DURATION $PERF_SECONDS"

    log "[4/4] puxando artefatos..."
    OUTDIR="$REPO/profile-out-haswell/$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$OUTDIR"
    scp_from_remote "$REMOTE_REPO/profile-out/*" "$OUTDIR/"

    log "artefatos: $OUTDIR"
    ls -lh "$OUTDIR"
    echo
    echo "k6 summary:"
    cat "$OUTDIR/k6-summary.txt" 2>/dev/null | sed 's/^/  /'
    echo
    log "lembre-se: $0 stop pra economizar"
}

# ---------------------------------------------------------------------------
# destroy
# ---------------------------------------------------------------------------

cmd_destroy() {
    require_provisioned
    echo "isso vai:"
    echo "  - terminar a instância $INSTANCE_ID"
    echo "  - remover SG $SG_ID"
    echo "  - remover IAM role $ROLE_NAME + instance profile $PROFILE_NAME"
    printf "tem certeza? (digite 'yes'): "
    read -r ans
    [ "$ans" = "yes" ] || { echo "abortado"; exit 1; }

    log "terminando instância..."
    aw ec2 terminate-instances --instance-ids "$INSTANCE_ID" >/dev/null
    aw ec2 wait instance-terminated --instance-ids "$INSTANCE_ID"

    log "removendo SG..."
    aw ec2 delete-security-group --group-id "$SG_ID" 2>/dev/null || true

    log "removendo IAM..."
    aw iam remove-role-from-instance-profile \
        --instance-profile-name "$PROFILE_NAME" --role-name "$ROLE_NAME" 2>/dev/null || true
    aw iam delete-instance-profile --instance-profile-name "$PROFILE_NAME" 2>/dev/null || true
    aw iam detach-role-policy --role-name "$ROLE_NAME" \
        --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore 2>/dev/null || true
    aw iam delete-role --role-name "$ROLE_NAME" 2>/dev/null || true

    rm -f "$STATE_FILE"
    log "destroyed"
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

case "${1:-status}" in
    doctor)            cmd_doctor ;;
    provision)         cmd_provision ;;
    start|up)          cmd_start ;;
    stop|down)         cmd_stop ;;
    status)            cmd_status ;;
    ssh)               shift; cmd_ssh "$@" ;;
    profile)           cmd_profile ;;
    destroy)           cmd_destroy ;;
    -h|--help|help)
        cat <<USAGE
uso: $0 <subcomando>

  doctor      checa pré-requisitos locais (aws cli, ssm plugin, creds)
  provision   one-time: IAM role + SG egress + lança $INSTANCE_TYPE + bootstrap (~6-8min)
  start       liga instância parada
  stop        para instância (preserva EBS)
  status      mostra estado + ping do SSM agent
  ssh [cmd]   shell via SSM Session (ou roda cmd)
  profile     rsync source → build remoto → k6+perf → traz artefatos
  destroy     termina tudo (irreversível)

acesso via SSM Session Manager — sem SG inbound, sem IP público dependente.
local precisa: aws-cli + session-manager-plugin (rode 'doctor' pra checar).

config via env:
  AWS_REGION=$REGION
  INSTANCE_TYPE=$INSTANCE_TYPE   # m4.xlarge=Haswell-EP 2.4GHz; c4.xlarge=2.9GHz
  VOLUME_SIZE=$VOLUME_SIZE
  K6_RATE=$K6_RATE
  K6_DURATION=$K6_DURATION
  PERF_SECONDS=$PERF_SECONDS

custo: ~\$0.20/hr ligada, ~\$2.50/mês parada (só EBS).
USAGE
        ;;
    *) err "subcomando desconhecido: $1 (rode '$0 help')" ;;
esac
