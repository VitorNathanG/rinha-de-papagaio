#!/usr/bin/env bash
# perf-prep.sh — Liga as knobs de kernel necessárias pra profiling
# de alta resolução com `perf` (PEBS, LBR, stack unwind, kernel symbols,
# tracepoints).
#
# Uso:
#   sudo ./perf-prep.sh check     mostra estado atual, não muda nada
#   sudo ./perf-prep.sh setup     aplica as mudanças (salva backup)
#   sudo ./perf-prep.sh restore   reverte aos valores anteriores
#
# As mudanças são runtime-only (sysfs/proc), sumem no próximo boot.
# Nenhuma edição em /etc; nenhum daemon reiniciado.

set -euo pipefail

STATE=/tmp/perf-prep-state
MODE=${1:-check}

# ---------------------------------------------------------------------------
# Definição das knobs.
#
# Cada knob é uma linha "label|path|desired_value|why".
# - path é um arquivo em /proc ou /sys que aceita `echo > path`.
# - desired_value é o valor pra aplicar no `setup`.
# - why é uma descrição curta pro log.
# ---------------------------------------------------------------------------

KNOBS=(
    "paranoid|/proc/sys/kernel/perf_event_paranoid|-1|userspace pode usar todos os eventos (kernel, raw PMU, LBR, tracepoints)"
    "kptr|/proc/sys/kernel/kptr_restrict|0|exporta endereços de símbolos do kernel em /proc/kallsyms"
    "max_rate|/proc/sys/kernel/perf_event_max_sample_rate|1000000|permite até 1M Hz de sampling (default 100k)"
    "cpu_pct|/proc/sys/kernel/perf_cpu_time_max_percent|95|perf pode gastar até 95% do CPU em sampling (default 25)"
    "max_stack|/proc/sys/kernel/perf_event_max_stack|1024|profundidade máx do stack unwind"
    "ptrace|/proc/sys/kernel/yama/ptrace_scope|0|permite anexar a qualquer processo (perf record -p PID)"
    "nmi|/proc/sys/kernel/nmi_watchdog|0|libera 1 contador PMC fixo que o watchdog consome"
)

# Governor é tratado à parte (vários cores).
GOVERNOR_DESIRED=performance

# ---------------------------------------------------------------------------

require_root() {
    if [ "$EUID" -ne 0 ]; then
        echo "perf-prep: precisa rodar como root (use sudo)" >&2
        exit 1
    fi
}

color() {
    # $1 = ok|warn|err, $2 = text
    case "$1" in
        ok)   printf '\033[32m%s\033[0m' "$2" ;;
        warn) printf '\033[33m%s\033[0m' "$2" ;;
        err)  printf '\033[31m%s\033[0m' "$2" ;;
        dim)  printf '\033[2m%s\033[0m'  "$2" ;;
        *)    printf '%s' "$2" ;;
    esac
}

read_path() {
    [ -e "$1" ] && cat "$1" 2>/dev/null || echo "<missing>"
}

# ---------------------------------------------------------------------------
# Action: check
# ---------------------------------------------------------------------------

do_check() {
    printf "%-12s %-50s %-12s %-12s\n" "KNOB" "PATH" "CURRENT" "DESIRED"
    printf '%.0s-' {1..90}; echo
    for entry in "${KNOBS[@]}"; do
        IFS='|' read -r label path desired why <<<"$entry"
        cur=$(read_path "$path")
        if [ "$cur" = "$desired" ]; then
            status=$(color ok "OK")
        elif [ "$cur" = "<missing>" ]; then
            status=$(color warn "N/A")
        else
            status=$(color warn "drift")
        fi
        printf "%-12s %-50s %-12s %-12s %s  %s\n" \
            "$label" "$path" "$cur" "$desired" "$status" "$(color dim "$why")"
    done

    echo
    # Governor
    local g
    g=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "?")
    if [ "$g" = "$GOVERNOR_DESIRED" ]; then
        echo "governor cpu0: $g $(color ok OK)"
    else
        echo "governor cpu0: $g (desired: $GOVERNOR_DESIRED) $(color warn drift)"
    fi

    # PMU info
    echo
    echo "PMU disponível:"
    if [ -d /sys/bus/event_source/devices/cpu ]; then
        ls /sys/bus/event_source/devices/ | sed 's/^/  /'
    fi

    # NMI watchdog (alternativa)
    if [ -e /sys/devices/system/cpu/intel_pstate ]; then
        echo
        echo "intel_pstate: $(cat /sys/devices/system/cpu/intel_pstate/status 2>/dev/null || echo unknown)"
        echo "no_turbo:     $(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo unknown)"
    fi
    if [ -e /sys/devices/system/cpu/cpufreq/boost ]; then
        echo "amd boost:    $(cat /sys/devices/system/cpu/cpufreq/boost)"
    fi

    # Debugfs/tracefs
    echo
    if mountpoint -q /sys/kernel/debug; then
        echo "debugfs:  $(color ok mounted) em /sys/kernel/debug"
    else
        echo "debugfs:  $(color warn 'not mounted') (tracepoints de kernel ficam limitados)"
    fi
    if mountpoint -q /sys/kernel/tracing; then
        echo "tracefs:  $(color ok mounted) em /sys/kernel/tracing"
    else
        echo "tracefs:  $(color warn 'not mounted')"
    fi

    # Backup presente?
    echo
    if [ -f "$STATE" ]; then
        echo "backup:   $(color ok present) em $STATE — restore disponível"
    else
        echo "backup:   $(color dim 'absent') — setup vai criar"
    fi
}

# ---------------------------------------------------------------------------
# Action: setup
# ---------------------------------------------------------------------------

do_setup() {
    require_root

    if [ -f "$STATE" ]; then
        echo "perf-prep: backup já existe em $STATE — rode 'restore' antes de re-aplicar"
        echo "           ou apague o arquivo manualmente se souber o que está fazendo."
        exit 1
    fi

    : > "$STATE"
    echo "# perf-prep backup created $(date -Iseconds)" >> "$STATE"

    for entry in "${KNOBS[@]}"; do
        IFS='|' read -r label path desired _why <<<"$entry"
        if [ ! -e "$path" ]; then
            echo "skip $label ($path absent)"
            continue
        fi
        cur=$(cat "$path")
        printf 'KNOB|%s|%s|%s\n' "$label" "$path" "$cur" >> "$STATE"
        if [ "$cur" = "$desired" ]; then
            echo "$label: já em $desired"
        else
            echo "$cur" > "$path" 2>/dev/null && \
                cur_orig=$cur || { echo "FAIL writing $path"; continue; }
            # ^ touch-and-save pra deixar claro o backup; agora aplica o desired
            echo "$desired" > "$path"
            new=$(cat "$path")
            echo "$label: $cur_orig -> $new"
        fi
    done

    # Governor: trata cada cpu individualmente
    echo
    for g_path in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        [ -e "$g_path" ] || continue
        cur=$(cat "$g_path")
        printf 'GOV|%s|%s\n' "$g_path" "$cur" >> "$STATE"
        if [ "$cur" != "$GOVERNOR_DESIRED" ]; then
            echo "$GOVERNOR_DESIRED" > "$g_path" 2>/dev/null || true
        fi
    done
    new_g=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "?")
    echo "governor: -> $new_g (todos os cpus)"

    # Mount debugfs/tracefs se faltar
    if ! mountpoint -q /sys/kernel/debug; then
        mount -t debugfs none /sys/kernel/debug && \
            echo "MOUNT|debugfs|/sys/kernel/debug" >> "$STATE" && \
            echo "debugfs: mounted /sys/kernel/debug"
    fi
    if ! mountpoint -q /sys/kernel/tracing; then
        mount -t tracefs none /sys/kernel/tracing 2>/dev/null && \
            echo "MOUNT|tracefs|/sys/kernel/tracing" >> "$STATE" && \
            echo "tracefs: mounted /sys/kernel/tracing"
    fi

    # tracefs default é mode 0700 — só root lê tracepoints. Pra perf trace,
    # perf stat -e syscalls:*, perf record -e sched:* funcionar como user,
    # destrava read em /sys/kernel/tracing recursivamente.
    if [ -d /sys/kernel/tracing ]; then
        cur_mode=$(stat -c %a /sys/kernel/tracing 2>/dev/null || echo "")
        if [ "$cur_mode" != "755" ]; then
            printf 'TRACEFS|%s\n' "$cur_mode" >> "$STATE"
            chmod -R o+rX /sys/kernel/tracing 2>/dev/null && \
                chmod 755 /sys/kernel/tracing && \
                echo "tracefs perms: $cur_mode -> 755 (recursive o+rX)"
        fi
    fi

    echo
    echo "perf-prep: setup OK. estado salvo em $STATE."
    echo "          'restore' pra reverter, 'check' pra ver."
}

# ---------------------------------------------------------------------------
# Action: restore
# ---------------------------------------------------------------------------

do_restore() {
    require_root
    if [ ! -f "$STATE" ]; then
        echo "perf-prep: nenhum backup em $STATE — nada pra restaurar"
        exit 1
    fi

    while IFS='|' read -r kind a b c; do
        case "$kind" in
            "#"*|"") ;;
            KNOB)
                # KNOB|label|path|old_value
                echo "$c" > "$b" 2>/dev/null && echo "restore $a: $(cat "$b") <- (was $c)"
                ;;
            GOV)
                # GOV|path|old_value
                echo "$b" > "$a" 2>/dev/null || true
                ;;
            MOUNT)
                # MOUNT|fs|mountpoint  (umount o que setup montou)
                umount "$b" 2>/dev/null && echo "umount $b"
                ;;
            TRACEFS)
                # TRACEFS|old_mode
                chmod "$a" /sys/kernel/tracing 2>/dev/null && \
                    echo "restore tracefs mode -> $a"
                ;;
        esac
    done < "$STATE"

    rm -f "$STATE"
    echo "perf-prep: restore OK."
}

# ---------------------------------------------------------------------------

case "$MODE" in
    check)   do_check ;;
    setup)   do_setup ;;
    restore) do_restore ;;
    -h|--help|help)
        echo "uso: sudo $0 [check|setup|restore]"
        echo "  check   - mostra estado atual"
        echo "  setup   - aplica tunings (salva backup em $STATE)"
        echo "  restore - reverte do backup"
        ;;
    *)
        echo "modo desconhecido: $MODE (use check|setup|restore)" >&2
        exit 2
        ;;
esac
