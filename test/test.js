// Cópia local do teste oficial da rinha (ramp 0→900 RPS em 120s sobre o
// test-data.json de 54.100 entradas), com a adição de uma série temporal por
// segundo da latência máxima observada. Produz:
//
//   test/results.json              — pontuação oficial idêntica ao test.js da rinha
//   test/latency_timeseries.csv    — t_sec, max_ms, p99_ms, count
//   test/latency_timeseries.svg    — gráfico x=tempo(s), y=latência máxima(ms)
//
// O bucket por segundo é feito atribuindo cada amostra a
// floor((Date.now() - exec.scenario.startTime) / 1000). Isso usa o instante
// em que a resposta chega; sob 900 RPS isso já é granular o suficiente para
// expor saltos de C-state / governor / boost.

import http from 'k6/http';
import { SharedArray } from 'k6/data';
import { Counter, Trend } from 'k6/metrics';
import exec from 'k6/execution';

const testData = new SharedArray('test-data', function () {
    return JSON.parse(open('../../rinha-de-backend-2026/test/test-data.json')).entries;
});
const statsArr = new SharedArray('test-stats', function () {
    return [JSON.parse(open('../../rinha-de-backend-2026/test/test-data.json')).stats];
});
const expectedStats = statsArr[0];

const tpCount = new Counter('tp_count');
const tnCount = new Counter('tn_count');
const fpCount = new Counter('fp_count');
const fnCount = new Counter('fn_count');
const errorCount = new Counter('error_count');

// Janela total do teste em segundos (120s de ramp + slack pra gracefulStop e
// requisições em voo). Cada segundo vira um Trend (latência) e um Counter
// (request count) separado, para que o handleSummary possa extrair os valores
// por bucket sem precisar de raw samples. Counters expõem 'count' em values;
// Trends expõem só as chaves de summaryTrendStats.
const MAX_SECONDS = 200;
const secondTrends = [];
const secondCounters = [];
for (let i = 0; i < MAX_SECONDS; i++) {
    const tag = String(i).padStart(3, '0');
    secondTrends.push(new Trend(`lat_sec_${tag}`));
    secondCounters.push(new Counter(`req_sec_${tag}`));
}

export const options = {
    // 'max' alimenta a série temporal; 'p(99)' alimenta o score oficial;
    // o resto é só pra exibir no painel de stats globais no SVG.
    summaryTrendStats: ['min', 'avg', 'med', 'p(90)', 'p(99)', 'p(99.9)', 'max'],
    systemTags: ['status', 'method'],
    dns: {
        ttl: '5m',
        select: 'roundRobin',
    },
    scenarios: {
        default: {
            executor: 'ramping-arrival-rate',
            startRate: 1,
            timeUnit: '1s',
            preAllocatedVUs: 100,
            maxVUs: 250,
            gracefulStop: '10s',
            stages: [
                { duration: '120s', target: 900 },
            ],
        },
    },
};

export function setup() {
    console.log(
        `Dataset: ${expectedStats.total} entries, `
        + `${expectedStats.fraud_count} fraud (${expectedStats.fraud_rate}%), `
        + `${expectedStats.legit_count} legit (${expectedStats.legit_rate}%), `
        + `edge cases: ${expectedStats.edge_case_rate}%`
    );
}

export default function () {
    const idx = exec.scenario.iterationInTest;
    if (idx >= testData.length) return;
    const entry = testData[idx];
    const expectedApproved = entry.expected_approved;

    const res = http.post(
        'http://localhost:9999/fraud-score',
        JSON.stringify(entry.request),
        { headers: { 'Content-Type': 'application/json' }, timeout: '2001ms' }
    );

    // Bucket por segundo decorrido desde o início do scenario.
    // exec.scenario.startTime é o timestamp Unix em ms.
    const elapsedSec = Math.floor((Date.now() - exec.scenario.startTime) / 1000);
    if (elapsedSec >= 0 && elapsedSec < MAX_SECONDS) {
        secondTrends[elapsedSec].add(res.timings.duration);
        secondCounters[elapsedSec].add(1);
    }

    if (res.status === 200) {
        const body = JSON.parse(res.body);
        if (expectedApproved === body.approved) {
            if (body.approved) tnCount.add(1);
            else tpCount.add(1);
        } else {
            if (body.approved) fnCount.add(1);
            else fpCount.add(1);
        }
    } else {
        errorCount.add(1);
    }
}

function buildSeries(data) {
    const series = [];
    for (let i = 0; i < MAX_SECONDS; i++) {
        const tag = String(i).padStart(3, '0');
        const t = data.metrics[`lat_sec_${tag}`];
        const c = data.metrics[`req_sec_${tag}`];
        // Em k6 v0.54 Trend.values só contém as chaves de summaryTrendStats;
        // não há 'count'. Filtra por typeof do max.
        if (!t || !t.values || typeof t.values.max !== 'number') continue;
        series.push({
            t: i,
            max: t.values.max,
            p99: t.values['p(99)'],
            count: c && c.values ? c.values.count : 0,
        });
    }
    return series;
}

function buildCsv(series) {
    let csv = 't_sec,max_ms,p99_ms,count\n';
    for (const s of series) {
        csv += `${s.t},${s.max.toFixed(3)},${s.p99.toFixed(3)},${s.count}\n`;
    }
    return csv;
}

// SVG self-contained, sem dependências externas. Eixo Y esquerdo = latência
// (ms), eixo Y direito = RPS (count). Eixo X = tempo decorrido em segundos.
// Três séries: max latência (vermelho), p99 latência (azul), RPS (verde).
// O painel `stats` (canto superior direito) traz os percentis globais do
// http_req_duration e contagens de TP/FP/etc do scoring oficial.
function buildSvg(series, stats) {
    const W = 1400;
    const H = 700;
    const M = { top: 50, right: 90, bottom: 60, left: 80 };
    const innerW = W - M.left - M.right;
    const innerH = H - M.top - M.bottom;

    if (series.length === 0) {
        return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}"><text x="${W / 2}" y="${H / 2}" text-anchor="middle">sem dados</text></svg>`;
    }

    const tMax = series[series.length - 1].t + 1;
    let yMaxLat = 0, yMaxRps = 0;
    for (const s of series) {
        if (s.max > yMaxLat) yMaxLat = s.max;
        if (s.count > yMaxRps) yMaxRps = s.count;
    }
    yMaxLat = yMaxLat * 1.1;
    yMaxRps = yMaxRps * 1.1;
    if (yMaxLat <= 0) yMaxLat = 1;
    if (yMaxRps <= 0) yMaxRps = 1;

    const sx = (t) => M.left + (t / tMax) * innerW;
    const syLat = (y) => M.top + innerH - (y / yMaxLat) * innerH;
    const syRps = (y) => M.top + innerH - (y / yMaxRps) * innerH;

    let xTicks = '';
    for (let t = 0; t <= tMax; t += 10) {
        const x = sx(t);
        xTicks += `<line x1="${x.toFixed(1)}" y1="${M.top}" x2="${x.toFixed(1)}" y2="${M.top + innerH}" stroke="#e8e8e8" stroke-width="1"/>`;
        xTicks += `<text x="${x.toFixed(1)}" y="${(M.top + innerH + 20).toFixed(1)}" text-anchor="middle" font-size="12" fill="#555">${t}</text>`;
    }

    // Eixo Y esquerdo: latência. Linhas-guia horizontais derivam desse eixo.
    let yLeftTicks = '';
    const latStep = yMaxLat / 6;
    for (let i = 0; i <= 6; i++) {
        const yVal = latStep * i;
        const y = syLat(yVal);
        yLeftTicks += `<line x1="${M.left}" y1="${y.toFixed(1)}" x2="${(M.left + innerW).toFixed(1)}" y2="${y.toFixed(1)}" stroke="#e8e8e8" stroke-width="1"/>`;
        yLeftTicks += `<text x="${(M.left - 8).toFixed(1)}" y="${(y + 4).toFixed(1)}" text-anchor="end" font-size="12" fill="#d62728">${yVal.toFixed(2)}</text>`;
    }

    // Eixo Y direito: RPS. Mesmas 6 divisões do eixo esquerdo, labels à direita.
    let yRightTicks = '';
    const rpsStep = yMaxRps / 6;
    for (let i = 0; i <= 6; i++) {
        const yVal = rpsStep * i;
        const y = M.top + innerH - (i / 6) * innerH;
        yRightTicks += `<text x="${(M.left + innerW + 8).toFixed(1)}" y="${(y + 4).toFixed(1)}" text-anchor="start" font-size="12" fill="#2ca02c">${yVal.toFixed(0)}</text>`;
    }

    let maxPath = '';
    let p99Path = '';
    let rpsPath = '';
    for (let i = 0; i < series.length; i++) {
        const s = series[i];
        const x = sx(s.t);
        maxPath += (i === 0 ? 'M' : 'L') + x.toFixed(1) + ' ' + syLat(s.max).toFixed(1) + ' ';
        p99Path += (i === 0 ? 'M' : 'L') + x.toFixed(1) + ' ' + syLat(s.p99).toFixed(1) + ' ';
        rpsPath += (i === 0 ? 'M' : 'L') + x.toFixed(1) + ' ' + syRps(s.count).toFixed(1) + ' ';
    }

    const lx = M.left + 12;
    const ly = M.top + 12;
    const legend = `
      <g font-size="13">
        <rect x="${lx}" y="${ly}" width="160" height="80" fill="white" stroke="#ccc"/>
        <line x1="${lx + 12}" y1="${ly + 20}" x2="${lx + 42}" y2="${ly + 20}" stroke="#d62728" stroke-width="2"/>
        <text x="${lx + 48}" y="${ly + 24}" fill="#333">max latência (ms)</text>
        <line x1="${lx + 12}" y1="${ly + 44}" x2="${lx + 42}" y2="${ly + 44}" stroke="#1f77b4" stroke-width="2"/>
        <text x="${lx + 48}" y="${ly + 48}" fill="#333">p99 latência (ms)</text>
        <line x1="${lx + 12}" y1="${ly + 68}" x2="${lx + 42}" y2="${ly + 68}" stroke="#2ca02c" stroke-width="2"/>
        <text x="${lx + 48}" y="${ly + 72}" fill="#333">requests/s</text>
      </g>`;

    // Painel de stats globais — colocado no canto superior direito do plot,
    // dentro da área de dados pra não competir com a margem do eixo direito.
    const sw = 240;
    const sh = 200;
    const sx0 = M.left + innerW - sw - 12;
    const sy0 = M.top + 12;
    const row = (i, label, value) =>
        `<text x="${sx0 + 12}" y="${sy0 + 36 + i * 18}" fill="#333">${label}</text>` +
        `<text x="${sx0 + sw - 12}" y="${sy0 + 36 + i * 18}" text-anchor="end" fill="#333" font-family="monospace">${value}</text>`;
    const fmtMs = (v) => (v == null ? '—' : `${v.toFixed(3)} ms`);
    const statsPanel = `
      <g font-size="13">
        <rect x="${sx0}" y="${sy0}" width="${sw}" height="${sh}" fill="white" stroke="#ccc"/>
        <text x="${sx0 + sw / 2}" y="${sy0 + 20}" text-anchor="middle" font-weight="bold" fill="#333">http_req_duration (global)</text>
        ${row(0, 'min',       fmtMs(stats.min))}
        ${row(1, 'avg',       fmtMs(stats.avg))}
        ${row(2, 'p50 (med)', fmtMs(stats.med))}
        ${row(3, 'p90',       fmtMs(stats.p90))}
        ${row(4, 'p99',       fmtMs(stats.p99))}
        ${row(5, 'p99.9',     fmtMs(stats.p999))}
        ${row(6, 'max',       fmtMs(stats.max))}
        <line x1="${sx0 + 8}" y1="${sy0 + 36 + 7 * 18 - 8}" x2="${sx0 + sw - 8}" y2="${sy0 + 36 + 7 * 18 - 8}" stroke="#ddd"/>
        ${row(7.4, 'total reqs',  String(stats.totalReqs))}
        ${row(8.4, 'erros',       String(stats.errors))}
        ${row(9.4, 'score',       String(stats.finalScore))}
      </g>`;

    return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" font-family="sans-serif">
  <rect width="${W}" height="${H}" fill="white"/>
  <text x="${(W / 2).toFixed(1)}" y="30" text-anchor="middle" font-size="18" font-weight="bold">Latência e throughput por segundo — rinha official load (ramp 0→900 RPS em 120s)</text>
  <text x="${(M.left - 60).toFixed(1)}" y="${(M.top + innerH / 2).toFixed(1)}" text-anchor="middle" font-size="13" fill="#d62728" transform="rotate(-90 ${(M.left - 60).toFixed(1)} ${(M.top + innerH / 2).toFixed(1)})">latência (ms)</text>
  <text x="${(M.left + innerW + 60).toFixed(1)}" y="${(M.top + innerH / 2).toFixed(1)}" text-anchor="middle" font-size="13" fill="#2ca02c" transform="rotate(90 ${(M.left + innerW + 60).toFixed(1)} ${(M.top + innerH / 2).toFixed(1)})">requests/s</text>
  <text x="${(M.left + innerW / 2).toFixed(1)}" y="${(H - 15).toFixed(1)}" text-anchor="middle" font-size="13" fill="#333">tempo decorrido (s)</text>
  ${xTicks}
  ${yLeftTicks}
  ${yRightTicks}
  <rect x="${M.left}" y="${M.top}" width="${innerW}" height="${innerH}" fill="none" stroke="#888" stroke-width="1"/>
  <path d="${rpsPath}" stroke="#2ca02c" stroke-width="1.5" fill="none" opacity="0.75"/>
  <path d="${p99Path}" stroke="#1f77b4" stroke-width="1.5" fill="none" opacity="0.85"/>
  <path d="${maxPath}" stroke="#d62728" stroke-width="1.5" fill="none"/>
  ${legend}
  ${statsPanel}
</svg>
`;
}

export function handleSummary(data) {
    const K = 1000;
    const T_MAX_MS = 1000;
    const P99_MIN_MS = 1;
    const P99_MAX_MS = 2000;
    const EPSILON_MIN = 0.001;
    const BETA = 300;
    const TX_CORTE = 0.15;
    const SCORE_P99_CORTE = -3000;
    const SCORE_DET_CORTE = -3000;

    const httpDuration = data.metrics.http_req_duration.values;
    const p99 = httpDuration['p(99)'];

    const tp = data.metrics.tp_count ? data.metrics.tp_count.values.count : 0;
    const tn = data.metrics.tn_count ? data.metrics.tn_count.values.count : 0;
    const fp = data.metrics.fp_count ? data.metrics.fp_count.values.count : 0;
    const fn = data.metrics.fn_count ? data.metrics.fn_count.values.count : 0;
    const errs = data.metrics.error_count ? data.metrics.error_count.values.count : 0;

    const N = tp + tn + fp + fn + errs;

    const E = (fp * 1) + (fn * 3) + (errs * 5);
    const failures = fp + fn + errs;
    const epsilon = N > 0 ? E / N : 0;
    const failureRate = N > 0 ? failures / N : 0;

    let p99Score;
    let p99CutTriggered = false;
    if (p99 <= 0) {
        p99Score = 0;
    } else if (p99 > P99_MAX_MS) {
        p99Score = SCORE_P99_CORTE;
        p99CutTriggered = true;
    } else {
        p99Score = K * Math.log10(T_MAX_MS / Math.max(p99, P99_MIN_MS));
    }

    let detScore;
    let rateComponent = 0;
    let absolutePenalty = 0;
    let cutTriggered = false;
    if (failureRate > TX_CORTE) {
        detScore = SCORE_DET_CORTE;
        cutTriggered = true;
    } else {
        rateComponent = K * Math.log10(1 / Math.max(epsilon, EPSILON_MIN));
        absolutePenalty = -BETA * Math.log10(1 + E);
        detScore = rateComponent + absolutePenalty;
    }

    const finalScore = p99Score + detScore;

    const result = {
        expected: expectedStats,
        p99: p99.toFixed(2) + 'ms',
        scoring: {
            breakdown: {
                false_positive_detections: fp,
                false_negative_detections: fn,
                true_positive_detections: tp,
                true_negative_detections: tn,
                http_errors: errs,
            },
            failure_rate: +(failureRate * 100).toFixed(2) + '%',
            weighted_errors_E: E,
            error_rate_epsilon: +epsilon.toFixed(6),
            p99_score: {
                value: +p99Score.toFixed(2),
                cut_triggered: p99CutTriggered,
            },
            detection_score: {
                value: +detScore.toFixed(2),
                rate_component: cutTriggered ? null : +rateComponent.toFixed(2),
                absolute_penalty: cutTriggered ? null : +absolutePenalty.toFixed(2),
                cut_triggered: cutTriggered,
            },
            final_score: +finalScore.toFixed(2),
        },
    };

    const series = buildSeries(data);
    const stats = {
        min:        httpDuration['min'],
        avg:        httpDuration['avg'],
        med:        httpDuration['med'],
        p90:        httpDuration['p(90)'],
        p99:        httpDuration['p(99)'],
        p999:       httpDuration['p(99.9)'],
        max:        httpDuration['max'],
        totalReqs:  N,
        errors:     errs,
        finalScore: +finalScore.toFixed(0),
    };

    return {
        'test/results.json': JSON.stringify(result, null, 2),
        'test/latency_timeseries.csv': buildCsv(series),
        'test/latency_timeseries.svg': buildSvg(series, stats),
    };
}
