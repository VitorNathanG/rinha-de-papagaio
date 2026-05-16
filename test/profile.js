// Parallel k6 test for profiling — sustained high-rate (no ramp) so the load
// is stationary while perf samples. Pushes well above the rinha SLA (default
// 10000 RPS vs the official 900 peak) to actually stress the backend.
// Reuses the canonical rinha test data via relative path.
//
// Tunables:
//   RATE      requests/sec target  (default 10000)
//   DURATION  sustained duration   (default 30s)
//   PREVUS    preAllocatedVUs      (default 400)
//   MAXVUS    upper bound on VUs   (default 2000)
//
// Difference from /projects/rinha-de-backend-2026/test/test.js:
//   - constant-arrival-rate sustained, no ramp
//   - cycles through test data with modulo (duration ≠ data size is fine)
//   - reports more latency percentiles so the tail is visible
//   - does NOT emit results.json/scoring (profiling != scoring;
//     the original test.js stays the source of truth for that)

import http from 'k6/http';
import { SharedArray } from 'k6/data';
import { Counter } from 'k6/metrics';
import exec from 'k6/execution';

const testData = new SharedArray('test-data', function () {
    return JSON.parse(open('../../rinha-de-backend-2026/test/test-data.json')).entries;
});

const tpCount = new Counter('tp_count');
const tnCount = new Counter('tn_count');
const fpCount = new Counter('fp_count');
const fnCount = new Counter('fn_count');
const errorCount = new Counter('error_count');

export const options = {
    summaryTrendStats: ['min', 'avg', 'med', 'p(90)', 'p(95)', 'p(99)', 'p(99.9)', 'max'],
    systemTags: ['status', 'method'],
    dns: {
        ttl: '5m',
        select: 'roundRobin',
    },
    scenarios: {
        sustained: {
            executor: 'constant-arrival-rate',
            rate: parseInt(__ENV.RATE || '10000'),
            timeUnit: '1s',
            duration: __ENV.DURATION || '30s',
            preAllocatedVUs: parseInt(__ENV.PREVUS || '400'),
            maxVUs: parseInt(__ENV.MAXVUS || '2000'),
            gracefulStop: '5s',
        },
    },
};

export default function () {
    const entry = testData[exec.scenario.iterationInTest % testData.length];
    const expectedApproved = entry.expected_approved;

    const res = http.post(
        'http://localhost:9999/fraud-score',
        JSON.stringify(entry.request),
        { headers: { 'Content-Type': 'application/json' }, timeout: '2001ms' }
    );

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
