// Short rinha peak test — 1s warmup ramp to 900 RPS, then 5s held at the
// peak. Quick smoke for "what's our latency RIGHT NOW under the rinha SLA?"
// without waiting 2 minutes of ramp like the official test.
//
// Reports the full latency percentile spread (p50/p90/p99/p99.9), unlike
// the official test.js which only emits p99 in results.json.

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
        peak: {
            executor: 'ramping-arrival-rate',
            startRate: 0,
            timeUnit: '1s',
            preAllocatedVUs: 50,
            maxVUs: 200,
            gracefulStop: '2s',
            stages: [
                { duration: '1s', target: 900 },  // warmup ramp
                { duration: '5s', target: 900 },  // hold at peak
            ],
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
