# Cold MMMU split comparison

The completed measurement comprises six waves on one four-H800 node.
Each wave starts fresh workers and empty EC storage. The proxy gives each
request its own media key, shared between encoder and PD. Every wave
recorded 720 visual forward calls, 720 encoded images, zero EC hits,
zero resident-cache hits, 720 completed requests, and zero failures.

| Round | Split | E completions/s | PD completions/s | P90 TTFT (ms) |
| --- | --- | ---: | ---: | ---: |
| 1 | 1E:3PD | 23.123 | 23.166 | 9626.98 |
| 1 | 2E:2PD | 33.705 | 25.900 | 6478.69 |
| 1 | 3E:1PD | 33.398 | 13.143 | 27906.09 |
| 2 | 3E:1PD | 33.696 | 13.445 | 27667.23 |
| 2 | 2E:2PD | 33.756 | 26.214 | 5692.18 |
| 2 | 1E:3PD | 23.013 | 23.252 | 9679.19 |

The nominal offered rate was 34.27 requests/s. Actual seeded arrivals averaged
34.473–34.474 requests/s. Completion rates use the middle 60% of actual
arrivals. Round two reverses split order. The same 720-image MMMU cohort,
preprocessing and fixed 64-token generation apply to every wave.

2E:2PD ranked first in both rounds, exceeding second-place 1E:3PD by
11.8% and 12.7%. The single E worker queued 173–175 requests at peak,
whereas the equal split's E workers queued at most 2–3 requests each.
The equal split still had PD queueing under this offered load.

The original job 785325 failed during the first wave because the cold-only
proxy modification gave the encoder and decoder inconsistent media keys.
Job 786457 uses the corrected key propagation. It also explicitly forwards
the frozen seed and request count to the vLLM client. The earlier custom
image adapter omitted those flags and used the client's default seed.
The current result therefore establishes a matched cold comparison within
this six-wave experiment. Cross-experiment differences also include node
and request-order differences.

See plan.json and results.json for the frozen conditions, code digests,
per-window counts and full-wave counts. This is a descriptive result for
the tested serving configuration.
