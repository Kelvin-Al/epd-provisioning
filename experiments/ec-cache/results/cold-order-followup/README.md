# MMMU cold order follow-up

The job replays the same frozen 720-image MMMU cohort and 64-token output
at a target 34.27 requests/s. Each split occupies each within-round
position once. Throughput is measured in the middle 60% of arrivals.

| Round | Position | Split | Actual offered req/s | E req/s | PD req/s | P90 TTFT ms | Visual forwards | EC hits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1E:3PD | 34.474 | 23.532 | 23.595 | 9106 | 720 | 0 |
| 1 | 2 | 2E:2PD | 34.475 | 33.706 | 26.443 | 6207 | 720 | 0 |
| 1 | 3 | 3E:1PD | 34.473 | 33.506 | 12.999 | 29083 | 720 | 0 |
| 2 | 1 | 2E:2PD | 34.473 | 33.650 | 26.037 | 6062 | 720 | 0 |
| 2 | 2 | 3E:1PD | 34.474 | 33.653 | 13.555 | 26945 | 720 | 0 |
| 2 | 3 | 1E:3PD | 34.474 | 23.459 | 23.524 | 9116 | 720 | 0 |
| 3 | 1 | 3E:1PD | 34.471 | 33.863 | 13.309 | 27695 | 720 | 0 |
| 3 | 2 | 1E:3PD | 34.473 | 23.365 | 23.444 | 9329 | 720 | 0 |
| 3 | 3 | 2E:2PD | 34.474 | 33.868 | 26.673 | 5703 | 720 | 0 |

Every wave encoded all 720 images, with zero EC or resident-cache hits, and completed all requests.

## PD completions per second by position

| Split | First | Second | Third | Mean |
| --- | --- | --- | --- | --- |
| 1E:3PD | 23.595 | 23.444 | 23.524 | 23.521 |
| 2E:2PD | 26.037 | 26.443 | 26.673 | 26.384 |
| 3E:1PD | 13.309 | 13.555 | 12.999 | 13.288 |

## Round rankings

- Round 1  2E:2PD > 1E:3PD > 3E:1PD (12.07% ahead of second place).
- Round 2  2E:2PD > 1E:3PD > 3E:1PD (10.68% ahead of second place).
- Round 3  2E:2PD > 1E:3PD > 3E:1PD (13.77% ahead of second place).

Each split-position cell has one observation. The cyclic schedule
balances positions and rounds, with limited coverage of predecessor
effects and between-wave variability. The run addresses ranking under
this workload and does not supply sustained saturation or SLO evidence.

The nine-wave rerun used one four-H800 allocation on the original
experiment node. The earlier attempt completed six waves before a PD
worker in wave seven failed to bind an occupied port. That attempt is
retained in [interrupted-results.json](interrupted-results.json) and is
excluded from the complete nine-wave comparison. The new launcher checks
that its serving ports are available and outside the ephemeral range
before worker startup.

The machine-readable evidence is in [plan.json](plan.json) and
[results.json](results.json). To repeat the order design, stage the
[balanced sweep](../../order_followup/cold_sweep.py) and its
[batch launcher](../../order_followup/audit.sbatch) with the existing
[audit helper](../../audit.py) and
[four-GPU experiment helper](../../../four-gpu/experiment.py) using
the [EC audit setup](../../README.md).

The two earlier reverse-order rounds kept 2E:2PD in the middle. Here it
ranks first once from each position, which addresses the fixed
middle-position concern for this workload. Each split-position cell
was observed once. Runtime, sample-size and acceptance limitations
remain, and the observation is author-reported.
