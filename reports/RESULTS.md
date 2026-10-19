# Onboard vision: measured results

Everything below is produced by `python scripts/run_experiments.py` and can be regenerated.

## Setups

* `nominal`: Clean footage, correct camera height and pitch.
* `noise_high`: Sensor noise quadrupled, as on a compressed broadcast feed.
* `texture_weak`: Road texture reduced to 35 percent, like fresh asphalt in shade.
* `texture_none`: Essentially untextured road, to show what the estimator does when it should give up.
* `roll_high`: Body roll of 0.06 rad per g enabled, so the camera rotates about the optical axis.
* `height_wrong`: Estimator assumes a camera 20 cm lower than it really is.
* `texture_weak_gated`: Same footage as texture_weak with the inlier gate raised to 100, showing that the feature count can be used to refuse a frame instead of guessing.

## Frame to frame accuracy

| setup | covered % | speed MAE m/s | speed MAE % | speed bias m/s | yaw MAE rad/s | yaw sign % | frames |
|---|---|---|---|---|---|---|---|
| nominal |  100.0 |   1.40 |   2.57 |   -0.76 |  0.030 |  100.0 | 90 |
| noise_high |  100.0 |   2.00 |   3.68 |   -1.74 |  0.048 |  100.0 | 90 |
| texture_weak |  100.0 |   6.70 |  12.29 |   -6.66 |  0.070 |  100.0 | 90 |
| texture_none |   92.2 |  11.56 |  21.23 |  -11.56 |  0.080 |  100.0 | 90 |
| roll_high |  100.0 |  14.31 |  26.26 |  -14.31 |  0.637 |  100.0 | 90 |
| height_wrong |  100.0 |  10.11 |  18.54 |  -10.11 |  0.030 |  100.0 | 90 |
| texture_weak_gated |   50.0 |   6.37 |  11.72 |   -6.31 |  0.057 |  100.0 | 90 |

## What the numbers say

* Best speed accuracy: nominal at 2.57 percent MAE over 90 of 90 frames.
* Worst case that still produced estimates: roll_high at 26.26 percent, coverage 100 percent.
* Systematic speed bias above 2 m/s: texture_weak (-6.7 m/s), texture_none (-11.6 m/s), roll_high (-14.3 m/s), height_wrong (-10.1 m/s), texture_weak_gated (-6.3 m/s).

## Resolution

| camera | covered % | speed MAE m/s | speed MAE % | yaw MAE rad/s |
|---|---|---|---|---|
| 320x180 | 100 | 1.62 | 2.96 | 0.081 |
| 640x360 | 100 | 1.40 | 2.57 | 0.030 |
| 960x540 | 100 | 1.91 | 3.50 | 0.029 |

## Temporal filtering

A centred five frame average of the same per frame estimates moves the speed MAE from 1.40 m/s to 1.29 m/s over 86 windows, which is the square root of five to within the spread of the estimate. The measurement noise is close to white, so filtering is nearly free.

## Tracking configuration sweep

| configuration | quality | min distance | window | coverage % | speed MAE % | ms |
|---|---|---|---|---|---|---|
| strong_long_window | 0.010 | 6 | 21 | 100 | 20.44 | 141 |
| few_weak_corners | 0.005 | 6 | 21 | 100 | 6.77 | 88 |
| many_short_window | 0.003 | 4 | 15 | 100 | 2.79 | 159 |
| selected | 0.002 | 3 | 11 | 100 | 2.57 | 77 |
| short_window | 0.002 | 3 | 9 | 100 | 2.55 | 63 |
| deep_pyramid | 0.002 | 3 | 11 | 100 | 2.64 | 81 |
| shallow_pyramid | 0.002 | 3 | 11 | 100 | 2.73 | 66 |
