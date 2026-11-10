# evidence/particle-filter

The particle filters' likelihood estimates against the Kalman filter on the AR(1) benchmark: unbiased on the natural scale (the mean of Z-hat / Z over replications is one within its error) for the bootstrap, auxiliary and guided proposals, with the guided one the least variable.

Run on 2026-11-10 at commit 908edcb3; seed 0, budget scale 1 (replications 80, particles 100, proposals 3), 1.6 s.

**Passed**: every proposal unbiased on the natural scale; the guided proposal is less variable than the bootstrap

AR(1) with phi = 0.8, 12 observations, 100 particles, 80 replications. The reference is the Kalman filter's exact log likelihood -19.2199.

| proposal                 | mean Z-hat / Z | se    | mean bias of log Z-hat | sd of log Z-hat | verdict |
| ------------------------ | -------------- | ----- | ---------------------- | --------------- | ------- |
| bootstrap                | 0.919          | 0.099 | -0.401                 | 0.785           | ok      |
| auxiliary                | 0.972          | 0.068 | -0.230                 | 0.664           | ok      |
| guided (locally optimal) | 1.008          | 0.056 | -0.081                 | 0.410           | ok      |

The log of an unbiased estimate is biased low by about half its variance, which is why the bias of log Z-hat is negative and the ratio is the test. The auxiliary filter's first stage uses a point prediction of the state; with process noise as large as the observation noise it is the most variable of the three.
