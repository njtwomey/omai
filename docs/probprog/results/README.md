# Validation results

Written by `make probprog-validate` on 2026-11-10 at commit 4ce8b576 (seed 0, budget scale 3). Each file is one campaign; a diff of these tables is a regression.

| campaign                                              | verdict                                                                                            | seconds |
| ----------------------------------------------------- | -------------------------------------------------------------------------------------------------- | ------- |
| [sbc/nuts-conjugate](sbc-nuts-conjugate.md)           | passed: calibrated over 300 replications                                                           | 8.0     |
| [sbc/svi-normal-guide](sbc-svi-normal-guide.md)       | passed: calibrated over 300 replications                                                           | 1.7     |
| [sbc/wrong-guide-delta](sbc-wrong-guide-delta.md)     | passed: detected: the deliberately wrong method failed calibration                                 | 0.5     |
| [sbc/wrong-guide-narrow](sbc-wrong-guide-narrow.md)   | passed: detected: the deliberately wrong method failed calibration                                 | 0.1     |
| [sbc/nuts-hierarchical](sbc-nuts-hierarchical.md)     | passed: calibrated over 120 replications                                                           | 90.1    |
| [agreement/logistic-grid](agreement-logistic-grid.md) | passed: NUTS agrees with the grid (log evidence -9.6273); Laplace and SVI are reported as they are | 5.9     |
| [funnel/non-centred](funnel-non-centred.md)           | passed: the centred model diverges, the non-centred one does not                                   | 10.6    |
| [geweke/kernels](geweke-kernels.md)                   | passed: both kernels pass                                                                          | 1.6     |
