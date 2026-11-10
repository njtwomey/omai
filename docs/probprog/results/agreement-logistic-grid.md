# agreement/logistic-grid

Logistic regression with two coefficients: NUTS, the Laplace approximation and mean-field SVI against the grid posterior.

Run on 2026-11-10 at commit 23783786; seed 0, budget scale 1 (draws 1000, gridPoints 121, gridEvaluations 22083), 2.6 s.

**Passed**: NUTS agrees with the grid (log evidence -9.6273); Laplace and SVI are reported as they are

| method  | quantity | mean  | sd    | mean error (sd) | mean z | sd ratio | KS     | W1 (sd) | KL      | ESS  |
| ------- | -------- | ----- | ----- | --------------- | ------ | -------- | ------ | ------- | ------- | ---- |
| nuts    | b0       | 0.408 | 0.655 | 0.0073          | 0.265  | 1.02     | 0.0217 | 0.0273  | 0.00671 | 1360 |
| laplace | b0       | 0.347 | 0.59  | -0.0868         | -6     | 0.916    | 0.0453 | 0.0918  | 0.0152  | 4000 |
| svi     | b0       | 0.313 | 0.583 | -0.141          | -9.84  | 0.904    | 0.0669 | 0.147   | 0.0222  | 4000 |
| nuts    | b1       | 1.96  | 0.813 | -0.00881        | -0.338 | 0.975    | 0.0142 | 0.0246  | 0.0034  | 1400 |
| laplace | b1       | 1.67  | 0.758 | -0.357          | -24.8  | 0.908    | 0.126  | 0.357   | 0.0871  | 4000 |
| svi     | b1       | 1.96  | 0.756 | -0.016          | -1.11  | 0.906    | 0.0416 | 0.0962  | 0.0209  | 4000 |
