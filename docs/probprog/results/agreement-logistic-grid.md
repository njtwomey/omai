# agreement/logistic-grid

Logistic regression with two coefficients: NUTS, the Laplace approximation and mean-field SVI against the grid posterior.

Run on 2026-11-10 at commit 4ce8b576; seed 0, budget scale 3 (draws 3000, gridPoints 121, gridEvaluations 22083), 5.9 s.

**Passed**: NUTS agrees with the grid (log evidence -9.6273); Laplace and SVI are reported as they are

| method  | quantity | mean  | sd    | mean error (sd) | mean z | sd ratio | KS      | W1 (sd) | KL      | ESS  |
| ------- | -------- | ----- | ----- | --------------- | ------ | -------- | ------- | ------- | ------- | ---- |
| nuts    | b0       | 0.404 | 0.64  | 0.000392        | 0.0242 | 0.993    | 0.0129  | 0.0167  | 0.00149 | 3760 |
| laplace | b0       | 0.347 | 0.59  | -0.0868         | -6     | 0.916    | 0.0453  | 0.0918  | 0.0152  | 4000 |
| svi     | b0       | 0.399 | 0.592 | -0.00738        | -0.507 | 0.92     | 0.0239  | 0.0673  | 0.00862 | 4000 |
| nuts    | b1       | 1.97  | 0.821 | -0.00348        | -0.229 | 0.985    | 0.00819 | 0.0143  | 0.00157 | 4210 |
| laplace | b1       | 1.67  | 0.758 | -0.357          | -24.8  | 0.908    | 0.126   | 0.357   | 0.0871  | 4000 |
| svi     | b1       | 2.01  | 0.796 | 0.0434          | 2.88   | 0.954    | 0.0593  | 0.102   | 0.0175  | 4000 |
