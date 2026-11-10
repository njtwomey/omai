# kernels/bimodal

The equal two-mode target: parallel tempering over a random walk recovers the one-half weight of each mode in every chain; NUTS stays in one mode per chain.

Run on 2026-11-10 at commit d30248ff; seed 0, budget scale 1 (chains 4, drawsPerChain 2000), 5.4 s.

**Passed**: parallel tempering recovers the mode weights in every chain; every NUTS chain stays in one mode

| method                                                                   | fraction of each chain in the upper mode (truth 0.5) |
| ------------------------------------------------------------------------ | ---------------------------------------------------- |
| NUTS                                                                     | 0.00, 0.00, 1.00, 1.00                               |
| parallel tempering over a random walk (6 temperatures down to beta 0.02) | 0.47, 0.50, 0.46, 0.47 (pooled 0.47)                 |
