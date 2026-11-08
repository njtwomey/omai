# A Gaussian process, tuned by a gradient

A Gaussian process (GP) fits a curve through noisy points and says how sure it is: a mean and a band around it. How
smooth the curve is, and how far it may stray from zero, is set by a few hyperparameters of the kernel, here an amplitude
and a length scale. Choosing them well matters, and the usual way is to maximise the marginal likelihood of the data.

## What the program does

- It draws 14 noisy points from a sine curve and fits a GP with the kernel `amplitude * rbf(length scale)`, starting from
  an amplitude of 1 and a length scale of 0.5. The noise variance is 0.03.
- It shows `grad` on the library's `negativeLogMarginalLikelihood`: the gradient with respect to the log length scale.
  For a small test problem it prints 0.1209288, and a finite difference gives the same number.
- It fits the hyperparameters with L-BFGS-B on that gradient, as a session, so each step is on the tape. It took 11 steps.
- It asks the session for the `model/posterior` record at step 0 and at the end. The render package draws both.

## What to look for

- The left panel is the GP given the 14 points with the **initial** hyperparameters. It is the posterior, not the prior: it
  has seen the data. With a length scale of 0.5 the curve can only change slowly, so it is one smooth arch that cannot reach
  the peaks and misses several points. The log marginal likelihood, which says how well the model explains the data, is -20.3.
- The right panel uses the **learned** hyperparameters: amplitude 0.42 and length scale 0.14. The curve now follows the data,
  the band is narrow near the points and wide where there are none, and the log marginal likelihood is -3.57.
- The thin orange lines are three curves drawn from each posterior.

The tape of the run is `tape.json`. To run it again: `make readme-examples ARGS="gp"`.
Source: [main.tsx](main.tsx). Figure: [figure.svg](figure.svg).
