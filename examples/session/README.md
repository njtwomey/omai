# Pause, jump back, compare with the truth

Most methods are run once and give a final answer. Here a method runs as a session: it can be run a few steps, stopped,
run on, and, because every step is recorded, asked what the model looked like at an earlier step.

## What the program does

- It draws 120 points from the two-moons data. The generator knows the truth, so the best possible accuracy is known: 0.94.
- It starts a support vector classifier (Gaussian kernel, C = 1) as a session and runs it for 3 steps of its solver. It then
  stops, and runs it on to the end. It took 45 steps in all.
- It asks the session for the model at step 3, and at the end, each as a decision surface: the score the classifier gives to
  every point of the plane.

## What to look for

- Both pictures are the classifier's decision surface, drawn the same way: the plane is shaded blue where it predicts
  class 0 and orange where it predicts class 1, and the black line is the decision boundary. The points are the training data.
- After 3 steps the classifier has not learned much: it shades a blue oval around the middle of the plane and everything
  else orange, which gets 53% of the points right, hardly better than a guess. After all 45 steps the boundary bends between
  the two moons and 89% are right.
- The gap between 0.89 and the best possible 0.94 is what noise and 120 points cost.

The tape of the run is `tape.json`. To run it again: `make readme-examples ARGS="session"`.
Source: [main.tsx](main.tsx). Figure: [figure.svg](figure.svg).
