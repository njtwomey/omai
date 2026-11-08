# A small neural network, trained and watched

A neural network is a function with many adjustable numbers, and training adjusts them to make the loss small. This example
trains a small network to separate two interleaved half-circles (the "two moons") and shows what it has learned at four
moments of training.

## What the program does

- It draws 200 points from the two-moons data. The generator knows the truth, so the best possible accuracy is known: 0.942.
- It builds a network with two inputs, two hidden layers of 16 units and two outputs, and trains it with cross-entropy
  and Adam (learning rate 0.02) on minibatches of 20 rows for 150 epochs. One step of the session is one epoch.
- It asks the session for the network as it was at steps 0, 11, 45 and 150, and for each one it asks for the decision
  surface: the predicted probability of class 1 over the plane.

## What to look for

- On the left is the training loss. It falls from 0.56 to about 0.12 within the first 20 steps and then flattens.
- Each square is the probability of class 1 (orange) or class 0 (blue) over the plane, and the black line is where it is
  one half. At step 0 the network knows nothing and the surface is almost flat. By step 11 the line already runs between
  the two moons. By steps 45 and 150 it bends round the moons. The accuracy on the training points goes from 0.500 to
  0.925, 0.945 and 0.960.

The tape of the run is `tape.json`. To run it again: `make readme-examples ARGS="neural-network"`.
Source: [main.tsx](main.tsx). Figure: [figure.svg](figure.svg).
