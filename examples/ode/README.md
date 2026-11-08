# Predators and prey, from an equation

A differential equation says how something changes: given the state now, it gives the rate of change. Solving it means
following those changes forward in time. Here the state is two numbers, a prey population and a predator population:

```
prey'      =  1.0 * prey  -  0.1 * prey * predators
predators' =  0.075 * prey * predators  -  1.5 * predators
```

Prey grow by themselves and are eaten in proportion to how often they meet predators. Predators grow by eating and die off
on their own. These are the Lotka-Volterra equations.

## What the program does

- It starts with 40 prey and 9 predators and solves the equations from time 0 to 20 with `solveIvp` (the RK45 method, a
  relative tolerance of 1e-6). The solver picks its own step size, and records every step on a tape.
- It reads the populations back at 401 evenly spaced times (the solver's dense output), and the step sizes from the tape.

## How to read the figure

- Left: the two populations. Prey rise, predators follow them up a little later, prey crash, and predators fall after them.
  The cycle repeats about every 5.5 time units.
- Middle: the same solution drawn as a path, prey against predators. It is a closed loop around the balance point (red dot,
  at 20 prey and 10 predators, where both rates are zero). The equations conserve a quantity, so the solution cannot spiral
  in or out; the solver's answer stays on its loop, with the conserved quantity drifting by only about 4e-6 over 20 time units.
- Right: the step size the solver chose. It takes small steps where the populations change fast, and larger ones where they
  change slowly. It took 95 accepted steps, and 17 attempts were rejected as too large.

The tape of the solve is `tape.json`. To run it again: `make readme-examples ARGS="ode"`.
Source: [main.tsx](main.tsx). Figure: [figure.svg](figure.svg).
