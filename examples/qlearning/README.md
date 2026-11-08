# An agent learning: Q-learning on the cliff

In the cliff-walking gridworld an agent walks from a start cell to a goal along the edge of a cliff. Each step costs 1, and
stepping off the cliff costs 100 and sends the agent back to the start. Q-learning learns, by trial, how good each move is
in each cell.

## What the program does

- It trains a tabular Q-learning agent (learning rate 0.5, discount 1, 10% random moves) for 300 episodes.
- The run is recorded as a trace, and the figure is drawn from the recorded episode returns.

## What to look for

- The return of an episode is its total reward, so higher is better and -13 would be the shortest safe walk along the cliff.
- The first three episodes score -2216, -41 and -88: the agent falls off the cliff many times before it learns. The mean
  of the last 50 episodes is -53.8, so it has learned to reach the goal. It is not -13 because 10% of its moves are random, which
  near a cliff sometimes costs a fall.
- Each faint dot is one episode. The line is the rolling mean of the returns and the shaded band is one rolling standard
  deviation around it. The axis stops at -200, so the first, worst episodes are cut off.

To run it again: `make readme-examples ARGS="qlearning"`. Source: [main.tsx](main.tsx). Figure: [figure.svg](figure.svg).
