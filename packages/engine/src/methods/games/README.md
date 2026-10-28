# Games (part 7, application domains)

Game theory computed and played: normal-form games and their equilibria (Nash by support enumeration and Lemke-Howson, correlated equilibria by linear programming, evolutionary dynamics), learning in games (fictitious play, regret matching, multiplicative weights), extensive-form games solved by counterfactual regret minimisation, and game-playing search (with `search-and-planning`), as sessions over `games`, `optim` and `agents`.

## Capabilities

- Equilibria: the best-response dynamics on a payoff matrix drawn in the simplex, the equilibrium found; options: the game (named or random), the solver.
- Learning: fictitious play and regret matching converging (or cycling) on rock-paper-scissors and matching pennies, the empirical strategies on the tape.
- Replicator dynamics: the population flowing in the simplex, the vector field as a probe.
- CFR on Kuhn poker: the regrets and the strategy at each information set, exploitability falling.
- Self-play with `reinforcement-learning` agents on small games.

## Examples to make

- Fictitious play on rock-paper-scissors: the strategies spiralling into the mixed equilibrium.
- Replicator dynamics in the simplex with the payoff matrix as knobs.
- CFR on Kuhn poker: exploitability on a log axis, the strategy at the viewed iteration.
- Lemke-Howson's path on a bimatrix game.

## Rendering

- Simplex plots with trajectories and fields, matrices as heatmaps, strategy bars per information set, exploitability curves.
- Missing: a simplex panel kind (a triangle with a trajectory and a field).

## Shared

- `shared/`: the game slot (every `games` game), the equilibrium probes.

## Problems

- New chapter; `lib/games` has the algorithms.
