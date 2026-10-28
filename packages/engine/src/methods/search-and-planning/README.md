# Search and planning (a methods chapter the encyclopedia spreads over optimisation, games and reinforcement learning)

Classical AI search watched: state-space search (breadth-first, depth-first, uniform-cost, greedy, A* with heuristics), local search (hill climbing, simulated annealing, tabu), evolutionary and swarm search (genetic algorithms, genetic programming, evolution strategies, particle swarms, ant colonies), adversarial search (minimax, alpha-beta, Monte Carlo tree search), constraint satisfaction (backtracking, forward checking, arc consistency, min-conflicts) and planning (STRIPS, graph plan, partial order), as sessions over `search`, `problems`, `worlds`, `logic` and `games`.

## Capabilities

- State-space search on any `problems` problem (grids and mazes, puzzles, route finding): the frontier, the explored set and the path as a board each step; options: the algorithm, the heuristic, the tie-breaking; log nodes expanded, the frontier size, the path cost.
- Local search on a landscape or a combinatorial problem (n-queens, TSP): the current state and its neighbours, the temperature schedule as a knob.
- Populations: a genetic algorithm's population over a landscape or a program tree evolving; options: selection, crossover, mutation, population size; log fitness statistics and the best individual.
- Adversarial: minimax's tree with alpha-beta cuts on a small game, MCTS's tree growing with visit counts.
- CSP: assignments and domains pruned by arc consistency on Sudoku or map colouring; min-conflicts on n-queens.
- Planning: a plan found step by step on a blocks-world domain.

## Examples to make

- A* against Dijkstra and greedy on the same maze: the explored cells as boards side by side, the heuristic as the knob (exists for A*).
- Simulated annealing on a landscape, the temperature and the acceptance of uphill moves (exists).
- A genetic algorithm on a landscape: the population as points moving, the best fitness curve.
- MCTS on tic-tac-toe: the tree's visit counts at the viewed step.
- Arc consistency on Sudoku: domains shrinking cell by cell.

## Rendering

- Boards (`boardSteps`), population steps (`populationSteps`), trees with counts (`tree`), domains as a grid of small sets, program trees.
- Missing: a game-tree panel with cuts marked; a domains grid panel.

## Shared

- `shared/`: the problem slot (every `problems` problem and `worlds` map), the heuristic slot, the population base.

## Problems

- `search/`, `evolutionary/` and the search half of `gofai/` move here. The encyclopedia has no such chapter; its derivative-free optimisation, games and planning sections would link here.
