# Reinforcement learning (part 5, learning paradigms)

Agents learning in environments: Markov decision processes and dynamic programming (value and policy iteration), Monte Carlo and temporal-difference learning, value-based methods (SARSA, Q-learning, DQN and its variants), exploration (bandits' strategies, bonuses, intrinsic rewards), policy methods (REINFORCE, actor-critic, PPO, SAC), model-based and offline methods, imitation and reward learning, hierarchical and goal-conditioned agents, and multi-agent learning, as sessions over `rl`, `environments`, `agents`, `nn` and `games`. Sections as subfolders following the encyclopedia.

## Capabilities

- The agent base: an environment slot (every `environments` task, grid worlds from `worlds`), an agent slot (every `agents` learner with its knobs), the schedule (episodes, steps per episode, evaluation episodes); log returns, losses, the policy's entropy; probes: the value and action maps on a grid world, an evaluation episode rendered, the replay buffer's state.
- Dynamic programming: sweeps of value and policy iteration on a tabular MDP, the Bellman backup of one state as a probe.
- Tabular learning: SARSA and Q-learning's Q-table updating, the step-size and exploration as knobs; eligibility traces drawn.
- Deep: DQN's target network and replay, the loss and the Q-values of a fixed state set over training; policy gradients' advantage estimates.
- Exploration: epsilon, UCB, Thompson, count bonuses, on a bandit and on a grid.
- Model-based: a learned model's rollouts against the truth; Dyna's planning steps.
- Imitation: behavioural cloning and DAgger from an expert policy; reward learning from preferences.
- Multi-agent: self-play on a small game (`games`), the policies' exploitability.

## Examples to make

- Value iteration on a grid world: the values and the greedy policy's arrows sweep by sweep.
- Q-learning on cliff walking against SARSA: the two paths and returns.
- DQN on cart-pole with the Q-values of ten fixed states on the tape and an evaluation episode as a probe.
- Thompson sampling on a bandit: the posteriors over arms narrowing.
- Self-play on rock-paper-scissors converging to the mixed equilibrium.

## Rendering

- Episodes (`EpisodePlayer`, `episodes`), agent steps (`agentSteps`), value and action maps as boards and fields, returns curves, posteriors over arms.
- Missing: a value-map panel kind over a grid world at the viewed step (the board with arrows); a replay-buffer probe view.

## Shared

- `shared/`: the agent base (environment, agent, schedule, standard probes), the environment slot shared with `control-theory` and `search-and-planning`.

## Problems

- `rl/` moves here with its name expanded. The agents and environments are in `lib` already; the methods are thin, and the value of the chapter is in the probes and the pages.
