# Reinforcement learning (part 5, learning paradigms)

Agents learning in environments: Markov decision processes and dynamic programming (value and policy iteration), Monte Carlo and temporal-difference learning, value-based methods (SARSA, Q-learning, DQN and its variants, the distributional ones), exploration (bandits' strategies, intrinsic rewards), policy methods (REINFORCE, A2C, TRPO, PPO, DDPG, TD3, SAC), model-based and offline methods with off-policy evaluation, imitation and inverse reinforcement learning, hierarchical and safe agents, multi-agent learning and the reliable evaluation of all of them, as sessions over `rl`, `agents`, `environments`, `worlds`, `nn` and `games`. The agents and the environments are in the library already, with a tagged registry of 52 environments; the methods here are thin, and the chapter's value is in the probes and the pages.

## Structure

Everything here is an agent acting in an environment, so it lives under `act/` (section 3 of `methods/README.md`): `tabular/` for the agents that keep a value per state, `deep/` for the networks, `bandits/` for the one-state case, `planning/` for agents that act from a model, and `control/` for the classical controllers, which are `control-theory`'s and get one row here. The studies that compare agents across seeds and tasks are `study/`. Every method of the chapter is the same session: an environment from the registry, an agent from the agent slot, a schedule of episodes, the standard probes; the folders differ in which agents the slot admits and which probes make sense.

```
act/
  tabular/
    dynamic-programming/   value and policy iteration sweep by sweep, exact policy evaluation, the Bellman backup of one state
    monte-carlo/           Monte Carlo prediction and control, first-visit and every-visit, exploring starts
    temporal-difference/   TD(0) and TD(λ) prediction; SARSA, expected SARSA, n-step SARSA, SARSA(λ) with its traces, Q-learning, double Q-learning
    dyna/                  Dyna-Q and Dyna-Q+, prioritised sweeping: planning steps from a learnt model
    approximation/         state aggregation and tile coding with semi-gradient TD, the deadly triad
  deep/
    value-based/           DQN with its variants as knobs; C51, QR-DQN, IQN and Rainbow
    policy-gradient/       REINFORCE, A2C, TRPO, PPO
    actor-critic/          DDPG, TD3, SAC
    model-based/           PETS, MBPO, Dreamer, MuZero; the learnt model against the truth
    exploration/           intrinsic rewards (counts, RND, ICM) on the hard-exploration chains and mazes
    offline/               behaviour cloning, CQL, BCQ, IQL, TD3+BC, decision transformers; off-policy evaluation
    imitation/             DAgger, GAIL, maximum-entropy inverse reinforcement learning
    hierarchical/          options, SMDP and intra-option Q-learning, option-critic; goal-conditioned agents
    safe/                  constrained MDPs by Lagrangian PPO, shields
    multi-agent/           independent learners, minimax-Q, self-play, neural fictitious self-play, VDN, QMIX, MADDPG
  bandits/                 the bandit agents on the bandit environments, regret against the Lai-Robbins bound; contextual bandits
  planning/                MCTS on an environment used as its model, CEM and MPPI planners over a differentiable model
  control/                 see control-theory (`act/control/`)
  shared/                  the agent base, the environment and agent slots, the tabular readers, the fixed-state probe (below)
study/
  reliable-evaluation/     rliable's aggregates and intervals over runs and tasks; paired comparisons; hyperparameter sweeps
```

The game-tree searches and AlphaZero are `games` (`search/games/`) and `search-and-planning` (`search/adversarial/`); the equilibria of multi-agent learning are checked against `games`.

## Methods

Tags on tracks: `s` a per-step scalar, `t` a per-step tensor at summary detail, `t!` a tensor kept in full, `p` a probe computed from the state at any step, `proj` a projection over the tape, `e` an event, `m` a message with a payload. The step of a learning agent is one environment step, with `episode/end` as the event that carries the return; the step of a planner is one sweep, simulation or plan. Status is about the method alone; what the library has is in the needs column.

### tabular/dynamic-programming/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `value-iteration` | One sweep per step on a grid world: the values spreading back from the goal, the greedy policy's arrows settling long before the values converge (the policy is right once the ordering is), the contraction at rate γ (the knob) visible as the log of the change; the Bellman backup of one chosen state drawn as a probe (its successors, their values, the max). Synchronous against in-place (Gauss-Seidel) sweeps as the mode; the comparison is with `policy-iteration` on the same model. | `sweep/delta` s, `sweep/values` t!, `sweep/policy` t!, `sweep/changed` s, `model/valueGrid` p (a `valueGrids` record over the world), `model/policyGrid` p, `state/backup` p | `rl/mdp.ts`: `valueIterationAlgorithm`, `qFromV`, `greedyPolicy`, `mdpFromModel`, `mdpFromEnv`; `agents/planning.ts`: `valueIterationAgent`, `sweepingPlannerAgent` (sweeps between episodes, so planning sits on a learning curve); `environments/gridworlds.ts`: `frozenLake`, `cliffWalking`, `gridWorld`; synchronous ready, in-place TODO (a sweep order in `rl/mdp.ts`) | |
| `policy-iteration` | Evaluate (exactly by a linear solve, or iteratively with a sweep budget as the knob) then improve: a handful of outer steps, each a policy that is never worse; against value iteration's many cheap sweeps on one model; modified policy iteration between them. | `iteration/policy` t!, `iteration/values` t!, `iteration/changedActions` s, `iteration/evaluationSweeps` s, `model/valueGrid` p, `model/policyGrid` p | `rl/mdp.ts`: `policyIterationAlgorithm`, `policyEvaluation`; `agents/planning.ts`: `policyIterationAgent` | |

### tabular/monte-carlo/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `monte-carlo-control` | Learning from complete episodes only: Blackjack's value surface (usable ace or not) forming as episodes accumulate, first-visit against every-visit as the mode, exploring starts against ε-soft policies as the knob; the returns' variance against TD's bias is the comparison with `td-control`. Off-policy Monte Carlo with importance sampling as the third mode: ordinary against weighted, the weights' variance. | `episode/return` s, `episode/length` s, `model/qTable` t!, `model/visits` t!, `model/valueGrid` p (the Blackjack surface), `model/policyGrid` p | `agents/tabular.ts`: `monteCarloControl`, `qTable`, `valueTable`, `greedyTable`; `environments/toy-text.ts`: `blackjack`; `rl/training.ts`: `trainingAlgorithm`; on-policy ready, off-policy by importance sampling TODO (`agents/tabular.ts`: `offPolicyMonteCarlo`) | |

### tabular/temporal-difference/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `td-prediction` | A fixed policy's values learnt as TD(0) and TD(λ) (λ as the knob, the trace kind as a mode: accumulating, replacing, dutch), Monte Carlo being λ = 1: on the random walk the RMS error against the exact values (from the linear solve) by λ and α, the U-shaped curve of Sutton and Barto; the traces drawn over the states at the viewed step. | `episode/rmsError` s, `step/tdError` s, `step/traces` t!, `model/values` t!, `truth/values` p, `study/lambdaCurve` proj | `agents/tabular.ts`: `td0`, `tdLambda`; `rl/mdp.ts`: `policyEvaluation` (the truth); `rl/returns.ts`: `tdErrors`, `lambdaReturns`, `nStepReturns`; the 19-state random walk TODO (`environments/toy-text.ts`: `randomWalk`) | |
| `td-control` | The learner as the mode (SARSA, expected SARSA, n-step SARSA, SARSA(λ) with its trace kind, Q-learning, double Q-learning) with α, ε and γ as knobs: cliff walking's two paths (SARSA's safe one, Q-learning's edge), the Q-table updating one step at a time, the TD error, the maximisation bias double Q-learning removes; windy grid world with king's moves. The comparison is two learners on one world under one cursor. | `episode/return` s, `episode/length` s, `step/tdError` s, `step/epsilon` s, `step/traces` t (SARSA(λ)), `model/qTable` t!, `model/valueGrid` p, `model/policyGrid` p, `eval/episode` p (a greedy episode rendered) | `agents/tabular.ts`: `qLearning`, `sarsa`, `expectedSarsa`, `nStepSarsa`, `sarsaLambda` (`TraceKind`), `doubleQLearning`, `qTable`, `greedyTable`; `environments/gridworlds.ts`: `cliffWalking`, `cliffWalkingSlippery`, `windyGridWorld`; `rl/training.ts`: `trainingAlgorithm`, `replayEpisode`; all modes ready | |

### tabular/dyna/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `dyna` | Planning from a learnt model: Dyna-Q with n planning updates per real step (the knob, from 0 up: the episodes to solve a maze falling), Dyna-Q+'s exploration bonus when the maze changes (a wall opening: the event, and Dyna-Q failing to notice), prioritised sweeping's queue of backups by priority as the mode; the model's learnt transitions as a board beside the true ones. | `episode/return` s, `episode/length` s, `step/planningUpdates` s, `queue/size` s (prioritised sweeping), `model/transitions` t!, `model/valueGrid` p, `world/changed` e | `agents/tabular.ts`: `dynaQ`, `prioritizedSweeping`; the blocking and shortcut mazes TODO (`environments/gridworlds.ts`: `changingMaze` over `worlds/rules.ts`) | |

### tabular/approximation/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `linear-td` | Function approximation on a continuous state: state aggregation and tile coding (tiles and tilings as knobs) with semi-gradient SARSA on mountain car, the cost-to-go surface forming over position and velocity; the deadly triad: off-policy learning with bootstrapping and linear features diverging on Baird's counterexample (the weights on a log axis), fixed by on-policy data or by gradient TD as the mode. | `episode/return` s, `step/tdError` s, `model/weights` t!, `model/costToGo` p (a field over the state plane), `model/weightNorm` s | `agents/_tabular.ts`: `tabularCoding` (state aggregation of a continuous observation; private, to expose); `environments/classic-control.ts`: `mountainCar`; tile coding and the linear semi-gradient learners TODO (`agents/approximation.ts`: `tileCoding`, `semiGradientSarsa`, `semiGradientTd`, `gradientTd`); Baird's counterexample TODO (`environments/toy-text.ts`: `bairdCounterexample`) | |

### deep/value-based/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `dqn` | The variants as knobs on one agent (double Q-learning, duelling heads, prioritised replay, n-step returns, noisy networks), the target network's sync period, ε's schedule: the loss, the Q-values of a fixed set of states rising over training (Mnih's figure), the evaluation return, the replay buffer's age; on cart-pole and the MinAtar games. Rainbow is every knob on. | `training/loss` s, `training/qFixed` t! (the fixed states' values), `training/epsilon` s, `training/targetSync` e, `episode/return` s, `eval/return` s, `buffer/size` s, `buffer/priorities` t (prioritised), `model/params` p, `eval/episode` p | `agents/value-based.ts`: `dqn`, `qNetwork`, `qValues`; `rl/buffers.ts`: `replayBuffer`, `prioritizedReplay`, `nStepBuffer`; `rl/training.ts`: `trainingAlgorithm`, `replayEpisode`; `environments/classic-control.ts`: `cartPole`; `environments/minatar.ts`: `breakoutMinAtar`, `freewayMinAtar`; `rl/observation-wrappers.ts`: `frameStack` | |
| `distributional` | C51 (atoms and the categorical projection of the Bellman target), QR-DQN (quantiles by the quantile Huber loss), IQN (sampled fractions, with a risk distortion as the knob: a risk-averse policy choosing the safer action) and Rainbow as modes: the return distribution of a chosen state drawn at every step, its mean against DQN's point estimate, the loss. | `training/loss` s, `training/distribution` t! (one state's atoms or quantiles), `episode/return` s, `eval/return` s, `model/returnDistribution` p, `model/params` p | `agents/distributional.ts`: `c51`, `qrDqn`, `iqn`, `rainbow`, `categoricalDistribution`, `quantileDistribution`, `implicitQuantiles`, `riskDistortion`, `categoricalProjection`; all modes ready | |

### deep/policy-gradient/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `policy-gradient` | REINFORCE (with and without a baseline: the gradient's variance as the picture), A2C (short rollouts, GAE's λ as the knob), TRPO (the KL trust region, the natural gradient step) and PPO (clipped or KL-penalised as the knob, the clip fraction and the approximate KL per update) as modes: the policy's entropy falling, the advantage estimates, the explained variance of the critic, the return; the policy's action probabilities on a fixed state set. | `training/policyLoss` s, `training/valueLoss` s, `training/entropy` s, `training/kl` s, `training/clipFraction` s, `training/explainedVariance` s, `training/gradNorm` s, `episode/return` s, `eval/return` s, `model/actionProbabilities` t! (fixed states), `model/params` p | `agents/policy-gradient.ts`: `reinforce`, `a2c`, `trpo`, `ppo`, `ppoClipObjective`, `ppoKlObjective`, `adaptKlCoefficient`, `trpoStep`; `rl/returns.ts`: `gae`, `discountedReturns`; `rl/buffers.ts`: `rolloutBuffer`, `rolloutAdvantages`; `agents/_on-policy.ts`: `explainedVariance`, `policyKl`; all modes ready | |

### deep/actor-critic/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `off-policy-actor-critic` | DDPG, TD3 and SAC as modes on the pendulum and the reacher: the critic's estimate of a fixed state's value against the Monte Carlo return actually obtained (DDPG's overestimation, TD3's twin critics removing it), target policy smoothing and the delayed actor as knobs, SAC's entropy temperature tuning itself to the target entropy; the exploration noise (Gaussian, Ornstein-Uhlenbeck) as the other knob; `sacDiscrete` on cart-pole for the comparison with DQN. | `training/criticLoss` s, `training/actorLoss` s, `training/alpha` s (SAC), `training/qEstimate` s, `training/qTrue` s, `episode/return` s, `eval/return` s, `model/params` p, `eval/episode` p | `agents/actor-critic.ts`: `ddpg`, `td3`, `sac`, `sacDiscrete`; `rl/policies.ts`: `gaussianPolicy`, `deterministicPolicy`, `ornsteinUhlenbeckNoise`, `gaussianNoise`; `environments/classic-control.ts`: `pendulum`; `environments/continuous-control.ts`: `reacher2d`, `pointMass`; all modes ready | |

### deep/model-based/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `model-based` | PETS (an ensemble of probabilistic networks, CEM planning over sampled trajectories), MBPO (short rollouts of the ensemble feeding SAC; the rollout length as the knob), Dreamer (a latent model, actor and critic trained in imagination) and MuZero (search over a learnt hidden-state model) as modes: the model's one-step error against the true dynamics, its uncertainty split into epistemic and aleatoric, imagined rollouts drawn beside real ones from the same state (the compounding error), the planner's chosen sequence; sample efficiency against the model-free agent on the same environment. | `model/loss` s, `model/stepError` s, `model/epistemic` s, `model/aleatoric` s, `plan/horizon` t!, `imagination/rollout` p (against the truth), `episode/return` s, `eval/return` s | `agents/model-based.ts`: `pets`, `mbpo`, `dreamer`, `muZero`; `rl/models.ts`: `ensemblePredict`, `ensembleSample`, `modelEnv`, `modelTargets`; `environments/models.ts`: `modelStepError`, `rolloutModel`; `rl/planning.ts`: `cemPlannerAlgorithm`; PETS and MBPO ready, Dreamer and MuZero ready as mechanics (they do not yet solve a task within a test budget) | |

### deep/exploration/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `intrinsic-reward` | The bonus module as the mode (visit counts with MBIE-EB, SimHash counts for vectors, random network distillation, the intrinsic curiosity module) wrapping any agent of the slot on the chains where ε-greedy fails (NChain, RiverSwim) and the key maze: the visit counts as a board, the bonus decaying as states become familiar, the extrinsic return against the same agent without the bonus; the bonus scale as the knob. | `episode/return` s, `episode/intrinsic` s, `step/bonus` s, `step/noveltyLoss` s (RND, ICM), `model/visitCounts` t!, `model/visitGrid` p, `model/valueGrid` p | `agents/exploration.ts`: `intrinsicReward`, `countBonusModule`, `rndModule`, `icmModule`; `rl/exploration.ts`: `countTable`, `countBonus`, `pseudoCount`, `simHash`, `rndBonus`, `curiosityBonus`; `environments/toy-text.ts`: `nChain`, `riverSwim`; `environments/gridworlds.ts`: `keyMaze`; all modes ready | |

### deep/offline/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `offline-rl` | The learner as the mode (behaviour cloning, offline DQN, CQL, BCQ, IQL, TD3+BC, the decision transformer) on a logged dataset whose quality is the knob (expert, medium, random, a mixture): offline DQN's Q-values climbing on actions the data never took against CQL's pushed-down ones, BCQ's behaviour model's support, IQL's expectile, the decision transformer's target return; the evaluated return against the behaviour policy's and the optimum (known for a tabular MDP). | `training/loss` s, `training/qUnsupported` s, `training/conservativePenalty` s (CQL), `eval/return` s, `data/behaviourReturn` s, `truth/optimalReturn` s, `model/params` p, `eval/episode` p | `agents/offline.ts`: `collectDataset`, `datasetEpisodes`, `offlineAlgorithm`, `offlineAgent`, `behaviourCloning`; `agents/offline-q-learning.ts`: `offlineDqn`, `cql`, `bcq`; `agents/offline-actor-critic.ts`: `iql`, `td3bc`; `agents/decision-transformer.ts`: `decisionTransformer`, `evaluateInContext`; `datasets/trajectories.ts`: `offlineMdp`, `policyEvaluation` (the truth); all modes ready | |
| `off-policy-evaluation` | Estimating a target policy's value from logged data without running it: inverse propensity scoring, its self-normalised and clipped forms, the direct method and the doubly robust estimator on logged bandit data (the logging policy's distance from the target as the knob: IPS's variance exploding, DR's holding), and fitted Q evaluation on episodes (tabular exactly, or with any regressor) one iteration per step; every estimate against the exact value the generator knows. The contextual-bandit side is shared with `online-experimentation`. | `estimate/value` s, `estimate/error` s, `estimate/weightsEss` s, `estimate/maxWeight` s, `fqe/iteration` s, `fqe/values` t!, `truth/value` s | `rl/off-policy.ts`: `ips`, `snips`, `clippedIps`, `directMethod`, `doublyRobust`, `importanceWeights`, `fittedQEvaluationAlgorithm`, `tabularFqeAlgorithm`; `datasets/trajectories.ts`: `loggedBandit`, `offlineMdp`, `policyEvaluation` | |

### deep/imitation/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `imitation` | Behaviour cloning's compounding error (the learner drifting into states the expert never showed) against DAgger (the expert labelling the learner's own states, one round per step, the mixing β as the knob) and GAIL (a discriminator telling the learner's transitions from the expert's, its accuracy on the tape, the policy trained against it) as modes; the expert is a solved MDP or a trained agent from the slot. | `round/expertLabels` s, `round/return` s, `training/discriminatorAccuracy` s (GAIL), `training/policyLoss` s, `eval/return` s, `expert/return` s, `eval/episode` p | `agents/imitation.ts`: `daggerAlgorithm`, `dagger`, `gail`; `agents/offline.ts`: `behaviourCloning`; `agents/_learner.ts`: `Demonstration`; the expert from `rl/mdp.ts`: `valueIteration` or a trained agent; all modes ready | |
| `inverse-rl` | Maximum-entropy IRL on a grid world: the reward map recovered one gradient step per step (the demonstrations' feature expectations against the policy's state-visitation frequencies, their difference being the gradient), the soft value iteration inside each step, the recovered reward against the true one and the policy it induces; reward learning from trajectory preferences (Bradley-Terry over pairs) as the other mode. | `iteration/gradientNorm` s, `iteration/reward` t!, `iteration/visitation` t!, `model/rewardGrid` p, `truth/rewardGrid` p, `model/policyGrid` p | `agents/imitation.ts`: `maxEntIrlAlgorithm`, `softValueIteration`, `stateVisitationFrequencies`, `featureExpectations`; preferences TODO (`agents/imitation.ts`: `preferenceReward`, a Bradley-Terry fit over trajectory pairs) | |

### deep/hierarchical/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `options` | Four rooms: the hallway options given (each a policy, an initiation set and a termination) with SMDP Q-learning over options against primitive actions (the episodes to the goal; when the goal moves, the knob, the options transfer), intra-option learning updating every option consistent with the step; option-critic learning the options and their terminations from reward alone, the termination map of each option as a probe. | `episode/return` s, `episode/length` s, `episode/optionsUsed` s, `model/optionValues` t!, `model/terminationGrid` p (per option), `model/optionPolicyGrid` p, `model/valueGrid` p | `agents/hierarchical.ts`: `hallwayOptions`, `subgoalOption`, `primitiveOptions`, `smdpQLearning`, `intraOptionQLearning`, `optionCritic`, `optionTerminations`, `optionValues`; `environments/gridworlds.ts`: `fourRooms`; all modes ready | |
| `goal-conditioned` | A universal value function over (state, goal) with hindsight experience replay: failed episodes relabelled with the goal they did reach (the relabelling strategy as the knob), the success rate on a sparse-reward reaching task climbing where the plain agent's stays flat. | `episode/success` s, `buffer/relabelled` s, `model/goalValueGrid` p (for a chosen goal) | TODO `agents/goal-conditioned.ts`: `her` (relabelling over `rl/buffers.ts`: `replayBuffer`), a goal-conditioned agent over `agents/value-based.ts`: `dqn` or `agents/actor-critic.ts`: `td3`; a reaching task from `environments/continuous-control.ts`: `pointMass` | |

### deep/safe/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `constrained-rl` | A cost added to every step of an environment (a lava region, a speed limit), the limit as the knob: Lagrangian PPO's multiplier rising until the expected cost sits under the limit, the return it gives up; shields as the other mode (a safety predicate, or a model lookahead) counting the actions overridden; the plain agent's cost beside both. | `episode/return` s, `episode/cost` s, `training/lagrangeMultiplier` s, `shield/overrides` s, `eval/costViolations` s, `eval/episode` p | `agents/safe.ts`: `costWrapper`, `transitionCost`, `lagrangianPpo`, `predicateShield`, `modelShield`, `shieldAction`, `shieldedEnv`; all modes ready | |

### deep/multi-agent/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `competitive-learning` | Independent Q-learners on a repeated matrix game (cycling, never settling: the joint strategy's path), minimax-Q on a zero-sum Markov game (the stage games solved exactly, the value converging), self-play against a pool of past snapshots (the pool's size as the knob), neural fictitious self-play on Kuhn poker (the average policies' exploitability falling, checked with `games`) as modes. | `episode/returns` t (per agent), `policy/profile` t!, `policy/exploitability` s, `policy/nashConv` s, `pool/size` s, `eval/episode` p | `agents/multi-agent.ts`: `independentLearners`, `minimaxQ`, `minimaxQStrategy`, `selfPlayAlgorithm`, `fictitiousSelfPlayAlgorithm`; `environments/multi-agent.ts`: `matrixGame`, `MATRIX_GAMES`, `pursuitEvasion`; `environments/games.ts`: `kuhnPoker`; `games/extensive-form/best-response.ts`: `exploitability`; `games/normal-form/game.ts`: `nashConv`; `rl/multi-agent.ts`: `playParallel`, `playAec`; all modes ready | |
| `cooperative-learning` | Centralised training, decentralised execution on cooperative navigation and pursuit-evasion: VDN's summed values against QMIX's monotonic mixing (a task where the sum is not enough: the knob), MADDPG's centralised critics; the team return, each agent's contribution, the coverage of the targets as a probe. | `episode/teamReturn` s, `episode/coverage` s, `training/mixerLoss` s, `model/agentValues` t, `eval/episode` p | `agents/cooperative.ts`: `vdn`, `qmix`, `maddpg`; `environments/multi-agent.ts`: `cooperativeNavigation`, `navigationCoverage`, `pursuitEvasion`; all modes ready | |

### bandits/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `bandit` | The agent as the mode (ε-greedy with its schedule, UCB1, KL-UCB, Thompson sampling for Bernoulli and Gaussian arms, EXP3, the gradient bandit) on the bandit slot (Bernoulli arms, the ten-armed Gaussian testbed, the adversarial bandit, the non-stationary one where sample averages lose to constant step sizes): cumulative regret against the Lai-Robbins bound on a log axis, the arms' estimates and counts, the posteriors over arms narrowing (Thompson) or the confidence indices (UCB) at the viewed pull; best-arm identification and the comparison with a fixed A/B allocation are `online-experimentation`'s. | `pull/arm` s, `pull/reward` s, `regret/cumulative` s, `regret/bound` s, `arms/estimates` t!, `arms/counts` t!, `arms/posterior` p, `arms/index` p | `agents/bandits.ts`: `epsilonGreedyBandit`, `ucb1`, `klUcb`, `thompsonBernoulli`, `thompsonGaussian`, `exp3`, `gradientBandit`, `runBandit`; `rl/bandits.ts`: `cumulativeRegret`, `laiRobbinsBound`, `ucb1Index`, `klUcbIndex`; `environments/bandits.ts`: `bernoulliBandit`, `gaussianBandit`, `adversarialBandit`, `nonStationaryBandit`; all modes ready | |
| `contextual-bandit` | LinUCB and linear Thompson sampling on the contextual linear bandit: the confidence ellipsoid around the weights shrinking, the chosen arm's upper bound, regret growing sublinearly, the exploration width α as the knob; against a non-contextual agent on the same stream. | `pull/arm` s, `regret/cumulative` s, `model/weights` t!, `model/covariance` t, `model/ellipsoid` p, `arms/upperBounds` t | `agents/bandits.ts`: `linUcb`, `linearThompson`; `environments/bandits.ts`: `contextualLinearBandit`; both modes ready | |

### planning/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `mcts` | UCT on an environment used as its own model: the tree growing one simulation per step (visit counts and values on the nodes), the exploration constant as the knob, rollouts against a value function (PUCT with priors as the mode), the root's action values converging to those of value iteration on the same model; the chosen action's visit share. The two-player version and AlphaZero are `games`. | `search/simulations` s, `search/rootValue` s, `search/depth` s, `search/visits` t!, `search/tree` p (a `tree` record), `truth/actionValues` p | `rl/planning.ts`: `mctsAlgorithm`, `mcts`, `rolloutPlanner`; `agents/planning.ts`: `mctsAgent`; `rl/mdp.ts`: `valueIteration` (the truth); both modes ready | |
| `sampling-planners` | The cross-entropy method and MPPI over a differentiable model of the plant: the sampled action sequences drawn ahead of the cart, the elite set narrowing (CEM) or the weighted average (MPPI), the planned horizon's cost, re-planning every step; the horizon and the sample count as knobs. The gradient-based planners (iLQR, MPC as a QP) are `control-theory`'s. | `plan/cost` s, `plan/samples` t, `plan/elite` t! (CEM), `plan/horizon` t!, `loop/state` t!, `episode/return` s | `rl/planning.ts`: `cemPlannerAlgorithm`, `cem`, `mppi`; `environments/models.ts`: `rolloutModel`, `sequenceCosts`, `cartPoleModel`, `pendulumModel`; `agents/control.ts`: `cemAgent`, `mpcAgent` (modes: iLQR, CEM, MPPI); both modes ready | |

### control/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `classical-control` | see control-theory (`act/control/`): LQR, the swing-up, MPC and the observers on the same plants, with `agents/control.ts`: `lqrAgent`, `swingUpAgent`, `mpcAgent` as agents of this chapter's slot | | | |

### study/reliable-evaluation/

| method | why it is interesting | tracks | needs from lib | status |
| --- | --- | --- | --- | --- |
| `reliable-evaluation` | Many seeds across several tasks as a study over the agent base: the interquartile mean against the mean and median, the optimality gap, the probability of improvement of one agent over another, each with stratified-bootstrap intervals; performance profiles; the learning curves aligned and aggregated with pointwise bands; paired comparison of two agents under common random numbers (the variance of the difference falling). | `run/return` s (per seed and task), `study/aggregates` proj, `study/intervals` proj, `study/profile` proj, `study/curves` proj | `rl/evaluation.ts`: `interquartileMean`, `aggregateScores`, `optimalityGap`, `probabilityOfImprovement`, `stratifiedBootstrap`, `performanceProfile`, `alignCurves`, `aggregateCurves`, `normalizeScores`; `rl/rollout.ts`: `compareAgents`, `evaluatePolicy` | |
| `sensitivity-sweep` | One knob of an agent (α, ε, γ, the learning rate, the buffer size) on the x axis as a `study/sweep` over the agent base: the final and the area-under-curve return per setting with intervals, click to open the run; the knob that matters and the one that does not. | `trial/setting` s, `trial/return` s, `trial/interval` t, `study/sweepCurve` proj | the agent base; `rl/evaluation.ts`: `aggregateScores`, `stratifiedBootstrap`; `learn/model-selection/search.ts` for the grid | |

## shared/

| helper | what it does | used by |
| --- | --- | --- |
| `agent.ts` | The agent base over `rl/training.ts`: `trainingAlgorithm` (episodes, `evalEvery`, `evalEpisodes`, checkpoints): the environment slot, the agent slot, the schedule; the standard paths (`episode/return`, `episode/length`, `eval/return`, the agent's own metrics as `training/<metric>`), the standard probes (`eval/episode` rendered through `rl/training.ts`: `replayEpisode` and `exposition/episodes.ts`: `episodeRecord`; `model/params`), and the per-step base for bandits (one pull per step) | every group |
| `environments.ts` | The environment slot over `environments/registry.ts`: `environmentRegistry`, `whereEnvironments` (family, action and observation kinds, episodic), with the wrappers as modifiers (`rl/wrappers.ts`: `timeLimit`, `normalizeReward`, `stickyAction`, `actionRepeat`; `rl/observation-wrappers.ts`: `frameStack`, `normalizeObservation`) and `rl/gym.ts`: `make`; shared with `control-theory` (the plants) and `search-and-planning` (the worlds) | every group |
| `agents.ts` | The agent slot over `lib/agents` as a mode table (`methods/shared/modes.ts`): every learner with its knobs, tagged by the spaces it supports (discrete or continuous actions; vector, grid or image observations) so the validity table greys an agent its environment cannot take | every group |
| `tabular.ts` | Readers of tabular state as records: `agents/tabular.ts`: `qTable`, `valueTable`, `greedyTable` laid over the world (`worlds/drawing.ts`: `toScene`) as `exposition/cells.ts`: `valueGrids` (`model/valueGrid`, `model/policyGrid`), the Bellman-backup probe of one state, the visit-count board | `tabular/*`, `exploration/`, `hierarchical/`, `imitation/inverse-rl` |
| `fixedStates.ts` | A fixed set of states sampled at the start of a run (from the first episodes or the model), kept so the Q-values, action probabilities and return distributions of the deep agents are tracked on the same states throughout | `deep/*` |
| `records.ts` | Records the generic ones do not cover: the replay buffer's snapshot (ages, priorities), the arm posteriors and indices as bars, the MCTS tree as a `tree` record, the imagined-against-real rollout pair | `deep/*`, `bandits/`, `planning/` |

## Examples to make

- Value iteration on a grid world: the values and the greedy policy's arrows sweep by sweep, γ as the knob.
- Q-learning against SARSA on cliff walking under one cursor: the two paths, the returns, the Q-tables.
- DQN on cart-pole with the Q-values of ten fixed states on the tape, the target syncs as events and an evaluation episode as a probe.
- Thompson sampling against UCB1 on a Bernoulli bandit: the posteriors over arms narrowing, regret against the Lai-Robbins bound.
- Neural fictitious self-play on Kuhn poker: exploitability on a log axis, checked with `games`' best response.
- Off-policy evaluation on logged bandit data: IPS, SNIPS and doubly robust converging (or not) on the known value as the logging policy moves away.

## Rendering

- Episodes (`episodes`, `filmstrip` of the rendered frames), agent steps (`agentSteps`), value and policy maps as `valueGrids` over the world, returns curves with bands, posteriors over arms, trees with visit counts (`tree`).
- Missing: a value-map panel that follows the cursor over a grid world (the board with the policy's arrows); a replay-buffer probe view; a return-distribution panel (atoms or quantiles of one state over training); a fixed-state Q-value panel; the comparison of two agents on one cursor.

## Problems

- The library has the agents and the environments; what is missing is small and listed per row: asynchronous sweeps, off-policy Monte Carlo, tile coding with the linear semi-gradient learners and Baird's counterexample, the changing maze, HER and a goal-conditioned agent, reward learning from preferences, the 19-state random walk.
- Cost: a deep agent at 60 steps a second on the main thread is the worker-producer problem of `methods/README.md` (section 6, problem 7); every row under `deep/` beyond cart-pole and MinAtar at toy size waits on it.
- The agent slot's validity table (which agents take which spaces) is the first use of the registry's tags beyond a selector; the tags are derived from the spaces already, the table is a lookup over them.
- Rows that belong to two chapters are split by the truth they compare with: the bandit against the Lai-Robbins bound is here, against an A/B allocation it is `online-experimentation`'s; MCTS against value iteration is here, against alpha-beta it is `games`'; LQR as an agent is `control-theory`'s.
