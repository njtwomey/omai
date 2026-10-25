/**
 * Experiment protocols for comparing RL agents, each a traced algorithm whose training runs are child runs of one
 * study. Learning-curve comparisons across tasks and seeds (the rliable protocol: interquartile means with stratified
 * bootstrap intervals, probabilities of improvement and performance profiles), hyperparameter sensitivity sweeps over a
 * search space, exploration studies that count the states each agent visits, and generalisation studies that train on
 * some procedural levels and test on held-out ones (Procgen's protocol).
 */

import { type Json, linspaceNumbers, maxOf, meanOf, minOf, WhiteboxError } from '../../lib/base/index.ts'
import { foldIn, type Key, key as makeKey } from '../../lib/random/index.ts'
import {
  type Agent,
  type AggregateMetric,
  aggregateCurves,
  aggregateScores,
  type BootstrapEstimate,
  type BootstrapOptions,
  type Env,
  type EnvDefinition,
  evaluatePolicy,
  greedyPolicyOf,
  type IntervalSeries,
  improvementEstimate,
  make,
  normalizeScores,
  performanceProfile,
  stratifiedBootstrap,
  type Transition,
  trainAgent,
} from '../../lib/rl/index.ts'
import { grid, type Space, type SpaceValue } from '../../lib/space/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm, type StepContext } from '../../lib/trace/index.ts'

/** Any environment, as studies see it. */
export type AnyEnv = Env<unknown, unknown, unknown>

/** Any agent, as studies see it. */
export type AnyAgent = Agent<unknown, unknown, unknown>

/** Builds an agent for an environment (each run of a study gets a fresh agent). */
export type AgentFactory = (env: AnyEnv) => AnyAgent

/** The training budget of every run: a number of episodes or of environment steps. */
export interface StudyBudget {
  readonly episodes?: number
  readonly maxSteps?: number
}

/** The run options a study accepts: a key or seed, a tracer and policy, or a parent run to nest under. */
export type StudyRunOptions = Partial<Pick<RunOptions, 'key' | 'seed' | 'tracer' | 'policy' | 'parent' | 'runId'>>

/** One training run of a study. */
export interface StudyRun {
  readonly agent: string
  /** The task (environment) index. */
  readonly task: number
  readonly env: string
  readonly seed: number
  readonly returns: readonly number[]
  readonly lengths: readonly number[]
  /** The run's final score (mean return of its last episodes, or of greedy evaluation episodes). */
  readonly final: number
  readonly runId: string
}

/** The state of a {@link studyAlgorithm}: the units run so far and their results. */
export interface StudyState<R> {
  readonly next: number
  readonly results: readonly R[]
}

/**
 * A study as an {@link Algorithm}: one step per unit of work (usually one training run), each run inside the step as a
 * child of the study's run, so a whole study is one trace. Unit `i` receives the study's root key; studies derive
 * training keys from the seed, not the unit, so every agent sees the same random numbers for the same seed.
 *
 * @param name - The study's name (the run is named `study:<name>`).
 * @param config - Its configuration, for the trace.
 * @param units - The units of work.
 * @param run - Runs one unit in a step (start child runs with `ctx.run`).
 * @param score - The number recorded as the `final` metric of each unit.
 * @returns The algorithm.
 * @concept studies
 * @example
 * ```ts
 * import { studyAlgorithm } from 'whitebox-engine/methods/rl'
 * import { runAlgorithm } from 'whitebox-engine/trace'
 * const alg = studyAlgorithm('squares', {}, [1, 2, 3], (u) => u * u, (r) => r)
 * runAlgorithm(alg, { maxSteps: 10 }).state.results // => [1, 4, 9]
 * ```
 */
export function studyAlgorithm<U, R>(
  name: string,
  config: Json,
  units: readonly U[],
  run: (unit: U, ctx: StepContext) => R,
  score: (result: R) => number,
): Algorithm<StudyState<R>> {
  return {
    name: `study:${name}`,
    config,
    schema: channels({ metrics: { final: { description: 'final score of the unit just run', direction: 'higher' } } }),
    init: () => ({ next: 0, results: [] }),
    step: (s, ctx) => {
      const unit = units[s.next] as U
      const r = run(unit, ctx)
      if (ctx.run.wants('metric', 'final')) ctx.run.metric(ctx.t + 1, 'final', score(r))
      return { next: s.next + 1, results: [...s.results, r] }
    },
    done: (s) => (s.next >= units.length ? 'completed' : undefined),
  }
}

function execute<U, R>(
  name: string,
  config: Json,
  units: readonly U[],
  run: (unit: U, ctx: StepContext, root: Key) => R,
  score: (r: R) => number,
  options: StudyRunOptions,
): { readonly results: readonly R[]; readonly runId: string } {
  const root = options.key ?? makeKey(options.seed ?? 0)
  const alg = studyAlgorithm(name, config, units, (u: U, ctx: StepContext) => run(u, ctx, root), score)
  const result = runAlgorithm(alg, { ...options, key: root, maxSteps: units.length })
  return { results: result.state.results, runId: result.runId }
}

const seedList = (seeds: number | readonly number[] | undefined): readonly number[] =>
  typeof seeds === 'number' ? Array.from({ length: seeds }, (_, i) => i) : (seeds ?? [0, 1, 2, 3, 4])

function budgetOf(budget: StudyBudget, name: string): StudyBudget {
  if (budget.episodes === undefined && budget.maxSteps === undefined) {
    throw new WhiteboxError('value', `${name}: give episodes or maxSteps`)
  }
  return {
    ...(budget.episodes === undefined ? {} : { episodes: budget.episodes }),
    ...(budget.maxSteps === undefined ? {} : { maxSteps: budget.maxSteps }),
  }
}

/** Options of how a run's final score is measured. */
export interface FinalScoreOptions {
  /** Average the last this many training returns (default 10). */
  readonly finalEpisodes?: number
  /** Instead, evaluate the greedy policy for this many episodes. */
  readonly evalEpisodes?: number
}

/** Train one agent on one task under one seed, as a child run, and measure its final score. */
function trainUnit(
  ctx: StepContext,
  root: Key,
  agentName: string,
  factory: AgentFactory,
  env: AnyEnv,
  task: number,
  seed: number,
  budget: StudyBudget,
  final: FinalScoreOptions,
  evalEnv: AnyEnv = env,
): StudyRun & { readonly agentState: unknown; readonly agentImpl: AnyAgent } {
  const agent = factory(env)
  // The key depends on the seed and task only: common random numbers across agents.
  const k = foldIn(foldIn(root, seed), task)
  const result = trainAgent(env, agent, { ...budget, key: k, parent: ctx.run, checkpoints: false })
  let score: number
  if (final.evalEpisodes !== undefined) {
    score = evaluatePolicy(evalEnv, greedyPolicyOf(agent, result.agent), {
      episodes: final.evalEpisodes,
      key: foldIn(k, 0x6576),
      parent: ctx.run,
    }).meanReturn
  } else {
    const n = Math.min(final.finalEpisodes ?? 10, result.returns.length)
    score = n === 0 ? Number.NaN : meanOf(result.returns.slice(result.returns.length - n))
  }
  return {
    agent: agentName,
    task,
    env: env.spec.id,
    seed,
    returns: result.returns,
    lengths: result.lengths,
    final: score,
    runId: result.runId,
    agentState: result.agent,
    agentImpl: agent,
  }
}

const strip = (r: StudyRun): StudyRun => ({
  agent: r.agent,
  task: r.task,
  env: r.env,
  seed: r.seed,
  returns: r.returns,
  lengths: r.lengths,
  final: r.final,
  runId: r.runId,
})

/** `[seeds][tasks]` final scores of one agent, normalised per task when bounds are given. */
function scoreMatrix(
  runs: readonly StudyRun[],
  agent: string,
  seeds: readonly number[],
  nTasks: number,
  bounds?: readonly (readonly [number, number])[],
): number[][] {
  const m = seeds.map((s) =>
    Array.from(
      { length: nTasks },
      (_, t) => runs.find((r) => r.agent === agent && r.seed === s && r.task === t)?.final ?? Number.NaN,
    ),
  )
  return bounds === undefined
    ? m
    : normalizeScores(
        m,
        bounds.map((b) => b[0]),
        bounds.map((b) => b[1]),
      )
}

const trailing = (xs: readonly number[], window: number): number[] =>
  xs.map((_, i) => meanOf(xs.slice(Math.max(0, i - window + 1), i + 1)))

// ---------------------------------------------------------------------------------------- Learning-curve comparisons

/** Options of {@link compareLearningCurves}. */
export interface CompareLearningCurvesOptions
  extends StudyBudget,
    FinalScoreOptions,
    BootstrapOptions,
    StudyRunOptions {
  /** The agents, by name. */
  readonly agents: Readonly<Record<string, AgentFactory>>
  /** The tasks (environments with their time limits, as from `make`). */
  readonly envs: readonly AnyEnv[]
  /** Seeds per agent and task: a count or the list (default 5). */
  readonly seeds?: number | readonly number[]
  /** Per-task `[low, high]` scores mapped to 0 and 1 (random and expert scores) before aggregating. */
  readonly normalization?: readonly (readonly [number, number])[]
  /** The curves' x axis: episodes (default with an episode budget) or environment steps. */
  readonly x?: 'episodes' | 'steps'
  /** Points of the common grid of the curves. */
  readonly points?: number
  /** Trailing mean over this many episodes before aggregating curves (default 1). */
  readonly window?: number
  /** The aggregate of curves and final scores (default `'iqm'`). */
  readonly aggregate?: AggregateMetric
  /** Thresholds of the performance profiles (default 11 points over the range of final scores). */
  readonly taus?: readonly number[]
}

/** Result of {@link compareLearningCurves}. */
export interface LearningCurveComparison {
  readonly runs: readonly StudyRun[]
  /** Final scores per agent, `[seeds][tasks]` (normalised when bounds were given). */
  readonly scores: Readonly<Record<string, number[][]>>
  /** The aggregate learning curve of each agent, with pointwise intervals. */
  readonly curves: Readonly<Record<string, IntervalSeries>>
  /** Mean, median, IQM and optimality gap of each agent's final scores, with intervals. */
  readonly aggregates: Readonly<Record<string, Partial<Record<AggregateMetric, BootstrapEstimate>>>>
  /** P(a > b) for every ordered pair of agents, keyed `'a > b'`. */
  readonly improvement: Readonly<Record<string, BootstrapEstimate>>
  /** Run-score performance profiles of each agent. */
  readonly profiles: Readonly<Record<string, IntervalSeries>>
  /** The agent with the highest aggregate final score. */
  readonly best: string
  readonly runId: string
}

/**
 * Compare agents by their learning curves (the rliable protocol): train every agent on every task under every seed
 * with {@link trainAgent} (child runs of one traced study; a seed gives every agent the same keys), then aggregate the
 * episode-return curves step by step and the final scores with stratified bootstrap intervals (IQM by default),
 * compute the probability of improvement between every pair of agents and their performance profiles.
 *
 * @param options - Agents, tasks, seeds, budget, final-score rule, normalisation, curve grid, aggregate, bootstrap and
 *   run options.
 * @returns Runs, scores, curves, aggregates, pairwise probabilities of improvement, profiles and the best agent.
 * @throws WhiteboxError with code `value` without agents, tasks or a budget.
 * @concept studies
 * @example
 * ```ts
 * import { make, policyAgent } from 'whitebox-engine/rl'
 * import { compareLearningCurves, frozenLake } from 'whitebox-engine/methods/rl'
 * const env = make(frozenLake, { isSlippery: false })
 * // The shortest path across the dry lake, and an agent that walks into the left wall.
 * const path = [1, 0, 0, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 2, 2]
 * const study = compareLearningCurves({
 *   agents: {
 *     path: () => policyAgent((o: number) => path[o] ?? 0) as never,
 *     left: () => policyAgent(() => 0) as never,
 *   },
 *   envs: [env], seeds: 3, episodes: 2, reps: 50,
 * })
 * study.best // => 'path'
 * study.improvement['path > left']?.value // => 1
 * ```
 */
export function compareLearningCurves(options: CompareLearningCurvesOptions): LearningCurveComparison {
  const names = Object.keys(options.agents)
  if (names.length === 0 || options.envs.length === 0) {
    throw new WhiteboxError('value', 'compareLearningCurves: give at least one agent and one environment')
  }
  const budget = budgetOf(options, 'compareLearningCurves')
  const seeds = seedList(options.seeds)
  const units = names.flatMap((agent) =>
    options.envs.flatMap((env, task) => seeds.map((seed) => ({ agent, env, task, seed }))),
  )
  const { results, runId } = execute(
    'compareLearningCurves',
    { agents: names, envs: options.envs.map((e) => e.spec.id), seeds: [...seeds], ...budget } as Json,
    units,
    (u, ctx, root) =>
      strip(
        trainUnit(ctx, root, u.agent, options.agents[u.agent] as AgentFactory, u.env, u.task, u.seed, budget, options),
      ),
    (r) => r.final,
    options,
  )
  const nTasks = options.envs.length
  const boot: BootstrapOptions = {
    ...(options.key === undefined ? { seed: options.seed ?? 0 } : { key: options.key }),
    ...(options.reps === undefined ? {} : { reps: options.reps }),
    ...(options.level === undefined ? {} : { level: options.level }),
    ...(options.taskBootstrap === undefined ? {} : { taskBootstrap: options.taskBootstrap }),
  }
  const aggregate = options.aggregate ?? 'iqm'
  const scores: Record<string, number[][]> = {}
  const curves: Record<string, IntervalSeries> = {}
  const aggregates: Record<string, Partial<Record<AggregateMetric, BootstrapEstimate>>> = {}
  const profiles: Record<string, IntervalSeries> = {}
  for (const name of names) scores[name] = scoreMatrix(results, name, seeds, nTasks, options.normalization)
  const all = Object.values(scores).flat(2).filter(Number.isFinite)
  const lo = all.length > 0 ? minOf(all) : 0
  const hi = all.length > 0 ? maxOf(all) : 1
  const taus = options.taus ?? linspaceNumbers(lo, hi, 11)
  const byEpisodes = (options.x ?? (budget.episodes !== undefined ? 'episodes' : 'steps')) === 'episodes'
  for (const name of names) {
    const m = scores[name] ?? []
    aggregates[name] = aggregateScores(m, boot)
    profiles[name] = performanceProfile(m, taus, boot)
    const mine = results.filter((r) => r.agent === name && r.returns.length > 0)
    const curveOf = (r: StudyRun): { steps: number[]; values: number[] } => {
      const b = options.normalization?.[r.task]
      const values = trailing(r.returns, options.window ?? 1).map((v) =>
        b === undefined ? v : (v - b[0]) / (b[1] - b[0]),
      )
      let total = 0
      const steps = byEpisodes ? values.map((_, i) => i + 1) : r.lengths.map((l) => (total += l))
      return { steps, values }
    }
    if (mine.length === results.filter((r) => r.agent === name).length && mine.length > 0) {
      curves[name] = aggregateCurves(mine.map(curveOf), {
        ...boot,
        aggregate,
        tasks: mine.map((r) => r.task),
        ...(options.points === undefined ? {} : { points: options.points }),
      })
    }
  }
  const improvement: Record<string, BootstrapEstimate> = {}
  for (const a of names) {
    for (const b of names) {
      if (a !== b) improvement[`${a} > ${b}`] = improvementEstimate(scores[a] ?? [], scores[b] ?? [], boot)
    }
  }
  let best = names[0] as string
  const rank = (n: string): number => {
    const v = aggregates[n]?.[aggregate]?.value ?? Number.NEGATIVE_INFINITY
    return aggregate === 'optimalityGap' ? -v : v
  }
  for (const n of names) if (rank(n) > rank(best)) best = n
  return { runs: results, scores, curves, aggregates, improvement, profiles, best, runId }
}

// ------------------------------------------------------------------------------------- Hyperparameter sensitivity

/** Options of {@link sensitivityStudy}. */
export interface SensitivityOptions extends StudyBudget, FinalScoreOptions, BootstrapOptions, StudyRunOptions {
  /** Builds an agent from one configuration of the space. */
  readonly agent: (config: SpaceValue, env: AnyEnv) => AnyAgent
  /** The hyperparameter space (a dict of scalar spaces), swept on a grid. */
  readonly space: Space
  /** Grid points per continuous dimension (default 3). */
  readonly points?: number
  /** Explicit configurations instead of the grid. */
  readonly configs?: readonly SpaceValue[]
  readonly envs: readonly AnyEnv[]
  readonly seeds?: number | readonly number[]
  readonly normalization?: readonly (readonly [number, number])[]
  /** The aggregate of each setting's final scores (default `'iqm'`). */
  readonly aggregate?: AggregateMetric
}

/** One setting of a {@link sensitivityStudy}. */
export interface SensitivitySetting {
  readonly config: SpaceValue
  /** `[seeds][tasks]` final scores. */
  readonly scores: number[][]
  /** The aggregate final score with its stratified bootstrap interval. */
  readonly estimate: BootstrapEstimate
}

/** Result of {@link sensitivityStudy}. */
export interface SensitivityResult {
  readonly settings: readonly SensitivitySetting[]
  /** The setting with the best aggregate score. */
  readonly best: SensitivitySetting
  readonly runs: readonly (StudyRun & { readonly setting: number })[]
  readonly runId: string
}

/**
 * Hyperparameter sensitivity: sweep an agent factory over a grid of a {@link Space} (`grid(space, points)`, or given
 * configurations), train each setting on every task under every seed, and summarise each setting's final performance
 * by an aggregate (IQM by default) with a stratified bootstrap interval, so flat and sharp regions of the
 * hyperparameter landscape are visible with their uncertainty.
 *
 * @param options - The agent factory, space and grid, tasks, seeds, budget, final-score rule, aggregate and bootstrap
 *   and run options.
 * @returns Every setting with its scores and estimate, the best setting and the runs.
 * @concept studies
 * @example
 * ```ts
 * import { make, policyAgent } from 'whitebox-engine/rl'
 * import { frozenLake, sensitivityStudy } from 'whitebox-engine/methods/rl'
 * import { dict, int } from 'whitebox-engine/space'
 * const env = make(frozenLake, { isSlippery: false })
 * const r = sensitivityStudy({
 *   agent: (c) => policyAgent(() => (c as { a: number }).a) as never,
 *   space: dict({ a: int(0, 3) }), points: 4, envs: [env], seeds: 2, episodes: 1, reps: 20,
 * })
 * r.settings.length // => 4
 * ```
 */
export function sensitivityStudy(options: SensitivityOptions): SensitivityResult {
  const budget = budgetOf(options, 'sensitivityStudy')
  const configs = options.configs ?? grid(options.space, options.points ?? 3)
  if (configs.length === 0) throw new WhiteboxError('value', 'sensitivityStudy: no configurations')
  const seeds = seedList(options.seeds)
  const units = configs.flatMap((config, setting) =>
    options.envs.flatMap((env, task) => seeds.map((seed) => ({ config, setting, env, task, seed }))),
  )
  const { results, runId } = execute(
    'sensitivity',
    { configs: configs as Json, envs: options.envs.map((e) => e.spec.id), seeds: [...seeds], ...budget } as Json,
    units,
    (u, ctx, root) => ({
      ...strip(
        trainUnit(
          ctx,
          root,
          `setting ${u.setting}`,
          (env) => options.agent(u.config, env),
          u.env,
          u.task,
          u.seed,
          budget,
          options,
        ),
      ),
      setting: u.setting,
    }),
    (r) => r.final,
    options,
  )
  const aggregate = options.aggregate ?? 'iqm'
  const boot: BootstrapOptions = {
    ...(options.key === undefined ? { seed: options.seed ?? 0 } : { key: options.key }),
    ...(options.reps === undefined ? {} : { reps: options.reps }),
    ...(options.level === undefined ? {} : { level: options.level }),
  }
  const settings = configs.map((config, i): SensitivitySetting => {
    const scores = scoreMatrix(
      results.filter((r) => r.setting === i),
      `setting ${i}`,
      seeds,
      options.envs.length,
      options.normalization,
    )
    return {
      config,
      scores,
      estimate: aggregateScores(scores, { ...boot, metrics: [aggregate] })[aggregate] as BootstrapEstimate,
    }
  })
  const better = (a: SensitivitySetting, b: SensitivitySetting): boolean =>
    aggregate === 'optimalityGap' ? a.estimate.value < b.estimate.value : a.estimate.value > b.estimate.value
  let best = settings[0] as SensitivitySetting
  for (const s of settings) if (better(s, best)) best = s
  return { settings, best, runs: results, runId }
}

// ------------------------------------------------------------------------------------------- Exploration studies

/** The state of a {@link visitCountingAgent}: the wrapped agent's state and what it has visited. */
export interface VisitCountingState<G> {
  readonly inner: G
  /** Visits per state index. */
  readonly visits: readonly number[]
  /** The step of the first visit to each state (-1 if never). */
  readonly firstVisit: readonly number[]
  /** Transitions learned from so far. */
  readonly steps: number
  /** The step at which the goal was first reached (-1 if never). */
  readonly goalAt: number
  readonly episodeStart: boolean
}

/**
 * Wrap an agent so that its state also counts the states it visits (from the transitions it learns from: the first
 * observation of every episode and every next observation) and records the first step at which a goal transition
 * occurred. Acting and learning are unchanged.
 *
 * @param agent - The agent.
 * @param nStates - The number of states.
 * @param index - The state index of an observation (default the observation itself, for discrete observations).
 * @param isGoal - Whether a transition reaches the goal (default: it terminates with positive reward).
 * @returns The counting agent.
 * @concept exploration
 * @example
 * ```ts
 * import { policyAgent } from 'whitebox-engine/rl'
 * import { visitCountingAgent } from 'whitebox-engine/methods/rl'
 * import { key } from 'whitebox-engine/random'
 * const a = visitCountingAgent(policyAgent(() => 0), 3)
 * const t = { observation: 0, action: 0, reward: 1, nextObservation: 2, terminated: true, truncated: false }
 * a.learn(a.init(key(0)), t, key(1)).visits // => [1, 0, 1]
 * ```
 */
export function visitCountingAgent<G, O, A>(
  agent: Agent<G, O, A>,
  nStates: number,
  index: (observation: O) => number = (o) => o as unknown as number,
  isGoal: (transition: Transition<O, A>) => boolean = (t) => t.terminated && t.reward > 0,
): Agent<VisitCountingState<G>, O, A> {
  return {
    name: agent.name,
    config: agent.config,
    ...(agent.metrics === undefined ? {} : { metrics: agent.metrics }),
    init: (k) => ({
      inner: agent.init(k),
      visits: new Array<number>(nStates).fill(0),
      firstVisit: new Array<number>(nStates).fill(-1),
      steps: 0,
      goalAt: -1,
      episodeStart: true,
    }),
    act: (s, o, k) => agent.act(s.inner, o, k),
    greedy: (s, o) => agent.greedy(s.inner, o),
    learn: (s, t, k) => {
      const visits = [...s.visits]
      const firstVisit = [...s.firstVisit]
      const visit = (o: O, step: number): void => {
        const i = index(o)
        if (i < 0 || i >= nStates) return
        visits[i] = (visits[i] ?? 0) + 1
        if (firstVisit[i] === -1) firstVisit[i] = step
      }
      if (s.episodeStart) visit(t.observation, s.steps)
      visit(t.nextObservation, s.steps + 1)
      return {
        inner: agent.learn(s.inner, t, k),
        visits,
        firstVisit,
        steps: s.steps + 1,
        goalAt: s.goalAt === -1 && isGoal(t) ? s.steps + 1 : s.goalAt,
        episodeStart: t.terminated || t.truncated,
      }
    },
    ...(agent.stats === undefined
      ? {}
      : { stats: (s: VisitCountingState<G>) => (agent.stats as (g: G) => Readonly<Record<string, number>>)(s.inner) }),
  }
}

/** Options of {@link explorationStudy}. */
export interface ExplorationOptions extends BootstrapOptions, StudyRunOptions {
  readonly agents: Readonly<Record<string, AgentFactory>>
  /** The environment (a grid world with discrete observations, or give `stateIndex` and `nStates`). */
  readonly env: AnyEnv
  /** Environment steps per run. */
  readonly maxSteps: number
  readonly seeds?: number | readonly number[]
  /** The number of states (default the size of a discrete observation space). */
  readonly nStates?: number
  /** The state index of an observation (default the observation). */
  readonly stateIndex?: (observation: unknown) => number
  /** Whether a transition reaches the goal (default: terminated with positive reward). */
  readonly isGoal?: (transition: Transition<unknown, unknown>) => boolean
  /** Points of the coverage curves (default 21). */
  readonly points?: number
}

/** The exploration summary of one agent. */
export interface ExplorationSummary {
  /** The mean fraction of states visited by each step, with pointwise intervals. */
  readonly coverage: IntervalSeries
  /** The fraction of states visited by the end of the run. */
  readonly finalCoverage: BootstrapEstimate
  /** Steps to first reach the goal, censored at the budget for runs that never do (mean, with an interval). */
  readonly timeToGoal: BootstrapEstimate
  /** The fraction of runs that reached the goal. */
  readonly reached: number
  /** Mean visits per state over the runs (a state-visitation heat map). */
  readonly visits: readonly number[]
}

/** Result of {@link explorationStudy}. */
export interface ExplorationResult {
  readonly agents: Readonly<Record<string, ExplorationSummary>>
  readonly runs: readonly {
    readonly agent: string
    readonly seed: number
    readonly visits: readonly number[]
    readonly firstVisit: readonly number[]
    readonly goalAt: number
    readonly runId: string
  }[]
  readonly runId: string
}

/**
 * An exploration study: train each agent for a fixed number of steps under several seeds while counting the states
 * its training transitions visit ({@link visitCountingAgent}); report state-visitation coverage over time, final
 * coverage, the time to first reach the goal (with bootstrap intervals over seeds) and mean visit counts per state.
 *
 * @param options - Agents, environment, step budget, seeds, state indexing, goal test and bootstrap and run options.
 * @returns Per-agent summaries and the runs.
 * @throws WhiteboxError with code `value` when the number of states is unknown.
 * @concept exploration
 * @example
 * ```ts
 * import { make, policyAgent } from 'whitebox-engine/rl'
 * import { explorationStudy, frozenLake } from 'whitebox-engine/methods/rl'
 * const env = make(frozenLake, { isSlippery: false })
 * const r = explorationStudy({ agents: { left: () => policyAgent(() => 0) as never }, env, maxSteps: 20, seeds: 2, reps: 20 })
 * r.agents['left']?.finalCoverage.value // => 0.0625
 * ```
 */
export function explorationStudy(options: ExplorationOptions): ExplorationResult {
  const space = options.env.observationSpace
  const nStates = options.nStates ?? (space.kind === 'discrete' ? space.n : undefined)
  if (nStates === undefined)
    throw new WhiteboxError('value', 'explorationStudy: give nStates for this observation space')
  const names = Object.keys(options.agents)
  const seeds = seedList(options.seeds)
  const units = names.flatMap((agent) => seeds.map((seed) => ({ agent, seed })))
  const index = options.stateIndex ?? ((o: unknown) => o as number)
  const { results, runId } = execute(
    'exploration',
    { agents: names, env: options.env.spec.id, seeds: [...seeds], maxSteps: options.maxSteps } as Json,
    units,
    (u, ctx, root) => {
      const agent = visitCountingAgent(
        (options.agents[u.agent] as AgentFactory)(options.env),
        nStates,
        index,
        options.isGoal,
      )
      const r = trainAgent(options.env, agent, {
        maxSteps: options.maxSteps,
        key: foldIn(root, u.seed),
        parent: ctx.run,
        checkpoints: false,
      })
      return {
        agent: u.agent,
        seed: u.seed,
        visits: r.agent.visits,
        firstVisit: r.agent.firstVisit,
        goalAt: r.agent.goalAt,
        runId: r.runId,
      }
    },
    (r) => r.visits.filter((v) => v > 0).length / nStates,
    options,
  )
  const boot: BootstrapOptions = {
    ...(options.key === undefined ? { seed: options.seed ?? 0 } : { key: options.key }),
    ...(options.reps === undefined ? {} : { reps: options.reps }),
    ...(options.level === undefined ? {} : { level: options.level }),
  }
  const gridSteps = linspaceNumbers(0, options.maxSteps, options.points ?? 21)
  const agents: Record<string, ExplorationSummary> = {}
  for (const name of names) {
    const mine = results.filter((r) => r.agent === name)
    const curves = mine.map((r) => gridSteps.map((t) => r.firstVisit.filter((f) => f >= 0 && f <= t).length / nStates))
    const coverage = aggregateCurves(curves, { ...boot, aggregate: 'mean' })
    const finals = mine.map((r) => [r.visits.filter((v) => v > 0).length / nStates])
    const times = mine.map((r) => [r.goalAt === -1 ? options.maxSteps : r.goalAt])
    const visits = Array.from({ length: nStates }, (_, i) => meanOf(mine.map((r) => r.visits[i] ?? 0)))
    agents[name] = {
      coverage: { ...coverage, x: gridSteps },
      finalCoverage: stratifiedBootstrap(finals, (m) => meanOf(m.flat()), boot),
      timeToGoal: stratifiedBootstrap(times, (m) => meanOf(m.flat()), boot),
      reached: mine.filter((r) => r.goalAt !== -1).length / mine.length,
      visits,
    }
  }
  return { agents, runs: results, runId }
}

// --------------------------------------------------------------------------------------- Generalisation studies

/** A range of procedural levels (`startLevel` and `numLevels` options). */
export interface LevelRange {
  readonly startLevel: number
  readonly numLevels: number
}

/** Options of {@link generalisationStudy}. */
export interface GeneralisationOptions extends StudyBudget, BootstrapOptions, StudyRunOptions {
  readonly agents: Readonly<Record<string, AgentFactory>>
  /** A procedural environment definition (levels chosen by `startLevel` and `numLevels`). */
  readonly definition: EnvDefinition
  /** Options of the definition shared by training and test environments. */
  readonly envOptions?: Readonly<Record<string, unknown>>
  /** The training levels. */
  readonly train: LevelRange
  /** The held-out test levels (disjoint from the training levels). */
  readonly test: LevelRange
  readonly seeds?: number | readonly number[]
  /** Greedy evaluation episodes on each level set (default 20). */
  readonly evalEpisodes?: number
}

/** The generalisation summary of one agent. */
export interface GeneralisationSummary {
  /** Mean greedy return on the training levels. */
  readonly train: BootstrapEstimate
  /** Mean greedy return on the held-out levels. */
  readonly test: BootstrapEstimate
  /** The generalisation gap, train minus test (paired by seed). */
  readonly gap: BootstrapEstimate
}

/** Result of {@link generalisationStudy}. */
export interface GeneralisationResult {
  readonly agents: Readonly<Record<string, GeneralisationSummary>>
  readonly runs: readonly {
    readonly agent: string
    readonly seed: number
    readonly train: number
    readonly test: number
    readonly returns: readonly number[]
    readonly runId: string
  }[]
  readonly runId: string
}

/**
 * A generalisation study (Procgen's protocol, Cobbe et al. 2020): train each agent on a fixed set of procedural levels,
 * then evaluate its greedy policy on those levels and on disjoint held-out levels, and report both scores and the
 * generalisation gap (train minus test, paired by seed) with bootstrap intervals.
 *
 * @param options - Agents, the procedural definition and its options, training and test level ranges, seeds, budget,
 *   evaluation episodes and bootstrap and run options.
 * @returns Per-agent train and test scores and gaps, and the runs.
 * @throws WhiteboxError with code `value` when the level ranges overlap or no budget is given.
 * @concept generalisation
 * @example
 * ```ts
 * import { policyAgent } from 'whitebox-engine/rl'
 * import { generalisationStudy, levelSplit, proceduralMaze } from 'whitebox-engine/methods/rl'
 * const split = levelSplit(5, 5)
 * const r = generalisationStudy({
 *   agents: { still: () => policyAgent(() => 0) as never }, definition: proceduralMaze,
 *   envOptions: { rows: 3, cols: 3 }, ...split, seeds: 2, episodes: 1, evalEpisodes: 2, reps: 20,
 * })
 * r.runs.length // => 2
 * ```
 */
export function generalisationStudy(options: GeneralisationOptions): GeneralisationResult {
  const budget = budgetOf(options, 'generalisationStudy')
  const { train, test } = options
  const overlap =
    train.numLevels === 0 ||
    test.numLevels === 0 ||
    (train.startLevel < test.startLevel + test.numLevels && test.startLevel < train.startLevel + train.numLevels)
  if (overlap)
    throw new WhiteboxError('value', 'generalisationStudy: training and test levels must be finite and disjoint')
  const trainEnv = make(options.definition, { ...options.envOptions, ...train })
  const testEnv = make(options.definition, { ...options.envOptions, ...test })
  const names = Object.keys(options.agents)
  const seeds = seedList(options.seeds)
  const evalEpisodes = options.evalEpisodes ?? 20
  const units = names.flatMap((agent) => seeds.map((seed) => ({ agent, seed })))
  const { results, runId } = execute(
    'generalisation',
    {
      agents: names,
      env: options.definition.id,
      train: { ...train },
      test: { ...test },
      seeds: [...seeds],
      ...budget,
    } as Json,
    units,
    (u, ctx, root) => {
      const run = trainUnit(ctx, root, u.agent, options.agents[u.agent] as AgentFactory, trainEnv, 0, u.seed, budget, {
        evalEpisodes,
      })
      const testScore = evaluatePolicy(testEnv, greedyPolicyOf(run.agentImpl, run.agentState), {
        episodes: evalEpisodes,
        key: foldIn(foldIn(root, u.seed), 0x7465),
        parent: ctx.run,
      }).meanReturn
      return { agent: u.agent, seed: u.seed, train: run.final, test: testScore, returns: run.returns, runId: run.runId }
    },
    (r) => r.test,
    options,
  )
  const boot: BootstrapOptions = {
    ...(options.key === undefined ? { seed: options.seed ?? 0 } : { key: options.key }),
    ...(options.reps === undefined ? {} : { reps: options.reps }),
    ...(options.level === undefined ? {} : { level: options.level }),
  }
  const mean = (m: readonly (readonly number[])[]): number => meanOf(m.flat())
  const agents: Record<string, GeneralisationSummary> = {}
  for (const name of names) {
    const mine = results.filter((r) => r.agent === name)
    agents[name] = {
      train: stratifiedBootstrap(
        mine.map((r) => [r.train]),
        mean,
        boot,
      ),
      test: stratifiedBootstrap(
        mine.map((r) => [r.test]),
        mean,
        boot,
      ),
      gap: stratifiedBootstrap(
        mine.map((r) => [r.train - r.test]),
        mean,
        boot,
      ),
    }
  }
  return { agents, runs: results, runId }
}
