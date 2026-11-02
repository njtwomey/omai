/**
 * The agent base: a method whose run is an agent learning in an environment. A particular declares the environment
 * (a registry name with options, or a definition), the agent (from `lib/agents`, built over the environment), and the
 * schedule (episodes or steps, the steps of an episode, how often and how long to evaluate); the base owns the loop
 * (`trainingAlgorithm`, one step an environment step or, as an option, an episode), the standard paths
 * (`episode/return`, `episode/length`, `train/loss`, `agent/<statistic>`, `eval/return`, `env/frame`), the probes (an
 * evaluation episode as an `episodes` record at any step, and for tabular agents the value and action maps as
 * `valueGrids`), serialisation of the whole run state so a session can seek, and `method`, `session` and `train`.
 *
 * @index structure Agent base: an RL agent learning in an environment, as a session.
 */

import { lookup, MdmlError } from '../../base/index.ts'
import { environmentRegistry } from '../../environments/index.ts'
import { type Exposition, valueGrids } from '../../exposition/index.ts'
import { foldIn, type Key, key as makeKey } from '../../random/index.ts'
import {
  type Agent,
  type Env,
  type EnvDefinition,
  greedyPolicyOf,
  make,
  type Policy,
  runEpisode,
  type TrainState,
  trainingAlgorithm,
} from '../../rl/index.ts'
import { type DictSpace, dict, type SpaceValue } from '../../space/index.ts'
import {
  type Algorithm,
  type Config,
  type Contract,
  defineMethod,
  derived,
  event,
  frame,
  Log,
  type Method,
  type Run,
  type RunStatus,
  type Session,
  type Subscription,
  scalar,
  session,
} from '../../trace/index.ts'
import { episodeRecordOf, type GridLayout, type GridMaps, gridMaps } from './_agent.ts'
import { extendSession, type SessionWith } from './_session.ts'

export type { GridLayout } from './_agent.ts'
export type { SessionWith } from './_session.ts'

/** Where an environment comes from: a registry name (with options), or a definition. */
export type EnvironmentChoice<O, A> =
  | string
  | EnvDefinition<unknown, O, A>
  | { readonly id: string; readonly options?: Readonly<Record<string, unknown>> }

/** The schedule of an agent's run, declared from its hyperparameters. */
export interface AgentSchedule {
  /** Stop after this many episodes. */
  readonly episodes?: number
  /** The budget in environment steps (default `episodes` times the episode length). */
  readonly steps?: number
  /** Cut episodes after this many steps (default the environment's own limit). */
  readonly maxEpisodeSteps?: number
  /** Evaluate the greedy policy every this many episodes, for `evalEpisodes` episodes (default 5), as `eval/return`. */
  readonly evalEvery?: number
  readonly evalEpisodes?: number
  /** What one session step is: an environment step (default) or an episode. */
  readonly unit?: 'step' | 'episode'
}

/**
 * The paths of a session of the agent base: the particular's own (`K`) and the base's (`episode/...`, `eval/return`,
 * `eval/episode`, `train/loss`, `rl/episode/end`, `env/frame`, `agent/<statistic>` for the agent's own, and the maps of
 * a tabular agent). With `K` left at `string` every path is allowed.
 *
 * @example
 * ```ts
 * import type { AgentPath } from 'mdml-engine/learn'
 * type P = AgentPath<'dqn/replay'>   // 'dqn/replay' | 'episode/return' | 'episode/length' | ...
 * ```
 */
export type AgentPath<K extends string = string> =
  | K
  | 'episode/return'
  | 'episode/length'
  | 'rl/episode/end'
  | 'env/frame'
  | 'eval/return'
  | 'eval/episode'
  | 'train/loss'
  | `agent/${string}`
  | 'agent/value-map'
  | 'agent/action-map'

/**
 * What a particular declares over the agent base. The last parameter is the type of the paths it adds to the base's
 * (`string` by default: unchecked); type them as for the iterative base (`IterativeParticular`).
 */
export interface AgentParticular<H extends object, G, O, A, K extends string = string> {
  readonly name: string
  readonly defaults: H
  readonly hyperSpace?: (hyper: H) => DictSpace
  environment(hyper: H): EnvironmentChoice<O, A>
  /** The agent over the made environment (deep agents take the key to initialise their networks). */
  agent(env: Env<unknown, O, A>, hyper: H, key: Key): Agent<G, O, A>
  schedule(hyper: H): AgentSchedule
  /** The layout of a tabular agent's states for `agent/value-map` and `agent/action-map` (default one row). */
  grid?(hyper: H): GridLayout
  /** What the state can tell beyond the standard paths. */
  contract?(hyper: H): Contract<TrainState<G, unknown, O>, K>
  readonly projections?: Readonly<Record<string, (run: Run, hyper: H) => Exposition | undefined>>
  /** What a run keeps by default (default the returns, lengths, the agent's loss and evaluations). */
  readonly logs?: Subscription<NoInfer<AgentPath<K>>>
}

/** A session of an agent's run: the verbs of a session, and the agent and its greedy policy at any reached step. */
export type AgentSession<G, O, A, K extends string = string> = SessionWith<
  TrainState<G, unknown, O>,
  { agentAt(t: number): G; policyAt(t: number): Policy<O, A> },
  K
>

/** The outcome of {@link AgentMethod.train}: the agent, the returns and the tape. */
export interface AgentTrained<G, O> {
  readonly agent: G
  readonly state: TrainState<G, unknown, O>
  readonly returns: readonly number[]
  readonly lengths: readonly number[]
  readonly evaluations: TrainState<G, unknown, O>['evaluations']
  readonly status: RunStatus | undefined
  readonly run: Run
  readonly exposition: readonly Exposition[]
}

/** A method on the agent base. The last parameter is the type of the paths the particular adds to the base's. */
export interface AgentMethod<H extends object, G, O, A, K extends string = string> {
  readonly name: string
  readonly hyper: H
  method(): Method<TrainState<G, unknown, O>, Config, AgentPath<K>>
  /** A session of the run: advance, seek, probe and fork it, and read the agent at any step. */
  session(options?: AgentSessionOptions<AgentPath<K>>): AgentSession<G, O, A, AgentPath<K>>
  /** The run to completion with the method's defaults. */
  train(options?: AgentSessionOptions<AgentPath<K>>): AgentTrained<G, O>
  with(changes: Partial<H>): AgentMethod<H, G, O, A, K>
}

/** Options of {@link AgentMethod.session}; the parameter is the type of the session's paths (`logs` keys are checked). */
export interface AgentSessionOptions<K extends string = string> {
  readonly seed?: number
  readonly key?: Key
  readonly logs?: Subscription<NoInfer<K>>
  readonly mode?: 'live' | 'eager'
  /** Override the budget in session steps. */
  readonly steps?: number
}

const metricNames = (agent: Agent<unknown, never, never>): string[] => Object.keys(agent.metrics ?? {})

/**
 * A method over the agent base.
 *
 * @param p - The particular.
 * @param hyper - Its hyperparameters (default the particular's defaults).
 * @returns The method, with `session` and `train`.
 * @concept log-contract
 * @example
 * ```ts
 * import { defineAgentMethod } from 'mdml-engine/learn'
 * import { qLearning } from 'mdml-engine/agents'
 * import { metricSeries } from 'mdml-engine/trace'
 * const q = defineAgentMethod({
 *   name: 'q-learning/frozen-lake', defaults: { alpha: 0.2 },
 *   environment: () => 'frozenLake',
 *   agent: (env, h) => qLearning(env, { alpha: h.alpha, gamma: 0.95, epsilon: 0.2 }),
 *   schedule: () => ({ episodes: 20 }),
 * })
 * metricSeries(q.train().run, 'episode/return').values.length // => 20
 * ```
 */
export function defineAgentMethod<H extends object, G, O, A, K extends string = string>(
  p: AgentParticular<H, G, O, A, K>,
  hyper: H = p.defaults,
): AgentMethod<H, G, O, A, K> {
  const space: DictSpace = p.hyperSpace?.(hyper) ?? dict({})
  const knobs = Object.keys(space.spaces)
  const hyperOf = (c: Config): H => ({ ...hyper, ...c }) as H
  const configOf = (h: H): Config =>
    Object.fromEntries(knobs.map((k) => [k, (h as Record<string, unknown>)[k] as SpaceValue])) as Config
  const envOf = (h: H): Env<unknown, O, A> => {
    const choice = p.environment(h)
    const named = typeof choice === 'string' ? { id: choice } : 'create' in choice ? undefined : choice
    const limit = p.schedule(h).maxEpisodeSteps
    const config = limit === undefined ? {} : { maxEpisodeSteps: limit }
    if (named === undefined) return make(choice as EnvDefinition<unknown, O, A>, undefined, config)
    const entry = lookup(environmentRegistry, named.id)
    if (entry.multiAgent !== undefined)
      throw new MdmlError('value', `${p.name}: ${named.id} is a multi-agent environment`)
    return make(
      entry.definition as EnvDefinition<unknown, O, A>,
      'options' in named ? named.options : undefined,
      config,
    )
  }
  const cache = new Map<string, { env: Env<unknown, O, A>; agent: Agent<G, O, A> }>()
  const partsOf = (h: H, key: Key): { env: Env<unknown, O, A>; agent: Agent<G, O, A> } => {
    const id = `${JSON.stringify(h)}|${key.words.join(',')}`
    const env = cache.get(id)?.env ?? envOf(h)
    const parts = cache.get(id) ?? { env, agent: p.agent(env, h, foldIn(key, 2)) }
    cache.set(id, parts)
    return parts
  }
  const budget = (h: H, env: Env<unknown, O, A>): number => {
    const s = p.schedule(h)
    if (s.unit === 'episode') {
      if (s.episodes === undefined) throw new MdmlError('value', `${p.name}: an episode unit needs schedule.episodes`)
      return s.episodes
    }
    const length = s.maxEpisodeSteps ?? env.spec.maxEpisodeSteps ?? 1000
    const steps = s.steps ?? (s.episodes === undefined ? undefined : s.episodes * length)
    if (steps === undefined) throw new MdmlError('value', `${p.name}: give schedule.episodes or schedule.steps`)
    return steps
  }
  const algorithmOf = (h: H, key: Key): Algorithm<TrainState<G, unknown, O>, AgentPath> => {
    const { env, agent } = partsOf(h, key)
    const s = p.schedule(h)
    const inner = trainingAlgorithm(env, agent, {
      checkpoints: false,
      ...(s.episodes === undefined ? {} : { episodes: s.episodes }),
      ...(s.evalEvery === undefined ? {} : { evalEvery: s.evalEvery }),
      ...(s.evalEpisodes === undefined ? {} : { evalEpisodes: s.evalEpisodes }),
    })
    return {
      ...inner,
      name: p.name,
      step: (st, ctx) => {
        let next = inner.step(st, ctx)
        while (s.unit === 'episode' && next.episode === st.episode && inner.done?.(next, ctx.t) === undefined)
          next = inner.step(next, ctx)
        if (next.episode > st.episode) {
          ctx.log('episode/return', next.returns.at(-1) ?? 0)
          ctx.log('episode/length', next.lengths.at(-1) ?? 0)
        }
        const evaluated = next.evaluations.at(-1)
        if (next.evaluations.length > st.evaluations.length && evaluated !== undefined)
          ctx.log('eval/return', evaluated.meanReturn)
        if (agent.stats !== undefined && ctx.run.enabled) {
          for (const [name, value] of Object.entries(agent.stats(next.agent))) {
            ctx.log(`agent/${name}`, value)
            if (name === 'loss') ctx.log('train/loss', value)
          }
        }
        return next
      },
    }
  }
  const contractOf = (h: H, key: Key): Contract<TrainState<G, unknown, O>, AgentPath<K>> => {
    const { env, agent } = partsOf(h, key)
    const s = p.schedule(h)
    const out: Record<string, Contract<TrainState<G, unknown, O>>[string]> = {
      'episode/return': scalar({ description: 'return of each finished episode', direction: 'higher' }),
      'episode/length': scalar({ description: 'steps of each finished episode' }),
      'rl/episode/end': event({ context: { episode: 'index', return: 'its return', length: 'its steps' } }),
      'env/frame': frame({ description: 'the environment’s scene' }),
    }
    if (s.evalEvery !== undefined)
      out['eval/return'] = scalar({ description: 'mean return of the greedy policy', direction: 'higher' })
    for (const name of metricNames(agent as Agent<unknown, never, never>))
      out[`agent/${name}`] = scalar({ description: agent.metrics?.[name]?.description ?? name })
    if (agent.metrics?.['loss'] !== undefined)
      out['train/loss'] = scalar({ description: 'the agent’s loss', direction: 'lower' })
    out['eval/episode'] = derived<TrainState<G, unknown, O>>({
      description: 'one greedy episode from the agent at this step, as an episodes record (args: seed)',
      cost: 'dear',
      compute: (st, args) => {
        const seed = (args as { readonly seed?: number } | undefined)?.seed ?? 0
        const ep = runEpisode(env, greedyPolicyOf(agent, st.agent), makeKey(seed), {
          maxSteps: s.maxEpisodeSteps ?? env.spec.maxEpisodeSteps ?? 1000,
        })
        return episodeRecordOf(env.spec.id, agent.name, st.episode, ep, gridMaps(st.agent, p.grid?.(h)))
      },
    })
    const mapsOf = (st: TrainState<G, unknown, O>): GridMaps => {
      const maps = gridMaps(st.agent, p.grid?.(h))
      if (maps === undefined) throw new MdmlError('value', `${p.name}: only tabular agents have value and action maps`)
      return maps
    }
    out['agent/value-map'] = derived<TrainState<G, unknown, O>>({
      description: 'a tabular agent’s state values as a grid',
      compute: (st) => valueGrids({ grids: [{ name: 'value', values: mapsOf(st).value }], valueName: 'value' }),
    })
    out['agent/action-map'] = derived<TrainState<G, unknown, O>>({
      description: 'a tabular agent’s action values, one grid per action, and its greedy action',
      compute: (st) => valueGrids({ grids: mapsOf(st).actions, valueName: 'action value', diverging: true }),
    })
    // The agent's statistics depend on the agent, so `out` is a plain record of `AgentPath` keys.
    return { ...out, ...(p.contract?.(h) ?? {}) } as Contract<TrainState<G, unknown, O>, AgentPath<K>>
  }
  const defaultLogs = (): Subscription<AgentPath<K>> => {
    const known = contractOf(hyper, makeKey(0))
    const out: Record<string, ReturnType<typeof Log>> = {}
    for (const path of ['episode/return', 'episode/length', 'eval/return', 'train/loss'])
      if (path in known) out[path] = Log()
    return out as Subscription<AgentPath<K>>
  }
  const method = (): Method<TrainState<G, unknown, O>, Config, AgentPath<K>> =>
    defineMethod<TrainState<G, unknown, O>, Config, AgentPath<K>>({
      name: p.name,
      config: space,
      make: (c, key) => algorithmOf(hyperOf(c), key),
      contract: (c, key) => contractOf(hyperOf(c), key),
      defaults: p.logs ?? defaultLogs(),
      ...(p.projections === undefined
        ? {}
        : {
            projections: Object.fromEntries(
              Object.entries(p.projections).map(([n, f]) => [n, (run: Run) => f(run, hyper)]),
            ),
          }),
    })
  const sessionOf = (options: AgentSessionOptions<AgentPath<K>>): AgentSession<G, O, A, AgentPath<K>> => {
    const root = options.key ?? makeKey(options.seed ?? 0)
    const steps = options.steps ?? budget(hyper, envOf(hyper))
    const s = session(method(), {
      key: root,
      ...(options.seed === undefined ? {} : { seed: options.seed }),
      config: configOf(hyper),
      logs: options.logs ?? 'defaults',
      mode: options.mode ?? 'live',
      steps,
      checkpointEvery: Math.max(1, Math.floor(steps / 100)),
    })
    const agentOf = (inner: Session<TrainState<G, unknown, O>, Config, AgentPath<K>>, t: number): G =>
      inner.stateAt(t).agent
    return extendSession(s, (inner) => {
      const { agent } = partsOf(hyperOf(inner.config), foldIn(root, 7))
      return {
        agentAt: (t: number): G => agentOf(inner, t),
        policyAt: (t: number): Policy<O, A> => greedyPolicyOf(agent, agentOf(inner, t)),
      }
    })
  }
  return {
    name: p.name,
    hyper,
    method,
    session: (options = {}) => sessionOf(options),
    train: (options = {}) => {
      const s = sessionOf({ ...options, mode: 'eager' })
      const st = s.state
      const run = s.run()
      const records = Object.values(p.projections ?? {}).flatMap((f) => f(run, hyper) ?? [])
      return {
        agent: st.agent,
        state: st,
        returns: st.returns,
        lengths: st.lengths,
        evaluations: st.evaluations,
        status: s.status,
        run,
        exposition: records,
      }
    },
    with: (changes) => defineAgentMethod(p, { ...hyper, ...changes }),
  }
}
