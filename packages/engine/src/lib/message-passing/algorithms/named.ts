/**
 * Expectation propagation, variational message passing and belief propagation, each as an algorithm of its own with
 * its own knobs, defaults and tape, and as a `Method` (a `Space` of the knobs and a typed contract of the tape's
 * paths) a session or a worker drives by name. They are three settings of the one engine of `engine.ts`; nothing here
 * picks among them for you.
 */

import { choice, type DictSpace, dict, int, real } from '../../space/index.ts'
import {
  type Algorithm,
  type Config,
  type Contract,
  defineContract,
  defineMethod,
  frame,
  type Method,
  message,
  scalar,
} from '../../trace/index.ts'
import type { CompiledGraph } from '../core/graph.ts'
import type { Mode } from '../core/types.ts'
import { klqpProjection, momentProjection, sumProductProjection } from '../engine/projections.ts'
import { messagePassingAlgorithm } from '../engine/run.ts'
import type { CustomOrder, Granularity, ScheduleKind } from '../engine/schedules.ts'
import type { EngineSettings, ImproperPolicy, InitialMessage, MpState } from '../engine/state.ts'

/** The knobs every message-passing algorithm has; the defaults differ by algorithm. */
export interface MessagePassingOptions {
  /** The order of a sweep. */
  readonly schedule?: ScheduleKind
  /** The order of a `custom` schedule: factors by index with one port or all (`null`). */
  readonly order?: CustomOrder
  /** How much of a sweep a step does: the whole sweep, one factor, or one message (the anytime granularity). */
  readonly granularity?: Granularity
  readonly damping?: number
  /** What to do with an update that makes a proper belief improper. */
  readonly improper?: ImproperPolicy
  /** Stop when no belief moves by more than this over a sweep. */
  readonly tol?: number
  readonly maxSweeps?: number
  /** Messages to start from instead of the flat ones. */
  readonly initial?: readonly InitialMessage[]
  /**
   * How many times a run that stalls or diverges is restarted from its start messages with a stronger damping (0.5, then
   * more) before it is reported as stalled: a stall is a run whose beliefs stopped moving because some updates could not be
   * computed (see `mpVerdict`). Each restart is counted on the tape (`mp/restarts`).
   */
  readonly restarts?: number
}

/**
 * The engine settings of a named algorithm: the projection and the algorithm's defaults, overridden by the options given.
 * What `epAlgorithm`, `vmpAlgorithm` and `bpAlgorithm` (and the variants beside them) build their run from.
 *
 * @param projection - The projection slot.
 * @param defaults - The algorithm's defaults.
 * @param o - The options given.
 * @returns The settings.
 */
export function messagePassingSettings(
  projection: EngineSettings['projection'],
  defaults: Required<Omit<MessagePassingOptions, 'order' | 'initial'>>,
  o: MessagePassingOptions,
): EngineSettings {
  return {
    projection,
    schedule: o.schedule ?? defaults.schedule,
    granularity: o.granularity ?? defaults.granularity,
    damping: o.damping ?? defaults.damping,
    improper: o.improper ?? defaults.improper,
    tol: o.tol ?? defaults.tol,
    maxSweeps: o.maxSweeps ?? defaults.maxSweeps,
    restarts: o.restarts ?? defaults.restarts,
    ...(o.order === undefined ? {} : { order: o.order }),
    ...(o.initial === undefined ? {} : { initial: o.initial }),
  }
}

/** The defaults of expectation propagation: serial sweeps, a repair that halves a step until the belief stays proper, one damped restart of a run that stalls. */
export const epDefaults: Required<Omit<MessagePassingOptions, 'order' | 'initial'>> = {
  schedule: 'serial',
  granularity: 'sweep',
  damping: 0,
  improper: 'damp',
  tol: 1e-8,
  maxSweeps: 200,
  restarts: 1,
}

/** The defaults of variational message passing: serial sweeps; an update that breaks a belief is dropped. */
export const vmpDefaults: Required<Omit<MessagePassingOptions, 'order' | 'initial'>> = {
  schedule: 'serial',
  granularity: 'sweep',
  damping: 0,
  improper: 'skip',
  tol: 1e-8,
  maxSweeps: 200,
  restarts: 0,
}

/** The defaults of belief propagation: flooding sweeps, undamped. */
export const bpDefaults: Required<Omit<MessagePassingOptions, 'order' | 'initial'>> = {
  schedule: 'flooding',
  granularity: 'sweep',
  damping: 0,
  improper: 'skip',
  tol: 1e-10,
  maxSweeps: 200,
  restarts: 0,
}

/** The paths on the tape of every message-passing run, apart from the evidence and the bound. */
type CommonPath =
  | 'mp/change'
  | 'mp/sweeps'
  | 'mp/updates'
  | 'mp/improper'
  | 'mp/skipped'
  | 'mp/damped'
  | 'mp/reset'
  | 'mp/unresolved'
  | 'mp/restarts'
  | 'mp/verdict'
  | 'mp/message'
  | 'mp/marginals'
  | 'mp/frame'
  | 'mp/snapshot'

/** The paths of an expectation propagation or belief propagation run: the common ones and the evidence estimate. */
export type EpPath = CommonPath | 'mp/evidence'

/** The paths of a variational message passing run: the common ones and the variational lower bound. */
export type VmpPath = CommonPath | 'mp/elbo'

const common = {
  'mp/change': scalar({ from: 'change', description: 'the largest change of a belief over the last completed sweep' }),
  'mp/sweeps': scalar({ from: 'sweeps', description: 'completed sweeps' }),
  'mp/updates': scalar({ from: 'updates', description: 'messages sent so far' }),
  'mp/improper': scalar({ from: 'improper', description: 'updates that would have made a belief improper, or failed' }),
  'mp/skipped': scalar({ from: 'skipped', description: 'updates dropped by the improper policy' }),
  'mp/damped': scalar({ from: 'damped', description: 'updates shortened until the belief stayed proper' }),
  'mp/reset': scalar({ from: 'reset', description: 'messages reset to flat' }),
  'mp/unresolved': scalar({
    from: 'unresolved',
    description: 'updates of the last completed sweep that could not be computed: those messages were not recomputed',
  }),
  'mp/restarts': scalar({
    from: 'restarts',
    description: 'automatic restarts from the start messages with a stronger damping',
  }),
  'mp/verdict': message({
    from: 'verdict',
    description: 'why a run that ended did not converge: stalled, degenerate, diverged or completed, with the reason',
    payload: { status: 'the run status it reaches', reason: 'what went wrong' },
  }),
  'mp/message': message({
    from: 'message',
    description: 'a message from a factor to a variable (ids as in graphStructure)',
    payload: {
      factor: 'the factor index',
      variable: 'the variable name',
      operator: 'its operator',
      eta: 'natural parameters',
    },
  }),
  'mp/marginals': message({ from: 'marginals', description: 'the current marginal of every variable, by variable id' }),
  'mp/frame': frame({ description: 'the marginals by variable id and the messages of the last step' }),
  'mp/snapshot': message({ description: 'an anytime snapshot (see runAnytime): where the run is and every marginal' }),
} as const

/**
 * What an expectation propagation or belief propagation run logs, by path.
 *
 * @concept message-passing
 */
export const epContract: Contract<MpState, EpPath> = defineContract({
  ...common,
  'mp/evidence': scalar({
    from: 'evidence',
    description: 'the EP (Bethe for BP) estimate of the log evidence',
    direction: 'higher',
  }),
})

/**
 * What a variational message passing run logs, by path.
 *
 * @concept message-passing
 */
export const vmpContract: Contract<MpState, VmpPath> = defineContract({
  ...common,
  'mp/elbo': scalar({ from: 'elbo', description: 'the variational lower bound', direction: 'higher' }),
})

/**
 * Expectation propagation (Minka 2001) on a factor graph: moment matching at every factor, by the closed-form rules
 * of the operators (or a numeric tilted-moment operator), over serial sweeps unless told otherwise, with the EP free
 * energy as the evidence estimate. Exact on a tree of Gaussian factors; on loopy or non-Gaussian graphs the usual
 * approximation, damped when `damping` is set and repaired (counted on the tape) when an update would make a belief
 * improper.
 *
 * @param g - The compiled graph.
 * @param options - Schedule, granularity, damping, improper policy, tolerance, sweep cap.
 * @returns The algorithm; its tape is described by {@link epContract}.
 * @concept expectation-propagation
 * @index algorithm Expectation propagation on a factor graph of exponential families.
 * @example
 * ```ts
 * import { compileGraph, defaultOperators, epAlgorithm, gaussianPrior, marginalsOf } from 'omai-engine/message-passing'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const g = compileGraph(
 *   {
 *     variables: [{ name: 'x', family: { kind: 'gaussian' } }, { name: 'y', family: { kind: 'bernoulli' }, observed: 1 }],
 *     factors: [gaussianPrior('x', 0, 1), { operator: 'probit', args: ['y', 'x'] }],
 *   },
 *   defaultOperators(),
 * )
 * const r = runAlgorithm(epAlgorithm(g), { maxSteps: 50 })
 * Number(marginalsOf(g, r.state).x?.mean).toFixed(4) // => '0.5642'
 * ```
 */
export function epAlgorithm(g: CompiledGraph, options: MessagePassingOptions = {}): Algorithm<MpState, EpPath> {
  return messagePassingAlgorithm(g, messagePassingSettings(momentProjection, epDefaults, options)) as Algorithm<
    MpState,
    EpPath
  >
}

/**
 * Variational message passing (Winn and Bishop 2005) on a factor graph: the expected-log message of every factor, so
 * each variable's belief is the exponential-family distribution that minimises KL(q || p) given the others, over
 * serial sweeps. The variational lower bound is on the tape and does not decrease on a conjugate-exponential graph.
 *
 * @param g - The compiled graph.
 * @param options - Schedule (a `custom` order for coordinate ascent in a chosen order), granularity, damping, tolerance, sweep cap.
 * @returns The algorithm; its tape is described by {@link vmpContract}.
 * @concept variational-message-passing
 * @index algorithm Variational message passing on a factor graph of exponential families.
 * @example
 * ```ts
 * import { compileGraph, defaultOperators, gammaPrior, gaussianPrior, marginalsOf, vmpAlgorithm } from 'omai-engine/message-passing'
 * import { runAlgorithm } from 'omai-engine/trace'
 * // x ~ N(0, 1) observed through N(y = 2; x, 1 / 4): the posterior mean is 4 / 5 * 2.
 * const g = compileGraph(
 *   {
 *     variables: [{ name: 'x', family: { kind: 'gaussian' } }, { name: 'y', family: { kind: 'gaussian' }, observed: 2 }],
 *     factors: [gaussianPrior('x', 0, 1), { operator: 'gaussian', args: ['y', 'x'], params: { precision: 4 } }],
 *   },
 *   defaultOperators(),
 * )
 * const r = runAlgorithm(vmpAlgorithm(g), { maxSteps: 50 })
 * Number(marginalsOf(g, r.state).x?.mean).toFixed(4) // => '1.6000'
 * ```
 */
export function vmpAlgorithm(g: CompiledGraph, options: MessagePassingOptions = {}): Algorithm<MpState, VmpPath> {
  return messagePassingAlgorithm(g, messagePassingSettings(klqpProjection, vmpDefaults, options)) as Algorithm<
    MpState,
    VmpPath
  >
}

/**
 * Belief propagation (sum-product) on a factor graph of discrete variables: the marginals are exact on a tree and
 * the loopy approximation otherwise, with the Bethe approximation of the log evidence.
 *
 * @param g - The compiled graph.
 * @param options - Schedule (`tree` is one pass in and out), granularity, damping, tolerance, sweep cap.
 * @returns The algorithm; its tape is described by {@link epContract}.
 * @concept belief-propagation
 * @index algorithm Sum-product belief propagation on a factor graph of discrete variables.
 * @example
 * ```ts
 * import { bpAlgorithm, compileGraph, defaultOperators, discretePrior, marginalsOf } from 'omai-engine/message-passing'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const g = compileGraph(
 *   {
 *     variables: [{ name: 'a', family: { kind: 'discrete', size: 2 } }, { name: 'b', family: { kind: 'discrete', size: 2 } }],
 *     factors: [discretePrior('a', [0.3, 0.7]), { operator: 'table', args: ['a', 'b'], params: { states: [2, 2], values: [0.9, 0.1, 0.2, 0.8] } }],
 *   },
 *   defaultOperators(),
 * )
 * const r = runAlgorithm(bpAlgorithm(g, { schedule: 'tree' }), { maxSteps: 10 })
 * (marginalsOf(g, r.state).b?.probs as number[]).map((p) => p.toFixed(2)) // => ['0.41', '0.59']
 * ```
 */
export function bpAlgorithm(g: CompiledGraph, options: MessagePassingOptions = {}): Algorithm<MpState, EpPath> {
  return messagePassingAlgorithm(g, messagePassingSettings(sumProductProjection, bpDefaults, options)) as Algorithm<
    MpState,
    EpPath
  >
}

/**
 * The defaults of a message rule's named algorithm: `epDefaults`, `vmpDefaults` or `bpDefaults`.
 *
 * @param mode - `ep`, `vmp` or `bp`.
 * @returns The defaults.
 * @example
 * ```ts
 * import { defaultsOf } from 'omai-engine/message-passing'
 * defaultsOf('bp').schedule // => 'flooding'
 * ```
 */
export function defaultsOf(mode: Mode): Required<Omit<MessagePassingOptions, 'order' | 'initial'>> {
  return mode === 'ep' ? epDefaults : mode === 'vmp' ? vmpDefaults : bpDefaults
}

const SCHEDULES = ['serial', 'flooding', 'tree', 'residual'] as const

/**
 * The `Space` of the knobs every message-passing algorithm has (schedule, granularity, damping, improper policy,
 * tolerance, sweep cap), with the given defaults; a variant adds its own knobs to the dictionary.
 *
 * @param defaults - The algorithm's defaults.
 * @returns The space.
 */
export function messagePassingSpace(defaults: Required<Omit<MessagePassingOptions, 'order' | 'initial'>>): DictSpace {
  return dict({
    schedule: choice(SCHEDULES, {
      default: defaults.schedule as (typeof SCHEDULES)[number],
      description: 'the order of a sweep',
    }),
    granularity: choice(['sweep', 'factor', 'message'] as const, {
      default: defaults.granularity,
      description: 'how much of a sweep one step does',
    }),
    damping: real(0, 0.95, { default: defaults.damping, description: 'weight of the old message in the new' }),
    improper: choice(['skip', 'damp', 'reset', 'error'] as const, {
      default: defaults.improper,
      description: 'the repair of an update that makes a belief improper',
    }),
    tol: real(1e-12, 1e-2, {
      log: true,
      default: defaults.tol,
      description: 'stop when no belief moves more than this over a sweep',
    }),
    maxSweeps: int(1, 10_000, { default: defaults.maxSweeps, description: 'the sweep cap' }),
    restarts: int(0, 5, {
      default: defaults.restarts,
      description:
        'automatic restarts with a stronger damping when the run stalls (its beliefs stop moving because updates cannot be computed)',
    }),
  })
}

/**
 * The options of a configuration (the values a `Space` of the knobs produced), ignoring entries that are not knobs of
 * the common message-passing set.
 *
 * @param c - The configuration.
 * @returns The options.
 */
export function messagePassingOptions(c: Config): MessagePassingOptions {
  const pick = <T>(name: string, ok: (v: unknown) => v is T): T | undefined =>
    ok(c[name]) ? (c[name] as T) : undefined
  const str = (v: unknown): v is string => typeof v === 'string'
  const num = (v: unknown): v is number => typeof v === 'number'
  const schedule = pick('schedule', str)
  const granularity = pick('granularity', str)
  const damping = pick('damping', num)
  const improper = pick('improper', str)
  const tol = pick('tol', num)
  const maxSweeps = pick('maxSweeps', num)
  const restarts = pick('restarts', num)
  return {
    ...(schedule === undefined ? {} : { schedule: schedule as ScheduleKind }),
    ...(granularity === undefined ? {} : { granularity: granularity as Granularity }),
    ...(damping === undefined ? {} : { damping }),
    ...(improper === undefined ? {} : { improper: improper as ImproperPolicy }),
    ...(tol === undefined ? {} : { tol }),
    ...(maxSweeps === undefined ? {} : { maxSweeps }),
    ...(restarts === undefined ? {} : { restarts }),
  }
}

/**
 * The knobs and the tape of a message rule, with no graph: the `Space` of its knobs (with the rule's defaults) and the
 * typed contract of its tape. What `epMethod`, `vmpMethod` and `bpMethod` carry, for a caller that has no graph yet (a
 * method built on a model spec that compiles its graph once it is run).
 *
 * @param mode - `ep`, `vmp` or `bp`.
 * @returns The space and the contract (`epContract` for `ep` and `bp`, `vmpContract` for `vmp`).
 * @concept message-passing
 * @index function The knobs (a Space) and the typed tape of a message rule, with no graph.
 * @example
 * ```ts
 * import { messagePassingKnobs } from 'omai-engine/message-passing'
 * Object.keys(messagePassingKnobs('vmp').contract).includes('mp/elbo') // => true
 * ```
 */
export function messagePassingKnobs(mode: Mode): {
  readonly config: DictSpace
  readonly contract: Contract<MpState, EpPath> | Contract<MpState, VmpPath>
} {
  return { config: messagePassingSpace(defaultsOf(mode)), contract: mode === 'vmp' ? vmpContract : epContract }
}

/**
 * Expectation propagation on a graph as a method: the knobs as a space (schedule, granularity, damping, improper
 * policy, tolerance, sweep cap), the tape as the typed contract {@link epContract}, so a session, a worker or a tuner
 * drives it by configuration.
 *
 * @param g - The compiled graph.
 * @returns The method, named `ep`.
 * @concept expectation-propagation
 * @index function The expectation propagation method of a graph: knobs as a space, typed tape.
 * @example
 * ```ts
 * import { compileGraph, defaultOperators, epMethod, gaussianPrior } from 'omai-engine/message-passing'
 * const g = compileGraph({ variables: [{ name: 'x', family: { kind: 'gaussian' } }], factors: [gaussianPrior('x', 0, 1)] }, defaultOperators())
 * epMethod(g).name // => 'ep'
 * ```
 */
export function epMethod(g: CompiledGraph): Method<MpState, Config, EpPath> {
  return defineMethod<MpState, Config, EpPath>({
    name: 'ep',
    config: messagePassingKnobs('ep').config,
    make: (c) => epAlgorithm(g, messagePassingOptions(c)),
    contract: epContract,
  })
}

/**
 * Variational message passing on a graph as a method (see {@link epMethod}); the tape is {@link vmpContract}.
 *
 * @param g - The compiled graph.
 * @returns The method, named `vmp`.
 * @concept variational-message-passing
 * @index function The variational message passing method of a graph: knobs as a space, typed tape.
 * @example
 * ```ts
 * import { compileGraph, defaultOperators, gaussianPrior, vmpMethod } from 'omai-engine/message-passing'
 * const g = compileGraph({ variables: [{ name: 'x', family: { kind: 'gaussian' } }], factors: [gaussianPrior('x', 0, 1)] }, defaultOperators())
 * vmpMethod(g).name // => 'vmp'
 * ```
 */
export function vmpMethod(g: CompiledGraph): Method<MpState, Config, VmpPath> {
  return defineMethod<MpState, Config, VmpPath>({
    name: 'vmp',
    config: messagePassingKnobs('vmp').config,
    make: (c) => vmpAlgorithm(g, messagePassingOptions(c)),
    contract: vmpContract,
  })
}

/**
 * Belief propagation on a graph as a method (see {@link epMethod}); the tape is {@link epContract}.
 *
 * @param g - The compiled graph.
 * @returns The method, named `bp`.
 * @concept belief-propagation
 * @index function The belief propagation method of a graph: knobs as a space, typed tape.
 * @example
 * ```ts
 * import { bpMethod, compileGraph, defaultOperators, discretePrior } from 'omai-engine/message-passing'
 * const g = compileGraph({ variables: [{ name: 'a', family: { kind: 'discrete', size: 2 } }], factors: [discretePrior('a', [0.5, 0.5])] }, defaultOperators())
 * bpMethod(g).name // => 'bp'
 * ```
 */
export function bpMethod(g: CompiledGraph): Method<MpState, Config, EpPath> {
  return defineMethod<MpState, Config, EpPath>({
    name: 'bp',
    config: messagePassingKnobs('bp').config,
    make: (c) => bpAlgorithm(g, messagePassingOptions(c)),
    contract: epContract,
  })
}
