/**
 * Expectation propagation from several starts, the runs weighted by their evidence, as an inference method on a model spec. One
 * EP run commits to one reading of a posterior that has several modes (which steps belong to a regime that is rarely visited,
 * which component is which), and reports the spread of that mode alone. Here the discrete variables of the graph (a regime
 * chain, a switch, a mixture's labels) are started from random messages, and from messages that rule out one category, EP runs
 * from each, the runs that reach the same fixed point are one mode, and the posterior is the mixture of the modes' beliefs
 * weighted by their EP evidence, which for a mode is its share of the posterior mass in the Gaussian (or gamma, beta)
 * approximation. A step is one start (the first is plain EP), so a snapshot is the mixture of the modes found so far.
 */

import { OmaiError } from '../../base/index.ts'
import { type CompiledGraph, epAlgorithm, type InitialMessage, type MpState } from '../../message-passing/index.ts'
import { foldIn, type Key, normal } from '../../random/index.ts'
import { bool, type DictSpace, dict, int, real } from '../../space/index.ts'
import { toFlat } from '../../tensor/index.ts'
import {
  type Algorithm,
  type Config,
  type Contract,
  defineContract,
  derived,
  Log,
  runAlgorithm,
  scalar,
} from '../../trace/index.ts'
import { factorGraphPosterior, factorGraphVerdict } from '../model/factors.ts'
import type { Posterior } from '../posterior.ts'
import { definePosterior } from '../posterior.ts'
import { recordOf, type SiteRecord, siteMixturePosterior } from './_site-mixture.ts'
import {
  flag,
  isProgramAnalysis,
  type MethodModel,
  type ModelAnalysis,
  no,
  num,
  type Prepared,
  posteriorJson,
  prepare,
  yes,
} from './_support.ts'
import { defineInferenceMethod, type InferenceMethod, type Verdict } from './method.ts'

/** The paths the multi-start method logs. */
export type EpMultiStartPath =
  | 'ep-multistart/starts'
  | 'ep-multistart/modes'
  | 'ep-multistart/logEvidence'
  | 'ep-multistart/snapshot'

/** One EP run from one start. */
export interface StartRun {
  /** Which start it was (0 is plain EP). */
  readonly start: number
  /** The EP estimate of log p(data) at the fixed point it reached (negative infinity when it has none). */
  readonly logEvidence: number
  readonly converged: boolean
  readonly sweeps: number
  readonly sites: Readonly<Record<string, SiteRecord>>
}

/** The state of {@link epMultiStartMethod}: the runs made so far. */
export interface EpMultiStartState {
  readonly runs: readonly StartRun[]
}

/** The knobs of {@link epMultiStartMethod} (the budget and snapshot knobs are added to them). */
export const epMultiStartKnobs: DictSpace = dict({
  starts: int(1, 2000, {
    default: 24,
    label: 'starts',
    description: 'the EP runs: the first is plain EP, then one per excluded category (when asked), then random ones',
  }),
  jitter: real(0, 10, {
    default: 2,
    label: 'start spread',
    description: 'the sd of the random initial log-probabilities of each discrete message',
  }),
  exclusions: bool({
    default: true,
    label: 'exclusion starts',
    description:
      'start also from messages that rule out one category of every discrete variable (a regime nobody visits)',
  }),
  merge: real(0.001, 1, {
    log: true,
    default: 0.05,
    label: 'merge tolerance',
    description: 'two runs are one mode when every site mean differs by less than this share of the larger sd',
  }),
  tol: real(1e-12, 1e-2, { log: true, default: 1e-9, description: 'the tolerance of each EP run' }),
  maxSweeps: int(1, 10_000, { default: 300, description: 'the sweeps of each EP run, at most' }),
  damping: real(0, 0.95, { default: 0, description: 'the damping of each EP run' }),
})

/** Options of {@link epMultiStartMethod}. */
export interface EpMultiStartOptions {
  /** The name of the method (default `ep-multistart`). */
  readonly name?: string
  /** How the rows of a plate are compiled (see `toFactorGraph`; default `vectorise`). */
  readonly plates?: 'vectorise' | 'expand'
  /** Whether scalars sharing a predictor are packed (default true). */
  readonly packScalars?: boolean
}

/** The ports of the graph that carry a discrete or Bernoulli message of a factor that is not a fixed prior: where a start is set. */
function discretePorts(g: CompiledGraph): { factor: number; port: number; size: number }[] {
  const out: { factor: number; port: number; size: number }[] = []
  g.factors.forEach((f, factor) => {
    if (f.operator.name === 'prior') return
    f.ports.forEach((p, port) => {
      if (p.edge >= 0 && (p.family.kind === 'discrete' || p.family.kind === 'bernoulli'))
        out.push({ factor, port, size: p.family.size })
    })
  })
  return out
}

/** The start messages of start number `s`: nothing for plain EP, else random log-probabilities, or ones that rule out category `off`. */
function startMessages(
  ports: readonly { factor: number; port: number; size: number }[],
  key: Key,
  jitter: number,
  off: number | undefined,
): InitialMessage[] {
  const total = ports.reduce((a, p) => a + p.size, 0)
  const noise = toFlat(normal(key, [total]))
  let at = 0
  return ports.map(({ factor, port, size }) => {
    const eta = Array.from({ length: size }, (_, q) => {
      const draw = (noise[at + q] ?? 0) * jitter
      return off === undefined ? draw : (q === off ? -6 : 0) + 0.25 * draw
    })
    at += size
    return { factor, port, eta }
  })
}

/** The runs grouped into modes: the best-evidence run of each set whose site means agree within `merge` of the larger sd. */
function modesOf(runs: readonly StartRun[], merge: number): StartRun[] {
  const usable = runs.filter((r) => Number.isFinite(r.logEvidence))
  const converged = usable.filter((r) => r.converged)
  const pool = (converged.length > 0 ? converged : usable).slice().sort((a, b) => b.logEvidence - a.logEvidence)
  const same = (a: StartRun, b: StartRun): boolean =>
    Object.entries(a.sites).every(([site, ra]) => {
      const rb = b.sites[site]
      if (rb === undefined) return false
      return ra.mean.every((m, i) => {
        const sd = Math.max(Math.sqrt(ra.variance[i] ?? 0), Math.sqrt(rb.variance[i] ?? 0), 1e-6)
        return Math.abs(m - (rb.mean[i] ?? 0)) <= merge * sd + 1e-9
      })
    })
  const modes: StartRun[] = []
  for (const r of pool) if (!modes.some((m) => same(m, r))) modes.push(r)
  return modes
}

/**
 * Expectation propagation from many starts, the distinct fixed points weighted by their evidence: the answer for a model whose
 * posterior over discrete structure has several modes (a regime that is hardly visited, the labels of mixture components), where
 * one run reports one mode with a spread that is far too small. The discrete messages of the graph are started at random and,
 * when `exclusions` is on, at messages that rule out one category; runs that end at the same fixed point are one mode; the
 * posterior is the mixture of the modes' beliefs with weights proportional to their EP evidence, and the evidence of the model is
 * the log sum of the modes' (the modes found: a mode that no start reaches, or the relabellings of a symmetric one, are not in it).
 *
 * @param model - A bound spec.
 * @param options - The name and how plates are compiled.
 * @returns The method, with `check` (the model compiles and has a discrete latent) and `posterior`.
 * @concept expectation-propagation
 * @index estimator Expectation propagation from several starts, the fixed points weighted by their evidence.
 * @example
 * ```ts
 * import { epMultiStartMethod } from 'omai-engine/probprog/inference'
 * import { defineModel, ops } from 'omai-engine/probprog/model'
 * import { bernoulli, normal } from 'omai-engine/probprog/model/dists'
 * import { session } from 'omai-engine/trace'
 * // A coin picks which of two means a reading is drawn around: a reading between them has two explanations.
 * const spec = defineModel('m', (b) => {
 *   const side = b.latent('side', bernoulli({ probs: 0.5 }))
 *   b.observe('y', normal(ops.where(side, 3, -3), 2), b.data('y'))
 * })
 * const m = epMultiStartMethod(spec.bind({ y: 0.5 }))
 * const run = session(m, { steps: 10, mode: 'eager', config: { starts: 6 } })
 * [run.status, Number(m.posterior?.(run.state, run.config).diagnostics['modes']) >= 1] // => ['completed', true]
 * ```
 */
export function epMultiStartMethod(
  model: MethodModel,
  options: EpMultiStartOptions = {},
): InferenceMethod<EpMultiStartState, Config, EpMultiStartPath, ModelAnalysis> {
  const name = options.name ?? 'ep-multistart'
  const prepared: Prepared | undefined =
    typeof model === 'function' ? undefined : prepare(model, options.plates ?? 'vectorise', options.packScalars ?? true)
  const check = (a: ModelAnalysis): Verdict => {
    if (prepared === undefined || isProgramAnalysis(a))
      return no(
        `${name} runs on a model spec: a closure program has no factor graph`,
        ['a model spec'],
        'write the model with defineModel',
      )
    if (prepared.failure !== undefined)
      return no(`the factor graph does not compile: ${prepared.failure}`, ['a valid factor graph'])
    const v = factorGraphVerdict(prepared.compiled, prepared.graph, 'ep')
    if (!v.ok) return no(v.reason, v.missing, 'see toFactorGraph(model).issues')
    if (prepared.graph === undefined || discretePorts(prepared.graph).length === 0)
      return no(
        `${name} starts the discrete messages of the graph from different values, and the model has no discrete or Bernoulli variable`,
        ['a discrete latent'],
        'use ep',
      )
    return yes(
      'expectation propagation from several starts of the discrete messages, the distinct fixed points weighted by their evidence',
    )
  }

  const posterior = (state: EpMultiStartState, config: Config): Posterior => {
    if (prepared?.graph === undefined || state.runs.length === 0)
      return definePosterior({
        form: 'variational',
        engine: name,
        sites: [],
        status: 'failed',
        diagnostics: { reason: 'no run yet' },
      })
    const modes = modesOf(state.runs, num(config, 'merge'))
    if (modes.length === 0)
      return definePosterior({
        form: 'variational',
        engine: name,
        sites: [],
        status: 'failed',
        diagnostics: { reason: 'no start reached a fixed point with an evidence' },
      })
    const top = Math.max(...modes.map((m) => m.logEvidence))
    const weights = modes.map((m) => Math.exp(m.logEvidence - top))
    const logEvidence = top + Math.log(weights.reduce((a, b) => a + b, 0))
    return siteMixturePosterior({
      engine: name,
      status: state.runs.length >= num(config, 'starts') ? 'completed' : 'stopped',
      logEvidence: { value: logEvidence, method: 'ep-multistart' },
      diagnostics: {
        starts: state.runs.length,
        modes: modes.length,
        convergedStarts: state.runs.filter((r) => r.converged).length,
        modeEvidence: modes.map((m) => m.logEvidence),
      },
      weights,
      runs: modes.map((m) => m.sites),
    })
  }

  return defineInferenceMethod<EpMultiStartState, Config, EpMultiStartPath, ModelAnalysis>({
    name,
    family: 'message-passing',
    description:
      'expectation propagation from several starts of the discrete messages, the distinct fixed points mixed by their evidence',
    config: epMultiStartKnobs,
    check,
    posterior,
    defaults: { 'ep-multistart/starts': Log(), 'ep-multistart/modes': Log(), 'ep-multistart/logEvidence': Log() },
    contract: (config): Contract<EpMultiStartState, EpMultiStartPath> =>
      defineContract({
        'ep-multistart/starts': scalar({ description: 'EP runs made so far' }),
        'ep-multistart/modes': scalar({ description: 'the distinct fixed points found so far' }),
        'ep-multistart/logEvidence': scalar({ description: "the log sum of the modes' EP evidence" }),
        'ep-multistart/snapshot': derived<EpMultiStartState>({
          cost: 'dear',
          description: 'the evidence-weighted mixture of the modes found so far as JSON',
          compute: (s) => posteriorJson(posterior(s, config)),
        }),
      }),
    make: (config, key): Algorithm<EpMultiStartState, EpMultiStartPath> => {
      const starts = num(config, 'starts')
      const jitter = num(config, 'jitter')
      const exclusions = flag(config, 'exclusions')
      const settings = { tol: num(config, 'tol'), maxSweeps: num(config, 'maxSweeps'), damping: num(config, 'damping') }
      const graph = (): CompiledGraph => {
        if (prepared?.graph === undefined)
          throw new OmaiError('value', `${name}: ${prepared?.failure ?? 'the model has no factor graph'}`)
        return prepared.graph
      }
      const run = (start: number): StartRun => {
        const g = graph()
        const ports = discretePorts(g)
        const categories = Math.max(0, ...ports.map((p) => p.size))
        // Start 0 is plain EP; with exclusions, starts 1..K rule out category 0..K-1; the rest are random.
        const off = exclusions && start >= 1 && start <= categories ? start - 1 : undefined
        const initial = start === 0 ? [] : startMessages(ports, foldIn(key, start), jitter, off)
        const r = runAlgorithm(epAlgorithm(g, { ...settings, ...(initial.length === 0 ? {} : { initial }) }), {
          maxSteps: settings.maxSweeps + 2,
          seed: 1,
        })
        const post = factorGraphPosterior((prepared as Prepared).compiled, g, r.state as MpState, { engine: name })
        const sites: Record<string, SiteRecord> = {}
        for (const s of post.sites) sites[s] = recordOf(post.marginal(s))
        const ev = post.logEvidence?.value
        return {
          start,
          logEvidence: ev === undefined || !Number.isFinite(ev) ? Number.NEGATIVE_INFINITY : ev,
          converged: r.status === 'converged',
          sweeps: r.steps,
          sites,
        }
      }
      return {
        name: 'probprog/ep-multistart-method',
        config: { starts, jitter, exclusions, merge: num(config, 'merge'), ...settings },
        init: (): EpMultiStartState => {
          graph()
          return { runs: [] }
        },
        step: (s, ctx) => {
          const next: EpMultiStartState = { runs: [...s.runs, run(s.runs.length)] }
          const modes = modesOf(next.runs, num(config, 'merge'))
          ctx.log('ep-multistart/starts', next.runs.length)
          ctx.log('ep-multistart/modes', modes.length)
          if (modes.length > 0) {
            const top = Math.max(...modes.map((m) => m.logEvidence))
            ctx.log(
              'ep-multistart/logEvidence',
              top + Math.log(modes.reduce((a, m) => a + Math.exp(m.logEvidence - top), 0)),
            )
          }
          return next
        },
        done: (s) => (s.runs.length >= starts ? 'completed' : undefined),
        serialise: (s) => JSON.parse(JSON.stringify(s)),
        deserialise: (j) => j as unknown as EpMultiStartState,
      }
    },
  })
}
