/**
 * Run expectation propagation on a bound model: `runEp` compiles the model, asks `epRoute` which method applies (plain EP, or EP in
 * a grid over the hyperparameters a covariance depends on), runs it sweep by sweep and returns the beliefs at every sweep, the
 * messages as a model-run record and the evidence, or the library's reason for refusing; the beliefs are read for the quantities
 * the caller names, or for every site of the model. Nothing here knows an atom: the suite's `atomRun` is this with the atom's
 * quantities, and a program page calls it on its own model.
 */

import { type ModelRun, modelRunFromTape } from '../../exposition/index.ts'
import { toFlat } from '../../tensor/index.ts'
import { type Config, type Session, session } from '../../trace/index.ts'
import {
  checkKnobKeys,
  type EpGridState,
  epRoute,
  epRunKnobs,
  epSpecMethod,
  type InferenceMethod,
  type ModelAnalysis,
} from '../inference/index.ts'
import { type BoundModel, bind, doSpec, toStructure } from '../model/index.ts'
import type { Posterior } from '../posterior.ts'
import type { AtomBuild, AtomQuantity, AtomSupport, Belief, EpRun, SweepBeliefs } from './atom-types.ts'

/** The messages of EP at the most probable node of a grid run: the model conditioned on the hyperparameters there, run as a plain EP session. */
function gridRecord(bound: BoundModel, state: EpGridState): ModelRun {
  let best = state.nodes[0]
  for (const n of state.nodes)
    if (best === undefined || n.logEvidence + n.logPrior > best.logEvidence + best.logPrior) best = n
  const values = Object.fromEntries(state.axes.map((a, k) => [a.name, best?.x[k] ?? 0]))
  const cond = bind({ spec: doSpec(bound.spec, values), customs: bound.customs }, bound.data, bound.options)
  const s = session(epSpecMethod(cond), { seed: 1, steps: 202, mode: 'eager', config: { tol: 1e-9 } })
  return modelRunFromTape(toStructure(cond, { view: 'factor' }), { run: s.run() }, { maxSteps: 120, maxMessages: 6 })
}

/** Moments of a quantity (`name` or `site[i]`) in a posterior. */
function momentsOf(
  posterior: Pick<Posterior, 'mean' | 'stddev' | 'sites'>,
  name: string,
  support: AtomSupport,
): Belief {
  const m = /^(.*)\[(\d+)\]$/.exec(name)
  const site = m?.[1] ?? name
  const i = Number(m?.[2] ?? 0)
  if (!posterior.sites.includes(site)) return { mean: Number.NaN, sd: Number.NaN, proper: false }
  const mean = toFlat(posterior.mean(site))[i] ?? Number.NaN
  const sd = toFlat(posterior.stddev(site))[i] ?? Number.NaN
  return {
    mean,
    sd,
    proper: Number.isFinite(mean) && Number.isFinite(sd) && sd < 1e5 && (sd > 0 || support === 'probability'),
  }
}

/**
 * The beliefs of an atom's quantities in a posterior: the mean and standard deviation of each (a derived quantity from the
 * beliefs before it), and whether each is a proper belief. What a run of any message-passing method reads at a sweep.
 *
 * @param problem - The problem (the quantities and their supports).
 * @param posterior - The posterior of a message-passing state.
 * @returns The beliefs by quantity name.
 * @example
 * ```ts
 * import { atomBeliefs, atomBound, epAtom } from 'omai-engine/probprog/ep'
 * import { epSpecMethod } from 'omai-engine/probprog/inference'
 * import { session } from 'omai-engine/trace'
 * const gaussian = epAtom('gaussian')
 * const problem = gaussian.problem(gaussian.defaults)
 * const method = epSpecMethod(atomBound(problem))
 * const s = session(method, { steps: 50, mode: 'eager' })
 * const posterior = method.posterior?.(s.state, s.config)
 * posterior === undefined ? false : atomBeliefs(problem, posterior).mu?.proper // => true
 * ```
 */
export function atomBeliefs(
  problem: Pick<AtomBuild, 'quantities'>,
  posterior: Pick<Posterior, 'mean' | 'stddev' | 'sites'>,
): Record<string, Belief> {
  const out: Record<string, Belief> = {}
  for (const q of problem.quantities) {
    const b = q.derive === undefined ? momentsOf(posterior, q.name, q.support) : q.derive(out)
    const spread = q.support === 'probability' ? b.sd >= 0 : b.sd > 0
    out[q.name] = { ...b, proper: b.proper && Number.isFinite(b.mean) && Number.isFinite(b.sd) && spread }
  }
  return out
}

/** What `runEp` is told besides the model. */
export interface EpRunOptions {
  /** The quantities to read the beliefs of, by site name (`mu`, `w[0]`); every site of the model, element by element, when absent. */
  readonly quantities?: readonly AtomQuantity[]
  /** Knobs of EP over its defaults (damping, schedule, tolerance, the sweep cap, the power); a key that is not a knob throws. */
  readonly ep?: Config
  /** Keep no tape and no per-sweep beliefs (only the final ones): what a calibration over many seeds needs. */
  readonly light?: boolean
}

/** Every site of a posterior, element by element, as quantities on the line. */
function sitesAsQuantities(posterior: Pick<Posterior, 'mean' | 'sites'>): readonly AtomQuantity[] {
  return posterior.sites.flatMap((site): AtomQuantity[] => {
    const size = toFlat(posterior.mean(site)).length
    return size === 1 && posterior.mean(site).shape.length === 0
      ? [{ name: site, label: site, support: 'real' }]
      : Array.from({ length: size }, (_, i) => ({
          name: `${site}[${i}]`,
          label: `${site}[${i}]`,
          support: 'real' as const,
        }))
  })
}

/**
 * Run expectation propagation on a bound model through the route that applies, sweep by sweep, and keep the beliefs at each sweep
 * and the messages as a model-run record. A model the library refuses is not run: the result carries the library's verdict.
 *
 * @param bound - The bound model (`definition.bind(data)`).
 * @param options - The quantities to read, the knobs of EP over its defaults, `light`.
 * @returns The run, the failure, or the refusal.
 * @throws OmaiError with code `value` for a key of `options.ep` that is not a knob of the method (a misspelled knob would run on the defaults).
 * @concept expectation-propagation
 * @index function Run EP on a bound model: beliefs per sweep, a record, or the refusal.
 * @example
 * ```ts
 * import { runEp } from 'omai-engine/probprog/ep'
 * import { defineModel } from 'omai-engine/probprog/model'
 * import * as d from 'omai-engine/probprog/model/dists'
 * const model = defineModel('mean', (m) => {
 *   const mu = m.latent('mu', d.normal(0, 2))
 *   m.observe('y', d.normal(mu, 1), m.data('y', { dims: ['i'], role: 'response' }))
 * })
 * const r = runEp(model.bind({ y: [1.2, 0.4, 1.9] }))
 * r.status === 'ran' ? Object.keys(r.beliefs) : r.status // => ['mu']
 * ```
 */
export function runEp(bound: BoundModel, options: EpRunOptions = {}): EpRun {
  const ep = options.ep ?? {}
  // region inference
  // compile to a factor graph and run EP on it: a rule for every factor
  // (a covariance that is a custom function of a random value makes it a grid over that value)
  const route = epRoute(bound)
  // endregion
  // a misspelled knob is an error, not a run on the defaults
  checkKnobKeys('runEp', ep, route.method?.config ?? epRunKnobs('ep'))
  if (route.via === 'none') return { status: 'refused', reason: route.verdict.reason, missing: route.verdict.missing }
  const config: Config = { tol: 1e-9, ...ep }
  const maxSweeps = typeof config['maxSweeps'] === 'number' ? config['maxSweeps'] : 200
  const logs = options.light === true ? 'off' : 'all'
  const run = { quantities: options.quantities, config, logs, light: options.light === true } as const
  return route.via === 'ep'
    ? runRoute(run, route.method, maxSweeps + 2, (s) => ({
        converged: s.status === 'converged',
        record: () =>
          modelRunFromTape(toStructure(bound, { view: 'factor' }), { run: s.run() }, { maxSteps: 120, maxMessages: 6 }),
      }))
    : runRoute<EpGridState>(run, route.method, 1000, (s, posterior) => ({
        converged: posterior?.status === 'converged',
        record: () => gridRecord(bound, s.state),
      }))
}

/** How a run of one route is read: whether it converged and the messages as a model-run record. */
interface RouteReading {
  readonly converged: boolean
  readonly record: () => ModelRun
}

/** Run the method of a route sweep by sweep (one step of the grid method is one EP run) and read the beliefs at the sweeps kept. */
function runRoute<S>(
  run: {
    readonly quantities: readonly AtomQuantity[] | undefined
    readonly config: Config
    readonly logs: 'off' | 'all'
    readonly light: boolean
  },
  method: InferenceMethod<S, Config, string, ModelAnalysis>,
  steps: number,
  reading: (s: Session<S, Config, string>, posterior: Posterior | undefined) => RouteReading,
): EpRun {
  const { quantities, config, logs, light } = run
  let s: Session<S, Config, string>
  try {
    // region inference
    // one step is one sweep of EP over the factors, recorded
    s = session(method, { seed: 1, steps, mode: 'eager', config, logs })
    // endregion
  } catch (e) {
    return { status: 'failed', reason: e instanceof Error ? e.message : String(e) }
  }
  const read = (state: S): Record<string, Belief> => {
    // region inference
    // the beliefs by site
    const posterior = method.posterior?.(state, s.config)
    // endregion
    return posterior === undefined
      ? {}
      : atomBeliefs({ quantities: quantities ?? sitesAsQuantities(posterior) }, posterior)
  }
  const last = s.t
  const keep = new Set<number>(light ? [last] : [0, last])
  const stride = Math.max(1, Math.ceil(last / 60))
  if (!light) for (let t = 0; t <= last; t += stride) keep.add(t)
  const history = [...keep]
    .filter((t) => t >= 0 && t <= last)
    .sort((a, b) => a - b)
    .map((t): SweepBeliefs => ({ sweep: t, beliefs: read(s.stateAt(t)) }))
  const final = read(s.state)
  const posterior = method.posterior?.(s.state, s.config)
  const evidence = posterior?.logEvidence?.value
  const { converged, record } = reading(s, posterior)
  return {
    status: 'ran',
    converged,
    sweeps: last,
    beliefs: final,
    history,
    record,
    ...(evidence === undefined ? {} : { evidence }),
  }
}
