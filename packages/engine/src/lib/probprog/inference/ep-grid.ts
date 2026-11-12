/**
 * Expectation propagation inside a grid over the hyperparameters, as an inference method on a model spec: the scalar latents
 * that a `custom` function reads (the length scale of a kernel whose Gram matrix is an opaque function, a bandwidth) are set
 * to each value of a grid in turn (`doSpec`), so the function's arguments are known and the compiler reads it as a constant
 * matrix (or, for a function that is affine in a random vector, as a matrix on it); expectation propagation runs on what is
 * left, and the log evidence of each run, which is the exact `log p(data | hyperparameter)` wherever EP is exact (a Gaussian
 * process with a Gaussian likelihood), weights the node together with the prior of the hyperparameter. The posterior is the
 * evidence-weighted mixture of the nodes' posteriors: the hyperparameters' own marginals on the grid, each other site's marginal
 * the mixture of the nodes' beliefs. One node is a step, in an order that refines the grid (the two ends, then bisections), so a
 * snapshot at any step is a coarser grid of the same posterior; when the posterior still has mass at an edge of the grid the
 * grid is extended outwards, within a budget of nodes. Two options make a fit with a few hundred sweeps per node cheap: `order:
 * 'walk'` starts in the middle of the range and steps outwards, best first, to the lattice neighbours of the nodes that still
 * have mass (within `window` nats of the best), so the nodes in the tails are never run, and `warm` starts each node from the
 * messages of the nearest settled one, which settles in a fraction of the sweeps. With one hyperparameter the walk is an
 * interval; with several it is a flood fill of the region of the lattice where the posterior has mass, whose cost follows the
 * volume of that region and not the product of the axes (three scalars, or an ARD length scale per input beside an amplitude and
 * a noise level, are a few hundred to a few thousand nodes of a few milliseconds each), and it is what a method over several
 * hyperparameters opens on. The hyperparameters are the scalar latents a `custom` function reads, and the components of a
 * plated vector of them with independent priors (an ARD kernel's length scale per input), or the latents named in `over`; a
 * noise level joins the grid by being read by a custom function too. A node that did not settle (an EP run that cycles) weighs
 * the grid only where its evidence does not exceed the best settled node's.
 */

import { OmaiError } from '../../base/index.ts'
import { bijectors, type Distribution } from '../../dists/index.ts'
import {
  type CompiledGraph,
  compileGraph,
  epAlgorithm,
  defaultOperators as factorOperators,
  type MpState,
} from '../../message-passing/index.ts'
import { bool, choice, type DictSpace, dict, int, real } from '../../space/index.ts'
import { add, arange, exp, item, log, max, sub, sum, tensor, toFlat, type Value } from '../../tensor/index.ts'
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
import { type BoundModel, bind } from '../model/bind.ts'
import { doSpec } from '../model/compose.ts'
import { distributionFamilies } from '../model/dists/families.ts'
import { factorGraphPosterior } from '../model/factor-graph/readback.ts'
import type { FactorGraphCompilation } from '../model/factor-graph/types.ts'
import { factorGraphVerdict } from '../model/factor-graph/verdict.ts'
import { toFactorGraph } from '../model/factors.ts'
import type { ExprSpec, ModelNode, ModelSpec } from '../model/spec.ts'
import { definePosterior, type Posterior } from '../posterior.ts'
import { walkNext } from './_ep-grid-walk.ts'
import { nearestSettled, settledMessages, warmStart } from './_ep-grid-warm.ts'
import { recordOf, type SiteRecord, siteMixturePosterior } from './_site-mixture.ts'
import { isProgramAnalysis, type MethodModel, type ModelAnalysis, no, num, posteriorJson, yes } from './_support.ts'
import { atPower, epRunKnobs, epRunSettings, powerOf } from './ep-knobs.ts'
import { defineInferenceMethod, type InferenceMethod, type Verdict } from './method.ts'

/** The paths the grid method logs. */
export type EpGridPath = 'ep-grid/nodes' | 'ep-grid/logEvidence' | 'ep-grid/sweeps' | 'ep-grid/snapshot'

/** What one node keeps of a site's belief (see `SiteRecord`). */
export type GridSiteRecord = SiteRecord

/** One run of EP at a value of the hyperparameters. */
export interface GridNode {
  /** The position on each axis (an integer index, as the axes number their points). */
  readonly index: readonly number[]
  /** The hyperparameters on the unconstrained scale and on their own. */
  readonly u: readonly number[]
  readonly x: readonly number[]
  /** The EP estimate of `log p(data | x)`: exact where EP is exact. */
  readonly logEvidence: number
  /** The log prior density of the hyperparameters on the unconstrained scale (the density of `x` times the Jacobian). */
  readonly logPrior: number
  readonly converged: boolean
  readonly sweeps: number
  readonly sites: Readonly<Record<string, GridSiteRecord>>
  /** The message on every edge of the run's last state: the start of a nearby node when `warm` is on. */
  readonly messages?: readonly (readonly number[])[]
}

/** One axis of the grid: the hyperparameter, where point 0 is, the spacing, and the range of point indices run so far. */
export interface GridAxis {
  readonly name: string
  readonly u0: number
  readonly du: number
  readonly from: number
  readonly to: number
}

/** EP inside a grid over hyperparameters, as a method. */
export type EpGridMethod = InferenceMethod<EpGridState, Config, EpGridPath, ModelAnalysis>

/** The state of {@link epGridMethod}: the axes, the nodes still to run (in order) and the nodes run. */
export interface EpGridState {
  readonly axes: readonly GridAxis[]
  readonly pending: readonly (readonly number[])[]
  readonly nodes: readonly GridNode[]
  /** How many times the grid has been extended. */
  readonly extensions: number
}

/**
 * The knobs of {@link epGridMethod} for one hyperparameter (the budget and snapshot knobs are added to them). A method over
 * several (`epGridKnobsFor(2)`, the method's own space) opens on a walk with warm starts, a coarser lattice and a budget that
 * fits its region: the product of the axes is out of reach, the region of the lattice where the posterior has mass is not.
 */
export const epGridKnobs: DictSpace = epGridKnobsFor(1)

/**
 * The knobs of EP inside a grid over `axes` hyperparameters: the same keys, with the defaults of a product grid for one and of
 * a walk over the region of mass for more (15 points per axis, warm starts, a window of 7 nats, a budget of 4000 nodes).
 *
 * @param axes - The number of axes of the grid (scalar hyperparameters, plus the components of the plated vectors).
 * @returns The space.
 * @example
 * ```ts
 * import { epGridKnobsFor } from 'omai-engine/probprog/inference'
 * import { defaults } from 'omai-engine/space'
 * const [one, three] = [defaults(epGridKnobsFor(1)), defaults(epGridKnobsFor(3))]
 * [one['order'], three['order'], three['points']] // => ['bisect', 'walk', 15]
 * ```
 */
export function epGridKnobsFor(axes: number): DictSpace {
  const several = axes > 1
  return dict({
    points: int(3, 401, {
      default: several ? 15 : 31,
      label: 'points',
      description:
        'points per hyperparameter on the first grid (spaced evenly on the unconstrained scale, so a positive one is log-spaced); with several hyperparameters the spacing of the lattice the walk visits',
    }),
    tail: real(1e-12, 0.45, {
      log: true,
      default: 1e-4,
      label: 'prior tail',
      description: 'the prior mass left out at each end of the first grid',
    }),
    maxNodes: int(3, 100_000, {
      default: several ? 4000 : 400,
      label: 'node budget',
      description: 'the most EP runs, extensions of the grid included',
    }),
    edge: real(1e-8, 1, {
      log: true,
      default: 1e-3,
      label: 'edge mass',
      description: 'extend the grid where an edge node has this share of the heaviest node',
    }),
    warm: bool({
      default: several,
      label: 'warm start',
      description:
        'start each node from the messages of the nearest settled node instead of the flat ones: a node near a settled one settles in a fraction of the sweeps, and keeps to the fixed point of its neighbours (one whose factors cycle or collapse from a flat start)',
    }),
    order: choice(['bisect', 'walk'], {
      default: several ? 'walk' : 'bisect',
      label: 'order',
      description:
        'the order of the nodes. bisect: the two ends, then midpoints (any snapshot is a coarser grid of the whole range; the product of the axes, so one or two hyperparameters). walk: start at the middle of the range and step outwards, best first, to the lattice neighbours of the nodes that are within `window` nats of the best (with one hyperparameter, on the side whose edge node weighs more), each node started from its neighbour (with warm); only the region where the posterior has mass is run, so the cost follows its volume and not the product of the axes (a snapshot covers the nodes run so far)',
    }),
    window: real(0.5, 1000, {
      log: true,
      default: several ? 7 : 10,
      label: 'evidence window',
      description:
        'walk order: stop on a side when its edge node has a log evidence plus log prior this many nats below the best node (the Gaussian tail is half the square of the sds it is out: 10 is 4.5 sds, 7 is 3.7)',
    }),
    // The knobs of each EP run (schedule, damping, improper, tol, maxSweeps, restarts, power): the shared declaration.
    ...epRunKnobs('ep-grid').spaces,
  })
}

/** Options of {@link epGridMethod}. */
export interface EpGridOptions {
  /** The name of the method (default `ep-grid`). */
  readonly name?: string
  /**
   * The latents to put on the grid (a scalar, or a plated vector whose components are independent and share a prior; the prior's
   * parameters are known numbers). Default: the scalar latents and plated vectors that a `custom` function reads directly.
   */
  readonly over?: readonly string[]
  /** How the rows of a plate are compiled (see `toFactorGraph`; default `vectorise`). */
  readonly plates?: 'vectorise' | 'expand'
  /** Whether scalars sharing a predictor are packed (default true). */
  readonly packScalars?: boolean
}

/** A hyperparameter on the grid: its node, the prior as a distribution, and the map from the real line. */
interface Hyper {
  readonly name: string
  readonly prior: Distribution
  readonly map: bijectors.Bijector
  /** The unconstrained interval of the first grid. */
  readonly lo: number
  readonly hi: number
}

const num1 = (v: Value): number => item(v)

/** The custom calls of a spec and the names they read directly. */
function customReads(spec: ModelSpec): Set<string> {
  const out = new Set<string>()
  const walk = (e: ExprSpec): void => {
    if (e.kind === 'custom') for (const a of e.args) if (a.kind === 'ref') out.add(a.name)
    if (e.kind === 'op' || e.kind === 'custom') for (const a of e.args) walk(a)
  }
  for (const n of spec.nodes) {
    if (n.kind === 'latent' || n.kind === 'observed' || n.kind === 'generated') {
      for (const e of Object.values(n.dist.params)) walk(e)
      if (n.kind === 'observed') walk(n.value)
    } else if (n.kind === 'derived' || n.kind === 'factor') walk(n.expr)
  }
  return out
}

/** The latents to put on the grid: the named ones, or the scalar latents a custom function reads. */
function hyperNames(spec: ModelSpec, over: readonly string[] | undefined): string[] {
  if (over !== undefined) return [...over]
  const reads = customReads(spec)
  return spec.nodes
    .filter((n) => n.kind === 'latent' && n.dims.length === 0 && n.eventRank === 0 && reads.has(n.name))
    .map((n) => n.name)
}

/** The plated vectors a custom function reads directly: candidates for the grid (an ARD kernel's length scales) when the scalars alone leave no graph. */
function vectorReads(spec: ModelSpec, sizes: Readonly<Record<string, number>>): string[] {
  const reads = customReads(spec)
  return spec.nodes
    .filter(
      (n) =>
        n.kind === 'latent' &&
        n.dims.length === 1 &&
        n.eventRank === 0 &&
        reads.has(n.name) &&
        sizes[n.dims[0] as string] !== undefined,
    )
    .map((n) => n.name)
}

/** One axis of the grid: a scalar latent, or one component of a plated vector of independent hyperparameters (`ell[1]`). */
interface Target {
  /** The latent node. */
  readonly node: string
  /** The component of a vector latent, or none for a scalar. */
  readonly component?: number
  /** The name of the axis (`ell`, `ell[1]`). */
  readonly label: string
}

/** The axes of the latents on the grid: one for a scalar, one per component for a vector (its size from the bound dimensions). */
function targetsOf(spec: ModelSpec, names: readonly string[], sizes: Readonly<Record<string, number>>): Target[] {
  return names.flatMap((name): Target[] => {
    const node = spec.nodes.find((x) => x.name === name)
    if (node === undefined || node.kind !== 'latent')
      throw new OmaiError('value', `ep-grid: '${name}' is not a latent node`)
    if (node.dims.length === 0) return [{ node: name, label: name }]
    const dim = node.dims[0] as string
    const size = sizes[dim]
    if (node.dims.length !== 1 || size === undefined)
      throw new OmaiError(
        'value',
        `ep-grid: '${name}' is not a scalar or a plated vector of known size: only those go on a grid`,
      )
    return Array.from({ length: size }, (_, i): Target => ({ node: name, component: i, label: `${name}[${i}]` }))
  })
}

/** The prior of a hyperparameter as a distribution, when its parameters are known numbers. */
function priorOf(node: ModelNode, tail: number, label: string = node.name): Hyper {
  if (node.kind !== 'latent') throw new OmaiError('value', `ep-grid: '${node.name}' is not a latent node`)
  if (node.dims.length > 1 || node.eventRank !== 0)
    throw new OmaiError(
      'value',
      `ep-grid: '${node.name}' is not a scalar or a plated vector: only scalar hyperparameters go on a grid`,
    )
  if (node.constraint !== undefined)
    throw new OmaiError('value', `ep-grid: '${node.name}' carries a constraint, which the grid does not read`)
  const family = distributionFamilies()[node.dist.family]
  if (family === undefined) throw new OmaiError('value', `ep-grid: unknown family '${node.dist.family}'`)
  const args: Record<string, ReturnType<typeof tensor>> = {}
  for (const [p, e] of Object.entries(node.dist.params)) {
    if (e.kind !== 'const' || typeof e.value !== 'number')
      throw new OmaiError(
        'value',
        `ep-grid: the prior of '${node.name}' has a parameter that is not a known number (${p})`,
      )
    args[p] = tensor(e.value)
  }
  for (const p of family.params) {
    const d = family.defaults[p]
    if (args[p] === undefined && d !== undefined) args[p] = tensor(d)
  }
  const prior = family.make(args)
  const map = bijectors.bijectorForSupport(prior.support)
  const lo = num1(map.inverse(prior.ppf(tail)))
  const hi = num1(map.inverse(prior.ppf(1 - tail)))
  if (!(Number.isFinite(lo) && Number.isFinite(hi) && hi > lo))
    throw new OmaiError('value', `ep-grid: the prior of '${node.name}' has no finite range at the tail ${tail}`)
  return { name: label, prior, map, lo, hi }
}

/** The log prior density of a hyperparameter on the unconstrained scale at `u`: the density at `x` times the Jacobian. */
function logPriorAt(h: Hyper, u: number): number {
  const x = h.map.forward(tensor(u))
  return num1(h.prior.logProb(x)) + num1(h.map.forwardLogDetJacobian(tensor(u)))
}

/** The order the points of one axis are run in: the two ends, then the midpoints of the gaps, level by level. */
function bisectionOrder(n: number): number[] {
  if (n <= 2) return toFlat(arange(n))
  const order = [0, n - 1]
  let gaps: [number, number][] = [[0, n - 1]]
  while (gaps.length > 0) {
    const next: [number, number][] = []
    for (const [a, b] of gaps) {
      if (b - a < 2) continue
      const mid = Math.floor((a + b) / 2)
      order.push(mid)
      next.push([a, mid], [mid, b])
    }
    gaps = next
  }
  return order
}

/** The tuples of point indices of a product grid, ordered so that the coarse nodes come first. */
function firstNodes(axes: readonly GridAxis[]): number[][] {
  const ranks = axes.map((a) => {
    const order = bisectionOrder(a.to - a.from + 1)
    const rank = new Map<number, number>()
    for (const [r, i] of order.entries()) rank.set(a.from + i, r)
    return rank
  })
  let tuples: number[][] = [[]]
  axes.forEach((a) => {
    const next: number[][] = []
    for (const t of tuples) for (let i = a.from; i <= a.to; i++) next.push([...t, i])
    tuples = next
  })
  const key = (t: readonly number[]): number[] => {
    const rs = t.map((i, k) => ranks[k]?.get(i) ?? 0)
    return [Math.max(...rs), rs.reduce((s, r) => s + r, 0)]
  }
  return tuples.sort((a, b) => {
    const [ka, kb] = [key(a), key(b)]
    return (ka[0] ?? 0) - (kb[0] ?? 0) || (ka[1] ?? 0) - (kb[1] ?? 0)
  })
}

/** The normalised weights of the nodes: the evidence of each EP run times the prior density, with the width of the cell of each node. */
function nodeWeights(state: EpGridState): { readonly logWeights: readonly number[]; readonly logEvidence: number } {
  const nodes = state.nodes
  const widths = state.axes.map((_, k) => {
    const values = [...new Set(nodes.map((n) => n.u[k] ?? 0))].sort((a, b) => a - b)
    return new Map(
      values.map((v, i) => {
        const left = i === 0 ? v : (values[i - 1] ?? v)
        const right = i === values.length - 1 ? v : (values[i + 1] ?? v)
        const w = (right - left) / 2
        return [v, w > 0 ? w : (state.axes[k]?.du ?? 1)] as const
      }),
    )
  })
  // A run that did not settle is not at a fixed point, and the EP energy at a state that cycles or has collapsed (a precision
  // sent to zero) can be far above the evidence of the nodes that did: when any node settled, a node that did not weighs the grid
  // only where its evidence does not exceed the best settled node's (a cycling state in the tail of the posterior is still a
  // fair estimate of the evidence there; one that overshoots the settled nodes is not to be trusted, and carries no weight).
  let bestSettled = Number.NEGATIVE_INFINITY
  for (const n of nodes)
    if (n.converged && n.logEvidence + n.logPrior > bestSettled) bestSettled = n.logEvidence + n.logPrior
  const settledOnly = nodes.some((n) => n.converged)
  const scores = nodes.map((n) => {
    const score = n.logEvidence + n.logPrior
    const trusted = !settledOnly || n.converged || score <= bestSettled
    let lw = trusted ? score : Number.NEGATIVE_INFINITY
    n.u.forEach((v, k) => {
      lw += Math.log(widths[k]?.get(v) ?? 1)
    })
    return Number.isNaN(lw) ? Number.NEGATIVE_INFINITY : lw
  })
  if (scores.length === 0) return { logWeights: [], logEvidence: Number.NEGATIVE_INFINITY }
  const logw = tensor(scores)
  const top = max(logw)
  const logZ = add(top, log(sum(exp(sub(logw, top)))))
  return { logWeights: toFlat(sub(logw, logZ)), logEvidence: item(logZ) }
}

/** Where the grid should grow: the axes and the side whose edge nodes still carry mass. */
function growth(state: EpGridState, edge: number): { axis: number; side: -1 | 1 }[] {
  const { logWeights } = nodeWeights(state)
  let top = Number.NEGATIVE_INFINITY
  for (const v of logWeights) if (v > top) top = v
  const out: { axis: number; side: -1 | 1 }[] = []
  state.axes.forEach((a, k) => {
    for (const side of [-1, 1] as const) {
      const at = side === -1 ? a.from : a.to
      let heaviest = Number.NEGATIVE_INFINITY
      state.nodes.forEach((n, i) => {
        if (n.index[k] === at && (logWeights[i] ?? Number.NEGATIVE_INFINITY) > heaviest)
          heaviest = logWeights[i] ?? heaviest
      })
      if (heaviest - top > Math.log(edge)) out.push({ axis: k, side })
    }
  })
  return out
}

/**
 * Expectation propagation inside a grid over hyperparameters, with the evidence of each EP run as the weight of its node: the
 * answer for a model whose factor graph exists only once some scalar latents are known, such as a Gaussian process with a random
 * length scale (the Gram matrix is a function of it). The grid is over the scalar latents a `custom` function reads (or the ones
 * named in `over`), laid out evenly on the unconstrained scale between the prior's tails (a positive hyperparameter is
 * log-spaced); each node conditions the spec on its values (`doSpec`) and runs EP on what is left; the posterior is the
 * mixture of the nodes' posteriors weighted by the EP log evidence plus the log prior, the hyperparameters' marginals read off
 * the grid. Where EP is exact given the hyperparameters (a Gaussian likelihood) the evidence of a node is the exact marginal
 * likelihood, so the grid is the textbook one. The grid grows outwards where its edge still has mass, within `maxNodes`.
 *
 * @param model - A bound spec.
 * @param options - The name, the hyperparameters (`over`), how plates are compiled.
 * @returns The method, with `check` (it applies when a scalar latent feeds a custom function and the model compiles with it known) and `posterior`.
 * @concept expectation-propagation
 * @index estimator Expectation propagation inside an evidence-weighted grid over the hyperparameters of a model.
 * @example
 * ```ts
 * import { epGridMethod } from 'omai-engine/probprog/inference'
 * import { defineModel } from 'omai-engine/probprog/model'
 * import { logNormal, normal } from 'omai-engine/probprog/model/dists'
 * import { mul } from 'omai-engine/tensor'
 * import { session } from 'omai-engine/trace'
 * // A Gaussian whose noise scale is half of a random value, through a function the graph cannot see through.
 * const spec = defineModel('m', (b) => {
 *   const s = b.latent('s', logNormal(0, 0.5))
 *   const scale = b.custom('half', ([v]) => mul((v as { value: never }).value, 0.5), [s], {})
 *   b.observe('y', normal(b.latent('mu', normal(0, 5)), scale), b.data('y', { dims: ['row'] }))
 * })
 * const m = epGridMethod(spec.bind({ y: [1, 2, 3] }))
 * const run = session(m, { steps: 100, mode: 'eager', config: { points: 15 } })
 * const post = m.posterior?.(run.state, run.config)
 * [run.status, post?.sites, Number(post?.mean('mu').data[0]).toFixed(2)] // => ['completed', ['s', 'mu'], '1.98']
 * ```
 */
export function epGridMethod(model: MethodModel, options: EpGridOptions = {}): EpGridMethod {
  const name = options.name ?? 'ep-grid'
  const plates = options.plates ?? 'vectorise'
  const packScalars = options.packScalars ?? true
  const bound: BoundModel | undefined = typeof model === 'function' ? undefined : model
  const priorsOf = (ts: readonly Target[], tail: number): Hyper[] =>
    ts.map((t) => priorOf((bound as BoundModel).spec.nodes.find((x) => x.name === t.node) as ModelNode, tail, t.label))

  /** The model with the hyperparameters of `ts` set to values: compiled to a graph (or the reason there is none). */
  const compileWith = (
    ts: readonly Target[],
    x: readonly number[],
  ): { c: FactorGraphCompilation; g: CompiledGraph | undefined; failure: string | undefined } => {
    const b = bound as BoundModel
    const values: Record<string, number | number[]> = {}
    for (const [k, t] of ts.entries()) {
      if (t.component === undefined) values[t.node] = x[k] as number
      else {
        const vector = (values[t.node] as number[] | undefined) ?? []
        vector[t.component] = x[k] as number
        values[t.node] = vector
      }
    }
    const conditioned = bind({ spec: doSpec(b.spec, values), customs: b.customs }, b.data, b.options)
    const c = toFactorGraph(conditioned, { plates, packScalars })
    if (c.graph === undefined) return { c, g: undefined, failure: undefined }
    try {
      return { c, g: compileGraph(c.graph, factorOperators()), failure: undefined }
    } catch (e) {
      return { c, g: undefined, failure: e instanceof Error ? e.message : String(e) }
    }
  }

  /** Whether the model has an EP graph with `ts` on the grid (at the middle of the first grid). */
  const compilesWith = (ts: readonly Target[]): boolean => {
    try {
      const mid = priorsOf(ts, 1e-4).map((h) => num1(h.map.forward(tensor((h.lo + h.hi) / 2))))
      const { c, g, failure } = compileWith(ts, mid)
      return failure === undefined && factorGraphVerdict(c, g, 'ep').ok
    } catch {
      return false
    }
  }

  // The axes: a scalar latent is one, a plated vector of them one per component. The scalars a custom function reads, or the
  // latents named in `over`; the plated vectors it reads (an ARD kernel's length scales) join only when the model has no EP graph
  // without them (a vector the function is affine in is a random vector for EP, not a hyperparameter).
  const targets: Target[] = (() => {
    if (bound === undefined) return []
    const scalars = targetsOf(bound.spec, hyperNames(bound.spec, options.over), bound.sizes)
    const vectors = options.over === undefined ? vectorReads(bound.spec, bound.sizes) : []
    if (vectors.length === 0 || compilesWith(scalars)) return scalars
    // in the order the model declares them
    const declared = (t: Target): number =>
      (bound.spec.nodes.findIndex((n) => n.name === t.node) as number) + (t.component ?? 0) / 1e6
    const wide = [...scalars, ...targetsOf(bound.spec, vectors, bound.sizes)].sort((a, b) => declared(a) - declared(b))
    return compilesWith(wide) ? wide : scalars
  })()
  const names = targets.map((t) => t.label)
  // The ranges of the hyperparameters, one set per tail (check() asks for 1e-4; a run takes the `tail` of its own config).
  const hypersByTail = new Map<number, Hyper[]>()
  const hypersFor = (tail: number): Hyper[] => {
    if (bound === undefined) throw new OmaiError('value', `${name}: needs a model spec`)
    let hypers = hypersByTail.get(tail)
    if (hypers === undefined) {
      hypers = priorsOf(targets, tail)
      hypersByTail.set(tail, hypers)
    }
    return hypers
  }
  const compileAt = (x: readonly number[]): ReturnType<typeof compileWith> => compileWith(targets, x)

  const check = (a: ModelAnalysis): Verdict => {
    if (bound === undefined || isProgramAnalysis(a))
      return no(
        `${name} runs on a model spec: a closure program has no factor graph`,
        ['a model spec'],
        'write the model with defineModel',
      )
    if (names.length === 0)
      return no(
        `${name} needs scalar latents that a custom function reads (the hyperparameters to put on a grid), and the model has none`,
        ['a hyperparameter read by a custom function'],
        'use ep when the model compiles as it is',
      )
    try {
      const hs = hypersFor(1e-4)
      const mid = hs.map((h) => num1(h.map.forward(tensor((h.lo + h.hi) / 2))))
      const { c, g, failure } = compileAt(mid)
      if (failure !== undefined)
        return no(`the factor graph does not compile at the hyperparameters: ${failure}`, ['a valid factor graph'])
      const v = factorGraphVerdict(c, g, 'ep')
      return v.ok
        ? yes(
            `expectation propagation on a grid over ${names.map((n) => `'${n}'`).join(', ')}: each value makes the custom function known, the graph compiles, and the EP evidence weights the grid`,
          )
        : no(
            `with ${names.join(', ')} known the model still has no EP graph: ${v.reason}`,
            v.missing,
            'see toFactorGraph(model).issues',
          )
    } catch (e) {
      return no(e instanceof Error ? e.message : String(e), ['scalar hyperparameters with known priors'])
    }
  }

  const posterior = (state: EpGridState, config: Config): Posterior => {
    if (state.nodes.length === 0)
      return definePosterior({
        form: 'variational',
        engine: name,
        sites: [],
        status: 'failed',
        diagnostics: { reason: 'no node run yet' },
      })
    const { logWeights, logEvidence } = nodeWeights(state)
    // No node with an evidence yet (every EP run so far failed or has none): nothing to weight, and the posterior says so.
    if (!logWeights.some((l) => Number.isFinite(l)))
      return definePosterior({
        form: 'variational',
        engine: name,
        sites: [],
        status: 'failed',
        diagnostics: { reason: 'no node has an EP evidence yet', nodes: state.nodes.length },
      })
    const converged = state.nodes.filter((n) => n.converged).length
    // A node whose EP run has no evidence (an improper cavity of a plated or tied factor) weighs nothing: the posterior is the mix of
    // the others, and says so (the single-run method falls back on the model with a factor per row; a grid node does not).
    const withoutEvidence = state.nodes.filter((n) => !Number.isFinite(n.logEvidence)).length
    const values: Record<string, number[]> = {}
    for (const [k, a] of state.axes.entries()) values[a.name] = state.nodes.map((n) => n.x[k] as number)
    // A walk that the budget cut short (the region still has an open edge) is not a finished grid, whatever the pending list says.
    const cut =
      config['order'] === 'walk' &&
      state.pending.length === 0 &&
      walkNext(state, num(config, 'window'), Number.POSITIVE_INFINITY).length > 0
    return siteMixturePosterior({
      engine: name,
      status:
        state.pending.length === 0 && !cut
          ? converged === state.nodes.length && withoutEvidence === 0
            ? 'converged'
            : 'completed'
          : 'stopped',
      logEvidence: { value: logEvidence, method: 'ep-grid' },
      diagnostics: {
        nodes: state.nodes.length,
        planned: state.nodes.length + state.pending.length,
        converged,
        extensions: state.extensions,
        ...(withoutEvidence === 0
          ? {}
          : {
              nodesWithoutEvidence: withoutEvidence,
              note: `${withoutEvidence} of ${state.nodes.length} nodes have no EP evidence and carry no weight: the posterior mixes the others`,
            }),
        ...(cut ? { budgetReached: true } : {}),
        hyperparameters: state.axes.map((a) => a.name),
        tol: num(config, 'tol'),
      },
      weights: Array.from(logWeights, (l) => Math.exp(l)),
      runs: state.nodes.map((n) => n.sites),
      values,
    })
  }

  return defineInferenceMethod<EpGridState, Config, EpGridPath, ModelAnalysis>({
    name,
    family: 'message-passing',
    description:
      'expectation propagation inside a grid over the hyperparameters a custom function reads, each node weighted by its EP evidence and the prior',
    config: epGridKnobsFor(targets.length),
    check,
    posterior,
    defaults: { 'ep-grid/nodes': Log(), 'ep-grid/logEvidence': Log() },
    contract: (config): Contract<EpGridState, EpGridPath> =>
      defineContract({
        'ep-grid/nodes': scalar({ description: 'EP runs made so far (one per grid node)' }),
        'ep-grid/logEvidence': scalar({
          description: 'the running estimate of log p(data): the grid integral of the EP evidence',
        }),
        'ep-grid/sweeps': scalar({ description: 'the sweeps the last EP run took' }),
        'ep-grid/snapshot': derived<EpGridState>({
          cost: 'dear',
          description: 'the evidence-weighted posterior of the nodes run so far as JSON',
          compute: (s) => posteriorJson(posterior(s, config)),
        }),
      }),
    make: (config): Algorithm<EpGridState, EpGridPath> => {
      const points = num(config, 'points')
      const tail = num(config, 'tail')
      const maxNodes = num(config, 'maxNodes')
      const edge = num(config, 'edge')
      const settings = epRunSettings(config)
      const maxSweeps = num(config, 'maxSweeps')
      const power = powerOf(config)
      const warm = config['warm'] === true
      const walk = config['order'] === 'walk'
      const window = num(config, 'window')
      const hs = (): Hyper[] => hypersFor(tail)
      const run = (index: readonly number[], axes: readonly GridAxis[], done: readonly GridNode[]): GridNode => {
        const h = hs()
        const u = index.map((i, k) => (axes[k] as GridAxis).u0 + (axes[k] as GridAxis).du * i)
        const x = u.map((v, k) => num1((h[k] as Hyper).map.forward(tensor(v))))
        const { c, g, failure } = compileAt(x)
        if (g === undefined)
          throw new OmaiError(
            'value',
            `${name}: the model does not compile at ${names.join(', ')} = ${x.join(', ')}: ${failure ?? c.issues.map((i) => i.reason).join('; ')}`,
          )
        const powered = atPower(g, power)
        const initial = warm ? warmStart(powered, nearestSettled(done, index)?.messages) : []
        const r = runAlgorithm(epAlgorithm(powered, { ...settings, ...(initial.length === 0 ? {} : { initial }) }), {
          maxSteps: maxSweeps + 2,
          seed: 0,
        })
        const post = factorGraphPosterior(c, powered, r.state, { engine: name })
        const sites: Record<string, GridSiteRecord> = {}
        for (const s of post.sites) sites[s] = recordOf(post.marginal(s))
        const ev = post.logEvidence?.value
        return {
          index,
          u,
          x,
          logEvidence: ev === undefined || !Number.isFinite(ev) ? Number.NEGATIVE_INFINITY : ev,
          logPrior: u.reduce((s, v, k) => s + logPriorAt(h[k] as Hyper, v), 0),
          converged: r.status === 'converged',
          sweeps: r.steps,
          sites,
          ...(warm ? settledMessages(r.status === 'converged', ev, (r.state as MpState).messages) : {}),
        }
      }
      return {
        name: 'probprog/ep-grid-method',
        config: { points, tail, maxNodes, edge, warm, order: walk ? 'walk' : 'bisect', window, ...settings, power },
        init: (): EpGridState => {
          const h = hs()
          const axes = h.map(
            (hy): GridAxis => ({
              name: hy.name,
              u0: hy.lo,
              du: (hy.hi - hy.lo) / (points - 1),
              from: 0,
              to: points - 1,
            }),
          )
          const state: EpGridState = { axes, pending: firstNodes(axes), nodes: [], extensions: 0 }
          return walk ? { ...state, pending: walkNext(state, window, maxNodes) } : state
        },
        step: (s, ctx) => {
          const [index, ...rest] = s.pending
          if (index === undefined) return s
          const node = run(index, s.axes, s.nodes)
          let next: EpGridState = { ...s, pending: rest, nodes: [...s.nodes, node] }
          if (walk) {
            next = { ...next, pending: walkNext(next, window, maxNodes) }
            // The walk reached the end of the range with mass still there: the range grows and the walk goes on.
            if (next.pending.length === 0 && next.nodes.length < maxNodes) {
              const grown = extend(next, edge, maxNodes, points, false)
              next = { ...grown, pending: walkNext(grown, window, maxNodes) }
            }
          } else if (next.pending.length === 0 && next.nodes.length < maxNodes) {
            // At the end of the grid, extend it where an edge still has mass.
            next = extend(next, edge, maxNodes, points, true)
          }
          ctx.log('ep-grid/nodes', next.nodes.length)
          ctx.log('ep-grid/sweeps', node.sweeps)
          const { logEvidence } = nodeWeights(next)
          if (Number.isFinite(logEvidence)) ctx.log('ep-grid/logEvidence', logEvidence)
          return next
        },
        done: (s) => (s.pending.length === 0 ? 'completed' : undefined),
        serialise: (s) => JSON.parse(JSON.stringify(s)),
        deserialise: (j) => j as unknown as EpGridState,
      }
    },
  })
}

/**
 * The grid with the axes extended outwards where an edge still has mass, and, with `product`, the new nodes pending (as many as
 * the budget leaves; a walk chooses its own and takes only the wider range).
 */
function extend(state: EpGridState, edge: number, maxNodes: number, points: number, product: boolean): EpGridState {
  const grow = growth(state, edge)
  if (grow.length === 0) return state
  const step = Math.max(2, Math.floor(points / 4))
  const axes = state.axes.map((a, k) => {
    const g = grow.filter((x) => x.axis === k)
    return {
      ...a,
      from: a.from - (g.some((x) => x.side === -1) ? step : 0),
      to: a.to + (g.some((x) => x.side === 1) ? step : 0),
    }
  })
  if (!product) return { ...state, axes, extensions: state.extensions + 1 }
  const have = new Set(state.nodes.map((n) => n.index.join(',')))
  const fresh = firstNodes(axes)
    .filter((t) => !have.has(t.join(',')))
    .slice(0, Math.max(0, maxNodes - state.nodes.length))
  return { ...state, axes, pending: fresh, extensions: state.extensions + 1 }
}
