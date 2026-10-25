/**
 * Exact inference for programs whose latent sites are all finite and discrete (Pyro's `TraceEnum` and
 * `infer_discrete`): the program is grounded into a factor graph from `pgm`, a junction tree is calibrated for
 * marginals and the evidence, and exact joint draws follow. Also the engine for the planner.
 */

import { MdmlError } from '../../base/index.ts'
import type { Distribution } from '../../dists/index.ts'
import {
  type CalibratedTree,
  calibrate,
  type DiscreteVariable,
  type Factor,
  type FactorGraph,
  factorGraph,
  factorMarginal,
  factorPower,
  logFactor,
  maxProductElimination,
  normaliseFactor,
  reduceFactor,
  unnormalisedLogProbability,
} from '../../pgm/index.ts'
import { categorical as drawCategorical, foldIn, type Key } from '../../random/index.ts'
import {
  asTensor,
  fromStorage,
  primalOf,
  sliceAxis,
  squeeze,
  type Tensor,
  type TensorLike,
  toFlat,
} from '../../tensor/index.ts'
import { traceFactors } from '../_contract.ts'
import { probabilityTensor, shiftedCategorical, unravel } from '../_discrete.ts'
import type { ProgramAnalysis } from '../analysis.ts'
import { logJoint } from '../density.ts'
import { enumerate, type SampleSite } from '../handlers.ts'
import { definePosterior, type Posterior } from '../posterior.ts'
import type { Model } from '../primitives.ts'
import { type EngineRunOptions, feasibility, type InferenceEngine, runTracing } from './engine.ts'

/** One enumerated site of a grounded program: its batch shape, states and the grounded variable of each element. */
export interface GroundedSite {
  readonly name: string
  readonly batchShape: readonly number[]
  readonly low: number
  readonly states: number
  /** The grounded variable name of each element, in row-major batch order. */
  readonly variables: readonly string[]
}

/** A discrete program as a factor graph (every plate element its own variable). */
export interface GroundedProgram {
  readonly graph: FactorGraph
  /** log of the factors that involve no discrete latent site (observations given the rest). */
  readonly constant: number
  readonly sites: readonly GroundedSite[]
}

function supportRange(site: SampleSite): { readonly low: number; readonly states: number } {
  const s = site.fn.support
  if (s.kind === 'boolean') return { low: 0, states: 2 }
  if (s.kind === 'integerInterval') return { low: s.low, states: s.high - s.low + 1 }
  throw new MdmlError('value', `enumeration: site ${site.name} is not finite and discrete`)
}

/**
 * Ground a program whose latent sites are all finite and discrete into a pgm factor graph: enumerate every discrete
 * site in parallel, turn each site's log density into a factor over enumerated sites and plates, and give each plate
 * element its own variable (`z[3]`, `z[1,2]`). Subsampled plates become factor powers.
 *
 * @param model - The program.
 * @param data - Its data.
 * @returns The factor graph, the constant log factor and the grounded sites.
 * @throws MdmlError with code `value` for continuous latent sites, or when a factor uses a plated discrete site
 *   outside its plate.
 * @concept exact-inference
 * @example
 * ```ts
 * import { groundDiscrete, observe, sample } from 'mdml-engine/probprog'
 * import { bernoulli, normal } from 'mdml-engine/dists'
 * const g = groundDiscrete((y: number) => { const c = sample('c', bernoulli({ probs: 0.5 })); observe('y', normal(c, 1), y) }, 0.3)
 * g.graph.variables.map((v) => v.name) // => ['c']
 * ```
 */
export function groundDiscrete<D = void>(model: Model<D>, data: D): GroundedProgram {
  const r = logJoint(enumerate(model), data)
  const t = r.trace
  for (const site of t.sites.values())
    if (site.type === 'sample' && !site.isObserved && site.enumDim === undefined)
      throw new MdmlError('value', `enumeration: site ${site.name} is not discrete; use another engine`, {
        site: site.name,
      })
  const depth = [...t.sites.values()].reduce((m, s) => s.plates.reduce((a, f) => Math.max(a, -f.dim), m), 0)
  const { factors, scales } = traceFactors(t, depth)
  const sites: GroundedSite[] = []
  const byName = new Map<string, { site: GroundedSite; axes: { plate: string; axis: number }[] }>()
  const variables: DiscreteVariable[] = []
  for (const site of t.sites.values()) {
    if (site.type !== 'sample' || site.enumDim === undefined) continue
    const { low, states } = supportRange(site)
    // The site's own plate axes (enumeration dimensions of other sites lie further left).
    const ownDepth = site.plates.reduce((m, f) => Math.max(m, -f.dim), 0)
    const full = site.fn.batchShape
    const batch = full.slice(Math.max(0, full.length - ownDepth))
    const axes: { plate: string; axis: number }[] = []
    batch.forEach((n, a) => {
      if (n > 1) axes.push({ plate: site.plates.find((f) => f.dim === a - batch.length)?.name ?? '', axis: a })
    })
    const count = batch.reduce((x, y) => x * y, 1)
    const names = Array.from({ length: count }, (_, i) => {
      if (axes.length === 0) return site.name
      const idx = unravel(i, batch)
      return `${site.name}[${axes.map((x) => idx[x.axis] ?? 0).join(',')}]`
    })
    const g: GroundedSite = { name: site.name, batchShape: [...batch], low, states, variables: names }
    sites.push(g)
    byName.set(site.name, { site: g, axes })
    const labels = Array.from({ length: states }, (_, k) => String(low + k))
    for (const n of new Set(names)) variables.push({ name: n, states: labels })
  }
  const grounded: Factor[] = []
  let constant = 0
  for (const f of factors) {
    const plateDims = f.dims.filter((d) => !byName.has(d))
    const varDims = f.dims.filter((d) => byName.has(d))
    const plateSizes = plateDims.map((d) => f.value.shape[f.dims.indexOf(d)] ?? 1)
    const power = f.plates.reduce((acc, p) => acc * (scales.get(p) ?? 1), 1)
    const total = plateSizes.reduce((a, b) => a * b, 1)
    const value = primalOf(f.value)
    for (let c = 0; c < total; c++) {
      const at = unravel(c, plateSizes)
      let slice: Tensor = value
      // Index the plate axes (from the last, so earlier axis positions stay valid).
      for (let j = plateDims.length - 1; j >= 0; j--) {
        const axis = f.dims.indexOf(plateDims[j] ?? '')
        slice = primalOf(squeeze(sliceAxis(slice, axis, at[j] ?? 0, (at[j] ?? 0) + 1), axis))
      }
      const logs = toFlat(slice)
      if (varDims.length === 0) {
        constant += power * (logs[0] ?? 0)
        continue
      }
      const vars = varDims.map((v) => {
        const info = byName.get(v)
        if (info === undefined) return v
        if (info.axes.length === 0) return info.site.name
        const idx = info.axes.map((x) => {
          const j = plateDims.indexOf(x.plate)
          if (j < 0)
            throw new MdmlError(
              'value',
              `enumeration: site ${f.site ?? '?'} depends on ${v} outside the plate ${x.plate}`,
            )
          return at[j] ?? 0
        })
        return `${v}[${idx.join(',')}]`
      })
      const cards = varDims.map((v) => byName.get(v)?.site.states ?? 1)
      const lf = logFactor(vars, cards, logs)
      grounded.push(power === 1 ? lf : factorPower(lf, power))
    }
  }
  return { graph: factorGraph(variables, grounded), constant, sites }
}

function sampleTree(calibrated: CalibratedTree, key: Key): Record<string, number> {
  const { tree, cliqueBeliefs } = calibrated
  const out: Record<string, number> = {}
  const done = new Set<number>()
  let draw = 0
  const visit = (i: number): void => {
    done.add(i)
    const belief = cliqueBeliefs[i] as Factor
    const fixed = Object.fromEntries(belief.variables.filter((v) => v in out).map((v) => [v, out[v] ?? 0]))
    const rest = normaliseFactor(reduceFactor(belief, fixed))
    if (rest.variables.length > 0) {
      const w = Array.from(rest.logValues, Math.exp)
      const flat = toFlat(drawCategorical(foldIn(key, draw), w, [1]))[0] ?? 0
      draw += 1
      const idx = unravel(flat, rest.cardinality)
      rest.variables.forEach((v, j) => {
        out[v] = idx[j] ?? 0
      })
    }
    for (const e of tree.edges) {
      const j = e.from === i ? e.to : e.to === i ? e.from : -1
      if (j >= 0 && !done.has(j)) visit(j)
    }
  }
  for (let i = 0; i < tree.cliques.length; i++) if (!done.has(i)) visit(i)
  return out
}

/** Options of {@link enumerationPosterior}. */
export interface EnumerationOptions {
  /** Junction-tree propagation (`shaferShenoy` default, or `hugin`). */
  readonly method?: 'shaferShenoy' | 'hugin'
  readonly tracer?: EngineRunOptions['tracer']
  readonly policy?: EngineRunOptions['policy']
}

/**
 * Exact inference for a discrete program (Pyro's `TraceEnum` and `infer_discrete`, done by `pgm`): ground it into a
 * factor graph, calibrate a junction tree (message events traced), and return exact marginals per site (categorical,
 * batched over plates), exact joint draws (sampling the calibrated cliques along the tree), the exact log posterior
 * and the evidence log p(data).
 *
 * @param model - The program (every latent site finite and discrete).
 * @param data - Its data.
 * @param options - Propagation method, tracer and policy.
 * @returns The exact posterior.
 * @concept exact-inference
 * @example
 * ```ts
 * import { enumerationPosterior, observe, sample } from 'mdml-engine/probprog'
 * import { bernoulli } from 'mdml-engine/dists'
 * import { item, take, tensor } from 'mdml-engine/tensor'
 * // Rain makes wet grass likely: P(rain | wet) = 0.2 * 0.9 / (0.2 * 0.9 + 0.8 * 0.1).
 * const sprinkler = () => {
 *   const rain = sample('rain', bernoulli({ probs: 0.2 }))
 *   observe('wet', bernoulli({ probs: take(tensor([0.1, 0.9]), rain) }), 1)
 * }
 * Math.abs(item(enumerationPosterior(sprinkler, undefined).mean('rain')) - 0.18 / 0.26) < 1e-12 // => true
 * ```
 */
export function enumerationPosterior<D = void>(model: Model<D>, data: D, options: EnumerationOptions = {}): Posterior {
  const g = groundDiscrete(model, data)
  const calibrated = calibrate(g.graph, {
    mode: 'sum',
    ...(options.method === undefined ? {} : { method: options.method }),
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
  })
  const logZ = g.constant + calibrated.logPartition
  // The most probable joint assignment, by max-product variable elimination.
  const mpe = maxProductElimination(g.graph, {
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
  })
  const map: Record<string, number[]> = {}
  for (const s of g.sites) map[s.name] = s.variables.map((v) => s.low + (mpe.indices[v] ?? 0))
  const marginalOf = (v: string, states: number): number[] => {
    const home = calibrated.tree.cliques.findIndex((c) => c.includes(v))
    if (home < 0) return new Array<number>(states).fill(1 / states)
    return Array.from(
      normaliseFactor(factorMarginal(calibrated.cliqueBeliefs[home] as Factor, [v])).logValues,
      Math.exp,
    )
  }
  const marginals: Record<string, Distribution> = {}
  for (const s of g.sites) {
    const rows = s.variables.map((v) => marginalOf(v, s.states))
    marginals[s.name] = shiftedCategorical(probabilityTensor(rows, s.batchShape, s.states), s.low)
  }
  const sampler = (key: Key, n: number): Record<string, Tensor> => {
    const draws = Array.from({ length: n }, (_, i) => sampleTree(calibrated, foldIn(key, i)))
    const out: Record<string, Tensor> = {}
    for (const s of g.sites) {
      const size = s.variables.length
      const data = new Float64Array(n * size)
      draws.forEach((d, i) => {
        s.variables.forEach((v, j) => {
          data[i * size + j] = s.low + (d[v] ?? 0)
        })
      })
      out[s.name] = fromStorage(data, [n, ...s.batchShape], 'float64')
    }
    return out
  }
  const logProb = (values: Readonly<Record<string, unknown>>): number => {
    const assignment: Record<string, number> = {}
    for (const s of g.sites) {
      const v = values[s.name]
      if (v === undefined) throw new MdmlError('value', `enumeration: no value for ${s.name}`)
      const flat = toFlat(primalOf(asTensor(v as TensorLike)))
      s.variables.forEach((name, j) => {
        assignment[name] = (flat[j] ?? flat[0] ?? 0) - s.low
      })
    }
    return unnormalisedLogProbability(g.graph, assignment) + g.constant - logZ
  }
  return definePosterior({
    form: 'exact',
    engine: 'enumeration',
    sites: g.sites.map((s) => s.name),
    status: 'completed',
    logEvidence: { value: logZ, standardError: 0, method: 'exact enumeration' },
    diagnostics: {
      variables: g.graph.variables.length,
      factors: g.graph.factors.length,
      largestClique: calibrated.tree.cliques.reduce((m, c) => Math.max(m, c.length), 0),
      map,
      mapLogProb: mpe.logValue + g.constant - logZ,
    },
    marginals,
    sampler,
    logProb: (v) => logProb(v),
  })
}

/** Options of {@link enumerationEngine}. */
export interface EnumerationEngineOptions {
  /** The largest number of joint states (or clique-table entries) the engine takes on (default 1e7). */
  readonly maxStates?: number
  readonly method?: 'shaferShenoy' | 'hugin'
}

/**
 * The enumeration engine: exact inference for programs whose latent sites are all finite and discrete, by `pgm`'s
 * junction tree on the grounded factor graph. Its cost is the size of the joint space when that is small, or the number
 * of elements times the squared cardinality for chain- and tree-shaped programs.
 *
 * @param options - The state limit and propagation method.
 * @returns The engine.
 * @concept inference-engines
 * @example
 * ```ts
 * import { analyse, enumerationEngine, sample } from 'mdml-engine/probprog'
 * import { bernoulli } from 'mdml-engine/dists'
 * enumerationEngine().applies(analyse(() => { sample('c', bernoulli({ probs: 0.5 })) }, undefined)).applies // => true
 * ```
 */
export function enumerationEngine(options: EnumerationEngineOptions = {}): InferenceEngine {
  const maxStates = options.maxStates ?? 1e7
  return {
    name: 'enumeration',
    form: 'exact',
    description: 'exact marginals and joint draws of discrete programs by junction-tree propagation',
    applies: (a: ProgramAnalysis) => {
      if (a.latent.length === 0) return feasibility(false, 'exact', 0, 'the program has no latent sites')
      if (!a.hasLikelihood) return feasibility(false, 'exact', 0, 'observed sites have no likelihood')
      if (a.continuous.length > 0)
        return feasibility(false, 'exact', 0, `continuous latent sites (${a.continuous.join(', ')})`)
      if (a.discrete.unbounded.length > 0)
        return feasibility(false, 'exact', 0, `unbounded discrete sites (${a.discrete.unbounded.join(', ')})`)
      const cards = Object.values(a.discrete.cardinalities)
      const k = cards.reduce((m, c) => Math.max(m, c), 1)
      const elements = a.sites.filter((s) => s.role === 'latent').reduce((n, s) => n + s.elements, 0)
      const local = a.graphShape === 'loopy' ? a.discrete.jointSize : elements * k * k
      const cost = Math.min(a.discrete.jointSize, local) * Math.max(1, a.observed.length)
      if (cost > maxStates)
        return feasibility(false, 'exact', cost, `about ${cost.toExponential(1)} states exceed the limit ${maxStates}`)
      return feasibility(
        true,
        'exact',
        cost,
        `every latent site is discrete (${elements} variables, ${k} states at most)`,
      )
    },
    run: <D>(model: Model<D>, data: D, _a: ProgramAnalysis, run: EngineRunOptions) =>
      enumerationPosterior(model, data, {
        ...(options.method === undefined ? {} : { method: options.method }),
        ...runTracing(run),
      }),
  }
}
