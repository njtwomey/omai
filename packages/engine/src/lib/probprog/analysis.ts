/**
 * Static analysis of a probabilistic program from one traced run: every site with its role, support, shapes, plates and
 * markov step, the dependency graph between sites (by tracking which values each one was computed from), and the
 * structures the inference planner looks for, such as conjugate prior-likelihood pairs, linear-Gaussian form and
 * enumerable discrete sites.
 *
 * @index function Static analysis of a program from one traced run.
 */

import { valueAndGrad } from '../autodiff/index.ts'
import { isMdmlError } from '../base/index.ts'
import { CONJUGATES, type ConjugatePair, type Distribution, exponentialFamily, type Support } from '../dists/index.ts'
import { inv } from '../numerics/linalg/index.ts'
import { connectedComponents, graphEdges, undirectedGraph } from '../pgm/index.ts'
import { foldIn, type Key, key as makeKey, normal as normalDraws } from '../random/index.ts'
import {
  add,
  broadcastShapes,
  broadcastTo,
  fromStorage,
  matmul,
  mul,
  primalOf,
  square,
  sub,
  swapaxes,
  type Tensor,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import { unexpanded } from './_expand.ts'
import { provenanceTrace } from './_provenance.ts'
import { potentialEnergy } from './density.ts'
import {
  handle,
  isDiscrete,
  isFiniteDiscrete,
  type ProgramTrace,
  type SampleSite,
  seed,
  substitute,
  traceOf,
} from './handlers.ts'
import type { Message, Model } from './primitives.ts'

/** What a site is in the program. */
export type SiteRole = 'latent' | 'observed' | 'factor' | 'param' | 'deterministic'

/** One site as analysis sees it. */
export interface SiteInfo {
  readonly name: string
  readonly role: SiteRole
  /** The distribution family (under any plate broadcasting), for sample sites. */
  readonly distribution?: string
  readonly support?: Support
  /** The value's shape, and the distribution's batch and event shapes. */
  readonly shape: readonly number[]
  readonly batchShape: readonly number[]
  readonly eventShape: readonly number[]
  readonly discrete: boolean
  /** The number of states of one element, for finite discrete sites. */
  readonly cardinality?: number
  /** How many independent elements the site has (the size of its batch). */
  readonly elements: number
  readonly plates: readonly string[]
  /** The markov loop and step the site belongs to. */
  readonly markov?: { readonly name: string; readonly step: number }
  readonly hasRsample: boolean
  readonly enumerated: boolean
  /** Whether the distribution is an exponential family (with a closed-form log partition). */
  readonly exponentialFamily: boolean
  /** Whether the site's log density can be evaluated (false for simulators). */
  readonly hasLogProb: boolean
  /** The sites this site's distribution (or value, for deterministic sites) depends on. */
  readonly parents: readonly string[]
}

/** A plate and the sites inside it. */
export interface PlateInfo {
  readonly name: string
  readonly size: number
  readonly subsampleSize: number
  readonly dim: number
  readonly sites: readonly string[]
}

/** A markov loop and the sites of its steps. */
export interface MarkovInfo {
  readonly name: string
  readonly length: number
  readonly history: number
  readonly sites: readonly string[]
  /** Latent sites outside the loop. */
  readonly outsideLatents: readonly string[]
  /** Sample sites that run after the loop. */
  readonly after: readonly string[]
}

/** A latent site with a conjugate prior for its observed children. */
export interface ConjugateStructure {
  readonly prior: string
  readonly priorFamily: string
  readonly likelihood: string
  readonly parameter: string
  readonly children: readonly string[]
  readonly pair: ConjugatePair
}

/** Whether the program is linear-Gaussian, and why not. */
export interface LinearGaussianInfo {
  readonly linear: boolean
  readonly reason: string
}

/**
 * What one traced run of a program reveals (see {@link analyse}).
 *
 * @concept model-analysis
 */
export interface ProgramAnalysis {
  readonly sites: readonly SiteInfo[]
  readonly latent: readonly string[]
  readonly observed: readonly string[]
  readonly params: readonly string[]
  readonly deterministic: readonly string[]
  readonly factors: readonly string[]
  readonly plates: readonly PlateInfo[]
  readonly maxPlateNesting: number
  readonly markov: readonly MarkovInfo[]
  /** Dependency edges between sites (parent to child). */
  readonly edges: readonly { readonly from: string; readonly to: string }[]
  /** False when dependencies could not be traced exactly and every earlier site is assumed a parent. */
  readonly dependenciesExact: boolean
  readonly discrete: {
    readonly sites: readonly string[]
    readonly cardinalities: Readonly<Record<string, number>>
    /** The number of joint states of every discrete latent element (Infinity when unbounded or too large). */
    readonly jointSize: number
    readonly unbounded: readonly string[]
  }
  readonly continuous: readonly string[]
  /** The number of unconstrained coordinates of the continuous latent sites. */
  readonly continuousDim: number
  readonly conjugate: readonly ConjugateStructure[]
  /** Every latent site is continuous and in a conjugate structure, each structure independent of the rest. */
  readonly fullyConjugate: boolean
  readonly linearGaussian: LinearGaussianInfo
  /** The shape of the moral graph of the latent sites. */
  readonly graphShape: 'empty' | 'chain' | 'tree' | 'loopy'
  readonly hasLikelihood: boolean
  readonly simulators: readonly string[]
  /** Whether the unconstrained log density has a finite gradient at the prototype point. */
  readonly differentiable: boolean
  /** The number of observed elements. */
  readonly observations: number
  /** The run the analysis was made from. */
  readonly prototype: ProgramTrace
}

/** Options of {@link analyse}. */
export interface AnalyseOptions {
  /** The key of the prototype run (default key(0)). */
  readonly key?: Key
}

const product = (shape: readonly number[]): number => shape.reduce((a, b) => a * b, 1)

function cardinalityOf(support: Support): number | undefined {
  if (support.kind === 'boolean') return 2
  if (support.kind === 'integerInterval' && Number.isFinite(support.high) && Number.isFinite(support.low))
    return support.high - support.low + 1
  return undefined
}

function hasLogProb(site: SampleSite): boolean {
  try {
    site.fn.logProb(site.value)
    return true
  } catch (e) {
    if (isMdmlError(e) && e.code === 'not-implemented') return false
    throw e
  }
}

function isExponentialFamily(d: Distribution): boolean {
  try {
    exponentialFamily(d)
    return true
  } catch {
    return false
  }
}

/** The program with every plate's subsample fixed to the one a trace used. */
function withSubsamples<D>(model: Model<D>, t: ProgramTrace): Model<D> {
  return handle(model, () => ({
    process: (msg: Message) => {
      if (msg.type !== 'plate' || msg.value !== undefined) return
      const s = t.sites.get(msg.name)
      if (s?.type === 'plate') msg.value = s.value
    },
  }))
}

/** Parents of every site by provenance, or undefined when the program cannot run on tagged values. */
function provenanceParents<D = void>(
  model: Model<D>,
  data: D,
  proto: ProgramTrace,
): Map<string, Set<string>> | undefined {
  const P = provenanceTrace()
  const deterministicSources = new Map<string, Set<string>>()
  const tagged = handle(model, () => ({
    process: (msg: Message) => {
      if (msg.type !== 'sample' || msg.isFactor) return
      if (msg.isObserved && msg.value !== undefined) {
        msg.value = P.tag(msg.value, msg.name)
        return
      }
      const p = proto.sites.get(msg.name)
      if (p?.type === 'sample') msg.value = P.tag(p.value, msg.name)
    },
    postprocess: (msg: Message) => {
      if (msg.value === undefined) return
      if (msg.type === 'param') msg.value = P.tag(msg.value, msg.name)
      else if (msg.type === 'deterministic') {
        deterministicSources.set(msg.name, new Set(P.sources(msg.value)))
        // Downstream sites depend on the deterministic site itself (direct edges); `latentDependencies` sees through.
        msg.value = P.tag(primalOf(msg.value), msg.name)
      }
    },
  }))
  try {
    const t = traceOf(tagged, data)
    const parents = new Map<string, Set<string>>()
    for (const site of t.sites.values()) {
      if (site.type === 'sample') {
        const lp = site.fn.logProb(primalOf(site.value))
        const s = new Set(P.sources(lp))
        s.delete(site.name)
        parents.set(site.name, s)
      } else if (site.type === 'deterministic') parents.set(site.name, deterministicSources.get(site.name) ?? new Set())
      else if (site.type === 'param') parents.set(site.name, new Set())
    }
    return parents
  } catch {
    return undefined
  }
}

/** The extraction of a likelihood's parameter from its distribution's parameters, for conjugacy checks. */
function parameterOf(d: Distribution, likelihood: string, parameter: string): Value | undefined {
  const p = d.params
  if (likelihood === 'normal' && parameter === 'variance')
    return p['scale'] === undefined ? undefined : square(p['scale'])
  if (likelihood === 'multivariateNormal' && (parameter === 'covariance' || parameter === 'precision')) {
    const l = p['scaleTril']
    if (l === undefined) return undefined
    const cov = matmul(l, swapaxes(l, -1, -2))
    return parameter === 'covariance' ? cov : inv(cov)
  }
  return p[parameter]
}

/**
 * The known parameters a conjugate pair needs, from a likelihood distribution (constant across its batch).
 *
 * @param d - The likelihood distribution.
 * @param pair - The conjugate pair.
 * @returns The fixed parameters by name, or undefined when one is missing.
 * @example
 * ```ts
 * import { fixedParameters } from 'mdml-engine/probprog'
 * import { conjugatePrior, normal } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * const pair = conjugatePrior('normal', 'loc')
 * pair === undefined ? 0 : item(fixedParameters(normal(0, 2), pair)?.['scale'] ?? 0) // => 2
 * ```
 */
export function fixedParameters(d: Distribution, pair: ConjugatePair): Record<string, Value> | undefined {
  const out: Record<string, Value> = {}
  for (const name of pair.fixed) {
    const v = parameterOf(d, pair.likelihood, name)
    if (v === undefined) return undefined
    out[name] = v
  }
  return out
}

/**
 * The extracted parameter of a pair, from a likelihood distribution.
 *
 * @param d - The likelihood distribution.
 * @param pair - The conjugate pair.
 * @returns The parameter's value, or undefined.
 * @example
 * ```ts
 * import { pairParameter } from 'mdml-engine/probprog'
 * import { conjugatePrior, normal } from 'mdml-engine/dists'
 * import { item } from 'mdml-engine/tensor'
 * // The variance of a normal is the square of its scale.
 * const pair = conjugatePrior('normal', 'variance')
 * pair === undefined ? 0 : item(pairParameter(normal(0, 3), pair) ?? 0) // => 9
 * ```
 */
export function pairParameter(d: Distribution, pair: ConjugatePair): Value | undefined {
  return parameterOf(d, pair.likelihood, pair.parameter)
}

function nearlyEqual(a: Value, b: Value): boolean {
  let shape: number[]
  try {
    shape = broadcastShapes(a.shape, b.shape)
  } catch {
    return false
  }
  const x = toFlat(broadcastTo(primalOf(a), shape))
  const y = toFlat(broadcastTo(primalOf(b), shape))
  return x.every((v, i) => Math.abs(v - (y[i] ?? Number.NaN)) <= 1e-10 * (1 + Math.abs(v)))
}

function findConjugate(
  proto: ProgramTrace,
  deps: ReadonlyMap<string, ReadonlySet<string>>,
  prior: SampleSite,
  isLatent: ReadonlySet<string>,
): ConjugateStructure | undefined {
  if ([...(deps.get(prior.name) ?? [])].some((p) => isLatent.has(p))) return undefined
  const children = [...deps.entries()]
    .filter(([name, ds]) => ds.has(prior.name) && proto.sites.get(name)?.type === 'sample')
    .map(([name]) => name)
  if (children.length === 0) return undefined
  const priorFamily = unexpanded(prior.fn).name
  let found: ConjugatePair | undefined
  let likelihood = ''
  for (const c of children) {
    const site = proto.sites.get(c)
    if (site?.type !== 'sample' || !site.isObserved || site.isFactor) return undefined
    if ([...(deps.get(c) ?? [])].filter((p) => isLatent.has(p)).length !== 1) return undefined
    const family = unexpanded(site.fn).name
    if (likelihood !== '' && family !== likelihood) return undefined
    likelihood = family
    const candidates = CONJUGATES.filter((p) => p.likelihood === family && p.prior === priorFamily)
    const pair = candidates.find((p) => {
      const v = pairParameter(site.fn, p)
      return v !== undefined && nearlyEqual(v, prior.value)
    })
    if (pair === undefined || (found !== undefined && pair !== found)) return undefined
    found = pair
  }
  if (found === undefined) return undefined
  return { prior: prior.name, priorFamily, likelihood, parameter: found.parameter, children, pair: found }
}

function linearGaussianCheck<D = void>(
  model: Model<D>,
  data: D,
  proto: ProgramTrace,
  parents: Map<string, Set<string>> | undefined,
  latent: readonly string[],
): LinearGaussianInfo {
  const gaussian = (d: Distribution): boolean => ['normal', 'multivariateNormal'].includes(unexpanded(d).name)
  if (latent.length === 0) return { linear: false, reason: 'no latent sites' }
  for (const site of proto.sites.values()) {
    if (site.type !== 'sample') continue
    if (site.isFactor) return { linear: false, reason: `factor ${site.name} is not Gaussian` }
    if (!gaussian(site.fn))
      return { linear: false, reason: `site ${site.name} is ${unexpanded(site.fn).name}, not Gaussian` }
    if (site.mask !== undefined) return { linear: false, reason: `site ${site.name} is masked` }
  }
  if (parents === undefined) return { linear: false, reason: 'dependencies could not be traced' }
  const spread = (k: Key): Record<string, Tensor> => {
    const out: Record<string, Tensor> = {}
    latent.forEach((name, i) => {
      const s = proto.sites.get(name)
      if (s?.type === 'sample') out[name] = normalDraws(foldIn(k, i), s.value.shape, { scale: 0.7 })
    })
    return out
  }
  const at = (shift: Record<string, Tensor>, sign: number): ProgramTrace => {
    const values: Record<string, Value> = {}
    for (const name of latent) {
      const s = proto.sites.get(name)
      const d = shift[name]
      if (s?.type === 'sample' && d !== undefined) values[name] = add(primalOf(s.value), mul(d, sign))
    }
    return traceOf(substitute(model, values), data)
  }
  const params = (t: ProgramTrace, name: string): { loc: Value; spread: Value } | undefined => {
    const s = t.sites.get(name)
    if (s?.type !== 'sample') return undefined
    const base = unexpanded(s.fn)
    const loc = base.params['loc']
    const sp = base.params['scale'] ?? base.params['scaleTril']
    return loc === undefined || sp === undefined ? undefined : { loc, spread: sp }
  }
  const centre = traceOf(
    substitute(model, Object.fromEntries(latent.map((n) => [n, (proto.sites.get(n) as SampleSite).value]))),
    data,
  )
  for (const k of [makeKey(7), makeKey(11)]) {
    const d = spread(k)
    const up = at(d, 1)
    const down = at(d, -1)
    for (const site of proto.sites.values()) {
      if (site.type !== 'sample') continue
      const c = params(centre, site.name)
      const u = params(up, site.name)
      const w = params(down, site.name)
      if (c === undefined || u === undefined || w === undefined)
        return { linear: false, reason: `site ${site.name} has no location and scale` }
      if (!nearlyEqual(u.spread, c.spread) || !nearlyEqual(w.spread, c.spread))
        return { linear: false, reason: `the scale of ${site.name} depends on latent sites` }
      const second = sub(add(primalOf(u.loc), primalOf(w.loc)), mul(primalOf(c.loc), 2))
      const scale = toFlat(primalOf(c.loc)).reduce((m, v) => Math.max(m, Math.abs(v)), 1)
      if (toFlat(second).some((v) => Math.abs(v) > 1e-8 * scale))
        return { linear: false, reason: `the location of ${site.name} is not affine in its parents` }
    }
  }
  return { linear: true, reason: 'every site is Gaussian with an affine mean and a constant scale' }
}

/** Each site's dependencies on sample and param sites, seeing through deterministic sites. */
function latentDependencies(sites: readonly SiteInfo[]): Map<string, Set<string>> {
  const byName = new Map(sites.map((s) => [s.name, s]))
  const out = new Map<string, Set<string>>()
  const visit = (name: string, seen: Set<string>): Set<string> => {
    const done = out.get(name)
    if (done !== undefined) return done
    const result = new Set<string>()
    for (const p of byName.get(name)?.parents ?? []) {
      if (seen.has(p)) continue
      const ps = byName.get(p)
      if (ps?.role === 'deterministic') for (const q of visit(p, new Set([...seen, p]))) result.add(q)
      else result.add(p)
    }
    out.set(name, result)
    return result
  }
  for (const s of sites) visit(s.name, new Set([s.name]))
  return out
}

function graphShapeOf(
  latent: readonly string[],
  sites: readonly SiteInfo[],
  deps: ReadonlyMap<string, ReadonlySet<string>>,
): ProgramAnalysis['graphShape'] {
  const isLatent = new Set(latent)
  const edges: [string, string][] = []
  for (const s of sites) {
    const ps = [...(deps.get(s.name) ?? [])].filter((p) => isLatent.has(p))
    if (isLatent.has(s.name)) for (const p of ps) edges.push([p, s.name])
    for (let i = 0; i < ps.length; i++) for (let j = i + 1; j < ps.length; j++) edges.push([ps[i] ?? '', ps[j] ?? ''])
  }
  const g = undirectedGraph(edges, latent)
  const e = graphEdges(g).length
  if (e === 0) return 'empty'
  const v = g.size
  const c = connectedComponents(g).length
  if (e !== v - c) return 'loopy'
  for (const ns of g.values()) if (ns.size > 2) return 'tree'
  return 'chain'
}

/**
 * Trace a program once and describe it (section 3.6 of the design): every site with its role, support, shapes, plates
 * and markov step; the dependency graph (by provenance tracking of tagged values); plates; discrete cardinalities and
 * the size of their joint space; conjugate prior-likelihood structures (from `dists`' conjugate pairs); whether it is
 * linear-Gaussian (Gaussian sites with affine means and constant scales, checked numerically); the shape of the latent
 * moral graph (chain, tree or loopy); whether a likelihood exists or only a simulator; and whether the unconstrained
 * density is differentiable. Inference engines decide whether they apply from this description.
 *
 * @param model - The program.
 * @param data - Its data.
 * @param options - The key of the prototype run.
 * @returns The analysis.
 * @concept model-analysis
 * @example
 * ```ts
 * import { analyse, observe, plate, sample } from 'mdml-engine/probprog'
 * import { beta, bernoulli } from 'mdml-engine/dists'
 * const coin = (flips: number[]) => {
 *   const p = sample('p', beta(2, 2))
 *   plate('flips', flips.length, () => observe('y', bernoulli({ probs: p }), flips))
 * }
 * const a = analyse(coin, [1, 0, 1, 1])
 * a.latent // => ['p']
 * a.conjugate[0]?.pair.prior // => 'beta'
 * a.edges // => [{ from: 'p', to: 'y' }]
 * ```
 */
export function analyse<D = void>(model: Model<D>, data: D, options: AnalyseOptions = {}): ProgramAnalysis {
  const raw = traceOf(seed(model, options.key ?? makeKey(0)), data)
  // Every later run uses the prototype's plate subsamples, so subsampled programs need no key.
  const fixed = withSubsamples(model, raw)
  const proto = raw
  const traced = provenanceParents(fixed, data, proto)
  const order = [...proto.sites.values()].filter((s) => s.type !== 'plate' && s.type !== 'markov')
  const parents =
    traced ??
    new Map(
      order.map((s, i) => [
        s.name,
        new Set(
          order
            .slice(0, i)
            .filter((p) => p.type !== 'sample' || !p.isObserved)
            .map((p) => p.name),
        ),
      ]),
    )
  const sites: SiteInfo[] = []
  const latent: string[] = []
  const observed: string[] = []
  const params: string[] = []
  const deterministic: string[] = []
  const factors: string[] = []
  const simulators: string[] = []
  const cardinalities: Record<string, number> = {}
  const unbounded: string[] = []
  const continuous: string[] = []
  let logJointSize = 0
  let observations = 0
  for (const site of order) {
    const plates = site.plates.map((f) => f.name)
    const frame = site.markov[0]
    const markov = frame === undefined ? {} : { markov: { name: frame.name, step: frame.step } }
    const ps = [...(parents.get(site.name) ?? [])].filter((p) => proto.sites.has(p))
    if (site.type === 'sample') {
      const role: SiteRole = site.isFactor ? 'factor' : site.isObserved ? 'observed' : 'latent'
      const discrete = isDiscrete(site.fn.support)
      const cardinality = cardinalityOf(site.fn.support)
      const elements = product(site.fn.batchShape)
      const logProbOk = hasLogProb(site)
      sites.push({
        name: site.name,
        role,
        distribution: unexpanded(site.fn).name,
        support: site.fn.support,
        shape: site.value.shape,
        batchShape: site.fn.batchShape,
        eventShape: site.fn.eventShape,
        discrete,
        ...(cardinality === undefined ? {} : { cardinality }),
        elements,
        plates,
        ...markov,
        hasRsample: site.fn.hasRsample,
        enumerated: site.infer.enumerate === 'parallel',
        exponentialFamily: !site.isFactor && isExponentialFamily(unexpanded(site.fn)),
        hasLogProb: logProbOk,
        parents: ps,
      })
      if (role === 'factor') factors.push(site.name)
      else if (role === 'observed') {
        observed.push(site.name)
        observations += product(site.value.shape)
        if (!logProbOk) simulators.push(site.name)
      } else {
        latent.push(site.name)
        if (discrete) {
          if (cardinality === undefined) unbounded.push(site.name)
          else {
            cardinalities[site.name] = cardinality
            logJointSize += product(site.value.shape) * Math.log(cardinality)
          }
        } else continuous.push(site.name)
      }
    } else if (site.type === 'param') {
      params.push(site.name)
      sites.push({
        name: site.name,
        role: 'param',
        support: site.support,
        shape: site.value.shape,
        batchShape: [],
        eventShape: [],
        discrete: false,
        elements: product(site.value.shape),
        plates,
        ...markov,
        hasRsample: false,
        enumerated: false,
        exponentialFamily: false,
        hasLogProb: true,
        parents: ps,
      })
    } else if (site.type === 'deterministic') {
      deterministic.push(site.name)
      sites.push({
        name: site.name,
        role: 'deterministic',
        shape: site.value.shape,
        batchShape: [],
        eventShape: [],
        discrete: false,
        elements: product(site.value.shape),
        plates,
        ...markov,
        hasRsample: false,
        enumerated: false,
        exponentialFamily: false,
        hasLogProb: true,
        parents: ps,
      })
    }
  }
  const edges = sites.flatMap((s) => s.parents.map((p) => ({ from: p, to: s.name })))
  const deps = latentDependencies(sites)
  const conjugate: ConjugateStructure[] = []
  if (traced !== undefined)
    for (const name of continuous) {
      const s = proto.sites.get(name)
      if (s?.type !== 'sample') continue
      const c = findConjugate(proto, deps, s, new Set(latent))
      if (c !== undefined) conjugate.push(c)
    }
  const plateMap = new Map<string, PlateInfo>()
  // Plates in the order they open (outer before inner), then their sites.
  for (const s of proto.sites.values())
    if (s.type === 'plate' && !plateMap.has(s.name))
      plateMap.set(s.name, { name: s.name, size: s.size, subsampleSize: s.subsampleSize, dim: -1, sites: [] })
  for (const s of proto.sites.values()) {
    if (s.type === 'plate') continue
    for (const f of s.plates) {
      const prev = plateMap.get(f.name)
      plateMap.set(f.name, {
        name: f.name,
        size: f.size,
        subsampleSize: f.subsampleSize,
        dim: f.dim,
        sites: [...(prev?.sites ?? []), s.name],
      })
    }
  }
  const markovInfo: MarkovInfo[] = []
  const all = [...proto.sites.values()]
  for (const [i, s] of all.entries()) {
    if (s.type !== 'markov') continue
    const inside = sites.filter((x) => x.markov?.name === s.name).map((x) => x.name)
    const after = all
      .slice(i + 1)
      .filter((x) => x.type === 'sample' && !x.markov.some((f) => f.name === s.name))
      .map((x) => x.name)
    markovInfo.push({
      name: s.name,
      length: s.length,
      history: s.history,
      sites: inside,
      outsideLatents: latent.filter((n) => !inside.includes(n)),
      after,
    })
  }
  let differentiable = false
  let continuousDim = 0
  if (continuous.length > 0 && unbounded.length === 0 && simulators.length === 0) {
    try {
      const um = potentialEnergy(fixed, data, options.key === undefined ? {} : { key: options.key })
      continuousDim = um.dim
      const u = new Float64Array(um.dim)
      const g = valueAndGrad((x: Value) => um.fn(x))(fromStorage(u, [um.dim], 'float64'))
      differentiable = toFlat(g.grad).every(Number.isFinite)
    } catch {
      differentiable = false
    }
  }
  const covered = new Set(conjugate.flatMap((c) => [c.prior]))
  const childSets = conjugate.map((c) => new Set(c.children))
  const disjoint = childSets.every((a, i) => childSets.every((b, j) => i === j || [...a].every((x) => !b.has(x))))
  const lg =
    unbounded.length > 0 || Object.keys(cardinalities).length > 0
      ? { linear: false, reason: 'the program has discrete latent sites' }
      : linearGaussianCheck(fixed, data, proto, traced, latent)
  return {
    sites,
    latent,
    observed,
    params,
    deterministic,
    factors,
    plates: [...plateMap.values()],
    maxPlateNesting: [...plateMap.values()].reduce((m, p) => Math.max(m, -p.dim), 0),
    markov: markovInfo,
    edges,
    dependenciesExact: traced !== undefined,
    discrete: {
      sites: latent.filter((n) => cardinalities[n] !== undefined || unbounded.includes(n)),
      cardinalities,
      jointSize:
        unbounded.length > 0 || logJointSize > 700 ? Number.POSITIVE_INFINITY : Math.round(Math.exp(logJointSize)),
      unbounded,
    },
    continuous,
    continuousDim,
    conjugate,
    fullyConjugate: latent.length > 0 && latent.every((n) => covered.has(n)) && disjoint && factors.length === 0,
    linearGaussian: lg,
    graphShape: graphShapeOf(latent, sites, deps),
    hasLikelihood: simulators.length === 0,
    simulators,
    differentiable,
    observations,
    prototype: proto,
  }
}

/**
 * Whether a site's support is finite and discrete.
 *
 * @param info - A site of an analysis.
 * @returns True when the site can be enumerated.
 * @example
 * ```ts
 * import { analyse, isEnumerable, sample } from 'mdml-engine/probprog'
 * import { bernoulli } from 'mdml-engine/dists'
 * const a = analyse(() => { sample('c', bernoulli({ probs: 0.5 })) }, undefined)
 * isEnumerable(a.sites[0]) // => true
 * ```
 */
export function isEnumerable(info: SiteInfo | undefined): boolean {
  return info?.support !== undefined && isFiniteDiscrete(info.support)
}
