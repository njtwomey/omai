/**
 * Expectation maximisation on probabilistic programs: point estimates of the parameters (and MAP values of the
 * remaining sites) with discrete or linear-Gaussian latent sites marginalised exactly in the E-step. A traced
 * `emAlgorithm`, the `em` function returning a posterior, and the engine for the planner.
 */

import { type Json, MdmlError } from '../../base/index.ts'
import { bijectors, compose, type Distribution, type Support } from '../../dists/index.ts'
import { cholesky } from '../../numerics/linalg/index.ts'
import { lbfgs } from '../../optim/index.ts'
import { foldIn, type Key, key as makeKey } from '../../random/index.ts'
import {
  add,
  asTensor,
  broadcastTo,
  cast,
  div,
  equal,
  eye,
  fromStorage,
  mul,
  primalOf,
  reshape,
  sliceAxis,
  sqrt,
  square,
  stack,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  take,
  tensor,
  toFlat,
  transpose,
  type Value,
  where,
} from '../../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../../trace/index.ts'
import { enumeratedMarginals, factorMarginals, type NamedFactor, nameLike, traceFactors } from '../_contract.ts'
import { shiftedCategorical } from '../_discrete.ts'
import { unexpanded } from '../_expand.ts'
import { analyse, type ProgramAnalysis } from '../analysis.ts'
import { logJoint } from '../density.ts'
import { condition, enumerate, type ProgramTrace, type SampleSite, seed, substitute, traceOf } from '../handlers.ts'
import { definePosterior, type Posterior } from '../posterior.ts'
import type { Model } from '../primitives.ts'
import { type EngineRunOptions, feasibility, type InferenceEngine } from './engine.ts'
import { gaussianCondition, gaussianInference, linearGaussianForm } from './gaussian.ts'
import { inferDiscrete } from './nuts.ts'

/** A quantity EM estimates: a param site or (MAP-EM) a continuous latent site. */
interface Estimated {
  readonly name: string
  readonly support: Support
  readonly bijector: bijectors.Bijector
  readonly kind: 'param' | 'latent'
}

/** A closed-form M-step: the family and parameter of the one child site, and the enumerated site that indexes it. */
interface ClosedForm {
  readonly family: 'normal.loc' | 'normal.scale' | 'poisson.rate' | 'exponential.rate' | 'probs'
  /** Every child site that uses the parameter, each directly or indexed by one enumerated site. */
  readonly roles: readonly ClosedRole[]
}

interface ClosedRole {
  readonly child: string
  readonly gather: string | undefined
  /** Whether the child is itself an enumerated site (its states are the data). */
  readonly latentChild: boolean
}

interface Problem<D> {
  readonly model: Model<D>
  readonly data: D
  readonly estimated: readonly Estimated[]
  readonly discrete: readonly string[]
  readonly gaussian: readonly string[]
  readonly depth: number
  readonly closed: ReadonlyMap<string, ClosedForm>
}

/** The state of {@link emAlgorithm}. */
export interface EmState {
  /** Current estimates (constrained) by name. */
  readonly values: Readonly<Record<string, Value>>
  /** log p(x, estimates) with the E-step sites summed or integrated out: the objective EM increases. */
  readonly objective: number
  readonly previous: number
}

/** Options of {@link emAlgorithm}. */
export interface EmOptions {
  /** Stop when the objective improves by less than this relative amount (default 1e-9). */
  readonly tol?: number
  /** `auto` (default) uses closed-form M-steps where recognised; `gradient` always maximises the expected log density. */
  readonly mStep?: 'auto' | 'gradient'
  /** L-BFGS iterations per gradient M-step (default 50; a partial M-step still increases the objective). */
  readonly gradientIterations?: number
  /** Accelerate with SQUAREM extrapolation (default true), falling back to plain EM steps when it does not help. */
  readonly accelerate?: boolean
  /** Initial values (constrained) by name; params otherwise start at their initialisers, latent sites at a draw. */
  readonly init?: Readonly<Record<string, TensorLike>>
  /** Continuous latent sites integrated in the E-step (default: the plated ones, when the program is linear-Gaussian in
   * them given the rest); the other continuous latent sites are MAP estimates. */
  readonly expectation?: readonly string[]
}

const FAMILIES = new Set(['normal', 'bernoulli', 'categorical', 'poisson', 'exponential'])

function sameValue(a: Value, b: Value): boolean {
  try {
    const x = toFlat(primalOf(a))
    const y = toFlat(primalOf(broadcastTo(b, a.shape)))
    return x.length === y.length && x.every((v, i) => Math.abs(v - (y[i] ?? Number.NaN)) <= 1e-10 * (1 + Math.abs(v)))
  } catch {
    return false
  }
}

function siteOf(t: ProgramTrace, name: string): SampleSite | undefined {
  const s = t.sites.get(name)
  return s?.type === 'sample' ? s : undefined
}

function enumTrace<D = void>(
  p: Pick<Problem<D>, 'model' | 'data' | 'depth'>,
  values: Readonly<Record<string, Value>>,
): ProgramTrace {
  return logJoint(enumerate(p.model), p.data, values, { maxPlateNesting: p.depth }).trace
}

/** Recognise a closed-form M-step for a param site by perturbing it and seeing which child parameter follows it. */
function detectClosedForm<D = void>(
  p: Pick<Problem<D>, 'model' | 'data' | 'depth'>,
  e: Estimated,
  values: Readonly<Record<string, Value>>,
): ClosedForm | undefined {
  const v0 = values[e.name]
  if (v0 === undefined) return undefined
  const u0 = primalOf(e.bijector.inverse(v0))
  const bump = fromStorage(
    Float64Array.from(toFlat(u0), (x, i) => x + 0.0137 * (1 + (i % 3))),
    u0.shape,
    'float64',
  )
  const v1 = primalOf(e.bijector.forward(bump))
  const t0 = enumTrace(p, values)
  const t1 = enumTrace(p, { ...values, [e.name]: v1 })
  const changed: SampleSite[] = []
  for (const s of t0.sites.values()) {
    if (s.type !== 'sample') continue
    const s1 = siteOf(t1, s.name)
    if (s1 === undefined) return undefined
    if (!sameValue(primalOf(s.fn.logProb(s.value)), primalOf(s1.fn.logProb(s1.value)))) changed.push(s)
  }
  if (changed.length === 0) return undefined
  let family: ClosedForm['family'] | undefined
  const roles: ClosedRole[] = []
  for (const child of changed) {
    const role = closedRole(p, child, t0, t1, v0, v1)
    if (role === undefined || (family !== undefined && role.family !== family)) return undefined
    family = role.family
    roles.push(role.role)
  }
  return family === undefined ? undefined : { family, roles }
}

function closedRole<D = void>(
  p: Pick<Problem<D>, 'depth'>,
  child: SampleSite,
  t0: ProgramTrace,
  t1: ProgramTrace,
  v0: Value,
  v1: Value,
): { readonly family: ClosedForm['family']; readonly role: ClosedRole } | undefined {
  const base0 = unexpanded(child.fn)
  const c1 = siteOf(t1, child.name)
  if (c1 === undefined || !FAMILIES.has(base0.name)) return undefined
  const base1 = unexpanded(c1.fn)
  const fields =
    base0.name === 'normal'
      ? ['loc', 'scale']
      : base0.name === 'poisson' || base0.name === 'exponential'
        ? ['rate']
        : ['probs']
  const moved = fields.filter((f) => {
    const a = base0.params[f]
    const b = base1.params[f]
    return a !== undefined && b !== undefined && !sameValue(a, b)
  })
  const field = moved[0]
  if (moved.length !== 1 || field === undefined) return undefined
  const P0 = base0.params[field] as Value
  const P1 = base1.params[field] as Value
  const family = (
    base0.name === 'normal'
      ? `normal.${field}`
      : base0.name === 'bernoulli' || base0.name === 'categorical'
        ? 'probs'
        : `${base0.name}.rate`
  ) as ClosedForm['family']
  const latentChild = child.enumDim !== undefined
  const fs = traceFactors(t0, p.depth).factors.find((f) => f.site === child.name)
  if (fs?.layout === undefined || fs.layout.loose.length > 0) return undefined
  // A parameter used directly must be one number (or one probability vector).
  const scalar =
    family === 'probs' && base0.name === 'categorical'
      ? v0.shape.length === 1
      : v0.shape.reduce((a, b) => a * b, 1) === 1
  if (scalar && sameValue(P0, v0) && sameValue(P1, v1))
    return { family, role: { child: child.name, gather: undefined, latentChild } }
  for (const s of t0.sites.values()) {
    // The indexing site must be one of the child's factor's variables (sites sharing a recycled dimension have
    // identical enumerated values, so the values alone cannot tell them apart).
    if (s.type !== 'sample' || s.enumDim === undefined || s.name === child.name || !fs.dims.includes(s.name)) continue
    const k = s.value.shape[0] ?? 0
    if ((v0.shape[0] ?? -1) !== k) continue
    const idx = cast(s.value, 'int32')
    if (sameValue(P0, take(v0, idx, 0)) && sameValue(P1, take(v1, idx, 0)))
      return { family, role: { child: child.name, gather: s.name, latentChild } }
  }
  return undefined
}

function setup<D = void>(
  model: Model<D>,
  data: D,
  options: EmOptions,
): { readonly problem: Problem<D>; readonly values: Record<string, Value> } {
  const a = analyse(model, data)
  if (a.discrete.unbounded.length > 0)
    throw new MdmlError('value', `em: discrete sites with unbounded support (${a.discrete.unbounded.join(', ')})`)
  const proto = a.prototype
  const discrete = a.discrete.sites
  let gaussian = options.expectation ?? []
  if (options.expectation === undefined && discrete.length === 0) {
    const plated = a.sites.filter((s) => s.role === 'latent' && !s.discrete && s.plates.length > 0).map((s) => s.name)
    if (plated.length > 0) {
      const fixedLatent: Record<string, Value> = {}
      const params: Record<string, Value> = {}
      for (const s of proto.sites.values()) {
        if (s.type === 'param') params[s.name] = s.value
        if (s.type === 'sample' && !s.isObserved && !plated.includes(s.name)) fixedLatent[s.name] = s.value
      }
      const inner = analyse(condition(substitute(model, params), fixedLatent), data)
      if (inner.linearGaussian.linear) gaussian = plated
    }
  }
  if (discrete.length > 0 && gaussian.length > 0)
    throw new MdmlError('not-implemented', 'em: discrete and Gaussian E-steps together are not supported')
  if (discrete.length === 0 && gaussian.length === 0)
    throw new MdmlError('value', 'em: nothing to integrate in the E-step (no discrete or linear-Gaussian latent sites)')
  const estimated: Estimated[] = []
  const values: Record<string, Value> = {}
  for (const s of proto.sites.values()) {
    if (s.type === 'param') {
      estimated.push({
        name: s.name,
        support: s.support,
        bijector: bijectors.bijectorForSupport(s.support),
        kind: 'param',
      })
      values[s.name] = s.value
    } else if (s.type === 'sample' && !s.isObserved && !discrete.includes(s.name) && !gaussian.includes(s.name)) {
      estimated.push({
        name: s.name,
        support: s.fn.support,
        bijector: bijectors.bijectorForSupport(s.fn.support),
        kind: 'latent',
      })
      values[s.name] = s.value
    }
  }
  for (const [k, v] of Object.entries(options.init ?? {})) values[k] = primalOf(asTensor(v))
  const depth = a.maxPlateNesting
  const closed = new Map<string, ClosedForm>()
  if (options.mStep !== 'gradient' && discrete.length > 0)
    for (const e of estimated) {
      if (e.kind !== 'param') continue
      const c = detectClosedForm({ model, data, depth }, e, values)
      if (c !== undefined) closed.set(e.name, c)
    }
  return { problem: { model, data, estimated, discrete, gaussian, depth, closed }, values }
}

/** The E-step: the objective at the current values and the expected log density as a function of new values. */
function eStep<D = void>(
  p: Problem<D>,
  values: Readonly<Record<string, Value>>,
): {
  readonly objective: number
  readonly expected: (next: Readonly<Record<string, Value>>) => Value
  readonly factors?: readonly NamedFactor[]
  readonly marginals?: readonly (Value | undefined)[]
  readonly trace?: ProgramTrace
} {
  if (p.discrete.length > 0) {
    const t = enumTrace(p, values)
    const { factors, sumVars, scales } = traceFactors(t, p.depth)
    const { logZ, marginals } = factorMarginals(factors, sumVars, scales)
    const expected = (next: Readonly<Record<string, Value>>): Value => {
      const fs = traceFactors(enumTrace(p, next), p.depth).factors
      let q: Value = tensor(0)
      fs.forEach((f, i) => {
        const m = marginals[i]
        const scale = f.plates.reduce((acc, pl) => acc * (scales.get(pl) ?? 1), 1)
        const term = m === undefined ? sum(f.value) : sum(mul(m, where(equal(m, 0), 0, f.value)))
        q = add(q, mul(term, scale))
      })
      return q
    }
    return { objective: logZ, expected, factors, marginals, trace: t }
  }
  const params: Record<string, Value> = {}
  const fixed: Record<string, Value> = {}
  for (const e of p.estimated) {
    const v = values[e.name] as Value
    if (e.kind === 'param') params[e.name] = v
    else fixed[e.name] = v
  }
  const conditioned = condition(substitute(p.model, params), fixed)
  const proto = traceOf(seed(conditioned, makeKey(0)), p.data)
  const g = gaussianCondition(linearGaussianForm(conditioned, p.data, proto))
  const points = sigmaPoints(g.mean, g.covariance, g.latent)
  const expected = (next: Readonly<Record<string, Value>>): Value => {
    let q: Value = tensor(0)
    for (const z of points) q = add(q, logJoint(p.model, p.data, { ...next, ...z }).logJoint)
    return mul(q, 1 / points.length)
  }
  return { objective: g.logEvidence, expected }
}

/**
 * Sigma points of the Gaussian posterior of the integrated sites (2k equally weighted points, exact for expectations of
 * quadratic log densities). When the posterior factorises into independent groups of equal size (plate elements),
 * every group moves along its own axes at once, so 2k points serve the whole plate.
 */
function sigmaPoints(
  mean: Tensor,
  cov: Tensor,
  blocks: readonly { readonly name: string; readonly shape: readonly number[]; readonly size: number }[],
): Record<string, Value>[] {
  const n = mean.shape[0] ?? 0
  const c = toFlat(cov)
  const m = toFlat(mean)
  // Connected groups of coordinates with non-zero posterior covariance.
  const group = new Int32Array(n).fill(-1)
  const groups: number[][] = []
  for (let i = 0; i < n; i++) {
    if ((group[i] ?? -1) >= 0) continue
    const members = [i]
    group[i] = groups.length
    for (let q = 0; q < members.length; q++) {
      const a = members[q] ?? 0
      for (let b = 0; b < n; b++) {
        if ((group[b] ?? -1) >= 0) continue
        const scale = Math.sqrt(Math.abs((c[a * n + a] ?? 0) * (c[b * n + b] ?? 0)))
        if (Math.abs(c[a * n + b] ?? 0) > 1e-12 * Math.max(scale, 1e-300)) {
          group[b] = groups.length
          members.push(b)
        }
      }
    }
    groups.push(members.sort((x, y) => x - y))
  }
  const sizes = new Set(groups.map((g) => g.length))
  const parts = sizes.size === 1 ? groups : [Array.from({ length: n }, (_, i) => i)]
  const k = parts[0]?.length ?? 0
  const factors = parts.map((g) => {
    const sub = fromStorage(
      Float64Array.from(g.flatMap((a) => g.map((b) => c[a * n + b] ?? 0))),
      [g.length, g.length],
      'float64',
    )
    return toFlat(primalOf(cholesky(add(sub, mul(eye(g.length), 1e-14)))))
  })
  const out: Record<string, Value>[] = []
  const r = Math.sqrt(k)
  for (let j = 0; j < k; j++)
    for (const sign of [1, -1]) {
      const z = Float64Array.from(m)
      parts.forEach((g, gi) => {
        const L = factors[gi] ?? []
        g.forEach((a, row) => {
          z[a] = (m[a] ?? 0) + sign * r * (L[row * k + j] ?? 0)
        })
      })
      const values: Record<string, Value> = {}
      let at = 0
      for (const b of blocks) {
        values[b.name] = reshape(fromStorage(z.slice(at, at + b.size), [b.size], 'float64'), b.shape)
        at += b.size
      }
      out.push(values)
    }
  if (n === 0) out.push({})
  return out
}

function namedChild(t: ProgramTrace, depth: number, child: string): NamedFactor {
  const f = traceFactors(t, depth).factors.find((x) => x.site === child)
  if (f === undefined) throw new MdmlError('value', `em: no factor for ${child}`)
  return f
}

/** Sum a named tensor over every dim except `keep` (in that order), returning `[...keep sizes]`. */
function keepDims(v: Value, dims: readonly string[], keep: readonly (string | undefined)[]): Value {
  const wanted = keep.filter((d): d is string => d !== undefined && dims.includes(d))
  const drop = dims.map((d, i) => (wanted.includes(d) ? -1 : i)).filter((i) => i >= 0)
  const reduced = drop.length === 0 ? v : sum(v, drop)
  const left = dims.filter((d) => wanted.includes(d))
  const perm = wanted.map((d) => left.indexOf(d))
  return perm.every((x, i) => x === i) ? reduced : transpose(reduced, perm)
}

function sliceIndex(v: Value, axis: number, i: number): Value {
  return reshape(
    sliceAxis(v, axis, i, i + 1),
    v.shape.filter((_, a) => a !== axis),
  )
}

/** The weighted sufficient statistics one child contributes to a closed-form update: a numerator and a denominator. */
function roleStatistics<D = void>(
  p: Problem<D>,
  family: ClosedForm['family'],
  role: ClosedRole,
  current: Value,
  e: {
    readonly factors?: readonly NamedFactor[]
    readonly marginals?: readonly (Value | undefined)[]
    readonly trace?: ProgramTrace
  },
  values: Readonly<Record<string, Value>>,
): { readonly num: Value; readonly den: Value } {
  const factors = e.factors ?? []
  const fi = factors.findIndex((f) => f.site === role.child)
  const f = factors[fi]
  const t = e.trace
  if (f === undefined || t === undefined) throw new MdmlError('value', `em: no factor for ${role.child}`)
  const m = e.marginals?.[fi] ?? broadcastTo(tensor(1), f.value.shape)
  const site = siteOf(t, role.child)
  if (site === undefined) throw new MdmlError('value', `em: no site ${role.child}`)
  const keep = [role.gather]
  const weight = (x: Value | undefined): Value => keepDims(x === undefined ? m : mul(m, x), f.dims, keep)
  const shaped = (v: Value, shape: readonly number[]): Value => reshape(v, shape)
  switch (family) {
    case 'normal.loc':
    case 'poisson.rate':
      return {
        num: shaped(weight(nameLike(f, site.value)), current.shape),
        den: shaped(weight(undefined), current.shape),
      }
    case 'exponential.rate':
      return {
        num: shaped(weight(undefined), current.shape),
        den: shaped(weight(nameLike(f, site.value)), current.shape),
      }
    case 'normal.scale': {
      const t2 = enumTrace(p, values)
      const f2 = namedChild(t2, p.depth, role.child)
      const s2 = siteOf(t2, role.child)
      const loc = s2 === undefined ? undefined : unexpanded(s2.fn).params['loc']
      if (loc === undefined) throw new MdmlError('value', 'em: no location for the scale update')
      const resid = square(sub(nameLike(f2, site.value), nameLike(f2, loc)))
      return { num: shaped(weight(resid), current.shape), den: shaped(weight(undefined), current.shape) }
    }
    case 'probs': {
      if (unexpanded(site.fn).name === 'bernoulli')
        return {
          num: shaped(weight(nameLike(f, site.value)), current.shape),
          den: shaped(weight(undefined), current.shape),
        }
      const k = current.shape[current.shape.length - 1] ?? 1
      const columns: Value[] = []
      for (let j = 0; j < k; j++) {
        if (role.latentChild) {
          // The child's own dim holds its states: take state j.
          const axis = f.dims.indexOf(role.child)
          const mj = sliceIndex(m, axis, j)
          columns.push(
            keepDims(
              mj,
              f.dims.filter((d) => d !== role.child),
              keep,
            ),
          )
        } else columns.push(weight(cast(equal(nameLike(f, site.value), j), 'float64')))
      }
      const counts = shaped(stack(columns, -1), current.shape)
      return { num: counts, den: broadcastTo(sum(counts, -1, true), current.shape) }
    }
  }
}

/** The closed-form update of one parameter: the summed statistics of all its children. */
function closedUpdate<D = void>(
  p: Problem<D>,
  form: ClosedForm,
  current: Value,
  e: Parameters<typeof roleStatistics>[4],
  values: Readonly<Record<string, Value>>,
): Value {
  let num: Value | undefined
  let den: Value | undefined
  for (const role of form.roles) {
    const r = roleStatistics(p, form.family, role, current, e, values)
    num = num === undefined ? r.num : add(num, r.num)
    den = den === undefined ? r.den : add(den, r.den)
  }
  if (num === undefined || den === undefined) return current
  const empty = equal(den, 0)
  const ratio = where(empty, current, div(num, where(empty, 1, den)))
  return primalOf(form.family === 'normal.scale' ? where(empty, current, sqrt(ratio)) : ratio)
}

/** One EM update: the E-step at `values`, closed-form M-steps where recognised, then a gradient M-step for the rest. */
function emMap<D>(p: Problem<D>, start: Readonly<Record<string, Value>>, iters: number): Record<string, Value> {
  const e = eStep(p, start)
  let values: Record<string, Value> = { ...start }
  // Closed forms: everything but scales from the E-step, then scales at the new locations.
  const order = [...p.closed.entries()].sort(
    (a, b) => Number(a[1].family === 'normal.scale') - Number(b[1].family === 'normal.scale'),
  )
  for (const [name, form] of order) {
    const current = values[name] as Value
    values = { ...values, [name]: closedUpdate(p, form, current, e, values) }
  }
  const rest = p.estimated.filter((x) => !p.closed.has(x.name))
  if (rest.length === 0) return values
  const shapes = rest.map((x) => primalOf(x.bijector.inverse(values[x.name] as Value)))
  const sizes = shapes.map((s) => s.data.length)
  const total = sizes.reduce((a, b) => a + b, 0)
  const x0 = fromStorage(Float64Array.from(shapes.flatMap((s) => toFlat(s))), [total], 'float64')
  const fixed = values
  const unpack = (u: Value): Record<string, Value> => {
    const out: Record<string, Value> = { ...fixed }
    let at = 0
    rest.forEach((x, i) => {
      const n = sizes[i] ?? 0
      out[x.name] = x.bijector.forward(reshape(sliceAxis(u, 0, at, at + n), (shapes[i] as Tensor).shape))
      at += n
    })
    return out
  }
  const r = lbfgs((u: Value) => mul(e.expected(unpack(u)), -1), x0, { maxIterations: iters, gradTol: 1e-10 })
  const u = fromStorage(Float64Array.from(toFlat(primalOf(r.x as Value))), [total], 'float64')
  return Object.fromEntries(Object.entries(unpack(u)).map(([k, v]) => [k, primalOf(v)]))
}

/** The estimates as one unconstrained vector, in a fixed order. */
function flatten<D>(p: Problem<D>, values: Readonly<Record<string, Value>>): number[] {
  return p.estimated.flatMap((e) => toFlat(primalOf(e.bijector.inverse(values[e.name] as Value))))
}

function unflatten<D>(
  p: Problem<D>,
  u: readonly number[],
  like: Readonly<Record<string, Value>>,
): Record<string, Value> {
  const out: Record<string, Value> = { ...like }
  let at = 0
  for (const e of p.estimated) {
    const shape = primalOf(e.bijector.inverse(like[e.name] as Value)).shape
    const n = shape.reduce((a, b) => a * b, 1)
    out[e.name] = primalOf(e.bijector.forward(fromStorage(Float64Array.from(u.slice(at, at + n)), shape, 'float64')))
    at += n
  }
  return out
}

/**
 * Expectation maximisation for programs with discrete or linear-Gaussian latent sites, as a traced algorithm (one step
 * per iteration): the E-step sums the discrete sites out exactly (posterior marginals by differentiating the
 * enumerated log density) or integrates linear-Gaussian plated sites (exact Gaussian conditioning and sigma points);
 * the M-step updates params in closed form where the program uses them directly or indexed by one enumerated site
 * (normal locations and scales, Bernoulli and categorical probabilities, Poisson and exponential rates: mixtures and
 * HMMs) and maximises the expected log density by L-BFGS otherwise, including MAP estimates of the remaining
 * continuous latent sites (MAP-EM, the prior included). The objective never decreases for exact M-steps; the run
 * stops with `converged` when it settles, and `diverged` when it is not finite.
 *
 * @param model - The program.
 * @param data - Its data.
 * @param options - Tolerance, M-step choice, gradient iterations, initial values and E-step sites.
 * @returns The algorithm.
 * @throws MdmlError with code `value` when there is nothing to integrate, or `not-implemented` for discrete and
 *   Gaussian E-steps together.
 * @concept expectation-maximisation
 * @example
 * ```ts
 * import { emAlgorithm, observe, param, plate, sample } from 'mdml-engine/probprog'
 * import { categorical, normal } from 'mdml-engine/dists'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * import { take } from 'mdml-engine/tensor'
 * const y = [-2.1, -1.9, -2.2, 2.0, 1.8, 2.3]
 * const gmm = () => {
 *   const w = param('w', [0.5, 0.5], { support: { kind: 'simplex' } })
 *   const mu = param('mu', [-1, 1])
 *   plate('n', y.length, () => { const z = sample('z', categorical({ probs: w })); observe('y', normal(take(mu, z), 0.5), y) })
 * }
 * runAlgorithm(emAlgorithm(gmm, undefined), { seed: 0, maxSteps: 50 }).status // => 'converged'
 * ```
 */
export function emAlgorithm<D = void>(model: Model<D>, data: D, options: EmOptions = {}): Algorithm<EmState> {
  return emParts(model, data, options).algorithm
}

function emParts<D = void>(
  model: Model<D>,
  data: D,
  options: EmOptions,
): { readonly algorithm: Algorithm<EmState>; readonly problem: () => Problem<D> } {
  const tol = options.tol ?? 1e-9
  const iters = options.gradientIterations ?? 50
  const accelerate = options.accelerate ?? true
  let problem: Problem<D> | undefined
  const get = (): Problem<D> => {
    if (problem === undefined) throw new MdmlError('value', 'em: not initialised')
    return problem
  }
  const algorithm: Algorithm<EmState> = {
    name: 'em',
    config: { tol, mStep: options.mStep ?? 'auto', gradientIterations: iters } as Json,
    schema: channels({
      metrics: { objective: { description: 'log p(x, estimates), latent sites marginalised', direction: 'higher' } },
    }),
    init: () => {
      const s = setup(model, data, options)
      problem = s.problem
      return { values: s.values, objective: eStep(s.problem, s.values).objective, previous: Number.NEGATIVE_INFINITY }
    },
    step: (state) => {
      const p = get()
      const once = emMap(p, state.values, iters)
      if (!accelerate || p.estimated.length === 0)
        return { values: once, objective: eStep(p, once).objective, previous: state.objective }
      // SQUAREM (Varadhan and Roland 2008, scheme S3): extrapolate two EM maps on the unconstrained scale, then
      // stabilise with one more map; fall back to the plain two-step result when that is not better.
      const twice = emMap(p, once, iters)
      const u0 = flatten(p, state.values)
      const u1 = flatten(p, once)
      const u2 = flatten(p, twice)
      const r = u1.map((v, i) => v - (u0[i] ?? 0))
      const v = u2.map((x, i) => x - 2 * (u1[i] ?? 0) + (u0[i] ?? 0))
      const nr = Math.sqrt(r.reduce((a, b) => a + b * b, 0))
      const nv = Math.sqrt(v.reduce((a, b) => a + b * b, 0))
      const plain = { values: twice, objective: eStep(p, twice).objective, previous: state.objective }
      if (!(nv > 0) || !(nr > 0)) return plain
      const alpha = Math.min(-1, -nr / nv)
      const jump = u0.map((x, i) => x - 2 * alpha * (r[i] ?? 0) + alpha * alpha * (v[i] ?? 0))
      let stabilised: Record<string, Value>
      let objective: number
      try {
        stabilised = emMap(p, unflatten(p, jump, state.values), iters)
        objective = eStep(p, stabilised).objective
      } catch {
        return plain
      }
      return Number.isFinite(objective) && objective >= plain.objective
        ? { values: stabilised, objective, previous: state.objective }
        : plain
    },
    done: (s) => {
      if (Number.isNaN(s.objective) || s.objective === Number.POSITIVE_INFINITY) return 'diverged'
      if (!Number.isFinite(s.previous)) return undefined
      return Math.abs(s.objective - s.previous) <= tol * Math.max(1, Math.abs(s.objective)) ? 'converged' : undefined
    },
    observe: (s) => ({ metrics: Number.isFinite(s.objective) ? { objective: s.objective } : {} }),
  }
  return { algorithm, problem: get }
}

/** Options of {@link em}. */
export interface EmRunOptions extends EmOptions {
  /** The iteration cap (default 500). */
  readonly maxIter?: number
}

/**
 * Run EM and return a posterior: point estimates of the params and MAP sites, exact marginals of the discrete sites
 * given them (or the Gaussian posterior of the integrated sites), and joint draws of the latent sites given the
 * estimates.
 *
 * @param model - The program.
 * @param data - Its data.
 * @param options - EM options, iteration cap, key and tracing.
 * @returns The posterior (form `point`); `diagnostics.closedForm` lists the closed-form parameters.
 * @concept expectation-maximisation
 * @example
 * ```ts
 * import { em, observe, param, plate, sample } from 'mdml-engine/probprog'
 * import { categorical, normal } from 'mdml-engine/dists'
 * import { key } from 'mdml-engine/random'
 * import { take, toArray } from 'mdml-engine/tensor'
 * const y = [-2.1, -1.9, -2.2, 2.0, 1.8, 2.3]
 * const gmm = () => {
 *   const mu = param('mu', [-1, 1])
 *   plate('n', y.length, () => { const z = sample('z', categorical({ probs: [0.5, 0.5] })); observe('y', normal(take(mu, z), 0.5), y) })
 * }
 * toArray(em(gmm, undefined, { key: key(0) }).mean('mu')).map((v) => Math.round(v * 100) / 100) // => [-2.07, 2.03]
 * ```
 */
export function em<D = void>(model: Model<D>, data: D, options: EmRunOptions & EngineRunOptions): Posterior {
  const parts = emParts(model, data, options)
  const r = runAlgorithm(parts.algorithm, {
    key: options.key,
    maxSteps: options.maxIter ?? 500,
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
    ...(options.parent === undefined ? {} : { parent: options.parent }),
  })
  const p = parts.problem()
  const values = r.state.values
  const marginals: Record<string, Distribution> = {}
  let sampler: ((key: Key, n: number) => Record<string, Tensor>) | undefined
  if (p.discrete.length > 0) {
    const t = enumTrace(p, values)
    const { factors, sumVars, scales } = traceFactors(t, p.depth)
    const { marginals: ms } = enumeratedMarginals(factors, sumVars, scales)
    for (const [name, mg] of ms) {
      const s = siteOf(t, name)
      const low = s?.fn.support.kind === 'integerInterval' ? s.fn.support.low : 0
      marginals[name] = shiftedCategorical(mg.probs, low)
    }
    sampler = (key, n) => {
      const out: Record<string, number[]> = {}
      const shapes: Record<string, readonly number[]> = {}
      for (let i = 0; i < n; i++) {
        const d = inferDiscrete(model, data, values, foldIn(key, i))
        for (const name of p.discrete) {
          const v = primalOf(asTensor(d[name] as TensorLike))
          shapes[name] = v.shape
          out[name] = [...(out[name] ?? []), ...toFlat(v)]
        }
      }
      const res: Record<string, Tensor> = {}
      for (const name of p.discrete)
        res[name] = fromStorage(Float64Array.from(out[name] ?? []), [n, ...(shapes[name] ?? [])], 'float64')
      for (const e of p.estimated) {
        const v = values[e.name] as Value
        res[e.name] = primalOf(broadcastTo(reshape(v, [1, ...v.shape]), [n, ...v.shape]))
      }
      return res
    }
  }
  const constants = (n: number): Record<string, Tensor> => {
    const res: Record<string, Tensor> = {}
    for (const e of p.estimated) {
      const v = values[e.name] as Value
      res[e.name] = primalOf(broadcastTo(reshape(v, [1, ...v.shape]), [n, ...v.shape]))
    }
    return res
  }
  if (p.gaussian.length > 0) {
    // The integrated sites' Gaussian posterior given the estimates.
    const params: Record<string, Value> = {}
    const fixed: Record<string, Value> = {}
    for (const e of p.estimated) (e.kind === 'param' ? params : fixed)[e.name] = values[e.name] as Value
    const g = gaussianInference(condition(substitute(model, params), fixed), data, { method: 'joint' })
    for (const name of p.gaussian) marginals[name] = g.marginal(name)
    sampler = (key, n) => ({ ...constants(n), ...g.sample(key, n) })
  }
  const pointMarginals: Record<string, Distribution> = {}
  for (const e of p.estimated) pointMarginals[e.name] = compose.deterministic(values[e.name] as Value)
  return definePosterior({
    form: 'point',
    engine: 'em',
    sites: [...p.estimated.map((e) => e.name), ...p.discrete, ...p.gaussian],
    status: r.status,
    marginals: { ...pointMarginals, ...marginals },
    ...(sampler === undefined ? {} : { sampler }),
    diagnostics: {
      iterations: r.steps,
      objective: r.state.objective,
      closedForm: [...p.closed.keys()],
      eStep: p.discrete.length > 0 ? 'enumeration' : 'gaussian',
    },
  })
}

/**
 * The EM engine: point estimates of params (and MAP sites) with the discrete or linear-Gaussian latent sites
 * marginalised; it applies when the program has params or continuous latent sites besides sites to integrate.
 *
 * @param options - EM options.
 * @returns The engine.
 * @concept inference-engines
 * @example
 * ```ts
 * import { emEngine } from 'mdml-engine/probprog'
 * emEngine().form // => 'point'
 * ```
 */
export function emEngine(options: EmRunOptions = {}): InferenceEngine {
  return {
    name: 'em',
    form: 'point',
    description: 'expectation maximisation with exact E-steps and closed-form or gradient M-steps',
    applies: (a: ProgramAnalysis) => {
      if (!a.hasLikelihood) return feasibility(false, 'approximate', 0, 'observed sites have no likelihood')
      if (a.discrete.unbounded.length > 0)
        return feasibility(false, 'approximate', 0, `unbounded discrete sites (${a.discrete.unbounded.join(', ')})`)
      if (a.params.length === 0) return feasibility(false, 'approximate', 0, 'no params to estimate')
      const discrete = a.discrete.sites.length
      const plated = a.sites.filter((s) => s.role === 'latent' && !s.discrete && s.plates.length > 0).length
      if (discrete === 0 && plated === 0)
        return feasibility(false, 'approximate', 0, 'no discrete or plated latent sites to integrate')
      const iters = options.maxIter ?? 500
      return feasibility(
        true,
        'approximate',
        iters *
          4 *
          Math.max(1, a.observations) *
          Math.max(
            1,
            Object.values(a.discrete.cardinalities).reduce((m, c) => Math.max(m, c), 1),
          ),
        discrete > 0
          ? 'params with discrete latent sites (closed-form M-steps where recognised)'
          : 'params with linear-Gaussian plated latents',
      )
    },
    run: <D>(model: Model<D>, data: D, _a: ProgramAnalysis, run: EngineRunOptions) =>
      em(model, data, { ...options, ...run }),
  }
}
