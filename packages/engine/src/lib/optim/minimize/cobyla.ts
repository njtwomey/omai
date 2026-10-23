import { WhiteboxError } from '../../base/index.ts'
import { daxpy, ddot, dnrm2, inv, solve } from '../../numerics/linalg/index.ts'
import { fromStorage, toFlat, type Value } from '../../tensor/index.ts'
import type { Algorithm } from '../../trace/index.ts'
import type { Tree } from '../transform.ts'
import { boundArrays, constraintValues } from './_bounds.ts'
import { configOf, observeVector, runVector, type VectorState, vectorDone, vectorSchema } from './_shared.ts'
import { trustStep } from './_trstlp.ts'
import type { Bounds } from './derivative-free.ts'
import {
  type FlatObjective,
  flatObjective,
  type MinimizeResult,
  type Objective,
  type Parameters,
  type VectorMethodOptions,
} from './objective.ts'

/** An inequality constraint c(x) >= 0 (scipy's `'ineq'` convention): one value, or a vector of values, of the parameters. */
export type InequalityConstraint = (params: Tree) => Value

/** Options of {@link cobyla}. */
export interface CobylaOptions extends VectorMethodOptions {
  /** Inequality constraints c(x) >= 0. */
  readonly constraints?: readonly InequalityConstraint[]
  /** Box bounds, handled as extra linear constraints. */
  readonly bounds?: Bounds
  /** The initial trust radius (default 1). */
  readonly rhobeg?: number
  /** The final trust radius, the accuracy sought (scipy's tol; default 1e-4). */
  readonly rhoend?: number
  /** The evaluation budget (default 1000). */
  readonly maxEvaluations?: number
  /** The largest constraint violation accepted in a solution (default 2e-4). */
  readonly catol?: number
}

/** One evaluated point: objective, constraint values and the largest violation. */
interface Vertex {
  readonly x: Float64Array
  readonly f: number
  readonly c: Float64Array
  readonly resmax: number
}

/** State of a COBYLA run. */
export interface CobylaState extends VectorState {
  /** The simplex of interpolation points, the best (by the merit function) first. */
  readonly vertices: readonly Vertex[]
  /** The trust radius. */
  readonly rho: number
  /** The penalty parameter of the merit function f + mu max(0, max_i -c_i). */
  readonly mu: number
  /** The largest constraint violation at x. */
  readonly maxcv: number
  /** What the next step does. */
  readonly mode: 'trust' | 'geometry'
  /** What the last step did. */
  readonly action: 'initial' | 'trust step' | 'geometry step' | 'reduce radius' | 'rejected step'
}

const ALPHA = 0.25
const BETA = 2.1
const GAMMA = 0.5
const DELTA = 1.1

const merit = (v: Vertex, mu: number): number => v.f + mu * v.resmax

function evaluate(
  obj: FlatObjective,
  constraints: readonly InequalityConstraint[],
  linear: readonly { readonly i: number; readonly sign: number; readonly bound: number }[],
  x: Float64Array,
): Vertex {
  const vals: number[] = []
  for (const c of constraints) vals.push(...constraintValues(c, obj, x))
  for (const l of linear) vals.push(l.sign * ((x[l.i] ?? 0) - l.bound))
  const c = Float64Array.from(vals)
  let resmax = 0
  for (const v of c) resmax = Math.max(resmax, -v)
  return { x, f: obj.value(x), c, resmax: Number.isNaN(resmax) ? Number.POSITIVE_INFINITY : resmax }
}

/** The best vertex by merit first; ties prefer the smaller violation (Powell's rule). */
function orderVertices(vertices: readonly Vertex[], mu: number): Vertex[] {
  let best = 0
  for (let j = 1; j < vertices.length; j++) {
    const a = merit(vertices[j] as Vertex, mu)
    const b = merit(vertices[best] as Vertex, mu)
    if (a < b || (a === b && (vertices[j] as Vertex).resmax < (vertices[best] as Vertex).resmax)) best = j
  }
  if (best === 0) return [...vertices]
  const out = [...vertices]
  out[0] = vertices[best] as Vertex
  out[best] = vertices[0] as Vertex
  return out
}

/** Linear models through the simplex: the gradient of f and of each constraint, and the inverse of the edge matrix. */
function models(vertices: readonly Vertex[], n: number, m: number) {
  const v0 = vertices[0] as Vertex
  const d = new Float64Array(n * n)
  const rhs = new Float64Array(n * (m + 1))
  for (let j = 1; j <= n; j++) {
    const vj = vertices[j] as Vertex
    for (let i = 0; i < n; i++) d[(j - 1) * n + i] = (vj.x[i] ?? 0) - (v0.x[i] ?? 0)
    rhs[(j - 1) * (m + 1)] = vj.f - v0.f
    for (let k = 0; k < m; k++) rhs[(j - 1) * (m + 1) + k + 1] = (vj.c[k] ?? 0) - (v0.c[k] ?? 0)
  }
  try {
    const dt = fromStorage(d, [n, n], 'float64')
    const w = Float64Array.from(toFlat(inv(dt)))
    const sol = Float64Array.from(toFlat(solve(dt, fromStorage(rhs, [n, m + 1], 'float64'))))
    const g = Float64Array.from({ length: n }, (_, i) => sol[i * (m + 1)] ?? 0)
    const a = Array.from({ length: m }, (_, k) =>
      Float64Array.from({ length: n }, (_, i) => sol[i * (m + 1) + k + 1] ?? 0),
    )
    if (![...g, ...w].every(Number.isFinite)) return undefined
    return { d, w, g, a }
  } catch {
    return undefined
  }
}

/** Distances of the vertices from x0 (veta) and from their opposite faces (vsig = 1 / |column j of D^-1|). */
function shape(d: Float64Array, w: Float64Array, n: number): { veta: number[]; vsig: number[] } {
  const veta: number[] = []
  const vsig: number[] = []
  for (let j = 0; j < n; j++) {
    let e = 0
    let s = 0
    for (let i = 0; i < n; i++) {
      e += (d[j * n + i] ?? 0) ** 2
      s += (w[i * n + j] ?? 0) ** 2
    }
    veta.push(Math.sqrt(e))
    vsig.push(1 / Math.sqrt(s))
  }
  return { veta, vsig }
}

/**
 * COBYLA (Powell 1994), constrained optimisation by linear approximations, as a traced algorithm: a simplex of n + 1
 * points interpolates the objective and the constraints linearly; each step solves the linearised problem in a trust
 * region of radius rho (least constraint violation first, then the objective), evaluates the step, updates the
 * penalty parameter of the merit function and replaces a vertex, or improves the simplex geometry, or halves rho.
 * It stops when rho reaches rhoend. Constraints follow scipy: c(x) >= 0.
 *
 * @param f - The objective.
 * @param x0 - The starting point (any parameter tree).
 * @param options - Constraints, bounds, rhobeg, rhoend, the evaluation budget and the feasibility tolerance.
 * @returns The algorithm.
 * @concept derivative-free-optimisation
 * @example
 * ```ts
 * import { cobylaAlgorithm } from 'whitebox-engine/optim'
 * import { square, sum, type Value } from 'whitebox-engine/tensor'
 * cobylaAlgorithm((x) => sum(square(x as Value)), [1, 2]).name // => 'cobyla'
 * ```
 */
export function cobylaAlgorithm(f: Objective, x0: Parameters, options: CobylaOptions = {}): Algorithm<CobylaState> {
  const obj = flatObjective(f, x0)
  const n = obj.n
  if (n === 0) throw new WhiteboxError('value', 'cobyla: no parameters')
  const constraints = options.constraints ?? []
  const linear: { i: number; sign: number; bound: number }[] = []
  if (options.bounds !== undefined) {
    const box = boundArrays(options.bounds, n, 'cobyla')
    for (let i = 0; i < n; i++) {
      if (Number.isFinite(box.lower[i] ?? 0)) linear.push({ i, sign: 1, bound: box.lower[i] ?? 0 })
      if (Number.isFinite(box.upper[i] ?? 0)) linear.push({ i, sign: -1, bound: box.upper[i] ?? 0 })
    }
  }
  const rhobeg = options.rhobeg ?? 1
  const rhoend = Math.min(options.rhoend ?? 1e-4, rhobeg)
  const maxfun = options.maxEvaluations ?? 1000
  const catol = options.catol ?? 2e-4
  const point = (x: Float64Array): Vertex => evaluate(obj, constraints, linear, x)

  const finish = (s: Omit<CobylaState, 'x' | 'value' | 'maxcv' | 'grad' | 'ngev' | 'nhev'>): CobylaState => {
    const best = s.vertices[0] as Vertex
    const base = { ...s, x: best.x, value: best.f, maxcv: best.resmax, grad: new Float64Array(0), ngev: 0, nhev: 0 }
    if (s.stop !== undefined) return base
    if (s.nfev >= maxfun)
      return { ...base, stop: { status: 'failed', message: 'maximum number of function evaluations reached' } }
    return base
  }
  const converge = (s: Omit<CobylaState, 'x' | 'value' | 'maxcv' | 'grad' | 'ngev' | 'nhev'>): CobylaState => {
    const best = s.vertices[0] as Vertex
    return finish({
      ...s,
      stop:
        best.resmax <= catol
          ? { status: 'converged', message: 'trust radius reached rhoend' }
          : { status: 'failed', message: `trust radius reached rhoend with constraint violation ${best.resmax}` },
    })
  }

  /** Halve rho (to rhoend at the end), reduce mu as Powell does, or stop when rho is already rhoend. */
  const reduce = (vertices: Vertex[], rho: number, mu: number, nfev: number): CobylaState => {
    if (rho <= rhoend) return converge({ vertices, rho, mu, nfev, step: rho, mode: 'trust', action: 'reduce radius' })
    let next = 0.5 * rho
    if (next <= 1.5 * rhoend) next = rhoend
    let nmu = mu
    if (mu > 0) {
      const m = (vertices[0] as Vertex).c.length
      let denom = 0
      for (let k = 0; k < m; k++) {
        let cmin = Number.POSITIVE_INFINITY
        let cmax = Number.NEGATIVE_INFINITY
        for (const v of vertices) {
          cmin = Math.min(cmin, v.c[k] ?? 0)
          cmax = Math.max(cmax, v.c[k] ?? 0)
        }
        if (cmin < 0.5 * cmax) {
          const temp = Math.max(cmax, 0) - cmin
          denom = denom <= 0 ? temp : Math.min(denom, temp)
        }
      }
      let fmin = Number.POSITIVE_INFINITY
      let fmax = Number.NEGATIVE_INFINITY
      for (const v of vertices) {
        fmin = Math.min(fmin, v.f)
        fmax = Math.max(fmax, v.f)
      }
      if (denom === 0) nmu = 0
      else if (fmax - fmin < mu * denom) nmu = (fmax - fmin) / denom
    }
    return finish({
      vertices: orderVertices(vertices, nmu),
      rho: next,
      mu: nmu,
      nfev,
      step: next,
      mode: 'trust',
      action: 'reduce radius',
    })
  }

  /** Replace vertex j (1..n) by a new point and put the best vertex first. */
  const replace = (vertices: readonly Vertex[], j: number, v: Vertex, mu: number): Vertex[] => {
    const out = [...vertices]
    out[j] = v
    return orderVertices(out, mu)
  }

  const geometry = (s: CobylaState, vertices: Vertex[], mdl: NonNullable<ReturnType<typeof models>>): CobylaState => {
    const { veta, vsig } = shape(mdl.d, mdl.w, n)
    let jdrop = 0
    if (veta.some((e) => e > BETA * s.rho)) {
      for (let j = 1; j < n; j++) if ((veta[j] ?? 0) > (veta[jdrop] ?? 0)) jdrop = j
    } else {
      for (let j = 1; j < n; j++) if ((vsig[j] ?? 0) < (vsig[jdrop] ?? 0)) jdrop = j
    }
    // Move the vertex along the normal of its opposite face, on the side the linear merit prefers.
    const col = Float64Array.from({ length: n }, (_, i) => mdl.w[i * n + jdrop] ?? 0)
    const dx = Float64Array.from(col, (v) => (GAMMA * s.rho * v) / dnrm2(col))
    const v0 = vertices[0] as Vertex
    const predicted = (sign: number): number => {
      let viol = 0
      for (const [k, ak] of mdl.a.entries()) viol = Math.max(viol, -((v0.c[k] ?? 0) + sign * ddot(ak, dx)))
      return sign * ddot(mdl.g, dx) + s.mu * viol
    }
    const sign = predicted(-1) < predicted(1) ? -1 : 1
    const v = point(daxpy(sign, dx, v0.x))
    return finish({
      vertices: replace(vertices, jdrop + 1, v, s.mu),
      rho: s.rho,
      mu: s.mu,
      nfev: s.nfev + 1,
      step: s.rho,
      mode: 'trust',
      action: 'geometry step',
    })
  }

  return {
    name: 'cobyla',
    config: configOf({ rhobeg, rhoend, maxEvaluations: maxfun, catol, constraints: constraints.length }),
    schema: vectorSchema({
      metrics: {
        rho: { description: 'trust radius' },
        mu: { description: 'penalty parameter of the merit function' },
        maxcv: { description: 'largest constraint violation at the best point', direction: 'lower' },
      },
      tensors: { simplex: { role: 'state', description: 'the interpolation simplex, best vertex first' } },
      messages: { action: { description: 'what the step did: trust step, geometry step or radius reduction' } },
    }),
    init: () => {
      const start = point(Float64Array.from(obj.x0))
      let vertices: Vertex[] = [start]
      for (let j = 0; j < n; j++) {
        const x = Float64Array.from(obj.x0)
        x[j] = (x[j] ?? 0) + rhobeg
        vertices.push(point(x))
      }
      vertices = orderVertices(vertices, 0)
      return finish({ vertices, rho: rhobeg, mu: 0, nfev: n + 1, step: rhobeg, mode: 'trust', action: 'initial' })
    },
    step: (s) => {
      let vertices = orderVertices(s.vertices, s.mu)
      let mdl = models(vertices, n, (vertices[0] as Vertex).c.length)
      if (mdl === undefined) {
        // A degenerate simplex: rebuild it around the best point.
        const v0 = vertices[0] as Vertex
        const rebuilt = [v0]
        for (let j = 0; j < n; j++) {
          const x = Float64Array.from(v0.x)
          x[j] = (x[j] ?? 0) + s.rho
          rebuilt.push(point(x))
        }
        return finish({
          vertices: orderVertices(rebuilt, s.mu),
          rho: s.rho,
          mu: s.mu,
          nfev: s.nfev + n,
          step: s.rho,
          mode: 'trust',
          action: 'geometry step',
        })
      }
      const acceptable = (): boolean => {
        const { veta, vsig } = shape((mdl as NonNullable<typeof mdl>).d, (mdl as NonNullable<typeof mdl>).w, n)
        return veta.every((e) => e <= BETA * s.rho) && vsig.every((g) => g >= ALPHA * s.rho)
      }
      if (s.mode === 'geometry') return geometry(s, vertices, mdl)
      let mu = s.mu
      let d: Float64Array = new Float64Array(n)
      let prerec = 0
      let sum = 0
      // Compute the step; raising mu may change the best vertex, in which case recompute (Powell's label 140).
      for (let pass = 0; pass <= n + 1; pass++) {
        const v0 = vertices[0] as Vertex
        d = trustStep(mdl.g, mdl.a, v0.c, s.rho)
        let resnew = 0
        for (const [k, ak] of mdl.a.entries()) resnew = Math.max(resnew, -((v0.c[k] ?? 0) + ddot(ak, d)))
        prerec = v0.resmax - resnew
        sum = ddot(mdl.g, d)
        const barmu = prerec > 0 ? sum / prerec : 0
        if (mu < 1.5 * barmu) {
          mu = 2 * barmu
          const reordered = orderVertices(vertices, mu)
          if (reordered[0] !== vertices[0]) {
            vertices = reordered
            const again = models(vertices, n, v0.c.length)
            if (again === undefined) break
            mdl = again
            continue
          }
        }
        break
      }
      if (dnrm2(d) < 0.5 * s.rho) {
        if (!acceptable()) return geometry({ ...s, mu }, vertices, mdl)
        return reduce(vertices, s.rho, mu, s.nfev)
      }
      const v0 = vertices[0] as Vertex
      const trial = point(daxpy(1, d, v0.x))
      const nfev = s.nfev + 1
      let prerem = mu * prerec - sum
      let actrem = merit(v0, mu) - merit(trial, mu)
      if (mu === 0 && trial.f === v0.f) {
        prerem = prerec
        actrem = v0.resmax - trial.resmax
      }
      // Choose the vertex to drop: the largest determinant change, weighted by distance from the trial point.
      const sigma = Float64Array.from({ length: n }, (_, j) => {
        let s2 = 0
        for (let i = 0; i < n; i++) s2 += (mdl.w[i * n + j] ?? 0) * (d[i] ?? 0)
        return s2
      })
      let jdrop = -1
      let best = actrem > 0 ? -1 : 1
      for (let j = 0; j < n; j++) {
        const vj = vertices[j + 1] as Vertex
        let dist = 0
        for (let i = 0; i < n; i++) dist += ((vj.x[i] ?? 0) - (trial.x[i] ?? 0)) ** 2
        const weight = Math.max(1, Math.sqrt(dist) / (DELTA * s.rho)) ** 3
        const score = Math.abs(sigma[j] ?? 0) * weight
        if (score > best) {
          best = score
          jdrop = j
        }
      }
      const kept = jdrop >= 0 ? replace(vertices, jdrop + 1, trial, mu) : vertices
      const action = jdrop >= 0 ? 'trust step' : 'rejected step'
      if (prerem > 0 && actrem > 0.1 * prerem) {
        return finish({ vertices: kept, rho: s.rho, mu, nfev, step: dnrm2(d), mode: 'trust', action })
      }
      const after = models(kept, n, v0.c.length)
      if (after === undefined)
        return finish({ vertices: kept, rho: s.rho, mu, nfev, step: dnrm2(d), mode: 'geometry', action })
      const { veta, vsig } = shape(after.d, after.w, n)
      const ok = veta.every((e) => e <= BETA * s.rho) && vsig.every((g) => g >= ALPHA * s.rho)
      if (!ok) return finish({ vertices: kept, rho: s.rho, mu, nfev, step: dnrm2(d), mode: 'geometry', action })
      const reduced = reduce(kept, s.rho, mu, nfev)
      return { ...reduced, action }
    },
    done: vectorDone,
    observe: (s) => {
      const flat = new Float64Array((n + 1) * n)
      for (const [k, v] of s.vertices.entries()) flat.set(v.x, k * n)
      return observeVector(s, {
        metrics: { rho: s.rho, mu: s.mu, maxcv: s.maxcv },
        tensors: { simplex: { role: 'state', value: flat, shape: [n + 1, n] } },
        messages: [{ kind: 'action', payload: { action: s.action, rho: s.rho } }],
      })
    },
  }
}

/**
 * Minimise subject to inequality constraints without derivatives, by COBYLA (scipy.optimize.minimize with method
 * 'COBYLA'; constraints c(x) >= 0, bounds as linear constraints).
 *
 * @param f - The objective.
 * @param x0 - The starting point (any parameter tree).
 * @param options - Constraints, bounds, rhobeg (default 1), rhoend (default 1e-4), the evaluation budget (default
 *   1000), the feasibility tolerance and run options.
 * @returns The result; `converged` requires the final violation to be within `catol`.
 * @concept derivative-free-optimisation
 * @example
 * ```ts
 * import { cobyla } from 'whitebox-engine/optim'
 * import { square, sub, sum, toFlat, type Value } from 'whitebox-engine/tensor'
 * // minimise |x - (2, 2)|^2 subject to 1 - x - y >= 0: the answer is (0.5, 0.5)
 * const r = cobyla((x) => sum(square(sub(x as Value, 2))), [0, 0], {
 *   constraints: [(x) => sub(1, sum(x as Value))],
 *   rhoend: 1e-8,
 * })
 * toFlat(r.x as Value).map((v) => Math.round(v * 1e6) / 1e6) // => [0.5, 0.5]
 * ```
 */
export function cobyla(f: Objective, x0: Parameters, options: CobylaOptions = {}): MinimizeResult {
  const obj = flatObjective(f, x0)
  const { maxIterations, ...rest } = options
  return runVector(cobylaAlgorithm(f, x0, rest), obj, { ...rest, maxIterations: maxIterations ?? 100_000 }, 100_000)
}
