/**
 * Item response theory for scored (0/1) test data: the 1PL (Rasch), 2PL and 3PL logistic models, P(correct | theta) = c +
 * (1 - c) sigmoid(a (theta - b)) with the ability theta ~ N(0, 1) integrated out on a quadrature grid (Bock and
 * Aitkin's marginal maximum likelihood). The marginal log-likelihood is smooth in the item parameters (difficulty b, log
 * discrimination, guessing on a logit scale), so the fit takes its solver from the slot: Bock-Aitkin EM (the model's own
 * mode) beside Newton, BFGS, L-BFGS and the other generic solvers, which reach the same items. Abilities are estimated
 * for a respondent by maximum likelihood, maximum a posteriori or the expected a posteriori mean, and the item and test
 * information functions give the precision of those estimates at each ability. Missing responses are `null`.
 */

import { MdmlError, maxOf } from '../base/index.ts'
import { gaussHermite } from '../numerics/quadrature/index.ts'
import {
  analyticObjective,
  familySolvers,
  finiteDifferenceHessian,
  type SlotState,
  type SmoothProblem,
  slotAlgorithm,
  type VectorParams,
} from '../optim/index.ts'
import { type Key, key as makeKey, uniform } from '../random/index.ts'
import { dict, type SpaceValue } from '../space/index.ts'
import { toFlat } from '../tensor/index.ts'
import { type Algorithm, type ModeSpec, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'

/** The item response model: which item parameters are free. */
export type IrtModel = '1pl' | '2pl' | '3pl'

/** The parameters of one item. */
export interface IrtItem {
  /** The discrimination (1 under the Rasch model). */
  readonly a: number
  /** The difficulty. */
  readonly b: number
  /** The guessing floor (0 below 3PL). */
  readonly c: number
}

/** Scored responses, one row per respondent, one column per item; `null` marks a response not observed. */
export type IrtResponses = readonly (readonly (0 | 1 | null)[])[]

/** The data and model of an IRT fit. */
export interface IrtProblem {
  readonly responses: IrtResponses
  readonly model: IrtModel
  /** The number of Gauss-Hermite nodes (default 21). */
  readonly nodes?: number
}

/** The largest guessing floor the logit parameterisation allows. */
const C_MAX = 0.5

const sigmoid = (z: number): number => 1 / (1 + Math.exp(-z))

/**
 * The probability of a correct response.
 *
 * @param item - The item.
 * @param theta - The ability.
 * @returns c + (1 - c) / (1 + exp(-a (theta - b))).
 * @example
 * ```ts
 * import { irtProbability } from 'mdml-engine/pgm'
 * irtProbability({ a: 1, b: 0, c: 0 }, 0) // => 0.5
 * ```
 */
export function irtProbability(item: IrtItem, theta: number): number {
  return item.c + (1 - item.c) * sigmoid(item.a * (theta - item.b))
}

/**
 * The information of an item at an ability: a^2 (P - c)^2 Q / ((1 - c)^2 P), the squared slope of the response curve over
 * its binomial variance. It peaks near the item's difficulty and is a^2 / 4 there under the Rasch/2PL models.
 *
 * @param item - The item.
 * @param theta - The ability.
 * @returns The Fisher information about theta from one response.
 * @example
 * ```ts
 * import { irtItemInformation } from 'mdml-engine/pgm'
 * irtItemInformation({ a: 1, b: 0, c: 0 }, 0) // => 0.25
 * ```
 */
export function irtItemInformation(item: IrtItem, theta: number): number {
  const p = irtProbability(item, theta)
  const s = (p - item.c) / (1 - item.c)
  return (item.a * item.a * s * s * (1 - p)) / p
}

/**
 * The test information at an ability: the sum of the item informations. The standard error of an ability estimate is
 * its reciprocal square root.
 *
 * @param items - The items.
 * @param theta - The ability.
 * @returns The information.
 * @example
 * ```ts
 * import { irtTestInformation } from 'mdml-engine/pgm'
 * irtTestInformation([{ a: 1, b: 0, c: 0 }, { a: 1, b: 0, c: 0 }], 0) // => 0.5
 * ```
 */
export function irtTestInformation(items: readonly IrtItem[], theta: number): number {
  return items.reduce((s, it) => s + irtItemInformation(it, theta), 0)
}

function layout(model: IrtModel): { readonly per: number } {
  return { per: model === '1pl' ? 1 : model === '2pl' ? 2 : 3 }
}

/** The items from the unconstrained vector: b, then log a (2PL, 3PL), then the guessing logit (3PL). */
function itemsOf(model: IrtModel, x: ArrayLike<number>): IrtItem[] {
  const { per } = layout(model)
  const nItems = x.length / per
  return Array.from({ length: nItems }, (_, j) => ({
    b: x[j * per] ?? 0,
    a: model === '1pl' ? 1 : Math.exp(x[j * per + 1] ?? 0),
    c: model === '3pl' ? C_MAX * sigmoid(x[j * per + 2] ?? 0) : 0,
  }))
}

function vectorOf(model: IrtModel, items: readonly IrtItem[]): number[] {
  const out: number[] = []
  for (const it of items) {
    out.push(it.b)
    if (model !== '1pl') out.push(Math.log(it.a))
    if (model === '3pl') out.push(Math.log(it.c / C_MAX / (1 - it.c / C_MAX)))
  }
  return out
}

interface Grid {
  readonly theta: Float64Array
  readonly logWeight: Float64Array
}

function gridOf(n: number): Grid {
  const rule = gaussHermite(n)
  const x = toFlat(rule.nodes)
  const w = toFlat(rule.weights)
  return {
    theta: Float64Array.from(x, (v) => v * Math.SQRT2),
    logWeight: Float64Array.from(w, (v) => Math.log(v / Math.sqrt(Math.PI))),
  }
}

/** The expected counts at each node (E-step) and the marginal log-likelihood. */
function expectations(
  problem: IrtProblem,
  items: readonly IrtItem[],
  grid: Grid,
): { n: Float64Array[]; r: Float64Array[]; ll: number } {
  const Q = grid.theta.length
  const J = items.length
  const n = Array.from({ length: J }, () => new Float64Array(Q))
  const r = Array.from({ length: J }, () => new Float64Array(Q))
  const logP = items.map((it) => Float64Array.from(grid.theta, (t) => Math.log(irtProbability(it, t))))
  const logQ = items.map((it) => Float64Array.from(grid.theta, (t) => Math.log(1 - irtProbability(it, t))))
  let ll = 0
  for (const row of problem.responses) {
    const lp = Float64Array.from(grid.logWeight)
    for (let j = 0; j < J; j++) {
      const x = row[j]
      if (x === null || x === undefined) continue
      for (let q = 0; q < Q; q++) lp[q] = (lp[q] ?? 0) + (x === 1 ? (logP[j]?.[q] ?? 0) : (logQ[j]?.[q] ?? 0))
    }
    const mx = maxOf(lp)
    let z = 0
    for (let q = 0; q < Q; q++) z += Math.exp((lp[q] ?? 0) - mx)
    ll += mx + Math.log(z)
    for (let q = 0; q < Q; q++) {
      const w = Math.exp((lp[q] ?? 0) - mx) / z
      for (let j = 0; j < J; j++) {
        const x = row[j]
        if (x === null || x === undefined) continue
        const nj = n[j] as Float64Array
        const rj = r[j] as Float64Array
        nj[q] = (nj[q] ?? 0) + w
        if (x === 1) rj[q] = (rj[q] ?? 0) + w
      }
    }
  }
  return { n, r, ll }
}

/** The gradient of an item's expected complete-data log-likelihood in its unconstrained parameters. */
function itemGradient(model: IrtModel, eta: ArrayLike<number>, n: Float64Array, r: Float64Array, grid: Grid): number[] {
  const it = itemsOf(model, eta)[0] as IrtItem
  const g = [0, 0, 0]
  for (let q = 0; q < grid.theta.length; q++) {
    const t = grid.theta[q] ?? 0
    const s = sigmoid(it.a * (t - it.b))
    const p = it.c + (1 - it.c) * s
    const dldp = (r[q] ?? 0) / p - ((n[q] ?? 0) - (r[q] ?? 0)) / (1 - p)
    const ds = s * (1 - s)
    g[0] = (g[0] ?? 0) + dldp * (-(1 - it.c) * it.a * ds)
    g[1] = (g[1] ?? 0) + dldp * ((1 - it.c) * (t - it.b) * ds * it.a)
    g[2] = (g[2] ?? 0) + dldp * (1 - s) * it.c * (1 - it.c / C_MAX)
  }
  return g.slice(0, layout(model).per)
}

/** The marginal negative log-likelihood per respondent and its gradient (Fisher's identity: the expected complete gradient). */
function evaluate(problem: IrtProblem, grid: Grid, x: ArrayLike<number>): { value: number; grad: Float64Array } {
  const { per } = layout(problem.model)
  const items = itemsOf(problem.model, x)
  const { n, r, ll } = expectations(problem, items, grid)
  const N = problem.responses.length
  const grad = new Float64Array(x.length)
  items.forEach((_, j) => {
    const eta = Array.from({ length: per }, (_, k) => x[j * per + k] ?? 0)
    const g = itemGradient(problem.model, eta, n[j] as Float64Array, r[j] as Float64Array, grid)
    g.forEach((v, k) => {
      grad[j * per + k] = -v / N
    })
  })
  return { value: -ll / N, grad }
}

/**
 * The marginal negative log-likelihood per respondent as a smooth problem in the item parameters, with its analytic
 * gradient. The start is the classical one: difficulty from the item's proportion correct, discrimination one, guessing
 * 0.1.
 *
 * @param problem - The responses and the model.
 * @returns The objective and the start.
 * @example
 * ```ts
 * import { irtSmoothProblem } from 'mdml-engine/pgm'
 * irtSmoothProblem({ model: '1pl', responses: [[1, 0], [1, 1], [0, 0], [1, 0]] }).x0.length // => 2
 * ```
 */
export function irtSmoothProblem(problem: IrtProblem): SmoothProblem {
  const grid = gridOf(problem.nodes ?? 21)
  const J = problem.responses[0]?.length ?? 0
  if (J < 2 || problem.responses.length < 3)
    throw new MdmlError('value', 'irt: need at least two items and three respondents')
  const start = Array.from({ length: J }, (_, j): IrtItem => {
    const seen = problem.responses.map((row) => row[j]).filter((v) => v !== null && v !== undefined) as number[]
    const p = Math.min(Math.max(seen.reduce((a, b) => a + b, 0) / Math.max(seen.length, 1), 0.02), 0.98)
    return { a: 1, b: -Math.log(p / (1 - p)), c: problem.model === '3pl' ? 0.1 : 0 }
  })
  return {
    objective: analyticObjective({
      value: (x) => evaluate(problem, grid, x).value,
      gradient: (x) => evaluate(problem, grid, x).grad,
    }),
    x0: vectorOf(problem.model, start),
  }
}

interface EmState {
  readonly x: Float64Array
  readonly value: number
  readonly grad: Float64Array
  readonly step: number
}

/** Newton's method with a ridge on one item's expected complete-data log-likelihood (1 to 3 parameters). */
function mStep(model: IrtModel, eta0: number[], n: Float64Array, r: Float64Array, grid: Grid): number[] {
  const Q = (eta: number[]): number => {
    const it = itemsOf(model, eta)[0] as IrtItem
    let q = 0
    for (let k = 0; k < grid.theta.length; k++) {
      const p = Math.min(Math.max(irtProbability(it, grid.theta[k] ?? 0), 1e-12), 1 - 1e-12)
      q += (r[k] ?? 0) * Math.log(p) + ((n[k] ?? 0) - (r[k] ?? 0)) * Math.log(1 - p)
    }
    return q
  }
  let eta = [...eta0]
  let q = Q(eta)
  for (let it = 0; it < 20; it++) {
    const g = itemGradient(model, eta, n, r, grid)
    if (maxOf(g.map(Math.abs)) < 1e-9) break
    const H = finiteDifferenceHessian((x) => itemGradient(model, x, n, r, grid), eta).data
    const m = eta.length
    let accepted = false
    for (let lam = 1e-8; lam < 1e6 && !accepted; lam *= 10) {
      // Solve (-H + lam I) d = g by Gaussian elimination.
      const A = Array.from({ length: m }, (_, i) =>
        Array.from({ length: m }, (_, j) => -(H[i * m + j] ?? 0) + (i === j ? lam : 0)),
      )
      const d = g.slice()
      for (let i = 0; i < m; i++) {
        let piv = i
        for (let k = i + 1; k < m; k++) if (Math.abs(A[k]?.[i] ?? 0) > Math.abs(A[piv]?.[i] ?? 0)) piv = k
        ;[A[i], A[piv]] = [A[piv] as number[], A[i] as number[]]
        ;[d[i], d[piv]] = [d[piv] as number, d[i] as number]
        const p = A[i]?.[i] ?? 0
        if (Math.abs(p) < 1e-14) break
        for (let k = i + 1; k < m; k++) {
          const f = (A[k]?.[i] ?? 0) / p
          for (let c = i; c < m; c++) (A[k] as number[])[c] = (A[k]?.[c] ?? 0) - f * (A[i]?.[c] ?? 0)
          d[k] = (d[k] ?? 0) - f * (d[i] ?? 0)
        }
      }
      for (let i = m - 1; i >= 0; i--) {
        let s = d[i] ?? 0
        for (let c = i + 1; c < m; c++) s -= (A[i]?.[c] ?? 0) * (d[c] ?? 0)
        d[i] = s / (A[i]?.[i] || 1)
      }
      const trial = eta.map((v, i) => v + Math.max(-2, Math.min(2, d[i] ?? 0)))
      const qt = Q(trial)
      if (qt >= q - 1e-12) {
        eta = trial
        q = qt
        accepted = true
      }
    }
    if (!accepted) break
  }
  return eta
}

const em: ModeSpec<IrtProblem, EmState, VectorParams> = {
  description:
    'Bock-Aitkin EM: the E-step gives the expected number of respondents at each ability node and how many of them answer each item correctly, the M-step maximises each item separately by Newton steps; the marginal likelihood never decreases',
  space: dict({}),
  make: (p): Algorithm<EmState> => {
    const grid = gridOf(p.nodes ?? 21)
    const smooth = irtSmoothProblem(p)
    const at = (x: Float64Array, step: number): EmState => ({ x, ...evaluate(p, grid, x), step })
    return {
      name: 'irt-em',
      config: { model: p.model } as never,
      init: () => at(Float64Array.from(smooth.x0 as number[]), 0),
      step: (s) => {
        const { per } = layout(p.model)
        const items = itemsOf(p.model, s.x)
        const { n, r } = expectations(p, items, grid)
        const next = new Float64Array(s.x.length)
        items.forEach((_, j) => {
          const eta = Array.from({ length: per }, (_, k) => s.x[j * per + k] ?? 0)
          mStep(p.model, eta, n[j] as Float64Array, r[j] as Float64Array, grid).forEach((v, k) => {
            next[j * per + k] = v
          })
        })
        return at(next, Math.hypot(...Array.from(next, (v, i) => v - (s.x[i] ?? 0))))
      },
      serialise: (s) => stateToJson(s),
      deserialise: (j) => stateFromJson(j) as EmState,
    }
  },
  params: (s) => ({ x: Array.from(s.x) }),
}

/** The ways to fit an IRT model: Bock-Aitkin `em`, then every generic smooth solver on the marginal likelihood. */
export const irtSolvers: ReturnType<typeof familySolvers<IrtProblem>> = familySolvers({ em }, irtSmoothProblem, {
  label: 'solver',
  description: 'how the marginal log-likelihood of the items is maximised',
  default: 'em',
})

/** The state of {@link irtAlgorithm}. */
export type IrtState = SlotState & {
  readonly u: number[]
  readonly objective: number
  readonly gradient: number
}

/** Options of {@link irtAlgorithm} and {@link fitIrt}. */
export interface IrtOptions {
  /** A value of `irtSolvers.space` (default `em`). */
  readonly solver?: SpaceValue
  /** The gradient tolerance (default 1e-5). */
  readonly tol?: number
  /** The iteration cap of `fitIrt` (default 500). */
  readonly maxSteps?: number
  readonly key?: Key
}

/**
 * The IRT fit as a traced algorithm over the solver slot, logging `objective` (the negative marginal log-likelihood per
 * respondent), `gradient`, `gradNorm`, `step` and the tensor `items` (the unconstrained parameters).
 *
 * @param problem - Responses and model.
 * @param options - The solver and tolerance.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { irtAlgorithm } from 'mdml-engine/pgm'
 * irtAlgorithm({ model: '2pl', responses: [[1, 0], [1, 1], [0, 0], [1, 0]] }).name // => 'irt'
 * ```
 */
export function irtAlgorithm(problem: IrtProblem, options: IrtOptions = {}): Algorithm<IrtState> {
  const solver = options.solver ?? { kind: 'em' }
  const size = (problem.responses[0]?.length ?? 0) * layout(problem.model).per
  return slotAlgorithm<IrtState, IrtProblem>({
    name: 'irt',
    config: { model: problem.model, solver } as never,
    schema: {
      metrics: {
        objective: { description: 'the negative marginal log-likelihood per respondent', direction: 'lower' },
        gradient: { description: 'the largest gradient entry', direction: 'lower' },
      },
      tensors: {
        items: {
          role: 'param',
          description: 'item parameters (difficulty, log discrimination, guessing logit)',
          shape: [size],
        },
      },
    },
    problem,
    table: irtSolvers,
    solver,
    ...(options.key === undefined ? {} : { key: options.key }),
    tol: options.tol ?? 1e-5,
    view: (r) => ({
      u: Array.from(r.x),
      objective: r.value,
      gradient: r.grad.reduce((a, b) => Math.max(a, Math.abs(b)), 0),
    }),
    observe: (s) => ({
      metrics: { objective: s.objective, gradient: s.gradient },
      tensors: { items: { role: 'param', value: s.u, shape: [size] } },
    }),
  })
}

/** A fitted IRT model. */
export interface IrtFit {
  readonly model: IrtModel
  readonly items: readonly IrtItem[]
  readonly logLikelihood: number
  readonly aic: number
  readonly bic: number
  readonly converged: boolean
  readonly steps: number
  readonly solver: string
}

/**
 * Fit an IRT model by marginal maximum likelihood.
 *
 * @param problem - Responses and model.
 * @param options - The solver (default Bock-Aitkin EM), tolerance and cap.
 * @returns The items and the likelihood.
 * @example
 * ```ts
 * import { fitIrt, irtSimulate } from 'mdml-engine/pgm'
 * import { key } from 'mdml-engine/random'
 * const items = [{ a: 1, b: -1, c: 0 }, { a: 1, b: 0, c: 0 }, { a: 1, b: 1, c: 0 }]
 * const { responses } = irtSimulate(key(0), 200, items)
 * fitIrt({ responses, model: '1pl' }).items.length // => 3
 * ```
 */
export function fitIrt(problem: IrtProblem, options: IrtOptions = {}): IrtFit {
  const algorithm = irtAlgorithm(problem, options)
  const run = runAlgorithm(algorithm, { maxSteps: options.maxSteps ?? 500, key: options.key ?? makeKey(0) })
  const items = itemsOf(problem.model, run.state.u)
  const N = problem.responses.length
  const k = run.state.u.length
  const ll = -run.state.objective * N
  return {
    model: problem.model,
    items,
    logLikelihood: ll,
    aic: -2 * ll + 2 * k,
    bic: -2 * ll + k * Math.log(N),
    converged: run.status === 'converged',
    steps: run.steps,
    solver: irtSolvers.modeOf(options.solver ?? { kind: 'em' }).name,
  }
}

/** An ability estimate and its precision. */
export interface IrtAbility {
  readonly theta: number
  /** The standard error: the posterior deviation (EAP, MAP) or 1 / sqrt(information) (MLE). */
  readonly se: number
}

/**
 * The ability of one respondent given the items: the expected a posteriori mean over the N(0, 1) prior (`eap`, the
 * default, always finite), the maximum a posteriori mode (`map`), or the maximum likelihood estimate (`mle`, undefined
 * for all-correct or all-wrong patterns, which return +-Infinity).
 *
 * @param responses - The respondent's responses (`null` for unseen items).
 * @param items - The items.
 * @param method - `eap`, `map` or `mle`.
 * @returns The estimate and its standard error.
 * @example
 * ```ts
 * import { irtAbility } from 'mdml-engine/pgm'
 * const items = [{ a: 1, b: -1, c: 0 }, { a: 1, b: 0, c: 0 }, { a: 1, b: 1, c: 0 }]
 * irtAbility([1, 0, 0], items, 'eap').theta < irtAbility([1, 1, 0], items, 'eap').theta // => true
 * ```
 */
export function irtAbility(
  responses: readonly (0 | 1 | null)[],
  items: readonly IrtItem[],
  method: 'eap' | 'map' | 'mle' = 'eap',
): IrtAbility {
  const observed = responses
    .map((x, j) => [x, items[j]] as const)
    .filter(([x]) => x !== null && x !== undefined) as (readonly [0 | 1, IrtItem])[]
  const logLik = (t: number): number =>
    observed.reduce((s, [x, it]) => {
      const p = Math.min(Math.max(irtProbability(it, t), 1e-12), 1 - 1e-12)
      return s + (x === 1 ? Math.log(p) : Math.log(1 - p))
    }, 0)
  if (method === 'mle') {
    const total = observed.reduce((a, [x]) => a + x, 0)
    if (total === 0) return { theta: Number.NEGATIVE_INFINITY, se: Number.POSITIVE_INFINITY }
    if (total === observed.length) return { theta: Number.POSITIVE_INFINITY, se: Number.POSITIVE_INFINITY }
  }
  if (method === 'eap') {
    const nodes = 81
    const ts = Array.from({ length: nodes }, (_, i) => -6 + (12 * i) / (nodes - 1))
    const lw = ts.map((t) => logLik(t) - 0.5 * t * t)
    const mx = maxOf(lw)
    const w = lw.map((v) => Math.exp(v - mx))
    const z = w.reduce((a, b) => a + b, 0)
    const mean = ts.reduce((a, t, i) => a + (t * (w[i] ?? 0)) / z, 0)
    const sd = Math.sqrt(ts.reduce((a, t, i) => a + ((t - mean) ** 2 * (w[i] ?? 0)) / z, 0))
    return { theta: mean, se: sd }
  }
  // Newton on the log-likelihood (plus the log prior for MAP) with numerical derivatives.
  const f = (t: number): number => logLik(t) - (method === 'map' ? 0.5 * t * t : 0)
  let t = 0
  for (let it = 0; it < 100; it++) {
    const h = 1e-4
    const g = (f(t + h) - f(t - h)) / (2 * h)
    const H = (f(t + h) - 2 * f(t) + f(t - h)) / (h * h)
    const step = H < -1e-8 ? -g / H : Math.sign(g) * 0.5
    t = Math.max(-8, Math.min(8, t + Math.max(-1, Math.min(1, step))))
    if (Math.abs(g) < 1e-9) break
  }
  const info =
    irtTestInformation(
      items.filter((_, j) => responses[j] !== null && responses[j] !== undefined),
      t,
    ) + (method === 'map' ? 1 : 0)
  return { theta: t, se: 1 / Math.sqrt(info) }
}

/**
 * Responses drawn from an IRT model with abilities from N(0, 1), with the abilities as the truth.
 *
 * @param key - The random key.
 * @param nRespondents - The number of respondents.
 * @param items - The true items.
 * @returns The responses and the abilities that produced them.
 * @example
 * ```ts
 * import { irtSimulate } from 'mdml-engine/pgm'
 * import { key } from 'mdml-engine/random'
 * irtSimulate(key(0), 4, [{ a: 1, b: 0, c: 0 }]).responses.length // => 4
 * ```
 */
export function irtSimulate(
  key: Key,
  nRespondents: number,
  items: readonly IrtItem[],
): { readonly responses: (0 | 1)[][]; readonly theta: readonly number[] } {
  const J = items.length
  const u = toFlat(uniform(key, [nRespondents * (J + 2)]))
  const theta = Array.from({ length: nRespondents }, (_, i) => {
    // Box-Muller from two uniforms.
    const a = Math.max(u[i * (J + 2)] ?? 0.5, 1e-12)
    const b = u[i * (J + 2) + 1] ?? 0.5
    return Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * b)
  })
  const responses = theta.map((t, i) =>
    items.map((it, j): 0 | 1 => ((u[i * (J + 2) + 2 + j] ?? 1) < irtProbability(it, t) ? 1 : 0)),
  )
  return { responses, theta }
}
