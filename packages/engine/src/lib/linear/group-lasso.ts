/**
 * Structured-sparsity linear models. `groupLasso` penalises the Euclidean norm of blocks of coefficients, so a whole
 * block of features enters or leaves the model together (Yuan and Lin); `multiTaskLasso` penalises the norm of each
 * feature's coefficients across several responses, so the tasks share one set of features (scikit-learn's
 * `MultiTaskLasso` and `MultiTaskElasticNet`). Both are traced `Algorithm`s whose state serialises (one block sweep or
 * one accelerated proximal step per iteration, with the KKT violation as the convergence measure) and estimators on
 * the iterative base, so a session shows groups appearing along the optimisation.
 */

import { MdmlError } from '../base/index.ts'
import {
  type IterativeEstimator,
  iterative,
  type LearnData,
  matrixView,
  sampleWeightsOf,
  withDefaultScore,
} from '../learn/index.ts'
import { ddot, dgemv, dsyev } from '../numerics/linalg/index.ts'
import { proxGroupLasso } from '../optim/index.ts'
import { dict, real } from '../space/index.ts'
import { fromStorage, primalOf, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, Log, stateFromJson, stateToJson } from '../trace/index.ts'
import { centre, type Design, designOf, interceptFor, linearPredictor, vectorTensor } from './design.ts'

// ---------------------------------------------------------------------------------------------------------------------
// Groups

/** How features are partitioned into groups, with each group's penalty weight. */
export interface GroupStructure {
  /** The distinct labels, sorted. */
  readonly labels: number[]
  /** The position (in `labels`) of each feature's group. */
  readonly positions: number[]
  /** The feature indices of each group. */
  readonly members: number[][]
  /** The penalty weight of each group. */
  readonly weights: number[]
}

/**
 * Partition `d` features into groups by label.
 *
 * @param d - The number of features.
 * @param groups - A label per feature (default each feature its own group).
 * @param groupWeights - A weight per group, in the order of the sorted labels (default sqrt of the group size when
 * `scaleByGroupSize`, else one).
 * @param scaleByGroupSize - Whether the default weights are sqrt(size).
 * @returns The structure.
 * @throws MdmlError `shape` for a wrong number of labels or weights, `value` for a negative weight.
 * @example
 * ```ts
 * import { groupStructure } from 'mdml-engine/linear'
 * const g = groupStructure(4, [7, 7, 3, 3])
 * g.labels // => [3, 7]
 * g.members // => [[2, 3], [0, 1]]
 * ```
 */
export function groupStructure(
  d: number,
  groups?: readonly number[],
  groupWeights?: readonly number[],
  scaleByGroupSize = true,
): GroupStructure {
  const raw = groups === undefined ? Array.from({ length: d }, (_, j) => j) : [...groups]
  if (raw.length !== d)
    throw new MdmlError('shape', `groupStructure: ${raw.length} group labels for ${d} features`, { d, got: raw.length })
  const labels = [...new Set(raw)].sort((a, b) => a - b)
  const position = new Map(labels.map((l, i) => [l, i]))
  const positions = raw.map((l) => position.get(l) ?? 0)
  const members: number[][] = labels.map(() => [])
  positions.forEach((g, j) => {
    members[g]?.push(j)
  })
  if (groupWeights !== undefined && groupWeights.length !== labels.length)
    throw new MdmlError('shape', `groupStructure: ${groupWeights.length} weights for ${labels.length} groups`)
  const weights = members.map((m, g) => {
    const w = groupWeights === undefined ? (scaleByGroupSize ? Math.sqrt(m.length) : 1) : (groupWeights[g] ?? 1)
    if (!(w >= 0)) throw new MdmlError('value', 'groupStructure: group weights must be non-negative')
    return w
  })
  return { labels, positions, members, weights }
}

// ---------------------------------------------------------------------------------------------------------------------
// Group lasso

/** Options of {@link groupLassoAlgorithm}. */
export interface GroupLassoOptions {
  /** A group label per feature (default each feature alone, which is the lasso). */
  readonly groups?: readonly number[]
  /** A weight per group, in the order of the sorted labels. */
  readonly groupWeights?: readonly number[]
  /** Whether the default weights are sqrt(group size) (default true). */
  readonly scaleByGroupSize?: boolean
  /** The penalty strength alpha (default 0.1). */
  readonly alpha?: number
  /** An extra ridge penalty ridge |beta|^2 / 2 on all coefficients (default 0). */
  readonly ridge?: number
  readonly fitIntercept?: boolean
  /** `bcd`: one proximal-gradient step per group per sweep; `fista`: accelerated proximal gradient on all groups (default `bcd`). */
  readonly solver?: 'bcd' | 'fista'
  /** Stop when the KKT violation falls below this (default 1e-9). */
  readonly tol?: number
}

/** The state of {@link groupLassoAlgorithm}. */
export type GroupLassoState = {
  readonly coef: number[]
  /** The extrapolated point of FISTA (the coefficients for `bcd`). */
  readonly z: number[]
  /** The FISTA momentum parameter. */
  readonly momentum: number
  /** The residual `yc - Xc coef` (centred, scaled by sqrt of the weights). */
  readonly residual: number[]
  /** The penalised objective at `coef`. */
  readonly objective: number
  /** The largest violation of the optimality conditions at `coef`. */
  readonly kkt: number
  readonly sweeps: number
}

const EPS = 2.220446049250313e-16

function maxEigenvalue(gram: ArrayLike<number>, n: number): number {
  if (n === 1) return gram[0] ?? 0
  const { w } = dsyev(gram, n)
  return w[n - 1] ?? 0
}

/** The Gram matrix X_S' X_S / scale of the columns `cols` of a row-major `[n, d]` matrix. */
function subGram(x: Float64Array, n: number, d: number, cols: readonly number[], scale: number): Float64Array {
  const m = cols.length
  const g = new Float64Array(m * m)
  for (let i = 0; i < n; i++)
    for (let a = 0; a < m; a++) {
      const va = (x[i * d + (cols[a] ?? 0)] ?? 0) / scale
      for (let b = a; b < m; b++) g[a * m + b] = (g[a * m + b] ?? 0) + va * (x[i * d + (cols[b] ?? 0)] ?? 0)
    }
  for (let a = 0; a < m; a++) for (let b = 0; b < a; b++) g[a * m + b] = g[b * m + a] ?? 0
  return g
}

/**
 * The group lasso as a traced algorithm: minimise `|y - X b|^2 / (2 W) + alpha sum_g w_g |b_g| + ridge |b|^2 / 2` over
 * coefficients (intercept by centring; `W` is the sum of the sample weights, so singleton groups of weight one are the
 * scikit-learn lasso). One step is a sweep over the groups (each a proximal-gradient step with the group's own
 * Lipschitz constant, exact for singletons) or one FISTA step with restarts. It logs the metrics `objective`, `kkt`
 * (largest violation of the optimality conditions, the stopping measure) and `active` (groups with a non-zero block)
 * and the tensor `coefficients`.
 *
 * @param design - Features, targets and weights.
 * @param options - Groups, penalty, solver and tolerance.
 * @returns The algorithm.
 * @reference M. Yuan and Y. Lin, Model selection and estimation in regression with grouped variables, JRSS B 68 (2006).
 * @example
 * ```ts
 * import { designOf, groupLassoAlgorithm } from 'mdml-engine/linear'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const design = designOf({ x: [[1, 0, 2], [0, 1, 1], [1, 1, 0], [2, 1, 1], [0, 2, 1]], y: [1, 2, 2, 4, 3] }, 'demo')
 * const run = runAlgorithm(groupLassoAlgorithm(design, { groups: [0, 0, 1], alpha: 5 }), { maxSteps: 50 })
 * run.state.coef // => [0, 0, 0]
 * ```
 */
export function groupLassoAlgorithm(design: Design, options: GroupLassoOptions = {}): Algorithm<GroupLassoState> {
  const { n, d } = design.x
  const alpha = options.alpha ?? 0.1
  const ridge = options.ridge ?? 0
  const solver = options.solver ?? 'bcd'
  const tol = options.tol ?? 1e-9
  if (!(alpha >= 0) || !(ridge >= 0)) throw new MdmlError('value', 'groupLasso: alpha and ridge must be >= 0')
  const structure = groupStructure(d, options.groups, options.groupWeights, options.scaleByGroupSize ?? true)
  const c = centre(design, options.fitIntercept ?? true)
  const x = c.x
  const scale = design.w.reduce((a, b) => a + b, 0)
  const nGroups = structure.members.length
  // Group Lipschitz constants for the block steps, the global one for FISTA.
  const lipschitz =
    solver === 'bcd'
      ? structure.members.map((m) => Math.max(maxEigenvalue(subGram(x, n, d, m, scale), m.length) + ridge, EPS))
      : []
  const lipschitzAll =
    solver === 'fista'
      ? Math.max(
          maxEigenvalue(
            subGram(
              x,
              n,
              d,
              Array.from({ length: d }, (_, j) => j),
              scale,
            ),
            d,
          ) + ridge,
          EPS,
        )
      : 1
  const residualOf = (coef: ArrayLike<number>): Float64Array => {
    const fit = dgemv(x, n, d, coef)
    return Float64Array.from(c.y, (v, i) => v - (fit[i] ?? 0))
  }
  const evaluate = (
    coef: ArrayLike<number>,
    residual: ArrayLike<number>,
  ): { objective: number; kkt: number; active: number } => {
    let rss = 0
    for (let i = 0; i < n; i++) rss += (residual[i] ?? 0) ** 2
    const grad = dgemv(x, n, d, residual, true)
    let penalty = 0
    let kkt = 0
    let active = 0
    let sq = 0
    for (let j = 0; j < d; j++) sq += (coef[j] ?? 0) ** 2
    for (let g = 0; g < nGroups; g++) {
      const m = structure.members[g] ?? []
      const w = alpha * (structure.weights[g] ?? 0)
      let bn = 0
      let gn = 0
      for (const j of m) {
        bn += (coef[j] ?? 0) ** 2
        gn += ((grad[j] ?? 0) / scale - ridge * (coef[j] ?? 0)) ** 2
      }
      bn = Math.sqrt(bn)
      penalty += w * bn
      if (bn > 0) {
        active++
        // Stationarity: -X_g'r / W + ridge b_g + w b_g / |b_g| = 0.
        let v = 0
        for (const j of m) v += ((grad[j] ?? 0) / scale - ridge * (coef[j] ?? 0) - (w * (coef[j] ?? 0)) / bn) ** 2
        kkt = Math.max(kkt, Math.sqrt(v))
      } else kkt = Math.max(kkt, Math.max(0, Math.sqrt(gn) - w))
    }
    return { objective: rss / (2 * scale) + penalty + (ridge * sq) / 2, kkt, active }
  }
  const zero = new Array<number>(d).fill(0)
  return {
    name: `group-lasso-${solver}`,
    config: { alpha, ridge, solver, tol, groups: structure.labels, positions: structure.positions },
    schema: channels({
      metrics: {
        objective: { description: 'the penalised objective', direction: 'lower' },
        kkt: { description: 'the largest violation of the optimality conditions', direction: 'lower' },
        active: { description: 'groups with a non-zero block' },
      },
      tensors: { coefficients: { role: 'param', description: 'the coefficients', shape: [d] } },
    }),
    init: () => {
      const residual = Array.from(c.y)
      const e = evaluate(zero, residual)
      return { coef: zero, z: zero, momentum: 1, residual, objective: e.objective, kkt: e.kkt, sweeps: 0 }
    },
    step: (s) => {
      let coef = s.coef
      let z = s.z
      let momentum = s.momentum
      let residual: ArrayLike<number> = s.residual
      if (solver === 'bcd') {
        const next = Float64Array.from(coef)
        const r = Float64Array.from(residual)
        for (let g = 0; g < nGroups; g++) {
          const m = structure.members[g] ?? []
          const lg = lipschitz[g] ?? 1
          // Gradient of the smooth part at the block, then the proximal step of the block norm.
          const u = m.map((j) => {
            let dot = 0
            for (let i = 0; i < n; i++) dot += (x[i * d + j] ?? 0) * (r[i] ?? 0)
            return (next[j] ?? 0) - (-dot / scale + ridge * (next[j] ?? 0)) / lg
          })
          let un = 0
          for (const v of u) un += v * v
          un = Math.sqrt(un)
          const t = (alpha * (structure.weights[g] ?? 0)) / lg
          const factor = un > t ? 1 - t / un : 0
          m.forEach((j, k) => {
            const nb = factor * (u[k] ?? 0)
            const delta = nb - (next[j] ?? 0)
            if (delta !== 0) for (let i = 0; i < n; i++) r[i] = (r[i] ?? 0) - (x[i * d + j] ?? 0) * delta
            next[j] = nb
          })
        }
        coef = Array.from(next)
        z = coef
        residual = r
      } else {
        const rz = residualOf(z)
        const grad = dgemv(x, n, d, rz, true)
        const u = Float64Array.from(z, (v, j) => v - (-(grad[j] ?? 0) / scale + ridge * v) / lipschitzAll)
        const next = toFlat(primalOf(proxGroupLasso(u, alpha / lipschitzAll, structure.positions, structure.weights)))
        const nextMomentum = (1 + Math.sqrt(1 + 4 * momentum * momentum)) / 2
        // Restart the momentum when it points against the descent direction.
        let against = 0
        for (let j = 0; j < d; j++) against += ((z[j] ?? 0) - (next[j] ?? 0)) * ((next[j] ?? 0) - (coef[j] ?? 0))
        if (against > 0) {
          momentum = 1
          z = Array.from(next)
        } else {
          const w = (momentum - 1) / nextMomentum
          z = Array.from(next, (v, j) => v + w * (v - (coef[j] ?? 0)))
          momentum = nextMomentum
        }
        coef = Array.from(next)
        residual = residualOf(coef)
      }
      const e = evaluate(coef, residual)
      return {
        coef,
        z,
        momentum,
        residual: Array.from(residual),
        objective: e.objective,
        kkt: e.kkt,
        sweeps: s.sweeps + 1,
      }
    },
    done: (s) => (s.kkt <= tol ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { objective: s.objective, kkt: s.kkt, active: evaluate(s.coef, s.residual).active },
      tensors: { coefficients: { role: 'param', value: s.coef, shape: [d] } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as GroupLassoState,
  }
}

/** The hyperparameters of {@link groupLasso}. */
export interface GroupLassoHyper {
  readonly groups?: readonly number[]
  readonly groupWeights?: readonly number[]
  readonly scaleByGroupSize: boolean
  readonly alpha: number
  readonly ridge: number
  readonly fitIntercept: boolean
  readonly solver: 'bcd' | 'fista'
  readonly tol: number
  readonly maxIter: number
}

/** The fitted parameters of {@link groupLasso}. */
export type GroupLassoParams = {
  readonly coef: number[]
  readonly intercept: number
  /** The group labels (sorted). */
  readonly groupLabels: number[]
  /** The Euclidean norm of each group's block. */
  readonly groupNorms: number[]
  /** Labels of the groups with a non-zero block. */
  readonly activeGroups: number[]
  readonly objective: number
  readonly kkt: number
  readonly iterations: number
  readonly converged: boolean
}

/** What a fitted group lasso offers. */
export interface GroupLassoVerbs {
  readonly nFeaturesIn: number
  readonly predict: (x: TensorLike) => Tensor
  readonly featureImportances: number[]
  readonly score: (data: LearnData) => number
}

const GROUP_LASSO_DEFAULTS: GroupLassoHyper = {
  scaleByGroupSize: true,
  alpha: 0.1,
  ridge: 0,
  fitIntercept: true,
  solver: 'bcd',
  tol: 1e-9,
  maxIter: 5000,
}

/**
 * The group lasso estimator: a linear model whose coefficient blocks (groups of features) are all zero or all
 * non-zero. A session adds one sweep per step, with `solver/objective`, `solver/kkt`, `solver/active` and
 * `solver/coefficients` logged, so `fittedAt(t)` is the model after `t` sweeps; `params.activeGroups` are the selected
 * groups.
 *
 * @param hyper - `groups` (a label per feature), `groupWeights`, `alpha`, `ridge`, `fitIntercept`, `solver`, `tol`, `maxIter`.
 * @returns The estimator.
 * @example
 * ```ts
 * import { groupLasso } from 'mdml-engine/linear'
 * const x = [[1, 0.9, 5], [2, 2.1, 1], [3, 2.8, 4], [4, 4.2, 2], [5, 4.9, 6], [6, 6.1, 3]]
 * const f = groupLasso({ groups: [0, 0, 1], alpha: 0.3 }).fit({ x, y: [1.1, 2.0, 3.2, 3.9, 5.1, 6.0] })
 * f.params.activeGroups // => [0]
 * ```
 */
export function groupLasso(
  hyper: Partial<GroupLassoHyper> = {},
): IterativeEstimator<GroupLassoHyper, GroupLassoState, GroupLassoParams, GroupLassoVerbs> {
  const optionsOf = (h: GroupLassoHyper): GroupLassoOptions => ({
    ...(h.groups === undefined ? {} : { groups: h.groups }),
    ...(h.groupWeights === undefined ? {} : { groupWeights: h.groupWeights }),
    scaleByGroupSize: h.scaleByGroupSize,
    alpha: h.alpha,
    ridge: h.ridge,
    fitIntercept: h.fitIntercept,
    solver: h.solver,
    tol: h.tol,
  })
  return iterative<GroupLassoHyper, GroupLassoState, GroupLassoParams, GroupLassoVerbs>(
    {
      name: 'group-lasso',
      kind: 'regressor',
      defaults: GROUP_LASSO_DEFAULTS,
      hyperSpace: (h) => dict({ alpha: real(1e-4, 10, { log: true, default: h.alpha }) }),
      algorithm: (data, h) => groupLassoAlgorithm(designOf(data, 'groupLasso'), optionsOf(h)),
      maxSteps: (h) => h.maxIter,
      params: (s, data, h) => {
        const design = designOf(data, 'groupLasso')
        const structure = groupStructure(design.x.d, h.groups, h.groupWeights, h.scaleByGroupSize)
        const norms = structure.members.map((m) => Math.sqrt(m.reduce((a, j) => a + (s.coef[j] ?? 0) ** 2, 0)))
        return {
          coef: [...s.coef],
          intercept: h.fitIntercept ? interceptFor(centre(design, true), s.coef) : 0,
          groupLabels: structure.labels,
          groupNorms: norms,
          activeGroups: structure.labels.filter((_, g) => (norms[g] ?? 0) > 0),
          objective: s.objective,
          kkt: s.kkt,
          iterations: s.sweeps,
          converged: s.kkt <= h.tol,
        }
      },
      verbs: (p) => {
        const predict = (x: TensorLike): Tensor => vectorTensor(linearPredictor(x, p.coef, p.intercept, 'groupLasso'))
        const base = { nFeaturesIn: p.coef.length, predict, featureImportances: p.coef.map(Math.abs) }
        return { ...base, score: withDefaultScore('regressor', base).score ?? ((): number => Number.NaN) }
      },
      logs: {
        'solver/objective': Log(),
        'solver/kkt': Log(),
        'solver/active': Log(),
        'solver/coefficients': Log({ detail: 'full' }),
      },
    },
    { ...GROUP_LASSO_DEFAULTS, ...hyper },
  )
}

/**
 * The smallest penalty at which every coefficient of the group lasso is zero: `max_g |X_g' y_c| / (W w_g)` on the
 * centred data.
 *
 * @param data - Features, targets and weights.
 * @param options - Groups, weights and the intercept.
 * @returns alpha_max.
 * @example
 * ```ts
 * import { groupLassoAlphaMax } from 'mdml-engine/linear'
 * groupLassoAlphaMax({ x: [[1], [2], [3]], y: [1, 2, 3] }) // => 0.6666666666666666
 * ```
 */
export function groupLassoAlphaMax(
  data: LearnData,
  options: Pick<GroupLassoOptions, 'groups' | 'groupWeights' | 'scaleByGroupSize' | 'fitIntercept'> = {},
): number {
  const design = designOf(data, 'groupLassoAlphaMax')
  const { n, d } = design.x
  const c = centre(design, options.fitIntercept ?? true)
  const structure = groupStructure(d, options.groups, options.groupWeights, options.scaleByGroupSize ?? true)
  const scale = design.w.reduce((a, b) => a + b, 0)
  const grad = dgemv(c.x, n, d, c.y, true)
  return structure.members.reduce((best, m, g) => {
    const w = structure.weights[g] ?? 0
    if (w <= 0) return best
    const norm = Math.sqrt(m.reduce((a, j) => a + (grad[j] ?? 0) ** 2, 0))
    return Math.max(best, norm / (scale * w))
  }, 0)
}

// ---------------------------------------------------------------------------------------------------------------------
// Multi-task lasso

/** Options of {@link multiTaskLassoAlgorithm}. */
export interface MultiTaskOptions {
  /** The penalty strength alpha (default 0.1). */
  readonly alpha?: number
  /** The share of the row-norm penalty (1 is the multi-task lasso, less adds a ridge: the multi-task elastic net; default 1). */
  readonly l1Ratio?: number
  readonly fitIntercept?: boolean
  /** Stop when the KKT violation falls below this (default 1e-9). */
  readonly tol?: number
}

/** The state of {@link multiTaskLassoAlgorithm}. */
export type MultiTaskState = {
  /** The coefficients `[d, k]`, row-major. */
  readonly coef: number[]
  /** The residual `[n, k]` of the centred, scaled responses. */
  readonly residual: number[]
  readonly objective: number
  readonly kkt: number
  readonly sweeps: number
}

/** Features, tasks and weights of a multi-task problem, centred. */
export interface MultiTaskDesign {
  readonly design: Design
  /** The responses `[n, k]`, row-major. */
  readonly y: Float64Array
  readonly k: number
}

/**
 * Read a multi-task problem: features `[n, d]`, responses `[n, k]` and weights.
 *
 * @param data - The data, with a matrix `y`.
 * @param who - The caller, for errors.
 * @returns The design (its `y` is the first task) and all the responses.
 * @throws MdmlError `shape` unless `y` is a matrix with one row per sample.
 * @example
 * ```ts
 * import { multiTaskDesign } from 'mdml-engine/linear'
 * multiTaskDesign({ x: [[1], [2]], y: [[1, 2], [3, 4]] }, 'demo').k // => 2
 * ```
 */
export function multiTaskDesign(data: LearnData, who: string): MultiTaskDesign {
  const x = matrixView(data.x, who)
  if (data.y === undefined) throw new MdmlError('value', `${who}: needs the responses y`)
  const y = matrixView(data.y, who, { minSamples: 1 })
  if (y.n !== x.n) throw new MdmlError('shape', `${who}: x has ${x.n} rows but y has ${y.n}`)
  const first = Float64Array.from({ length: y.n }, (_, i) => y.data[i * y.d] ?? 0)
  return { design: { x, y: first, w: sampleWeightsOf(data.weights, x.n, who) }, y: y.data, k: y.d }
}

/**
 * The multi-task lasso as a traced algorithm: minimise `|Y - X B|_F^2 / (2 W) + alpha l1Ratio sum_j |B_j|_2 +
 * alpha (1 - l1Ratio) |B|_F^2 / 2` over the coefficient matrix `B` `[d, k]`, whose row `B_j` holds feature `j`'s
 * coefficients on all tasks (scikit-learn's `MultiTaskLasso` and `MultiTaskElasticNet`). One step is a cyclic
 * coordinate sweep over the features, each row updated exactly by block soft-thresholding. It logs `objective`, `kkt`
 * and `active` (features with a non-zero row) and the tensor `coefficients` `[d, k]`.
 *
 * @param problem - Features, responses and weights.
 * @param options - Penalty, mixing, intercept and tolerance.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { multiTaskDesign, multiTaskLassoAlgorithm } from 'mdml-engine/linear'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const problem = multiTaskDesign({ x: [[1, 0], [0, 1], [1, 1], [2, 1]], y: [[1, 2], [0, 1], [1, 3], [2, 5]] }, 'demo')
 * runAlgorithm(multiTaskLassoAlgorithm(problem, { alpha: 100 }), { maxSteps: 20 }).state.coef // => [0, 0, 0, 0]
 * ```
 */
export function multiTaskLassoAlgorithm(
  problem: MultiTaskDesign,
  options: MultiTaskOptions = {},
): Algorithm<MultiTaskState> {
  const { design, k } = problem
  const { n, d } = design.x
  const alpha = options.alpha ?? 0.1
  const l1Ratio = options.l1Ratio ?? 1
  const tol = options.tol ?? 1e-9
  if (!(alpha >= 0) || !(l1Ratio >= 0 && l1Ratio <= 1))
    throw new MdmlError('value', 'multiTaskLasso: alpha >= 0 and l1Ratio in [0, 1]')
  const l1 = alpha * l1Ratio
  const l2 = alpha * (1 - l1Ratio)
  const c = centre(design, options.fitIntercept ?? true)
  const x = c.x
  const scale = design.w.reduce((a, b) => a + b, 0)
  // Centre each task like the first: weighted means, then rows scaled by sqrt(w).
  const y = new Float64Array(n * k)
  for (let t = 0; t < k; t++) {
    const col = Float64Array.from({ length: n }, (_, i) => problem.y[i * k + t] ?? 0)
    const yc = centre(design, options.fitIntercept ?? true, col).y
    for (let i = 0; i < n; i++) y[i * k + t] = yc[i] ?? 0
  }
  const colNorm = Float64Array.from({ length: d }, (_, j) => {
    let s = 0
    for (let i = 0; i < n; i++) s += (x[i * d + j] ?? 0) ** 2
    return s / scale
  })
  const evaluate = (
    coef: ArrayLike<number>,
    r: ArrayLike<number>,
  ): { objective: number; kkt: number; active: number } => {
    let rss = 0
    for (let i = 0; i < n * k; i++) rss += (r[i] ?? 0) ** 2
    let penalty = 0
    let ridge = 0
    let kkt = 0
    let active = 0
    for (let j = 0; j < d; j++) {
      let bn = 0
      const g = new Float64Array(k)
      for (let t = 0; t < k; t++) {
        bn += (coef[j * k + t] ?? 0) ** 2
        let dot = 0
        for (let i = 0; i < n; i++) dot += (x[i * d + j] ?? 0) * (r[i * k + t] ?? 0)
        g[t] = dot / scale - l2 * (coef[j * k + t] ?? 0)
      }
      ridge += bn
      bn = Math.sqrt(bn)
      penalty += l1 * bn
      let v = 0
      if (bn > 0) {
        active++
        for (let t = 0; t < k; t++) v += ((g[t] ?? 0) - (l1 * (coef[j * k + t] ?? 0)) / bn) ** 2
        kkt = Math.max(kkt, Math.sqrt(v))
      } else {
        for (let t = 0; t < k; t++) v += (g[t] ?? 0) ** 2
        kkt = Math.max(kkt, Math.max(0, Math.sqrt(v) - l1))
      }
    }
    return { objective: rss / (2 * scale) + penalty + (l2 * ridge) / 2, kkt, active }
  }
  const zero = new Array<number>(d * k).fill(0)
  return {
    name: 'multi-task-lasso',
    config: { alpha, l1Ratio, tol, tasks: k },
    schema: channels({
      metrics: {
        objective: { description: 'the penalised objective', direction: 'lower' },
        kkt: { description: 'the largest violation of the optimality conditions', direction: 'lower' },
        active: { description: 'features with a non-zero row' },
      },
      tensors: { coefficients: { role: 'param', description: 'the coefficients, one row per feature', shape: [d, k] } },
    }),
    init: () => {
      const residual = Array.from(y)
      const e = evaluate(zero, residual)
      return { coef: zero, residual, objective: e.objective, kkt: e.kkt, sweeps: 0 }
    },
    step: (s) => {
      const coef = Float64Array.from(s.coef)
      const r = Float64Array.from(s.residual)
      for (let j = 0; j < d; j++) {
        const cj = colNorm[j] ?? 0
        if (cj <= 0) continue
        // z = X_j' (r + x_j B_j) / W = X_j' r / W + c_j B_j
        const z = new Float64Array(k)
        for (let t = 0; t < k; t++) {
          let dot = 0
          for (let i = 0; i < n; i++) dot += (x[i * d + j] ?? 0) * (r[i * k + t] ?? 0)
          z[t] = dot / scale + cj * (coef[j * k + t] ?? 0)
        }
        let zn = 0
        for (const v of z) zn += v * v
        zn = Math.sqrt(zn)
        const factor = zn > l1 ? (1 - l1 / zn) / (cj + l2) : 0
        for (let t = 0; t < k; t++) {
          const nb = factor * (z[t] ?? 0)
          const delta = nb - (coef[j * k + t] ?? 0)
          if (delta !== 0) for (let i = 0; i < n; i++) r[i * k + t] = (r[i * k + t] ?? 0) - (x[i * d + j] ?? 0) * delta
          coef[j * k + t] = nb
        }
      }
      const e = evaluate(coef, r)
      return {
        coef: Array.from(coef),
        residual: Array.from(r),
        objective: e.objective,
        kkt: e.kkt,
        sweeps: s.sweeps + 1,
      }
    },
    done: (s) => (s.kkt <= tol ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { objective: s.objective, kkt: s.kkt, active: evaluate(s.coef, s.residual).active },
      tensors: { coefficients: { role: 'param', value: s.coef, shape: [d, k] } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as MultiTaskState,
  }
}

/** The hyperparameters of {@link multiTaskLasso}. */
export interface MultiTaskLassoHyper {
  readonly alpha: number
  readonly l1Ratio: number
  readonly fitIntercept: boolean
  readonly tol: number
  readonly maxIter: number
}

/** The fitted parameters of {@link multiTaskLasso}. */
export type MultiTaskLassoParams = {
  /** The coefficients `[k][d]` (scikit-learn's `coef_` layout: a row per task). */
  readonly coef: number[][]
  readonly intercept: number[]
  /** The indices of the features with a non-zero row (shared by all tasks). */
  readonly support: number[]
  readonly objective: number
  readonly kkt: number
  readonly iterations: number
  readonly converged: boolean
}

/** What a fitted multi-task lasso offers. */
export interface MultiTaskLassoVerbs {
  readonly nFeaturesIn: number
  /** Predictions `[m, k]`. */
  readonly predict: (x: TensorLike) => Tensor
  readonly featureImportances: number[]
  /** The R^2 averaged over the tasks. */
  readonly score: (data: LearnData) => number
}

const MULTI_TASK_DEFAULTS: MultiTaskLassoHyper = {
  alpha: 0.1,
  l1Ratio: 1,
  fitIntercept: true,
  tol: 1e-9,
  maxIter: 5000,
}

/**
 * The multi-task lasso estimator: several responses (`y` of shape `[n, k]`) fitted at once with a shared sparsity
 * pattern, a feature being in the model for all tasks or none. `l1Ratio` below one gives the multi-task elastic net.
 *
 * @param hyper - `alpha`, `l1Ratio`, `fitIntercept`, `tol`, `maxIter`.
 * @returns The estimator.
 * @example
 * ```ts
 * import { multiTaskLasso } from 'mdml-engine/linear'
 * const x = [[1, 0.2], [2, -0.1], [3, 0.3], [4, 0], [5, 0.1]]
 * const y = [[1, 2], [2, 4.1], [3, 5.9], [4, 8], [5, 10.1]]
 * multiTaskLasso({ alpha: 0.05 }).fit({ x, y }).params.support // => [0]
 * ```
 */
export function multiTaskLasso(
  hyper: Partial<MultiTaskLassoHyper> = {},
): IterativeEstimator<MultiTaskLassoHyper, MultiTaskState, MultiTaskLassoParams, MultiTaskLassoVerbs> {
  return iterative<MultiTaskLassoHyper, MultiTaskState, MultiTaskLassoParams, MultiTaskLassoVerbs>(
    {
      name: 'multi-task-lasso',
      kind: 'regressor',
      defaults: MULTI_TASK_DEFAULTS,
      hyperSpace: (h) => dict({ alpha: real(1e-4, 10, { log: true, default: h.alpha }) }),
      algorithm: (data, h) =>
        multiTaskLassoAlgorithm(multiTaskDesign(data, 'multiTaskLasso'), {
          alpha: h.alpha,
          l1Ratio: h.l1Ratio,
          fitIntercept: h.fitIntercept,
          tol: h.tol,
        }),
      maxSteps: (h) => h.maxIter,
      params: (s, data, h) => {
        const { design, y, k } = multiTaskDesign(data, 'multiTaskLasso')
        const { d, n } = design.x
        const c = centre(design, h.fitIntercept)
        const coef = Array.from({ length: k }, (_, t) => Array.from({ length: d }, (__, j) => s.coef[j * k + t] ?? 0))
        const intercept = coef.map((row, t) => {
          if (!h.fitIntercept) return 0
          const col = Float64Array.from({ length: n }, (_, i) => y[i * k + t] ?? 0)
          return centre(design, true, col).yMean - ddot(c.xMean, row)
        })
        return {
          coef,
          intercept,
          support: Array.from({ length: d }, (_, j) => j).filter((j) => coef.some((row) => (row[j] ?? 0) !== 0)),
          objective: s.objective,
          kkt: s.kkt,
          iterations: s.sweeps,
          converged: s.kkt <= h.tol,
        }
      },
      verbs: (p) => {
        const k = p.coef.length
        const predict = (x: TensorLike): Tensor => {
          const cols = p.coef.map((row, t) => linearPredictor(x, row, p.intercept[t] ?? 0, 'multiTaskLasso'))
          const m = cols[0]?.length ?? 0
          const out = new Float64Array(m * k)
          cols.forEach((col, t) => {
            col.forEach((v, i) => {
              out[i * k + t] = v
            })
          })
          return fromStorage(out, [m, k], 'float64')
        }
        const d = p.coef[0]?.length ?? 0
        return {
          nFeaturesIn: d,
          predict,
          featureImportances: Array.from({ length: d }, (_, j) =>
            Math.sqrt(p.coef.reduce((a, row) => a + (row[j] ?? 0) ** 2, 0)),
          ),
          score: (data) => {
            const truth = matrixView(data.y ?? [], 'multiTaskLasso')
            const pred = toFlat(predict(data.x))
            let total = 0
            for (let t = 0; t < k; t++) {
              let mean = 0
              for (let i = 0; i < truth.n; i++) mean += (truth.data[i * k + t] ?? 0) / truth.n
              let res = 0
              let tot = 0
              for (let i = 0; i < truth.n; i++) {
                res += ((truth.data[i * k + t] ?? 0) - (pred[i * k + t] ?? 0)) ** 2
                tot += ((truth.data[i * k + t] ?? 0) - mean) ** 2
              }
              total += tot === 0 ? (res === 0 ? 1 : 0) : 1 - res / tot
            }
            return total / k
          },
        }
      },
      logs: {
        'solver/objective': Log(),
        'solver/kkt': Log(),
        'solver/active': Log(),
        'solver/coefficients': Log({ detail: 'full' }),
      },
    },
    { ...MULTI_TASK_DEFAULTS, ...hyper },
  )
}
