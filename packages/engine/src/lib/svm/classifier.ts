/**
 * Support vector classification (scikit-learn's `SVC` and `NuSVC`, libsvm's C-SVC and nu-SVC) as estimators on the
 * iterative base: the maximum-margin classifier in a kernel's feature space, found by the library's sequential minimal
 * optimisation with one working-pair update per step, one-versus-one for more than two classes (the pairs solved in
 * turn) and Platt-scaled probabilities with pairwise coupling as a last step. The fitted model is the kernel expansion
 * over the support vectors, so the model at any step of the dual is available. The solver is a slot (`solver`, see
 * `svcSolvers`): SMO by default; with `loss: 'squaredHinge'` the primal is smooth and any solver of `smoothSolvers`
 * minimises it.
 */

import { MdmlError } from '../base/index.ts'
import { marginRecord } from '../exposition/index.ts'
import {
  type Kernel,
  type KernelChoice,
  kernelMatrix,
  kernelSpace,
  type PairwiseKernelMetric,
  type ResolvedKernel,
  resolveKernel,
} from '../kernels/index.ts'
import {
  encodeLabels,
  type IterativeEstimator,
  iterative,
  type LearnData,
  matrixView,
  plattScaling,
  sampleWeightsOf,
} from '../learn/index.ts'
import { type SmoState, nuSvcDual as solveNuSvc, solverConfigOf, svcDual as solveSvc } from '../optim/index.ts'
import { foldIn, type Key, key as makeKey, permutation } from '../random/index.ts'
import { choice, dict, real, type SpaceValue } from '../space/index.ts'
import { asTensor, fromStorage, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  channels,
  defineContract,
  derived,
  Log,
  scalar,
  stateFromJson,
  stateToJson,
} from '../trace/index.ts'
import { cSvcDual, type Dual, nuSvcDual, submatrix } from './_dual.ts'
import {
  checkSolver,
  type SvcLoss,
  solveSquaredSvc,
  squaredDualFor,
  squaredHingePrimal,
  svcSolvers,
} from './_squared.ts'
import { pairwiseCoupling } from './probability.ts'

export { type SvcLoss, svcSolvers } from './_squared.ts'

/** The hyperparameters of {@link svc}. */
export interface SvcHyper {
  /** `c` (the penalty C) or `nu` (a bound on margin errors and support vectors). */
  readonly form: 'c' | 'nu'
  /** The penalty of the C form (scikit-learn's C). */
  readonly C: number
  /** The nu of the nu form, in (0, 1]. */
  readonly nu: number
  readonly kernel: PairwiseKernelMetric | Kernel
  readonly gamma: number | 'scale' | 'auto'
  readonly degree: number
  readonly coef0: number
  /** The solver's tolerance on the KKT violation (libsvm's eps, default 1e-3). */
  readonly tol: number
  readonly selection: 'second-order' | 'first-order'
  /** The most steps (SMO pair updates over all the pairs of classes) a fit may take. */
  readonly maxIter: number
  /** Fit Platt probabilities by internal cross-validation as a last step (the C form only needs it for `predictProba`). */
  readonly probability: boolean
  /** The folds of the cross-validation behind the probabilities. */
  readonly folds: number
  /** Per-class multipliers of C (by label), or `balanced` (n / (k n_c)). */
  readonly classWeight: 'balanced' | Readonly<Record<number, number>> | null
  /** The shape of `decisionFunction` for more than two classes: scikit-learn's `ovr` transform of the votes, or the raw `ovo` pairs. */
  readonly decisionShape: 'ovr' | 'ovo'
  /** `hinge` (the C-SVC) or `squaredHinge` (the L2-loss SVM, whose primal is smooth; the C form only). */
  readonly loss: SvcLoss
  /**
   * How a pair is fitted: a value of `svcSolvers.space`. `smo` (default) is libsvm's dual solver and reaches either
   * loss; with `loss: 'squaredHinge'` every solver of `smoothSolvers` minimises the primal over the kernel expansion.
   */
  readonly solver: SpaceValue
}

/**
 * The paths the support vector machines (`svc`, `svr`, `oneClassSvm`, `svdd`) log beside the base's: the change of the
 * working pair and the model so far (the SMO's own channels are the base's `solver/...`).
 *
 * @example
 * ```ts
 * import type { SvmPath } from 'mdml-engine/svm'
 * const path: SvmPath = 'smo/alphaChanged'
 * path // => 'smo/alphaChanged'
 * ```
 */
export type SvmPath = 'smo/alphaChanged' | 'model/support-vectors'

/** One machine of the one-versus-one decomposition. */
export type SvcPair = {
  /** The class index (into `classes`) that decision values above zero vote for. */
  readonly positive: number
  readonly negative: number
  /** The rows of the pair's problem: the training rows of `negative`/`positive` in libsvm's order. */
  readonly rows: number[]
  /** Positions of the pair's support vectors among `vectors`. */
  readonly support: number[]
  /** The expansion coefficients (y_i alpha_i), one per support vector. */
  readonly coef: number[]
  readonly intercept: number
  /** Platt's a and b: P(positive) = 1 / (1 + exp(a f + b)), when probabilities were fitted. */
  readonly platt: { readonly a: number; readonly b: number } | null
}

/** The fitted parameters of {@link svc}: plain JSON. */
export type SvcParams = {
  readonly classes: number[]
  readonly kernel: ResolvedKernel
  /** The support vectors (rows of the training data), ordered by class and then by row (scikit-learn's order). */
  readonly vectors: number[][]
  /** The training row of each support vector (`support_`). */
  readonly supportIndex: number[]
  /** The number of support vectors of each class (`n_support_`). */
  readonly nSupport: number[]
  readonly pairs: SvcPair[]
}

/** What a fitted SVC offers. */
export interface SvcVerbs {
  readonly predict: (x: TensorLike) => Tensor
  /** `[m]` for two classes (positive for the second class); `[m, k]` votes plus confidences (`ovr`) or `[m, k (k - 1) / 2]` pairs (`ovo`). */
  readonly decisionFunction: (x: TensorLike) => Tensor
  /** Present when the probabilities were fitted. */
  readonly predictProba?: (x: TensorLike) => Tensor
  /** The mean accuracy on a labelled dataset. */
  readonly score: (data: LearnData) => number
  readonly classes: readonly number[]
  readonly nFeaturesIn: number
}

/** A solved pair, kept in the state. */
export type PairSolution = { readonly coef: number[]; readonly rho: number; readonly iterations: number }

/** The state of {@link svc}: the pair being solved, the pairs solved, and the probabilities once fitted. */
export interface SvcState {
  readonly phase: 'fit' | 'calibrate' | 'done'
  /** The pair being solved (the last one once all are). */
  readonly current: number
  readonly smo: SmoState
  /** The expansion of `smo`: coefficients over the pair's rows, and rho. */
  readonly live: { readonly coef: number[]; readonly rho: number }
  /** The margins `y_i f(x_i)` of the pair's rows at `smo`. */
  readonly margins: number[]
  readonly solved: PairSolution[]
  readonly platt: ({ readonly a: number; readonly b: number } | null)[]
}

/** The layout of a fit: classes, the pairs of the decomposition with their rows, the resolved kernel. */
interface Layout {
  readonly n: number
  readonly d: number
  readonly x: Float64Array
  readonly classes: readonly number[]
  readonly codes: Int32Array
  readonly pairs: { readonly positive: number; readonly negative: number; readonly rows: number[] }[]
  readonly kernel: ResolvedKernel
}

const WHO = 'svc'

function layoutOf(data: LearnData, h: SvcHyper): Layout {
  const m = matrixView(data.x, WHO)
  const { classes, codes } = encodeLabels(data.y ?? [], WHO, m.n)
  if (classes.length < 2) throw new MdmlError('value', `${WHO}: needs at least two classes`)
  const members = classes.map((_, c) => Array.from(codes).flatMap((v, i) => (v === c ? [i] : [])))
  const pairs: Layout['pairs'] = []
  for (let a = 0; a < classes.length; a++) {
    for (let b = a + 1; b < classes.length; b++) {
      const rows = [...(members[a] ?? []), ...(members[b] ?? [])]
      // Two classes: scikit-learn's convention, decision values above zero favour the second class.
      pairs.push(classes.length === 2 ? { positive: 1, negative: 0, rows } : { positive: a, negative: b, rows })
    }
  }
  return { n: m.n, d: m.d, x: m.data, classes, codes, pairs, kernel: resolveKernel(h as KernelChoice, data.x) }
}

function penalties(data: LearnData, h: SvcHyper, layout: Layout): Float64Array {
  const w = sampleWeightsOf(data.weights, layout.n, WHO)
  const counts = layout.classes.map((_, c) => layout.codes.reduce((s, v) => s + (v === c ? 1 : 0), 0))
  return Float64Array.from({ length: layout.n }, (_, i) => {
    const c = layout.codes[i] ?? 0
    let weight = 1
    if (h.classWeight === 'balanced') weight = layout.n / (layout.classes.length * (counts[c] ?? 1))
    else if (h.classWeight !== null) weight = h.classWeight[layout.classes[c] ?? 0] ?? 1
    return h.C * weight * (w[i] ?? 1)
  })
}

/** The pairs' decision values on held-out rows by a sub-fit on the rest (libsvm's cross-validation for Platt). */
function crossValidated(
  Kp: Float64Array,
  y: Float64Array,
  C: Float64Array,
  h: SvcHyper,
  order: readonly number[],
): Float64Array {
  const m = y.length
  const dec = new Float64Array(m)
  const k = Math.max(2, Math.min(h.folds, m))
  for (let f = 0; f < k; f++) {
    const held = order.filter((_, pos) => pos % k === f)
    const train = order.filter((_, pos) => pos % k !== f)
    const ys = Float64Array.from(train, (i) => y[i] ?? 1)
    const positives = ys.reduce((s, v) => s + (v > 0 ? 1 : 0), 0)
    if (positives === 0 || positives === ys.length) {
      for (const i of held) dec[i] = positives === 0 ? -1 : 1
      continue
    }
    const raw = submatrix(Kp, m, train)
    const sub = fromStorage(raw, [train.length, train.length], 'float64')
    const options = { tol: h.tol, selection: h.selection }
    const r =
      h.loss === 'squaredHinge'
        ? solveSquaredSvc(
            raw,
            ys,
            Float64Array.from(train, (i) => C[i] ?? 0),
            options,
          )
        : h.form === 'nu'
          ? solveNuSvc(sub, ys, Math.min(h.nu, (2 * Math.min(positives, ys.length - positives)) / ys.length), options)
          : solveSvc(
              sub,
              ys,
              Float64Array.from(train, (i) => C[i] ?? 0),
              options,
            )
    const coef = toFlat(r.coef)
    for (const i of held) {
      let s = r.bias
      for (let t = 0; t < train.length; t++) s += (coef[t] ?? 0) * (Kp[i * m + (train[t] ?? 0)] ?? 0)
      dec[i] = s
    }
  }
  return dec
}

/** Whether any pair carries a Platt map. */
const probabilistic = (p: SvcParams): boolean => p.pairs.length > 0 && p.pairs.every((pair) => pair.platt !== null)

function paramsOf(state: SvcState, data: LearnData, h: SvcHyper): SvcParams {
  const layout = layoutOf(data, h)
  const solutions = layout.pairs.map((pair, k) => {
    const s = state.solved[k]
    if (s !== undefined) return s.coef
    if (k === state.current) return state.live.coef
    return new Array<number>(pair.rows.length).fill(0)
  })
  const rhos = layout.pairs.map((_, k) => state.solved[k]?.rho ?? (k === state.current ? state.live.rho : 0))
  // A row is a support vector if any pair gives it a coefficient; vectors are ordered by class, then by row.
  const isSupport = new Uint8Array(layout.n)
  layout.pairs.forEach((pair, k) => {
    pair.rows.forEach((row, t) => {
      if ((solutions[k]?.[t] ?? 0) !== 0) isSupport[row] = 1
    })
  })
  const supportIndex: number[] = []
  const nSupport = layout.classes.map(() => 0)
  layout.classes.forEach((_, c) => {
    for (let i = 0; i < layout.n; i++) {
      if (isSupport[i] === 1 && layout.codes[i] === c) {
        supportIndex.push(i)
        nSupport[c] = (nSupport[c] ?? 0) + 1
      }
    }
  })
  const position = new Map(supportIndex.map((row, t) => [row, t]))
  const pairs = layout.pairs.map((pair, k): SvcPair => {
    const support: number[] = []
    const coef: number[] = []
    pair.rows.forEach((row, t) => {
      const c = solutions[k]?.[t] ?? 0
      if (c !== 0) {
        support.push(position.get(row) ?? 0)
        coef.push(c)
      }
    })
    return { ...pair, support, coef, intercept: -(rhos[k] ?? 0), platt: state.platt[k] ?? null }
  })
  return {
    classes: [...layout.classes],
    kernel: layout.kernel,
    vectors: supportIndex.map((row) => Array.from(layout.x.subarray(row * layout.d, (row + 1) * layout.d))),
    supportIndex,
    nSupport,
    pairs,
  }
}

const vector = (v: ArrayLike<number>): Tensor => fromStorage(Float64Array.from(v), [v.length], 'float64')

function verbsOf(p: SvcParams, h: SvcHyper): SvcVerbs {
  const k = p.classes.length
  const d = p.vectors[0]?.length ?? 0
  const decisions = (x: TensorLike): { m: number; f: Float64Array[] } => {
    const m = matrixView(x, WHO, d > 0 ? { nFeatures: d } : {})
    if (p.vectors.length === 0) return { m: m.n, f: p.pairs.map((pair) => new Float64Array(m.n).fill(pair.intercept)) }
    const nv = p.vectors.length
    const K = kernelMatrix(h, p.kernel, x, p.vectors)
    return {
      m: m.n,
      f: p.pairs.map((pair) =>
        Float64Array.from({ length: m.n }, (_, i) => {
          let s = pair.intercept
          for (let t = 0; t < pair.support.length; t++)
            s += (pair.coef[t] ?? 0) * (K[i * nv + (pair.support[t] ?? 0)] ?? 0)
          return s
        }),
      ),
    }
  }
  const votes = (x: TensorLike): { m: number; v: Float64Array; confidence: Float64Array } => {
    const { m, f } = decisions(x)
    const v = new Float64Array(m * k)
    const confidence = new Float64Array(m * k)
    p.pairs.forEach((pair, s) => {
      for (let i = 0; i < m; i++) {
        const dec = f[s]?.[i] ?? 0
        const winner = dec > 0 ? pair.positive : pair.negative
        v[i * k + winner] = (v[i * k + winner] ?? 0) + 1
        confidence[i * k + pair.positive] = (confidence[i * k + pair.positive] ?? 0) + dec
        confidence[i * k + pair.negative] = (confidence[i * k + pair.negative] ?? 0) - dec
      }
    })
    return { m, v, confidence }
  }
  const probabilities = (x: TensorLike): Tensor => {
    const { m, f } = decisions(x)
    const out = new Float64Array(m * k)
    for (let i = 0; i < m; i++) {
      const r = Array.from({ length: k }, () => new Array<number>(k).fill(0))
      p.pairs.forEach((pair, s) => {
        const platt = pair.platt ?? { a: -1, b: 0 }
        const z = platt.a * (f[s]?.[i] ?? 0) + platt.b
        const q = z >= 0 ? Math.exp(-z) / (1 + Math.exp(-z)) : 1 / (1 + Math.exp(z))
        const clipped = Math.min(Math.max(q, 1e-7), 1 - 1e-7)
        ;(r[pair.positive] as number[])[pair.negative] = clipped
        ;(r[pair.negative] as number[])[pair.positive] = 1 - clipped
      })
      const probs = k === 2 ? [r[0]?.[1] ?? 0.5, r[1]?.[0] ?? 0.5] : pairwiseCoupling(r)
      for (let c = 0; c < k; c++) out[i * k + c] = probs[c] ?? 0
    }
    return fromStorage(out, [m, k], 'float64')
  }
  const predict = (x: TensorLike): Tensor => {
    const { m, v } = votes(x)
    return vector(
      Float64Array.from({ length: m }, (_, i) => {
        let best = 0
        for (let c = 1; c < k; c++) if ((v[i * k + c] ?? 0) > (v[i * k + best] ?? 0)) best = c
        return p.classes[best] ?? 0
      }),
    )
  }
  return {
    predict,
    decisionFunction: (x) => {
      if (k === 2 || h.decisionShape === 'ovo') {
        const { m, f } = decisions(x)
        if (k === 2) return vector(f[0] ?? new Float64Array(m))
        const out = new Float64Array(m * f.length)
        for (let s = 0; s < f.length; s++) for (let i = 0; i < m; i++) out[i * f.length + s] = f[s]?.[i] ?? 0
        return fromStorage(out, [m, f.length], 'float64')
      }
      const { m, v, confidence } = votes(x)
      return fromStorage(
        Float64Array.from(v, (vote, idx) => {
          const c = confidence[idx] ?? 0
          return vote + c / (3 * (Math.abs(c) + 1))
        }),
        [m, k],
        'float64',
      )
    },
    score: (data) => {
      const predicted = toFlat(predict(data.x))
      const truth = toFlat(asTensor(data.y ?? []))
      return predicted.reduce((s, v, i) => s + (v === truth[i] ? 1 : 0), 0) / Math.max(1, predicted.length)
    },
    ...(probabilistic(p) ? { predictProba: probabilities } : {}),
    classes: p.classes,
    nFeaturesIn: d,
  }
}

/** Everything the algorithm needs from the data, built once: the layout, the Gram matrix, the penalties and the duals. */
function prepare(
  data: LearnData,
  h: SvcHyper,
): {
  readonly layout: Layout
  readonly K: Float64Array
  readonly C: Float64Array
  dualOf(k: number, key?: Key): Dual
} {
  const layout = layoutOf(data, h)
  if (h.form === 'nu' && data.weights !== undefined)
    throw new MdmlError('value', `${WHO}: sample weights are not supported by the nu form`)
  if (h.loss === 'squaredHinge' && h.form === 'nu')
    throw new MdmlError('value', `${WHO}: the squared hinge is a C form (the nu form has no squared-hinge dual)`)
  checkSolver(WHO, { kind: svcSolvers.modeOf(h.solver).name }, h.loss)
  const K = kernelMatrix(h, layout.kernel, data.x)
  const C = penalties(data, h, layout)
  const options = { tol: h.tol, selection: h.selection }
  const duals = new Map<number, Dual>()
  const dualOf = (k: number, key: Key = makeKey(0)): Dual => {
    let dual = duals.get(k)
    if (dual === undefined) {
      const pair = layout.pairs[k] as Layout['pairs'][number]
      const Kp = submatrix(K, layout.n, pair.rows)
      const y = Float64Array.from(pair.rows, (i) => (layout.codes[i] === pair.positive ? 1 : -1))
      const Cp = Float64Array.from(pair.rows, (i) => C[i] ?? 0)
      dual =
        h.loss === 'squaredHinge'
          ? squaredDualFor(squaredHingePrimal(Kp, y, Cp, options), h.solver, key)
          : h.form === 'nu'
            ? nuSvcDual(Kp, y, h.nu, options)
            : cSvcDual(Kp, y, Cp, options)
      duals.set(k, dual)
    }
    return dual
  }
  return { layout, K, C, dualOf }
}

const toLive = (e: { readonly coef: Float64Array; readonly rho: number }): SvcState['live'] => ({
  coef: Array.from(e.coef),
  rho: e.rho,
})

/**
 * Support vector classification (scikit-learn `SVC`, libsvm's C-SVC; `form: 'nu'` is `NuSVC`). The dual is solved by
 * the library's SMO, one working-pair update per step: `solver/dual` rises and `solver/gap` falls to zero, `smo/alphaChanged`
 * says how far the pair moved, `solver/pair` names it, and `model/support-vectors` is the model so far. More than two
 * classes are solved one pair at a time (one-versus-one); `probability: true` adds a last step fitting Platt maps by
 * internal cross-validation. The fitted value's parameters hold the support vectors, their dual coefficients and the
 * intercepts, so it predicts by the kernel expansion alone.
 *
 * @concept support-vector-machines
 * @reference C.-C. Chang and C.-J. Lin, LIBSVM: a library for support vector machines (2011).
 * @index estimator Support vector classifier by SMO over any kernel.
 * @example
 * ```ts
 * import { svc } from 'mdml-engine/svm'
 * import { toFlat } from 'mdml-engine/tensor'
 * const fitted = svc.with({ kernel: 'linear', C: 10 }).fit({ x: [[0], [1], [2], [3]], y: [0, 0, 1, 1] })
 * toFlat(fitted.predict([[0.2], [2.8]])) // => [0, 1]
 * fitted.params.nSupport // => [1, 1]
 * ```
 */
export const svc: IterativeEstimator<SvcHyper, SvcState, SvcParams, SvcVerbs, SvmPath> = iterative<
  SvcHyper,
  SvcState,
  SvcParams,
  SvcVerbs,
  SvmPath
>({
  name: 'svm/svc',
  kind: 'classifier',
  defaults: {
    form: 'c',
    C: 1,
    nu: 0.5,
    kernel: 'rbf',
    gamma: 'scale',
    degree: 3,
    coef0: 0,
    tol: 1e-3,
    selection: 'second-order',
    maxIter: 200_000,
    probability: false,
    folds: 5,
    classWeight: null,
    decisionShape: 'ovr',
    loss: 'hinge',
    solver: solverConfigOf(svcSolvers, { kind: 'smo' }),
  },
  hyperSpace: (h) =>
    dict({
      loss: choice(['hinge', 'squaredHinge'] as const, { default: h.loss, label: 'loss' }),
      solver: svcSolvers.space,
      ...(h.form === 'c'
        ? { C: real(1e-2, 1e3, { log: true, default: Math.min(Math.max(h.C, 1e-2), 1e3), label: 'C' }) }
        : { nu: real(0.01, 1, { default: Math.min(Math.max(h.nu, 0.01), 1), label: 'nu' }) }),
      ...kernelSpace(h),
    }),
  algorithm: (data, h): Algorithm<SvcState, SvmPath> => {
    const { layout, K, C, dualOf } = prepare(data, h)
    const pairCount = layout.pairs.length
    const start = (k: number, key: Key): Pick<SvcState, 'smo' | 'live' | 'margins'> => {
      const dual = dualOf(k, key)
      const smo = dual.algorithm.init(key)
      return { smo, live: toLive(dual.expansion(smo)), margins: Array.from(dual.margins(smo)) }
    }
    return {
      name: 'svc',
      config: {
        form: h.form,
        C: h.C,
        nu: h.nu,
        kernel: layout.kernel.kernel,
        gamma: layout.kernel.gamma,
        tol: h.tol,
        loss: h.loss,
        solver: h.solver,
      },
      schema: channels({
        metrics: {
          dual: { description: 'the dual objective of the pair being solved (to be maximised)', direction: 'higher' },
          gap: {
            description: 'the primal-dual gap of the pair being solved (zero at the optimum)',
            direction: 'lower',
          },
          violation: { description: 'the largest KKT violation m(alpha) - M(alpha) of the working-set selection' },
          support: { description: 'the support vectors of the pair being solved' },
        },
        tensors: { alphas: { role: 'param', description: 'the dual variables of the pair being solved' } },
        messages: {
          pair: {
            description: 'the working pair (i, j) of the step, as training rows, and the classes being separated',
          },
        },
      }),
      init: (key) => ({
        phase: 'fit',
        current: 0,
        ...start(0, key),
        solved: [],
        platt: layout.pairs.map(() => null),
      }),
      step: (s, ctx) => {
        if (s.phase === 'calibrate') {
          const platt = layout.pairs.map((pair, k) => {
            const y = Float64Array.from(pair.rows, (i) => (layout.codes[i] === pair.positive ? 1 : -1))
            const Kp = submatrix(K, layout.n, pair.rows)
            const order = Array.from(permutation(foldIn(ctx.key, k), pair.rows.length).data)
            const dec = crossValidated(
              Kp,
              y,
              Float64Array.from(pair.rows, (i) => C[i] ?? 0),
              h,
              order,
            )
            return plattScaling(
              dec,
              y.map((v) => (v > 0 ? 1 : 0)),
            ).params
          })
          return { ...s, phase: 'done', platt }
        }
        const dual = dualOf(s.current, ctx.key)
        const before = s.smo
        const smo = dual.algorithm.step(before, ctx)
        if (smo.pair !== undefined) {
          const [i, j] = smo.pair
          ctx.log(
            'smo/alphaChanged',
            Math.abs((smo.alpha[i] ?? 0) - (before.alpha[i] ?? 0)) +
              Math.abs((smo.alpha[j] ?? 0) - (before.alpha[j] ?? 0)),
          )
        }
        const live = toLive(dual.expansion(smo))
        const margins = Array.from(dual.margins(smo))
        if (!smo.finished) return { ...s, smo, live, margins }
        const solved = [...s.solved, { coef: live.coef, rho: live.rho, iterations: smo.iterations }]
        if (s.current + 1 < pairCount) return { ...s, current: s.current + 1, ...start(s.current + 1, ctx.key), solved }
        return { ...s, smo, live, margins, solved, phase: h.probability ? 'calibrate' : 'done' }
      },
      done: (s) => (s.phase === 'done' ? 'converged' : undefined),
      observe: (s) => {
        const dual = dualOf(s.current)
        const pair = layout.pairs[s.current] as Layout['pairs'][number]
        return {
          metrics: {
            dual: dual.dual(s.smo),
            gap: dual.gap(s.smo),
            ...(Number.isFinite(s.smo.violation) ? { violation: s.smo.violation } : {}),
            support: s.smo.alpha.reduce((c, v) => c + (v > 0 ? 1 : 0), 0),
          },
          tensors: { alphas: { role: 'param', value: s.smo.alpha } },
          ...(s.smo.pair === undefined
            ? {}
            : {
                messages: [
                  {
                    kind: 'pair',
                    payload: {
                      i: pair.rows[s.smo.pair[0]] ?? 0,
                      j: pair.rows[s.smo.pair[1]] ?? 0,
                      positive: layout.classes[pair.positive] ?? 0,
                      negative: layout.classes[pair.negative] ?? 0,
                    },
                  },
                ],
              }),
        }
      },
      serialise: (s) => stateToJson(s),
      deserialise: (j) => stateFromJson(j) as SvcState,
    }
  },
  maxSteps: (h) => h.maxIter,
  params: paramsOf,
  verbs: verbsOf,
  contract: (data, h): Contract<SvcState, SvmPath> => {
    const layout = layoutOf(data, h)
    return defineContract({
      'smo/alphaChanged': scalar({
        description: 'the change of the two dual variables updated in the step, |da_i| + |da_j|',
      }),
      'model/support-vectors': derived<SvcState>({
        description:
          'the support vectors, dual coefficients, intercept and margins of the pair being solved (the whole model for two classes)',
        compute: (s) => {
          const pair = layout.pairs[s.current] as Layout['pairs'][number]
          const support: number[] = []
          const dualCoefficients: number[] = []
          pair.rows.forEach((row, t) => {
            const c = s.live.coef[t] ?? 0
            if (c !== 0) {
              support.push(row)
              dualCoefficients.push(c)
            }
          })
          return marginRecord({ support, dualCoefficients, intercept: -s.live.rho, margins: s.margins })
        },
      }),
    })
  },
  logs: { 'solver/dual': Log(), 'solver/gap': Log() },
})

/**
 * The nu form of {@link svc} (scikit-learn's `NuSVC`).
 *
 * @example
 * ```ts
 * import { nuSvc } from 'mdml-engine/svm'
 * nuSvc.fit({ x: [[0], [1], [2], [3]], y: [0, 0, 1, 1] }).classes // => [0, 1]
 * ```
 */
export const nuSvc: IterativeEstimator<SvcHyper, SvcState, SvcParams, SvcVerbs, SvmPath> = svc.with({ form: 'nu' })
