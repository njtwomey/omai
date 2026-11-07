/**
 * Support vector regression (scikit-learn's `SVR` and `NuSVR`, libsvm's epsilon-SVR and nu-SVR) as estimators on the
 * iterative base: the epsilon-insensitive tube around the fit, whose width is given (epsilon) or found (nu), solved by
 * the library's SMO over the 2n variables of the dual one working pair per step. The fitted model is the kernel
 * expansion over the support vectors, the points on or outside the tube. The solver is a slot (`solver`, see
 * `svrSolvers`): SMO by default; with `loss: 'squaredEpsilonInsensitive'` the primal is smooth and any solver of
 * `smoothSolvers` minimises it.
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
  type IterativeEstimator,
  iterative,
  type LearnData,
  matrixView,
  sampleWeightsOf,
  targetVector,
} from '../learn/index.ts'
import { type SmoState, solverConfigOf } from '../optim/index.ts'
import type { Key } from '../random/index.ts'
import { key as makeKey } from '../random/index.ts'
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
import { type Dual, nuSvrDual, svrDual, toLive } from './_dual.ts'
import { checkSmoothSolver, type SvrLoss, squaredSvrDualFor, squaredSvrPrimal, svrSolvers } from './_squared-slack.ts'
import type { SvmPath } from './classifier.ts'

export { type SvrLoss, svrSolvers } from './_squared-slack.ts'

/** The hyperparameters of {@link svr}. */
export interface SvrHyper {
  /** `epsilon` (a given tube width) or `nu` (the width found, nu bounding the points outside it). */
  readonly form: 'epsilon' | 'nu'
  readonly C: number
  /** The half-width of the tube of the epsilon form. */
  readonly epsilon: number
  /** The nu of the nu form, in (0, 1]. */
  readonly nu: number
  readonly kernel: PairwiseKernelMetric | Kernel
  readonly gamma: number | 'scale' | 'auto'
  readonly degree: number
  readonly coef0: number
  readonly tol: number
  readonly selection: 'second-order' | 'first-order'
  readonly maxIter: number
  /** `epsilonInsensitive` (libsvm's) or `squaredEpsilonInsensitive` (the L2-loss SVR, whose primal is smooth; the epsilon form only). */
  readonly loss: SvrLoss
  /**
   * How the model is fitted: a value of `svrSolvers.space`. `smo` (default) is libsvm's dual solver and reaches either
   * loss; with `loss: 'squaredEpsilonInsensitive'` every solver of `smoothSolvers` minimises the primal over the kernel
   * expansion.
   */
  readonly solver: SpaceValue
}

/** The fitted parameters of {@link svr}: plain JSON. */
export type SvrParams = {
  readonly kernel: ResolvedKernel
  /** The support vectors: training rows with a nonzero coefficient. */
  readonly vectors: number[][]
  /** The training row of each support vector (`support_`). */
  readonly supportIndex: number[]
  /** alpha - alpha* of each support vector (`dual_coef_`). */
  readonly coef: number[]
  readonly intercept: number
  /** The half-width of the tube. */
  readonly tube: number
}

/** What a fitted SVR offers. */
export interface SvrVerbs {
  readonly predict: (x: TensorLike) => Tensor
  /** The coefficient of determination on a labelled dataset. */
  readonly score: (data: LearnData) => number
  readonly nFeaturesIn: number
}

/** The state of {@link svr}. */
export interface SvrState {
  readonly smo: SmoState
  /** alpha - alpha* over the training rows, and rho, at `smo`. */
  readonly live: { readonly coef: number[]; readonly rho: number }
  /** The residuals `t_i - f(x_i)` at `smo`. */
  readonly residuals: number[]
  readonly tube: number
}

const WHO = 'svr'

function prepare(
  data: LearnData,
  h: SvrHyper,
): {
  readonly n: number
  readonly kernel: ResolvedKernel
  readonly t: Float64Array
  dualOf(key?: Key): Dual
} {
  const m = matrixView(data.x, WHO)
  const t = targetVector(data.y ?? [], WHO, m.n)
  if (h.form === 'nu' && data.weights !== undefined)
    throw new MdmlError('value', `${WHO}: sample weights are not supported by the nu form`)
  const squared = h.loss === 'squaredEpsilonInsensitive'
  if (squared && h.form === 'nu')
    throw new MdmlError(
      'value',
      `${WHO}: the squared epsilon-insensitive loss is an epsilon form (the nu form has no squared dual)`,
    )
  checkSmoothSolver(
    WHO,
    { kind: svrSolvers.modeOf(h.solver).name },
    squared,
    'epsilon-insensitive',
    'squaredEpsilonInsensitive',
  )
  const kernel = resolveKernel(h as KernelChoice, data.x)
  const K = kernelMatrix(h, kernel, data.x)
  const w = sampleWeightsOf(data.weights, m.n, WHO)
  const options = { tol: h.tol, selection: h.selection }
  let dual: Dual | undefined
  const dualOf = (key: Key = makeKey(0)): Dual => {
    if (dual === undefined) {
      const C = w.map((v) => v * h.C)
      dual = squared
        ? squaredSvrDualFor(squaredSvrPrimal(K, t, C, h.epsilon, options), h.solver, key)
        : h.form === 'nu'
          ? nuSvrDual(K, t, h.C, h.nu, options)
          : svrDual(K, t, C, h.epsilon, options)
    }
    return dual
  }
  return { n: m.n, kernel, t, dualOf }
}

const snapshot = (dual: Dual, smo: SmoState): Pick<SvrState, 'smo' | 'live' | 'residuals' | 'tube'> => ({
  smo,
  live: toLive(dual.expansion(smo)),
  residuals: Array.from(dual.margins(smo)),
  tube: dual.tube?.(smo) ?? 0,
})

/**
 * Support vector regression (scikit-learn `SVR`, libsvm's epsilon-SVR; `form: 'nu'` is `NuSVR`). One SMO working pair
 * per step on the dual over (alpha, alpha*): `solver/dual` rises, `solver/gap` falls to zero, `smo/alphaChanged` says
 * how far the pair moved, and `model/support-vectors` is the tube so far, with each point's residual. The fitted
 * value's parameters hold the support vectors, their coefficients, the intercept and the tube width.
 *
 * The solver is a hyperparameter (`solver`, from `svrSolvers`): `smo` (default) solves the dual and reaches both losses;
 * `loss: 'squaredEpsilonInsensitive'` (the L2-loss SVR, epsilon form) has a smooth primal over the kernel-expansion
 * coefficients and the intercept, which every solver of `smoothSolvers` minimises. The epsilon-insensitive loss is not
 * smooth, so a smooth solver with it throws.
 *
 * @concept support-vector-machines
 * @reference A. J. Smola and B. Schoelkopf, A tutorial on support vector regression (2004).
 * @index estimator Support vector regression.
 * @example
 * ```ts
 * import { svr } from 'mdml-engine/svm'
 * import { toFlat } from 'mdml-engine/tensor'
 * const fitted = svr.with({ kernel: 'linear', C: 100, epsilon: 0.01 }).fit({ x: [[0], [1], [2], [3]], y: [1, 3, 5, 7] })
 * toFlat(fitted.predict([[4]])).map((v) => Math.round(v * 10) / 10) // => [9]
 * ```
 */
export const svr: IterativeEstimator<SvrHyper, SvrState, SvrParams, SvrVerbs, SvmPath> = iterative<
  SvrHyper,
  SvrState,
  SvrParams,
  SvrVerbs,
  SvmPath
>({
  name: 'svm/svr',
  kind: 'regressor',
  defaults: {
    form: 'epsilon',
    C: 1,
    epsilon: 0.1,
    nu: 0.5,
    kernel: 'rbf',
    gamma: 'scale',
    degree: 3,
    coef0: 0,
    tol: 1e-3,
    selection: 'second-order',
    maxIter: 200_000,
    loss: 'epsilonInsensitive',
    solver: solverConfigOf(svrSolvers, { kind: 'smo' }),
  },
  hyperSpace: (h) =>
    dict({
      loss: choice(['epsilonInsensitive', 'squaredEpsilonInsensitive'] as const, { default: h.loss, label: 'loss' }),
      solver: svrSolvers.space,
      C: real(1e-2, 1e3, { log: true, default: Math.min(Math.max(h.C, 1e-2), 1e3), label: 'C' }),
      ...(h.form === 'epsilon'
        ? { epsilon: real(0, 2, { default: Math.min(Math.max(h.epsilon, 0), 2), label: 'tube half-width' }) }
        : { nu: real(0.01, 1, { default: Math.min(Math.max(h.nu, 0.01), 1), label: 'nu' }) }),
      ...kernelSpace(h),
    }),
  algorithm: (data, h): Algorithm<SvrState, SvmPath> => {
    const { kernel, dualOf } = prepare(data, h)
    return {
      name: 'svr',
      config: {
        form: h.form,
        C: h.C,
        epsilon: h.epsilon,
        nu: h.nu,
        kernel: kernel.kernel,
        gamma: kernel.gamma,
        loss: h.loss,
        solver: h.solver,
      },
      schema: channels({
        metrics: {
          dual: { description: 'the dual objective (to be maximised)', direction: 'higher' },
          gap: { description: 'the primal-dual gap (zero at the optimum)', direction: 'lower' },
          violation: { description: 'the largest KKT violation of the working-set selection' },
          support: { description: 'the support vectors (variables of the 2n with alpha > 0)' },
          tube: { description: 'the half-width of the tube (given, or found by the nu form)' },
        },
        tensors: { alphas: { role: 'param', description: 'the dual variables (alpha, alpha*)' } },
        messages: { pair: { description: 'the working pair (i, j) of the step, as variables of the 2n' } },
      }),
      init: (key) => snapshot(dualOf(key), dualOf(key).algorithm.init(key)),
      step: (s, ctx) => {
        const dual = dualOf(ctx.key)
        const smo = dual.algorithm.step(s.smo, ctx)
        if (smo.pair !== undefined) {
          const [i, j] = smo.pair
          ctx.log(
            'smo/alphaChanged',
            Math.abs((smo.alpha[i] ?? 0) - (s.smo.alpha[i] ?? 0)) +
              Math.abs((smo.alpha[j] ?? 0) - (s.smo.alpha[j] ?? 0)),
          )
        }
        return snapshot(dual, smo)
      },
      done: (s) => (s.smo.finished ? 'converged' : undefined),
      observe: (s) => ({
        metrics: {
          dual: dualOf().dual(s.smo),
          gap: dualOf().gap(s.smo),
          ...(Number.isFinite(s.smo.violation) ? { violation: s.smo.violation } : {}),
          support: s.live.coef.reduce((c, v) => c + (v !== 0 ? 1 : 0), 0),
          tube: s.tube,
        },
        tensors: { alphas: { role: 'param', value: s.smo.alpha } },
        ...(s.smo.pair === undefined ? {} : { messages: [{ kind: 'pair', payload: [s.smo.pair[0], s.smo.pair[1]] }] }),
      }),
      serialise: (s) => stateToJson(s),
      deserialise: (j) => stateFromJson(j) as SvrState,
    }
  },
  maxSteps: (h) => h.maxIter,
  params: (s, data, h) => {
    const m = matrixView(data.x, WHO)
    const supportIndex = s.live.coef.flatMap((c, i) => (c !== 0 ? [i] : []))
    return {
      kernel: resolveKernel(h as KernelChoice, data.x),
      vectors: supportIndex.map((i) => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d))),
      supportIndex,
      coef: supportIndex.map((i) => s.live.coef[i] ?? 0),
      intercept: -s.live.rho,
      tube: s.tube,
    }
  },
  verbs: (p, h) => {
    const d = p.vectors[0]?.length ?? 0
    const predict = (x: TensorLike): Tensor => {
      const m = matrixView(x, WHO, d > 0 ? { nFeatures: d } : {})
      if (p.vectors.length === 0) return fromStorage(new Float64Array(m.n).fill(p.intercept), [m.n], 'float64')
      const nv = p.vectors.length
      const K = kernelMatrix(h, p.kernel, x, p.vectors)
      return fromStorage(
        Float64Array.from({ length: m.n }, (_, i) => {
          let s = p.intercept
          for (let t = 0; t < nv; t++) s += (p.coef[t] ?? 0) * (K[i * nv + t] ?? 0)
          return s
        }),
        [m.n],
        'float64',
      )
    }
    return {
      predict,
      score: (data) => {
        const truth = toFlat(asTensor(data.y ?? []))
        const fit = toFlat(predict(data.x))
        const mean = truth.reduce((a, b) => a + b, 0) / Math.max(1, truth.length)
        let rss = 0
        let tss = 0
        truth.forEach((v, i) => {
          rss += (v - (fit[i] ?? 0)) ** 2
          tss += (v - mean) ** 2
        })
        return tss === 0 ? 0 : 1 - rss / tss
      },
      nFeaturesIn: d,
    }
  },
  contract: (): Contract<SvrState, SvmPath> =>
    defineContract({
      'smo/alphaChanged': scalar({
        description: 'the change of the two dual variables updated in the step, |da_i| + |da_j|',
      }),
      'model/support-vectors': derived<SvrState>({
        description:
          'the support vectors of the tube so far: coefficients, intercept and the residual of every training row',
        compute: (s) => {
          const support: number[] = []
          const dualCoefficients: number[] = []
          s.live.coef.forEach((c, i) => {
            if (c !== 0) {
              support.push(i)
              dualCoefficients.push(c)
            }
          })
          return marginRecord({ support, dualCoefficients, intercept: -s.live.rho, margins: s.residuals })
        },
      }),
    }),
  logs: { 'solver/dual': Log(), 'solver/gap': Log() },
})

/**
 * The nu form of {@link svr} (scikit-learn's `NuSVR`): the tube width is found, nu bounding the fraction of points
 * outside it above and the fraction of support vectors below.
 *
 * @example
 * ```ts
 * import { nuSvr } from 'mdml-engine/svm'
 * nuSvr.fit({ x: [[0], [1], [2], [3]], y: [1, 3, 5, 7] }).params.tube >= 0 // => true
 * ```
 */
export const nuSvr: IterativeEstimator<SvrHyper, SvrState, SvrParams, SvrVerbs, SvmPath> = svr.with({ form: 'nu' })
