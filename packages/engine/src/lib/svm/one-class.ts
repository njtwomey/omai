/**
 * Novelty detection by support vector machines as estimators on the iterative base: the one-class machine (scikit-learn's
 * `OneClassSVM`, libsvm's ONE_CLASS), which separates the data from the origin in feature space with nu bounding the
 * fraction of outliers, and the support vector data description (Tax and Duin), the smallest enclosing ball in feature
 * space with a penalty C on points outside it. Both solve their duals with the library's SMO, one working pair per step;
 * with a kernel of constant diagonal (the RBF) they are the same machine. The solver of the SVDD is a slot (`solver`,
 * see `svddSolvers`): SMO by default; with `loss: 'squaredSlack'` (the L2-SVDD of the core vector machines) the primal
 * is smooth and any solver of `smoothSolvers` minimises it. The one-class machine is SMO-only: its nu form has no
 * standard smooth variant (a squared slack would trade nu, the bound on the outlier fraction, for a penalty C, which is
 * the squared-slack SVDD).
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
import { type IterativeEstimator, iterative, type LearnData, matrixView } from '../learn/index.ts'
import { type SmoState, solverConfigOf } from '../optim/index.ts'
import type { Key } from '../random/index.ts'
import { key as makeKey } from '../random/index.ts'
import { choice, dict, real, type SpaceValue } from '../space/index.ts'
import { fromStorage, type Tensor, type TensorLike } from '../tensor/index.ts'
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
import { type Dual, oneClassDual, svddDual } from './_dual.ts'
import {
  checkSmoothSolver,
  type SvddLoss,
  squaredSvddDualFor,
  squaredSvddPrimal,
  svddSolvers,
} from './_squared-slack.ts'
import type { SvmPath } from './classifier.ts'

export { type SvddLoss, svddSolvers } from './_squared-slack.ts'

/** The hyperparameters shared by {@link oneClassSvm} and {@link svdd}. */
export interface NoveltyHyper {
  readonly kernel: PairwiseKernelMetric | Kernel
  readonly gamma: number | 'scale' | 'auto'
  readonly degree: number
  readonly coef0: number
  readonly tol: number
  readonly selection: 'second-order' | 'first-order'
  readonly maxIter: number
}

/** The hyperparameters of {@link oneClassSvm}. */
export interface OneClassHyper extends NoveltyHyper {
  /** An upper bound on the fraction of training points outside the region and a lower bound on the support fraction, in (0, 1]. */
  readonly nu: number
}

/** The hyperparameters of {@link svdd}. */
export interface SvddHyper extends NoveltyHyper {
  /** The penalty on points outside the ball, at least 1 / n; 1 is the hard ball (every point inside). */
  readonly C: number
  /** `slack` (Tax and Duin's linear slack) or `squaredSlack` (the L2-SVDD, whose primal is smooth; C need only be positive). */
  readonly loss: SvddLoss
  /**
   * How the ball is fitted: a value of `svddSolvers.space`. `smo` (default) solves the dual and reaches either loss;
   * with `loss: 'squaredSlack'` every solver of `smoothSolvers` minimises the primal over the centre's kernel expansion
   * and the squared radius.
   */
  readonly solver: SpaceValue
}

/** The fitted parameters of a novelty detector: plain JSON. */
export type NoveltyParams = {
  readonly kernel: ResolvedKernel
  /** The support vectors: training rows with a nonzero coefficient. */
  readonly vectors: number[][]
  readonly supportIndex: number[]
  /** The dual coefficient of each support vector. */
  readonly coef: number[]
  readonly rho: number
  /** The squared radius of the ball (SVDD only). */
  readonly radiusSquared: number | null
}

/** What a fitted novelty detector offers. */
export interface NoveltyVerbs {
  /** +1 for points inside the learnt region, -1 for outliers. */
  readonly predict: (x: TensorLike) => Tensor
  /** The signed score `[m]`, positive inside the region. */
  readonly decisionFunction: (x: TensorLike) => Tensor
  readonly nFeaturesIn: number
}

/** The state of the novelty detectors. */
export interface NoveltyState {
  readonly smo: SmoState
  readonly live: { readonly coef: number[]; readonly rho: number }
  /** The score of each training row at `smo`. */
  readonly scores: number[]
}

type Hyper = NoveltyHyper & {
  readonly nu?: number
  readonly C?: number
  readonly loss?: SvddLoss
  readonly solver?: SpaceValue
}

const WHO = 'novelty'

function prepare(
  data: LearnData,
  h: Hyper,
  svdd: boolean,
): { kernel: ResolvedKernel; dualOf: (key?: Key) => Dual; n: number } {
  const m = matrixView(data.x, WHO)
  const kernel = resolveKernel(h as KernelChoice, data.x)
  const K = kernelMatrix(h, kernel, data.x)
  const options = { tol: h.tol, selection: h.selection }
  if (svdd) {
    const C = h.C ?? 1
    const squared = h.loss === 'squaredSlack'
    if (squared ? !(C > 0) : !(C * m.n >= 1 - 1e-12))
      throw new MdmlError('domain', `svdd: C must be at least ${squared ? '0' : `1/n = ${1 / m.n}`}, got ${C}`)
    const solver = h.solver ?? solverConfigOf(svddSolvers, { kind: 'smo' })
    checkSmoothSolver('svdd', { kind: svddSolvers.modeOf(solver).name }, squared, 'slack', 'squaredSlack')
    let dual: Dual | undefined
    const dualOf = (key: Key = makeKey(0)): Dual => {
      dual ??= squared ? squaredSvddDualFor(squaredSvddPrimal(K, C, options), solver, key) : svddDual(K, C, options)
      return dual
    }
    return { kernel, n: m.n, dualOf }
  }
  const nu = h.nu ?? 0.5
  if (!(nu > 0 && nu <= 1)) throw new MdmlError('domain', `oneClassSvm: nu must be in (0, 1], got ${nu}`)
  const dual = oneClassDual(K, nu, options)
  return { kernel, n: m.n, dualOf: () => dual }
}

const snapshot = (dual: Dual, smo: SmoState): NoveltyState => {
  const e = dual.expansion(smo)
  return { smo, live: { coef: Array.from(e.coef), rho: e.rho }, scores: Array.from(dual.margins(smo)) }
}

/** The estimator over one of the two duals. */
function noveltyEstimator<H extends Hyper>(
  name: string,
  svdd: boolean,
  defaults: H,
  knob: (h: H) => ReturnType<typeof dict>['spaces'],
): IterativeEstimator<H, NoveltyState, NoveltyParams, NoveltyVerbs, SvmPath> {
  return iterative<H, NoveltyState, NoveltyParams, NoveltyVerbs, SvmPath>({
    name,
    kind: 'outlier',
    defaults,
    hyperSpace: (h) => dict({ ...knob(h), ...kernelSpace(h) }),
    algorithm: (data, h): Algorithm<NoveltyState, SvmPath> => {
      const { kernel, dualOf } = prepare(data, h, svdd)
      return {
        name: svdd ? 'svdd' : 'one-class-svm',
        config: {
          ...(svdd ? { C: h.C ?? 1, loss: h.loss ?? 'slack', solver: h.solver ?? 'smo' } : { nu: h.nu ?? 0.5 }),
          kernel: kernel.kernel,
          gamma: kernel.gamma,
          tol: h.tol,
        },
        schema: channels({
          metrics: {
            dual: { description: 'the dual objective (to be maximised)', direction: 'higher' },
            gap: { description: 'the primal-dual gap (zero at the optimum)', direction: 'lower' },
            violation: { description: 'the largest KKT violation of the working-set selection' },
            support: { description: 'the support vectors (alpha > 0)' },
          },
          tensors: { alphas: { role: 'param', description: 'the dual variables' } },
          messages: { pair: { description: 'the working pair (i, j) of the step, as training rows' } },
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
            support: s.smo.alpha.reduce((c, v) => c + (v > 0 ? 1 : 0), 0),
          },
          tensors: { alphas: { role: 'param', value: s.smo.alpha } },
          ...(s.smo.pair === undefined
            ? {}
            : { messages: [{ kind: 'pair', payload: [s.smo.pair[0], s.smo.pair[1]] }] }),
        }),
        serialise: (s) => stateToJson(s),
        deserialise: (j) => stateFromJson(j) as NoveltyState,
      }
    },
    maxSteps: (h) => h.maxIter,
    params: (s, data, h) => {
      const m = matrixView(data.x, WHO)
      const supportIndex = s.live.coef.flatMap((c, i) => (c > 0 ? [i] : []))
      const coef = supportIndex.map((i) => s.live.coef[i] ?? 0)
      // The squared radius is a^T K a - rho, with the quadratic form over the support vectors.
      let radiusSquared: number | null = null
      if (svdd) {
        const kernel = resolveKernel(h as KernelChoice, data.x)
        const vectors = supportIndex.map((i) => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d)))
        const Ks = kernelMatrix(h, kernel, vectors)
        let q = 0
        for (let a = 0; a < coef.length; a++)
          for (let b = 0; b < coef.length; b++) q += (coef[a] ?? 0) * (coef[b] ?? 0) * (Ks[a * coef.length + b] ?? 0)
        radiusSquared = q - s.live.rho
      }
      return {
        kernel: resolveKernel(h as KernelChoice, data.x),
        vectors: supportIndex.map((i) => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d))),
        supportIndex,
        coef,
        rho: s.live.rho,
        radiusSquared,
      }
    },
    verbs: (p, h) => {
      const d = p.vectors[0]?.length ?? 0
      const decisionFunction = (x: TensorLike): Tensor => {
        const m = matrixView(x, WHO, d > 0 ? { nFeatures: d } : {})
        const nv = p.vectors.length
        const K = nv === 0 ? new Float64Array(0) : kernelMatrix(h, p.kernel, x, p.vectors)
        const out = Float64Array.from({ length: m.n }, (_, i) => {
          let s = 0
          for (let t = 0; t < nv; t++) s += (p.coef[t] ?? 0) * (K[i * nv + t] ?? 0)
          if (!svdd) return s - p.rho
          // The SVDD score R^2 - |phi(x) - centre|^2 = 2 sum a_j K(x_j, x) - K(x, x) - rho.
          const row = Array.from(m.data.subarray(i * m.d, (i + 1) * m.d))
          return 2 * s - (kernelMatrix(h, p.kernel, [row], [row])[0] ?? 0) - p.rho
        })
        return fromStorage(out, [m.n], 'float64')
      }
      return {
        decisionFunction,
        predict: (x) => {
          const f = decisionFunction(x)
          return fromStorage(
            Float64Array.from(f.data, (v) => (v > 0 ? 1 : -1)),
            [f.shape[0] ?? 0],
            'float64',
          )
        },
        nFeaturesIn: d,
      }
    },
    contract: (): Contract<NoveltyState, SvmPath> =>
      defineContract({
        'smo/alphaChanged': scalar({
          description: 'the change of the two dual variables updated in the step, |da_i| + |da_j|',
        }),
        'model/support-vectors': derived<NoveltyState>({
          description: 'the support vectors so far, their coefficients, the offset and the score of every training row',
          compute: (s) => {
            const support: number[] = []
            const dualCoefficients: number[] = []
            s.live.coef.forEach((c, i) => {
              if (c > 0) {
                support.push(i)
                dualCoefficients.push(c)
              }
            })
            return marginRecord({ support, dualCoefficients, intercept: -s.live.rho, margins: s.scores })
          },
        }),
      }),
    logs: { 'solver/dual': Log(), 'solver/gap': Log() },
  })
}

const base: NoveltyHyper = {
  kernel: 'rbf',
  gamma: 'scale',
  degree: 3,
  coef0: 0,
  tol: 1e-3,
  selection: 'second-order',
  maxIter: 200_000,
}

/**
 * The one-class support vector machine (scikit-learn `OneClassSVM`): the region of feature space holding most of the
 * data, from the dual min 1/2 a^T K a with sum a = nu n and 0 <= a <= 1. `decisionFunction` is positive inside the
 * region; at most a fraction nu of the training points fall outside it.
 *
 * It solves its dual by SMO only: the nu form has no standard smooth variant (a squared slack replaces nu, the bound on the
 * outlier fraction, by a penalty C, which is the squared-slack {@link svdd}, the same machine for a kernel of constant
 * diagonal), so there is no `solver` slot here.
 *
 * @concept support-vector-machines
 * @reference B. Schoelkopf et al., Estimating the support of a high-dimensional distribution (2001).
 * @index estimator One-class SVM for novelty detection.
 * @example
 * ```ts
 * import { oneClassSvm } from 'mdml-engine/svm'
 * import { toFlat } from 'mdml-engine/tensor'
 * const x = [[0], [0.1], [-0.1], [0.2], [-0.2], [0.05]]
 * toFlat(oneClassSvm.with({ nu: 0.2, gamma: 1 }).fit({ x }).predict([[0], [10]])) // => [1, -1]
 * ```
 */
export const oneClassSvm: IterativeEstimator<OneClassHyper, NoveltyState, NoveltyParams, NoveltyVerbs, SvmPath> =
  noveltyEstimator<OneClassHyper>('svm/one-class', false, { ...base, nu: 0.5 } as OneClassHyper, (h) => ({
    nu: real(0.01, 1, { default: Math.min(Math.max(h.nu, 0.01), 1), label: 'nu' }),
  }))

/**
 * The support vector data description (Tax and Duin): the smallest ball in feature space enclosing the data up to a
 * penalty C on points outside it. `decisionFunction` is R^2 minus the squared distance to the centre, positive inside.
 * `loss: 'squaredSlack'` is the L2-SVDD, whose primal is smooth: every solver of `smoothSolvers` (the `solver`
 * hyperparameter, from `svddSolvers`) minimises it, and `smo` (default) solves its dual; the linear slack admits `smo` only.
 *
 * @concept support-vector-machines
 * @reference D. Tax and R. Duin, Support vector data description (2004).
 * @example
 * ```ts
 * import { svdd } from 'mdml-engine/svm'
 * import { toFlat } from 'mdml-engine/tensor'
 * const x = [[0], [0.1], [-0.1], [0.2], [-0.2], [0.05]]
 * toFlat(svdd.with({ C: 0.5, gamma: 1 }).fit({ x }).predict([[0], [10]])) // => [1, -1]
 * ```
 */
export const svdd: IterativeEstimator<SvddHyper, NoveltyState, NoveltyParams, NoveltyVerbs, SvmPath> =
  noveltyEstimator<SvddHyper>(
    'svm/svdd',
    true,
    { ...base, C: 1, loss: 'slack', solver: solverConfigOf(svddSolvers, { kind: 'smo' }) } as SvddHyper,
    (h) => ({
      loss: choice(['slack', 'squaredSlack'] as const, { default: h.loss, label: 'loss' }),
      solver: svddSolvers.space,
      C: real(0.01, 1, { log: true, default: Math.min(Math.max(h.C, 0.01), 1), label: 'C' }),
    }),
  )
