/**
 * The mixture of linear experts as a particular of the iterative base, for regression on one feature: EM from random
 * chords through the data, with the log likelihood, responsibilities and gate of every step on the tape, the most
 * responsible expert of each training row, each expert's line over the rows and the gate's probabilities against x as
 * probes.
 */

import { curveSet } from '../../lib/exposition/index.ts'
import { type IterativeEstimator, iterative, matrixView } from '../../lib/learn/index.ts'
import { matrixTensor, vectorTensor } from '../../lib/linear/index.ts'
import {
  type MoeState,
  mixtureOfExpertsAlgorithm,
  moeAssignments,
  moeExpertMeans,
  moeGate,
  moePredict,
} from '../../lib/mixture/index.ts'
import { dict, int, real } from '../../lib/space/index.ts'
import type { Tensor, TensorLike } from '../../lib/tensor/index.ts'
import { type Contract, defineContract, derived, Log } from '../../lib/trace/index.ts'

/** The hyperparameters of {@link moe}. */
export interface MoeHyper {
  /** The number of linear experts. */
  readonly nExperts: number
  readonly maxIter: number
  /** The mean log-likelihood change that stops EM. */
  readonly tol: number
  /** The ridge penalty on the gate's feature weights (0 is plain maximum likelihood). */
  readonly gateRidge: number
  /** Starts from fresh random chords; the highest log likelihood wins (a `fit`; a session runs one). */
  readonly nInit: number
}

/** The fitted parameters of {@link moe}: plain JSON. Coefficients run over `[1, x...]`, intercept first. */
export type MoeParams = {
  readonly gate: number[][]
  readonly experts: number[][]
  readonly variances: number[]
  readonly logLikelihood: number
}

/** What a fitted mixture of experts offers. */
export interface MoeVerbs {
  /** The mixture's mean at each row: the experts' lines averaged by the gate. */
  readonly predict: (x: TensorLike) => Tensor
  /** The gate's probabilities `[m, k]`: how much each expert is trusted at each row. */
  readonly predictGate: (x: TensorLike) => Tensor
  readonly nFeaturesIn: number
}

/** The state's parameters in the form the library's readers take. */
const paramsOf = (s: MoeState): Pick<MoeState, 'gate' | 'experts' | 'variances'> => ({
  gate: s.gate,
  experts: s.experts,
  variances: s.variances,
})

/**
 * The paths of the mixture of experts beyond the solver’s, all derived: each row’s expert, the experts’ lines and the gate against the first feature.
 *
 * @example
 * ```ts
 * import type { MoePath } from 'omai-engine/methods/mixture'
 * const path: MoePath = 'model/gating'
 * path // => 'model/gating'
 * ```
 */
export type MoePath = 'model/assignments' | 'model/experts' | 'model/gating'

/**
 * A mixture of linear experts (Jacobs and Jordan): a gate that shares the rows among linear-Gaussian experts by a
 * softmax of linear functions of x, fitted by expectation-maximisation.
 *
 * @example
 * ```ts
 * import { moe } from 'omai-engine/methods/mixture'
 * const x = Array.from({ length: 40 }, (_, i) => [i / 40])
 * const y = x.map(([v]) => (v < 0.5 ? 4 * v : 2 - 2 * v))
 * moe.fit({ x, y }).params.experts.length // => 2
 * ```
 */
export const moe: IterativeEstimator<MoeHyper, MoeState, MoeParams, MoeVerbs, MoePath> = iterative<
  MoeHyper,
  MoeState,
  MoeParams,
  MoeVerbs,
  MoePath
>({
  name: 'mixture/moe',
  kind: 'regressor',
  defaults: { nExperts: 2, maxIter: 100, tol: 1e-6, gateRidge: 0.01, nInit: 3 },
  hyperSpace: (h) =>
    dict({
      nExperts: int(1, 6, { default: h.nExperts, label: 'experts' }),
      maxIter: int(5, 300, { default: h.maxIter, label: 'iterations' }),
      tol: real(1e-8, 1e-2, { log: true, default: h.tol, label: 'tolerance' }),
    }),
  algorithm: (data, h, key) =>
    mixtureOfExpertsAlgorithm(data.x, data.y ?? [], { nExperts: h.nExperts, tol: h.tol, gateRidge: h.gateRidge }, key),
  maxSteps: (h) => h.maxIter,
  restarts: { count: (h) => h.nInit, better: (a, b) => a.logLikelihood > b.logLikelihood },
  params: (s) => ({ ...paramsOf(s), logLikelihood: s.logLikelihood }),
  verbs: (p) => ({
    predict: (x) => vectorTensor(moePredict(x, p)),
    predictGate: (x) => {
      const g = moeGate(x, p.gate)
      return matrixTensor(g.flat(), g.length, p.gate.length)
    },
    nFeaturesIn: (p.gate[0]?.length ?? 1) - 1,
  }),
  contract: (data): Contract<MoeState, MoePath> =>
    defineContract({
      'model/assignments': derived<MoeState>({
        description: 'the most responsible expert of each training row',
        compute: (s) => moeAssignments(data.x, data.y ?? [], s),
      }),
      'model/experts': derived<MoeState>({
        description: 'each expert’s line over the training rows: lines[k][i] is expert k’s prediction at row i',
        compute: (s) => ({ lines: transposed(moeExpertMeans(data.x, s.experts)) }),
      }),
      'model/gating': derived<MoeState>({
        description: 'each expert’s gate probability against the first feature, over the training rows in order of x',
        compute: (s) => {
          const m = matrixView(data.x, 'mixture/moe')
          const xs = Array.from({ length: m.n }, (_, i) => m.data[i * m.d] ?? 0)
          const order = xs.map((_, i) => i).sort((a, b) => (xs[a] ?? 0) - (xs[b] ?? 0))
          const gate = moeGate(data.x, s.gate)
          return curveSet({
            x: order.map((i) => xs[i] ?? 0),
            xLabel: 'x',
            yLabel: 'gate probability',
            series: s.gate.map((_, k) => ({ name: `expert ${k}`, y: order.map((i) => gate[i]?.[k] ?? 0) })),
          })
        },
      }),
    }),
  logs: {
    'solver/logLikelihood': Log(),
    'solver/responsibilities': Log({ detail: 'full' }),
    'solver/gate': Log({ detail: 'full' }),
  },
})

/** A `[rows][experts]` matrix as `[experts][rows]`. */
function transposed(m: readonly (readonly number[])[]): number[][] {
  const k = m[0]?.length ?? 0
  return Array.from({ length: k }, (_, j) => m.map((row) => row[j] ?? 0))
}
