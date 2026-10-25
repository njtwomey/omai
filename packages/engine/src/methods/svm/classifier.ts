/**
 * C-support vector classification (as in scikit-learn's `SVC`, libsvm's C-SVC): the maximum-margin classifier in a
 * kernel's feature space, found by solving the dual problem with the core's traced SMO for each pair of classes
 * (one-versus-one), with Platt-scaled probabilities. Records the support vectors, margins and the decision surface as
 * SMO proceeds.
 */

import { WhiteboxError } from '../../lib/base/index.ts'
import { resolveKey } from '../../lib/context/index.ts'
import { categorical, type Distribution } from '../../lib/dists/index.ts'
import { type Exposition, marginRecord } from '../../lib/exposition/index.ts'
import {
  classNamesOf,
  decisionSurfaceOf,
  defineExplainedEstimator,
  type ExplainedEstimator,
  ExplainedEstimatorClass,
  encodeLabels,
  type FitOptions,
  type Fitted,
  featureNamesOf,
  geometricSteps,
  type LearnData,
  matrixView,
  plattScaling,
  reviveCalibrator,
  type SurfaceStepsInput,
  sampleWeightsOf,
} from '../../lib/learn/index.ts'
import { kFold, splitIndices } from '../../lib/learn/model-selection/index.ts'
import { foldIn } from '../../lib/random/index.ts'
import { bool, choice, dict, real, type Space } from '../../lib/space/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat } from '../../lib/tensor/index.ts'
import { type DualRun, type KernelHyper, type KernelParams, kernelMatrix, resolveKernel, runDual } from './_dual.ts'

// TODO: nu-SVC.

/** Hyperparameters of {@link svc}. */
export interface SvcHyper extends KernelHyper {
  readonly C: number
  readonly tol: number
  readonly maxIter: number
  /** Platt probabilities from 5-fold out-of-fold decision values (pairwise coupling beyond two classes). */
  readonly probability: boolean
  /** Per-class multipliers of C, by class label. */
  readonly classWeight?: Readonly<Record<number, number>>
}

/** One binary problem of the one-versus-one decomposition: class `positive` (label +1) against `negative`. */
export type SvcPair = {
  readonly positive: number
  readonly negative: number
  /** Indices of the support vectors among the stored vectors. */
  readonly support: number[]
  /** y_i alpha_i of each support vector. */
  readonly coef: number[]
  readonly intercept: number
  /** Platt's a and b, when probabilities were fitted. */
  readonly platt: { a: number; b: number } | null
}

/** Parameters of a fitted SVC. */
export type SvcParams = {
  readonly kernel: KernelParams
  readonly classes: number[]
  readonly vectors: number[][]
  readonly pairs: SvcPair[]
}

/** Verbs of a fitted SVC. */
export interface SvcVerbs {
  readonly predict: (x: TensorLike) => Tensor
  /** The binary decision value `[m]` for two classes; one-versus-one votes plus confidences `[m, k]` otherwise. */
  readonly decisionFunction: (x: TensorLike) => Tensor
  readonly predictProba?: (x: TensorLike) => Tensor
  readonly predictDistribution?: (x: TensorLike) => Distribution
  readonly classes: number[]
  readonly nFeaturesIn: number
}

/** A fitted SVC. */
export type FittedSvc = Fitted<SvcParams> & SvcVerbs

/** Decision values of every pair, `[pairs][m]`. */
function pairDecisions(p: SvcParams, x: TensorLike): number[][] {
  const k = kernelMatrix(p.kernel, x, p.vectors)
  const nv = p.vectors.length
  const m = nv === 0 ? 0 : k.length / nv
  return p.pairs.map((pair) =>
    Array.from({ length: m }, (_, i) => {
      let s = pair.intercept
      for (const [a, v] of pair.support.entries()) s += (pair.coef[a] ?? 0) * (k[i * nv + v] ?? 0)
      return s
    }),
  )
}

/** Pairwise coupling of pairwise probabilities r[i][j] = P(i | i or j) into class probabilities (libsvm's method). */
function couple(r: number[][], k: number): number[] {
  const q = Array.from({ length: k }, (_, t) =>
    Array.from({ length: k }, (_, j) =>
      t === j
        ? r.reduce((a, row, s) => (s === t ? a : a + (row[t] ?? 0) ** 2), 0)
        : -(r[j]?.[t] ?? 0) * (r[t]?.[j] ?? 0),
    ),
  )
  let p = new Array<number>(k).fill(1 / k)
  for (let iter = 0; iter < Math.max(100, k); iter++) {
    const qp = q.map((row) => row.reduce((a, v, j) => a + v * (p[j] ?? 0), 0))
    const pqp = p.reduce((a, v, t) => a + v * (qp[t] ?? 0), 0)
    let maxErr = 0
    for (let t = 0; t < k; t++) maxErr = Math.max(maxErr, Math.abs((qp[t] ?? 0) - pqp))
    if (maxErr < 0.005 / k) break
    for (let t = 0; t < k; t++) {
      const diff = (-(qp[t] ?? 0) + pqp) / (q[t]?.[t] ?? 1)
      p = p.map((v, j) => (j === t ? v + diff : v))
      const s = 1 + diff
      p = p.map((v) => v / s)
      for (let j = 0; j < k; j++) qp[j] = ((qp[j] ?? 0) + diff * (q[t]?.[j] ?? 0)) / s
    }
  }
  return p
}

function svcVerbs(p: SvcParams): SvcVerbs {
  const k = p.classes.length
  const d = p.vectors[0]?.length ?? 0
  const votes = (x: TensorLike): { m: number; v: Float64Array; conf: Float64Array } => {
    matrixView(x, 'svc', { nFeatures: d })
    const dec = pairDecisions(p, x)
    const m = dec[0]?.length ?? matrixView(x, 'svc').n
    const v = new Float64Array(m * k)
    const conf = new Float64Array(m * k)
    p.pairs.forEach((pair, s) => {
      for (let i = 0; i < m; i++) {
        const f = dec[s]?.[i] ?? 0
        const winner = f > 0 ? pair.positive : pair.negative
        v[i * k + winner] = (v[i * k + winner] ?? 0) + 1
        conf[i * k + pair.positive] = (conf[i * k + pair.positive] ?? 0) + f
        conf[i * k + pair.negative] = (conf[i * k + pair.negative] ?? 0) - f
      }
    })
    return { m, v, conf }
  }
  const probabilities = (x: TensorLike): Tensor => {
    const dec = pairDecisions(p, x)
    const m = dec[0]?.length ?? 0
    const out = new Float64Array(m * k)
    for (let i = 0; i < m; i++) {
      const r = Array.from({ length: k }, () => new Array<number>(k).fill(0))
      p.pairs.forEach((pair, s) => {
        const cal = reviveCalibrator('sigmoid', pair.platt ?? { a: -1, b: 0 })
        const q = toFlat(cal.predict([dec[s]?.[i] ?? 0]))[0] ?? 0.5
        const clipped = Math.min(Math.max(q, 1e-7), 1 - 1e-7)
        ;(r[pair.positive] as number[])[pair.negative] = clipped
        ;(r[pair.negative] as number[])[pair.positive] = 1 - clipped
      })
      const probs = k === 2 ? [r[0]?.[1] ?? 0.5, r[1]?.[0] ?? 0.5] : couple(r, k)
      for (let c = 0; c < k; c++) out[i * k + c] = probs[c] ?? 0
    }
    return fromStorage(out, [m, k], 'float64')
  }
  const probabilistic = p.pairs.every((pair) => pair.platt !== null)
  return {
    predict: (x) => {
      const { m, v } = votes(x)
      return fromStorage(
        Float64Array.from({ length: m }, (_, i) => {
          // The most votes; ties go to the lower class, as libsvm.
          let best = 0
          for (let c = 1; c < k; c++) if ((v[i * k + c] ?? 0) > (v[i * k + best] ?? 0)) best = c
          return p.classes[best] ?? 0
        }),
        [m],
        'float64',
      )
    },
    decisionFunction: (x) => {
      if (k === 2) {
        const dec = pairDecisions(p, x)[0] ?? []
        // The single pair is (class 0 positive, class 1 negative); report it for class 1, as scikit-learn.
        return fromStorage(
          Float64Array.from(dec, (v) => -v),
          [dec.length],
          'float64',
        )
      }
      const { m, v, conf } = votes(x)
      // scikit-learn's ovr shape: votes plus confidences squashed into (-1/3, 1/3).
      return fromStorage(
        Float64Array.from(v, (vote, idx) => {
          const c = conf[idx] ?? 0
          return vote + c / (3 * (Math.abs(c) + 1))
        }),
        [m, k],
        'float64',
      )
    },
    ...(probabilistic
      ? {
          predictProba: probabilities,
          predictDistribution: (x: TensorLike) => categorical({ probs: probabilities(x) }),
        }
      : {}),
    classes: p.classes,
    nFeaturesIn: d,
  }
}

/**
 * C-support vector classification (sklearn.svm.SVC, libsvm's C-SVC): for each pair of classes, the dual
 * max sum alpha - 1/2 sum alpha_i alpha_j y_i y_j K_ij with 0 <= alpha <= C solved by the core's traced SMO, combined
 * by one-versus-one voting; Platt probabilities from out-of-fold decision values on request. Exposition: one margin
 * record per pair (support vectors, dual coefficients, intercept, margins y_i f(x_i), the dual objective and SMO's
 * working pair at each step) and, for two features, the decision surface.
 *
 * @param hyper - C (default 1), kernel (rbf, linear, poly, sigmoid) with gamma ('scale'), degree and coef0, tolerance,
 * iterations, probabilities and class weights.
 * @returns The estimator.
 */
export function svc(hyper: Partial<SvcHyper> = {}): ExplainedEstimator<SvcHyper, FittedSvc> {
  const est: ExplainedEstimator<SvcHyper, FittedSvc> = defineExplainedEstimator(
    {
      name: 'svc',
      kind: 'classifier',
      hyperSpace: (h: SvcHyper): Space =>
        dict({
          C: real(1e-2, 1e3, { log: true, default: h.C, label: 'C' }),
          kernel: choice(['rbf', 'linear', 'poly', 'sigmoid'], {
            label: 'kernel',
            description: 'The kernel measuring similarity between rows.',
            default: h.kernel,
          }),
          gamma: real(1e-3, 10, {
            label: 'kernel γ',
            description: 'The kernel width: larger γ makes similarity more local.',
            log: true,
            default: typeof h.gamma === 'number' ? h.gamma : 1,
            when: { key: 'kernel', in: ['rbf', 'poly', 'sigmoid'] },
          }),
          degree: real(2, 5, {
            label: 'polynomial degree',
            description: 'The degree of the polynomial kernel.',
            step: 1,
            default: h.degree,
            when: { key: 'kernel', in: ['poly'] },
          }),
          probability: bool({
            label: 'probabilities',
            description: 'Calibrate probabilities by Platt scaling.',
            default: h.probability,
          }),
        }),
      fit: (data: LearnData, h: SvcHyper, options: FitOptions) => {
        const m = matrixView(data.x, 'svc')
        const { classes, codes } = encodeLabels(data.y ?? [], 'svc', m.n)
        if (classes.length < 2) throw new WhiteboxError('value', 'svc: needs at least two classes')
        const w = sampleWeightsOf(data.weights, m.n, 'svc')
        const kp = resolveKernel(h, m.data, m.d)
        const rows = Array.from({ length: m.n }, (_, i) => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d)))
        const K = kernelMatrix(kp, rows)
        const key = resolveKey(options, 'svc')
        const used = new Map<number, number>()
        const vectors: number[][] = []
        const exposition: Exposition[] = []
        const pairs: SvcPair[] = []
        const solvePair = (
          idx: readonly number[],
          labels: readonly number[],
          weights: readonly number[],
          track: boolean,
          snapshots = false,
        ) => {
          const n = idx.length
          const Q = Array.from({ length: n }, (_, a) =>
            Array.from(
              { length: n },
              (_, b) => (labels[a] ?? 1) * (labels[b] ?? 1) * (K[(idx[a] ?? 0) * m.n + (idx[b] ?? 0)] ?? 0),
            ),
          )
          const C = idx.map((i, a) => h.C * (w[i] ?? 1) * (weights[a] ?? 1))
          return runDual({ Q, p: new Float64Array(n).fill(-1), y: labels, C }, track ? options : {}, {
            tol: h.tol,
            maxIter: h.maxIter,
            snapshots,
          })
        }
        // Two classes and two features: snapshot the dual through SMO, to record the surface as it trained.
        const record = options.explain !== false && classes.length === 2 && m.d === 2
        let binary: { readonly dual: DualRun; readonly labels: readonly number[] } | undefined
        let pairIndex = 0
        for (let a = 0; a < classes.length; a++)
          for (let b = a + 1; b < classes.length; b++) {
            const idx = Array.from(codes.keys()).filter((i) => codes[i] === a || codes[i] === b)
            const labels = idx.map((i) => (codes[i] === a ? 1 : -1))
            const cw = idx.map((i) => h.classWeight?.[classes[codes[i] ?? 0] ?? 0] ?? 1)
            const dual = solvePair(idx, labels, cw, true, record)
            if (record) binary = { dual, labels }
            const support: number[] = []
            const coef: number[] = []
            dual.alpha.forEach((al, t) => {
              if (al <= 0) return
              const row = idx[t] ?? 0
              let v = used.get(row)
              if (v === undefined) {
                v = vectors.length
                used.set(row, v)
                vectors.push(rows[row] as number[])
              }
              support.push(v)
              coef.push(al * (labels[t] ?? 1))
            })
            let platt: { a: number; b: number } | null = null
            if (h.probability) {
              const folds = splitIndices(kFold(5, { shuffle: true }), { n: idx.length }, foldIn(key, pairIndex))
              const oof = new Float64Array(idx.length)
              for (const fold of folds) {
                const tr = Array.from(fold.train)
                const d2 = solvePair(
                  tr.map((t) => idx[t] ?? 0),
                  tr.map((t) => labels[t] ?? 1),
                  tr.map((t) => cw[t] ?? 1),
                  false,
                )
                for (const t of fold.test) {
                  let s = -d2.rho
                  tr.forEach((u, j) => {
                    const al = d2.alpha[j] ?? 0
                    if (al > 0) s += al * (labels[u] ?? 1) * (K[(idx[t] ?? 0) * m.n + (idx[u] ?? 0)] ?? 0)
                  })
                  oof[t] = s
                }
              }
              const cal = plattScaling(
                oof,
                labels.map((l) => (l > 0 ? 1 : 0)),
              )
              platt = { a: cal.params.a, b: cal.params.b }
            }
            pairs.push({ positive: a, negative: b, support, coef, intercept: -dual.rho, platt })
            const decision = (row: number): number => {
              let s = -dual.rho
              dual.alpha.forEach((al, t) => {
                if (al > 0) s += al * (labels[t] ?? 1) * (K[row * m.n + (idx[t] ?? 0)] ?? 0)
              })
              return s
            }
            exposition.push(
              marginRecord({
                support: idx.filter((_, t) => (dual.alpha[t] ?? 0) > 0),
                dualCoefficients: coef,
                intercept: -dual.rho,
                margins: idx.map((row, t) => (labels[t] ?? 1) * decision(row)),
                dualObjective: dual.objective,
                workingSets: dual.workingSets.map(([i, j]) => [idx[i] ?? i, idx[j] ?? j]),
                steps: dual.steps,
              }),
            )
            pairIndex++
          }
        const params: SvcParams = { kernel: kp, classes: [...classes], vectors, pairs }
        if (m.d === 2) {
          const fitted = est.revive(params)
          const steps =
            binary === undefined ? undefined : smoSteps(binary.dual, binary.labels, rows, kp, classes, fitted)
          const surface = decisionSurfaceOf(fitted, data.x, {
            featureNames: featureNamesOf(data, m.d),
            classNames: classNamesOf(data, classes),
            ...(data.y === undefined ? {} : { y: data.y }),
            levels: [-1, 1],
            ...(steps === undefined || binary === undefined
              ? {}
              : {
                  steps,
                  stepName: 'SMO step',
                  objective: { name: 'dual objective', iterations: binary.dual.steps, values: binary.dual.objective },
                }),
          })
          // The margin records index the same rows the surface carries: give them the rows and the surface.
          for (const [i, r] of exposition.entries())
            if (r.kind === 'margin')
              exposition[i] = {
                ...r,
                ...(surface.rows === undefined ? {} : { rows: surface.rows }),
                ...(classes.length === 2 ? { surface } : {}),
              }
          exposition.push(surface)
        }
        return { params, exposition }
      },
      methods: (p: SvcParams) => svcVerbs(p),
    },
    {
      C: hyper.C ?? 1,
      kernel: hyper.kernel ?? 'rbf',
      gamma: hyper.gamma ?? 'scale',
      degree: hyper.degree ?? 3,
      coef0: hyper.coef0 ?? 0,
      tol: hyper.tol ?? 1e-3,
      maxIter: hyper.maxIter ?? 100000,
      probability: hyper.probability ?? false,
      ...(hyper.classWeight === undefined ? {} : { classWeight: hyper.classWeight }),
    },
  )
  return est
}

// The models at up to 40 of SMO's snapshots (spaced geometrically in the step), for a binary SVC: each a one-pair model
// from the snapshot's dual, the last the fitted model itself, with the support vectors at each step.
function smoSteps(
  dual: DualRun,
  labels: readonly number[],
  rows: readonly (readonly number[])[],
  kernel: KernelParams,
  classes: readonly number[],
  fitted: FittedSvc,
): SurfaceStepsInput | undefined {
  const snaps = dual.snapshots ?? []
  if (snaps.length < 2) return undefined
  const last = snaps[snaps.length - 1]?.step ?? 0
  const chosen = new Set<number>()
  for (const target of geometricSteps(last, 40)) {
    let best = 0
    for (let k = 1; k < snaps.length; k++)
      if (Math.abs((snaps[k]?.step ?? 0) - target) < Math.abs((snaps[best]?.step ?? 0) - target)) best = k
    chosen.add(best)
  }
  const kept = [...chosen].sort((a, b) => a - b).map((k) => snaps[k] as NonNullable<(typeof snaps)[number]>)
  const models = kept.map((s, t) => {
    if (t === kept.length - 1) return fitted
    // Before any support vector (SMO's first step), one row with a zero coefficient keeps the model's shape.
    const active = Array.from(s.alpha.keys()).filter((i) => (s.alpha[i] ?? 0) > 0)
    const support = active.length > 0 ? active : [0]
    const params: SvcParams = {
      kernel,
      classes: [...classes],
      vectors: support.map((i) => [...(rows[i] ?? [])]),
      pairs: [
        {
          positive: 0,
          negative: 1,
          support: support.map((_, a) => a),
          coef: support.map((i) => (s.alpha[i] ?? 0) * (labels[i] ?? 1)),
          intercept: -s.rho,
          platt: null,
        },
      ],
    }
    return { estimator: 'svc', kind: 'classifier' as const, params, ...svcVerbs(params) }
  })
  // Each step's decision function from one kernel matrix per set of rows: f(x) = -(sum_i alpha_i y_i K(x, x_i) - rho)
  // for class 1 (scikit-learn's sign), the matrix computed once and reused by every step.
  const cache = new Map<Tensor, number[]>()
  const decisionAt = (t: number, x: Tensor): Float64Array => {
    let K = cache.get(x)
    if (K === undefined) {
      K = kernelMatrix(kernel, x, rows as number[][])
      cache.set(x, K)
    }
    const s = kept[t] as NonNullable<(typeof kept)[number]>
    const n = rows.length
    const m = K.length / Math.max(1, n)
    const out = new Float64Array(m)
    for (let c = 0; c < m; c++) {
      let f = -s.rho
      for (let i = 0; i < n; i++) {
        const a = s.alpha[i] ?? 0
        if (a > 0) f += a * (labels[i] ?? 1) * (K[c * n + i] ?? 0)
      }
      out[c] = -f
    }
    return out
  }
  return {
    models,
    decisionAt,
    iterations: kept.map((s) => s.step),
    objective: kept.map((s) => s.objective),
    highlight: kept.map((s) => Array.from(s.alpha.keys()).filter((i) => (s.alpha[i] ?? 0) > 0)),
    highlightName: 'support vectors',
  }
}

/** {@link svc} as a class. */
export class SVC extends ExplainedEstimatorClass<SvcHyper, FittedSvc> {
  /** @param hyper - As {@link svc}. */
  constructor(hyper: Partial<SvcHyper> = {}) {
    super(svc(hyper))
  }
}
