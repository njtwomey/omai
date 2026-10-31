/**
 * Learning from weak labels: votes from noisy sources instead of one clean label. Majority vote and the label models
 * that weigh the sources (Dawid-Skene by EM with its one-coin and GLAD forms, an agreement-based model for labelling
 * functions), the noise-transition matrix and confident learning that find what is wrong with given labels, positive-
 * unlabelled risks (unbiased and non-negative) and the Elkan-Noto estimator, learning from label proportions of bags,
 * and multiple-instance SVMs. Votes are a matrix `[items, sources]` of class labels with `-1` where a source abstains;
 * the iterative ones are estimators on the iterative base, one EM or gradient step per step.
 */

import { type Json, logSumExpOf, MdmlError } from '../base/index.ts'
import { logSigmoid } from '../numerics/special/index.ts'
import { type LbfgsState, lbfgsAlgorithm } from '../optim/index.ts'
import { foldIn, type Key, key as makeKey, permutation, uniform } from '../random/index.ts'
import { choice, dict, int, real } from '../space/index.ts'
import {
  add,
  asTensor,
  fromStorage,
  item,
  log,
  matmul,
  maximum,
  mul,
  neg,
  reshape,
  sigmoid,
  softmax,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import { type Algorithm, channels, Log, stateFromJson, stateToJson } from '../trace/index.ts'
import { accuracyOf } from './_classify.ts'
import { confidence } from './_staged.ts'
import { type IterativeEstimator, iterative } from './bases/index.ts'
import { type Estimator, type Fitted, type LearnData, requireTargets } from './estimator.ts'
import { matrixView, takeData } from './validation.ts'

/** The vote of a source that does not label an item. */
export const ABSTAIN = -1

// ---------------------------------------------------------------------------------------------------------------------
// Votes

/** A votes matrix read once: class indices (or -1) row-major `[n, m]`, and the classes. */
type Votes = { readonly n: number; readonly m: number; readonly v: Int32Array; readonly classes: number[] }

function readVotes(votes: TensorLike, who: string, classes?: readonly number[]): Votes {
  const mv = matrixView(votes, who)
  const seen = new Set<number>()
  for (const x of mv.data) if (x !== ABSTAIN) seen.add(x)
  const cls = classes === undefined ? Array.from(seen).sort((a, b) => a - b) : [...classes]
  if (cls.length < 2) throw new MdmlError('value', `${who}: needs at least two classes among the votes`)
  const index = new Map(cls.map((c, i) => [c, i]))
  const v = Int32Array.from(mv.data, (x) => {
    if (x === ABSTAIN) return -1
    const i = index.get(x)
    if (i === undefined) throw new MdmlError('value', `${who}: vote ${x} is not among the classes ${cls.join(', ')}`)
    return i
  })
  return { n: mv.n, m: mv.d, v, classes: cls }
}

/** What {@link labellingMatrixSummary} reports about a votes matrix. */
export interface LabellingSummary {
  /** The fraction of items each source votes on. */
  readonly coverage: number[]
  /** The fraction of items each source votes on that some other source votes on too. */
  readonly overlap: number[]
  /** The fraction of items each source votes on that another source votes differently on. */
  readonly conflict: number[]
  /** The classes each source ever votes for. */
  readonly polarity: number[][]
  /** The fraction of items with at least one vote. */
  readonly totalCoverage: number
  /** The agreement rate of every pair of sources over the items both vote on (NaN when they never overlap). */
  readonly agreement: number[][]
  /** The accuracy of each source on the items it votes on, when the truth is given. */
  readonly accuracy?: number[]
}

/**
 * The standard summary of a labelling matrix (as Snorkel's `LFAnalysis`): per source the coverage, the overlap with the
 * others, the conflicts, the classes it votes for, and the pairwise agreement; with the true labels, its accuracy.
 *
 * @param votes - Votes `[items, sources]` of class labels, `-1` for an abstention.
 * @param options - `classes` (default the labels seen) and `truth` (the true label of each item).
 * @returns The summary.
 * @example
 * ```ts
 * import { labellingMatrixSummary } from 'mdml-engine/learn'
 * const s = labellingMatrixSummary([[0, 0], [1, -1], [0, 1], [-1, -1]])
 * s.coverage // => [0.75, 0.5]
 * s.conflict // => [0.25, 0.25]
 * s.totalCoverage // => 0.75
 * ```
 */
export function labellingMatrixSummary(
  votes: TensorLike,
  options: { readonly classes?: readonly number[]; readonly truth?: TensorLike } = {},
): LabellingSummary {
  const { n, m, v, classes } = readVotes(votes, 'labellingMatrixSummary', options.classes)
  const truth = options.truth === undefined ? undefined : toFlat(asTensor(options.truth))
  const coverage: number[] = []
  const overlap: number[] = []
  const conflict: number[] = []
  const polarity: number[][] = []
  const agreement = Array.from({ length: m }, () => new Array<number>(m).fill(Number.NaN))
  for (let j = 0; j < m; j++) {
    let voted = 0
    let over = 0
    let conf = 0
    const seen = new Set<number>()
    for (let i = 0; i < n; i++) {
      const a = v[i * m + j] ?? -1
      if (a < 0) continue
      voted++
      seen.add(classes[a] ?? a)
      let others = false
      let differs = false
      for (let k = 0; k < m; k++) {
        const b = v[i * m + k] ?? -1
        if (k === j || b < 0) continue
        others = true
        if (b !== a) differs = true
      }
      if (others) over++
      if (differs) conf++
    }
    coverage.push(voted / n)
    overlap.push(voted === 0 ? 0 : over / n)
    conflict.push(voted === 0 ? 0 : conf / n)
    polarity.push(Array.from(seen).sort((a, b) => a - b))
  }
  for (let j = 0; j < m; j++)
    for (let k = 0; k < m; k++) {
      let both = 0
      let same = 0
      for (let i = 0; i < n; i++) {
        const a = v[i * m + j] ?? -1
        const b = v[i * m + k] ?? -1
        if (a >= 0 && b >= 0) {
          both++
          if (a === b) same++
        }
      }
      const row = agreement[j]
      if (row !== undefined) row[k] = both === 0 ? Number.NaN : same / both
    }
  let any = 0
  for (let i = 0; i < n; i++) if (Array.from({ length: m }, (_, j) => v[i * m + j] ?? -1).some((a) => a >= 0)) any++
  const accuracy =
    truth === undefined
      ? undefined
      : Array.from({ length: m }, (_, j) => {
          let voted = 0
          let right = 0
          for (let i = 0; i < n; i++) {
            const a = v[i * m + j] ?? -1
            if (a < 0) continue
            voted++
            if (classes[a] === truth[i]) right++
          }
          return voted === 0 ? Number.NaN : right / voted
        })
  return {
    coverage,
    overlap,
    conflict,
    polarity,
    totalCoverage: any / n,
    agreement,
    ...(accuracy === undefined ? {} : { accuracy }),
  }
}

/** The result of {@link majorityVote}. */
export interface MajorityVoteResult {
  /** The winning class of each item; `-1` for an item with no votes (or a tie under `ties: 'abstain'`). */
  readonly labels: number[]
  /** The weighted vote share of each class `[n][k]`; uniform for an item with no votes. */
  readonly proba: number[][]
  /** Whether each item received a vote. */
  readonly covered: boolean[]
  readonly classes: number[]
}

/**
 * Majority vote over sources that vote or abstain, optionally weighted (the weighted vote with known accuracies, e.g.
 * their log-odds). The vote share of each class is the probability the vote gives.
 *
 * @param votes - Votes `[items, sources]`, `-1` for an abstention.
 * @param options - `classes`; `weights` per source (default 1); `ties`: `abstain` (default), `lowest` class, or `random` (with `key`).
 * @returns The labels, the shares and which items were covered.
 * @example
 * ```ts
 * import { majorityVote } from 'mdml-engine/learn'
 * const r = majorityVote([[0, 0, 1], [1, 1, -1], [0, 1, -1], [-1, -1, -1]])
 * r.labels // => [0, 1, -1, -1]
 * r.covered // => [true, true, true, false]
 * majorityVote([[0, 1, 1]], { weights: [3, 1, 1] }).labels // => [0]
 * ```
 */
export function majorityVote(
  votes: TensorLike,
  options: {
    readonly classes?: readonly number[]
    readonly weights?: readonly number[]
    readonly ties?: 'abstain' | 'lowest' | 'random'
    readonly key?: Key
  } = {},
): MajorityVoteResult {
  const { n, m, v, classes } = readVotes(votes, 'majorityVote', options.classes)
  const k = classes.length
  const w = options.weights ?? new Array<number>(m).fill(1)
  if (w.length !== m) throw new MdmlError('shape', `majorityVote: ${w.length} weights for ${m} sources`)
  const ties = options.ties ?? 'abstain'
  const draws = ties === 'random' ? uniform(options.key ?? makeKey(0), [n]) : undefined
  const labels: number[] = []
  const proba: number[][] = []
  const covered: boolean[] = []
  for (let i = 0; i < n; i++) {
    const tally = new Array<number>(k).fill(0)
    let any = false
    for (let j = 0; j < m; j++) {
      const a = v[i * m + j] ?? -1
      if (a >= 0) {
        tally[a] = (tally[a] ?? 0) + (w[j] ?? 1)
        any = true
      }
    }
    covered.push(any)
    const total = tally.reduce((a, b) => a + b, 0)
    proba.push(total > 0 ? tally.map((t) => t / total) : new Array<number>(k).fill(1 / k))
    if (!any) {
      labels.push(ABSTAIN)
      continue
    }
    const best = Math.max(...tally)
    const tied = tally.map((t, c) => (t === best ? c : -1)).filter((c) => c >= 0)
    if (tied.length === 1) labels.push(classes[tied[0] ?? 0] ?? ABSTAIN)
    else if (ties === 'abstain') labels.push(ABSTAIN)
    else if (ties === 'lowest') labels.push(classes[tied[0] ?? 0] ?? ABSTAIN)
    else
      labels.push(
        classes[tied[Math.min(tied.length - 1, Math.floor((draws?.data[i] ?? 0) * tied.length))] ?? 0] ?? ABSTAIN,
      )
  }
  return { labels, proba, covered, classes }
}

// ---------------------------------------------------------------------------------------------------------------------
// Dawid-Skene

/** The hyperparameters of {@link dawidSkene}. */
export interface DawidSkeneHyper {
  /** `full` confusion matrices (Dawid and Skene 1979), `one-coin` accuracies (Raykar et al. 2010), or `glad` (Whitehill et al. 2009: ability times item easiness). */
  readonly mode: 'full' | 'one-coin' | 'glad'
  /** The classes (default the labels seen in the votes). */
  readonly classes: readonly number[] | null
  /** A Dirichlet pseudo-count added to every confusion entry (and to the one-coin counts): a MAP estimate. */
  readonly smoothing: number
  readonly maxIter: number
  /** Stop when the log-likelihood gains less than this (relative). */
  readonly tol: number
  /** GLAD: the gradient steps of the M-step per EM iteration. */
  readonly mSteps: number
}

/** The state of {@link dawidSkeneAlgorithm}: a model and the posterior it gives, always consistent. */
export type DawidSkeneState = {
  readonly priors: number[]
  /** The confusion matrices `[m][k][k]` flat: entry `(j, true, vote)`. */
  readonly confusions: number[]
  /** GLAD: the ability of each source. */
  readonly alpha: number[]
  /** GLAD: the log inverse difficulty of each item (`beta = exp(logBeta)`). */
  readonly logBeta: number[]
  /** The posterior over each item's true class `[n][k]` flat. */
  readonly posterior: number[]
  /** The marginal log-likelihood of the votes under this model. */
  readonly logLikelihood: number
  readonly change: number
  readonly iter: number
}

/** The fitted parameters of {@link dawidSkene}. */
export type DawidSkeneParams = {
  readonly classes: number[]
  readonly mode: 'full' | 'one-coin' | 'glad'
  readonly priors: number[]
  /** The confusion matrices `[source][true][vote]`; for GLAD at an item of easiness one. */
  readonly confusions: number[][][]
  readonly alpha: number[]
  readonly logBeta: number[]
  readonly posterior: number[][]
  readonly logLikelihood: number
  readonly iterations: number
}

const logOf = (x: number): number => Math.log(Math.max(x, 1e-300))

/** The posterior and the log-likelihood of every item under a model given as confusion log-probabilities. */
function expectation(
  vt: Votes,
  priors: readonly number[],
  logConf: (j: number, item: number, truth: number, vote: number) => number,
): { readonly posterior: Float64Array; readonly logLikelihood: number } {
  const { n, m, v, classes } = vt
  const k = classes.length
  const posterior = new Float64Array(n * k)
  let ll = 0
  const row = new Float64Array(k)
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < k; c++) {
      let s = logOf(priors[c] ?? 0)
      for (let j = 0; j < m; j++) {
        const a = v[i * m + j] ?? -1
        if (a >= 0) s += logConf(j, i, c, a)
      }
      row[c] = s
    }
    const z = logSumExpOf(row)
    ll += z
    for (let c = 0; c < k; c++) posterior[i * k + c] = Math.exp((row[c] ?? 0) - z)
  }
  return { posterior, logLikelihood: ll }
}

const sigmoidOf = (z: number): number => 1 / (1 + Math.exp(-z))
const logSigmoidOf = (z: number): number => (z >= 0 ? -Math.log1p(Math.exp(-z)) : z - Math.log1p(Math.exp(z)))

function majorityPosterior(vt: Votes): Float64Array {
  const { n, m, v, classes } = vt
  const k = classes.length
  const out = new Float64Array(n * k)
  for (let i = 0; i < n; i++) {
    let total = 0
    for (let j = 0; j < m; j++) {
      const a = v[i * m + j] ?? -1
      if (a >= 0) {
        out[i * k + a] = (out[i * k + a] ?? 0) + 1
        total++
      }
    }
    for (let c = 0; c < k; c++) out[i * k + c] = total === 0 ? 1 / k : (out[i * k + c] ?? 0) / total
  }
  return out
}

/** The confusion matrices of a GLAD model at item easiness one, `[m][k][k]` flat. */
function gladConfusions(alpha: readonly number[], k: number): number[] {
  const out = new Array<number>(alpha.length * k * k).fill(0)
  alpha.forEach((a, j) => {
    const right = sigmoidOf(a)
    for (let c = 0; c < k; c++)
      for (let l = 0; l < k; l++) out[(j * k + c) * k + l] = c === l ? right : (1 - right) / (k - 1)
  })
  return out
}

/** The M-step for the full and one-coin models: confusions and priors from a posterior. */
function maximisation(
  vt: Votes,
  posterior: ArrayLike<number>,
  mode: 'full' | 'one-coin',
  smoothing: number,
): { readonly priors: number[]; readonly confusions: number[] } {
  const { n, m, v, classes } = vt
  const k = classes.length
  const priors = new Array<number>(k).fill(0)
  for (let i = 0; i < n; i++) for (let c = 0; c < k; c++) priors[c] = (priors[c] ?? 0) + (posterior[i * k + c] ?? 0) / n
  const confusions = new Array<number>(m * k * k).fill(0)
  for (let j = 0; j < m; j++) {
    if (mode === 'full') {
      const counts = new Array<number>(k * k).fill(smoothing)
      for (let i = 0; i < n; i++) {
        const a = v[i * m + j] ?? -1
        if (a < 0) continue
        for (let c = 0; c < k; c++) counts[c * k + a] = (counts[c * k + a] ?? 0) + (posterior[i * k + c] ?? 0)
      }
      for (let c = 0; c < k; c++) {
        let z = 0
        for (let l = 0; l < k; l++) z += counts[c * k + l] ?? 0
        for (let l = 0; l < k; l++) confusions[(j * k + c) * k + l] = z > 0 ? (counts[c * k + l] ?? 0) / z : 1 / k
      }
    } else {
      let right = smoothing
      let total = 2 * smoothing
      for (let i = 0; i < n; i++) {
        const a = v[i * m + j] ?? -1
        if (a < 0) continue
        right += posterior[i * k + a] ?? 0
        total += 1
      }
      const acc = total > 0 ? right / total : 1 / k
      for (let c = 0; c < k; c++)
        for (let l = 0; l < k; l++) confusions[(j * k + c) * k + l] = c === l ? acc : (1 - acc) / (k - 1)
    }
  }
  return { priors, confusions }
}

/**
 * Dawid-Skene as an algorithm: EM over the true class of every item and the reliability of every source. `full` learns a
 * confusion matrix per source (Dawid and Skene 1979), `one-coin` one accuracy per source with errors uniform over the
 * other classes (Raykar et al. 2010), `glad` an ability per source and an easiness per item, `P(correct) =
 * sigmoid(ability * easiness)` (Whitehill et al. 2009, with gradient steps in the M-step that never lower the expected
 * log-likelihood). Step 0 is the M-step from the majority-vote shares; each step is E then M. Logs `logLikelihood` (which
 * never decreases) and `change`.
 *
 * @param votes - Votes `[items, sources]`, `-1` for an abstention.
 * @param hyper - The mode, smoothing and stopping rule.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { dawidSkeneAlgorithm } from 'mdml-engine/learn'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const votes = [[0, 0, 1], [1, 1, 1], [0, 0, 0], [1, 0, 1], [1, 1, 0], [0, 0, 0]]
 * const alg = dawidSkeneAlgorithm(votes, { mode: 'one-coin', classes: null, smoothing: 0, maxIter: 50, tol: 1e-10, mSteps: 5 })
 * runAlgorithm(alg, { maxSteps: 50 }).state.logLikelihood < 0 // => true
 * ```
 */
export function dawidSkeneAlgorithm(votes: TensorLike, hyper: DawidSkeneHyper): Algorithm<DawidSkeneState> {
  const vt = readVotes(votes, 'dawidSkene', hyper.classes ?? undefined)
  const { n, m, v, classes } = vt
  const k = classes.length
  const logConfOf =
    (confusions: readonly number[]) =>
    (j: number, _i: number, c: number, l: number): number =>
      logOf(confusions[(j * k + c) * k + l] ?? 0)
  // GLAD M-step: gradient ascent on the expected log-likelihood in (alpha, logBeta), by backtracking.
  const gladStep = (
    posterior: ArrayLike<number>,
    alpha: readonly number[],
    logBeta: readonly number[],
  ): { alpha: number[]; logBeta: number[] } => {
    const r = new Float64Array(n * m)
    for (let i = 0; i < n; i++)
      for (let j = 0; j < m; j++) {
        const a = v[i * m + j] ?? -1
        r[i * m + j] = a < 0 ? -1 : (posterior[i * k + a] ?? 0)
      }
    const q = (al: readonly number[], lb: readonly number[]): number => {
      let s = 0
      for (let i = 0; i < n; i++)
        for (let j = 0; j < m; j++) {
          const rij = r[i * m + j] ?? -1
          if (rij < 0) continue
          const z = (al[j] ?? 0) * Math.exp(lb[i] ?? 0)
          s += rij * logSigmoidOf(z) + (1 - rij) * logSigmoidOf(-z)
        }
      return s
    }
    let al = [...alpha]
    let lb = [...logBeta]
    let value = q(al, lb)
    for (let s = 0; s < hyper.mSteps; s++) {
      const ga = new Array<number>(m).fill(0)
      const gb = new Array<number>(n).fill(0)
      for (let i = 0; i < n; i++)
        for (let j = 0; j < m; j++) {
          const rij = r[i * m + j] ?? -1
          if (rij < 0) continue
          const beta = Math.exp(lb[i] ?? 0)
          const z = (al[j] ?? 0) * beta
          const d = rij - sigmoidOf(z)
          ga[j] = (ga[j] ?? 0) + d * beta
          gb[i] = (gb[i] ?? 0) + d * z
        }
      let step = 1
      let moved = false
      for (let tries = 0; tries < 30; tries++) {
        const na = al.map((x, j) => Math.max(-20, Math.min(20, x + step * (ga[j] ?? 0))))
        const nb = lb.map((x, i) => Math.max(-6, Math.min(6, x + step * (gb[i] ?? 0))))
        const nv = q(na, nb)
        if (nv >= value) {
          al = na
          lb = nb
          value = nv
          moved = true
          break
        }
        step /= 2
      }
      if (!moved) break
    }
    return { alpha: al, logBeta: lb }
  }
  const evaluate = (
    priors: number[],
    confusions: number[],
    alpha: number[],
    logBeta: number[],
    previous: DawidSkeneState | undefined,
  ): DawidSkeneState => {
    const e =
      hyper.mode === 'glad'
        ? expectation(vt, priors, (j, i, c, l) => {
            const z = (alpha[j] ?? 0) * Math.exp(logBeta[i] ?? 0)
            return c === l ? logSigmoidOf(z) : logSigmoidOf(-z) - Math.log(k - 1)
          })
        : expectation(vt, priors, logConfOf(confusions))
    return {
      priors,
      confusions,
      alpha,
      logBeta,
      posterior: Array.from(e.posterior),
      logLikelihood: e.logLikelihood,
      change: previous === undefined ? Number.POSITIVE_INFINITY : e.logLikelihood - previous.logLikelihood,
      iter: previous === undefined ? 0 : previous.iter + 1,
    }
  }
  const mstep = (posterior: ArrayLike<number>, prev?: DawidSkeneState): DawidSkeneState => {
    if (hyper.mode === 'glad') {
      const { alpha, logBeta } = gladStep(
        posterior,
        prev?.alpha ?? new Array<number>(m).fill(1),
        prev?.logBeta ?? new Array<number>(n).fill(0),
      )
      const priors = new Array<number>(k).fill(0)
      for (let i = 0; i < n; i++)
        for (let c = 0; c < k; c++) priors[c] = (priors[c] ?? 0) + (posterior[i * k + c] ?? 0) / n
      return evaluate(priors, gladConfusions(alpha, k), alpha, logBeta, prev)
    }
    const { priors, confusions } = maximisation(vt, posterior, hyper.mode, hyper.smoothing)
    return evaluate(priors, confusions, [], [], prev)
  }
  return {
    name: `dawid-skene-${hyper.mode}`,
    config: { mode: hyper.mode, smoothing: hyper.smoothing, items: n, sources: m, classes: k },
    schema: channels({
      metrics: {
        logLikelihood: { description: 'marginal log-likelihood of the votes under the model', direction: 'higher' },
        change: { description: 'gain in log-likelihood this step' },
      },
    }),
    init: () => {
      const start =
        hyper.mode === 'glad'
          ? evaluate(
              new Array<number>(k).fill(1 / k),
              gladConfusions(new Array<number>(m).fill(1), k),
              new Array<number>(m).fill(1),
              new Array<number>(n).fill(0),
              undefined,
            )
          : mstep(majorityPosterior(vt))
      if (hyper.mode === 'glad') return start
      return { ...start, change: Number.POSITIVE_INFINITY }
    },
    step: (s) => mstep(s.posterior, s),
    done: (s) =>
      Number.isFinite(s.change) && Math.abs(s.change) < hyper.tol * (Math.abs(s.logLikelihood) + 1)
        ? 'converged'
        : s.iter >= hyper.maxIter
          ? 'completed'
          : undefined,
    observe: (s) => ({
      metrics: { logLikelihood: s.logLikelihood, ...(Number.isFinite(s.change) ? { change: s.change } : {}) },
    }),
    serialise: (s) => stateToJson({ ...s, change: Number.isFinite(s.change) ? s.change : 1e300 }),
    deserialise: (j) => {
      const s = stateFromJson(j) as DawidSkeneState
      return { ...s, change: s.change >= 1e300 ? Number.POSITIVE_INFINITY : s.change }
    },
  }
}

/** What a fitted Dawid-Skene model offers. */
export interface DawidSkeneVerbs {
  /** The most probable true class of each row of new votes. */
  readonly predict: (x: TensorLike) => Tensor
  /** The posterior over the true class given new votes `[m, k]`. */
  readonly predictProba: (x: TensorLike) => Tensor
  /** Accuracy of the labels against `data.y`. */
  readonly score: (data: LearnData) => number
  readonly classes: readonly number[]
  readonly nFeaturesIn: number
  /** The confusion matrices `[source][true][vote]`. */
  readonly confusions: readonly (readonly (readonly number[])[])[]
  readonly priors: readonly number[]
  /** The posterior over the true class of each training item. */
  readonly posterior: readonly (readonly number[])[]
  /** The most probable true class of each training item. */
  readonly labels: readonly number[]
  readonly logLikelihood: number
  /** GLAD: the ability of each source (empty otherwise). */
  readonly abilities: readonly number[]
  /** GLAD: the easiness `exp(logBeta)` of each training item (empty otherwise). */
  readonly easiness: readonly number[]
}

function posteriorFor(
  p: { priors: readonly number[]; confusions: readonly (readonly (readonly number[])[])[]; classes: readonly number[] },
  x: TensorLike,
): Float64Array {
  const vt = readVotes(x, 'dawidSkene', p.classes)
  if (vt.m !== p.confusions.length)
    throw new MdmlError('shape', `dawidSkene: ${vt.m} sources, fitted on ${p.confusions.length}`)
  return expectation(vt, p.priors, (j, _i, c, l) => logOf(p.confusions[j]?.[c]?.[l] ?? 0)).posterior
}

const argmaxIndex = (a: ArrayLike<number>, from: number, k: number): number => {
  let best = 0
  for (let c = 1; c < k; c++) if ((a[from + c] ?? 0) > (a[from + best] ?? 0)) best = c
  return best
}

function dawidSkeneVerbs(p: DawidSkeneParams): DawidSkeneVerbs {
  const k = p.classes.length
  const predict = (x: TensorLike): Tensor => {
    const post = posteriorFor(p, x)
    const n = post.length / k
    return fromStorage(
      Float64Array.from({ length: n }, (_, i) => p.classes[argmaxIndex(post, i * k, k)] ?? 0),
      [n],
      'float64',
    )
  }
  return {
    predict,
    predictProba: (x) => {
      const post = posteriorFor(p, x)
      return fromStorage(post, [post.length / k, k], 'float64')
    },
    score: (data) => {
      const truth = toFlat(asTensor(requireTargets(data, 'dawidSkene.score')))
      const got = toFlat(predict(data.x))
      return got.reduce((a, g, i) => a + (g === truth[i] ? 1 : 0), 0) / got.length
    },
    classes: p.classes,
    nFeaturesIn: p.confusions.length,
    confusions: p.confusions,
    priors: p.priors,
    posterior: p.posterior,
    labels: p.posterior.map((row) => p.classes[argmaxIndex(row, 0, k)] ?? 0),
    logLikelihood: p.logLikelihood,
    abilities: p.alpha,
    easiness: p.logBeta.map((b) => Math.exp(b)),
  }
}

const DS_DEFAULTS: DawidSkeneHyper = {
  mode: 'full',
  classes: null,
  smoothing: 0,
  maxIter: 200,
  tol: 1e-10,
  mSteps: 5,
}

/**
 * The Dawid-Skene label model as an estimator whose `x` is the votes matrix (`-1` abstains): EM over the unknown true
 * labels and the reliability of the sources, in three modes (`full` confusion matrices, `one-coin`, `glad`). The fitted
 * value has the `confusions`, `priors`, the training `posterior` and `labels`, and `predict`/`predictProba` label new
 * rows of votes with the learned model. The log-likelihood never decreases from step to step.
 *
 * @param hyper - `mode`, `classes`, `smoothing`, `maxIter`, `tol`, `mSteps`.
 * @returns The estimator.
 * @example
 * ```ts
 * import { dawidSkene } from 'mdml-engine/learn'
 * import { toFlat } from 'mdml-engine/tensor'
 * const votes = [[0, 0, 1], [1, 1, 1], [0, 0, 0], [1, 0, 1], [1, 1, 0], [0, 0, 0]]
 * const f = dawidSkene({ mode: 'one-coin' }).fit({ x: votes })
 * f.labels // => [0, 1, 0, 1, 1, 0]
 * toFlat(f.predict([[0, 0, 0], [1, 1, 1]])) // => [0, 1]
 * ```
 */
export function dawidSkene(
  hyper: Partial<DawidSkeneHyper> = {},
): IterativeEstimator<DawidSkeneHyper, DawidSkeneState, DawidSkeneParams, DawidSkeneVerbs> {
  const h = { ...DS_DEFAULTS, ...hyper }
  return iterative<DawidSkeneHyper, DawidSkeneState, DawidSkeneParams, DawidSkeneVerbs>(
    {
      name: 'dawid-skene',
      kind: 'other',
      defaults: h,
      hyperSpace: (c) =>
        dict({
          mode: choice(['full', 'one-coin', 'glad'] as const, { default: c.mode, label: 'model' }),
          smoothing: real(0, 5, { default: c.smoothing, label: 'smoothing' }),
        }),
      algorithm: (data, c) => dawidSkeneAlgorithm(data.x, c),
      maxSteps: (c) => c.maxIter + 1,
      params: (s, data, c) => {
        const vt = readVotes(data.x, 'dawidSkene', c.classes ?? undefined)
        const k = vt.classes.length
        const confusions = Array.from({ length: vt.m }, (_, j) =>
          Array.from({ length: k }, (_, a) => s.confusions.slice((j * k + a) * k, (j * k + a + 1) * k)),
        )
        return {
          classes: vt.classes,
          mode: c.mode,
          priors: s.priors,
          confusions,
          alpha: s.alpha,
          logBeta: s.logBeta,
          posterior: Array.from({ length: vt.n }, (_, i) => s.posterior.slice(i * k, (i + 1) * k)),
          logLikelihood: s.logLikelihood,
          iterations: s.iter,
        }
      },
      verbs: (p) => dawidSkeneVerbs(p),
      logs: { 'solver/logLikelihood': Log(), 'solver/change': Log() },
    },
    h,
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// The agreement label model

/** The hyperparameters of {@link labelModel}. */
export interface LabelModelHyper {
  readonly classes: readonly number[] | null
  /** Pairs of sources known to be correlated, left out of the agreement moments the accuracies are fitted to. */
  readonly ignore: readonly (readonly [number, number])[]
  /** The class balance (default estimated from the posterior). */
  readonly classBalance: readonly number[] | null
  readonly maxIter: number
  readonly tol: number
}

/** The state of {@link labelModelAlgorithm}: the centred accuracy of every source and the fit of the moments. */
export type LabelModelState = {
  /** `(K a - 1) / (K - 1)` for each source's accuracy `a`: the quantity whose products the pairwise agreements give. */
  readonly c: number[]
  /** The weighted squared error between the observed agreement moments and `c_i c_j`. */
  readonly objective: number
  readonly change: number
  readonly iter: number
}

/** The fitted parameters of {@link labelModel}. */
export type LabelModelParams = {
  readonly classes: number[]
  /** The estimated accuracy of each source on the items it votes on. */
  readonly accuracies: number[]
  readonly priors: number[]
  /** The excess agreement of every pair over what the other sources predict for it: large entries mark correlated sources. */
  readonly residuals: number[][]
  /** The probabilistic label of every training item `[n][k]`. */
  readonly posterior: number[][]
  readonly objective: number
  readonly iterations: number
}

/** The pairwise moments `m_ij = (K p_ij - 1) / (K - 1)` of the agreement rates, and the number of items both vote on. */
function agreementMoments(
  vt: Votes,
  ignore: readonly (readonly [number, number])[],
): { moment: Float64Array; count: Float64Array } {
  const { n, m, v, classes } = vt
  const k = classes.length
  const moment = new Float64Array(m * m)
  const count = new Float64Array(m * m)
  const skip = new Set(ignore.flatMap(([a, b]) => [a * m + b, b * m + a]))
  for (let a = 0; a < m; a++)
    for (let b = 0; b < m; b++) {
      if (a === b || skip.has(a * m + b)) continue
      let both = 0
      let same = 0
      for (let i = 0; i < n; i++) {
        const x = v[i * m + a] ?? -1
        const y = v[i * m + b] ?? -1
        if (x >= 0 && y >= 0) {
          both++
          if (x === y) same++
        }
      }
      count[a * m + b] = both
      moment[a * m + b] = both === 0 ? 0 : ((k * same) / both - 1) / (k - 1)
    }
  return { moment, count }
}

/**
 * The agreement label model as an algorithm: under conditionally independent sources whose errors are uniform over the
 * wrong classes, the agreement rate of two sources is `1/K + (K-1)/K c_i c_j` with `c = (K a - 1)/(K - 1)` the centred
 * accuracy, so the accuracies follow from the agreements alone, with no true labels (the triplet method of Fu et al.
 * 2020, refined by the matrix-completion objective of Ratner et al. 2019). Step 0 is the median of the triplet estimates
 * `c_i^2 = m_ij m_ik / m_jk`; each step is an exact coordinate minimisation of `sum n_ij (m_ij - c_i c_j)^2`. Logs
 * `objective`, which never increases.
 *
 * @param votes - Votes `[items, sources]`, `-1` for an abstention.
 * @param hyper - The classes, pairs to ignore, the class balance and the stopping rule.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { labelModelAlgorithm } from 'mdml-engine/learn'
 * const votes = [[0, 0, 0], [1, 1, 1], [0, 0, 1], [1, 1, 0], [0, 0, 0], [1, 1, 1]]
 * labelModelAlgorithm(votes, { classes: null, ignore: [], classBalance: null, maxIter: 50, tol: 1e-12 }).name // => 'label-model'
 * ```
 */
export function labelModelAlgorithm(votes: TensorLike, hyper: LabelModelHyper): Algorithm<LabelModelState> {
  const vt = readVotes(votes, 'labelModel', hyper.classes ?? undefined)
  const m = vt.m
  const { moment, count } = agreementMoments(vt, hyper.ignore)
  const objectiveOf = (c: readonly number[]): number => {
    let num = 0
    let den = 0
    for (let a = 0; a < m; a++)
      for (let b = a + 1; b < m; b++) {
        const w = count[a * m + b] ?? 0
        num += w * ((moment[a * m + b] ?? 0) - (c[a] ?? 0) * (c[b] ?? 0)) ** 2
        den += w
      }
    return den === 0 ? 0 : num / den
  }
  const clampC = (x: number): number => Math.max(1e-3, Math.min(1, x))
  const triplet = (a: number): number => {
    const estimates: number[] = []
    for (let b = 0; b < m; b++)
      for (let c = b + 1; c < m; c++) {
        if (b === a || c === a) continue
        const mab = moment[a * m + b] ?? 0
        const mac = moment[a * m + c] ?? 0
        const mbc = moment[b * m + c] ?? 0
        if (
          (count[a * m + b] ?? 0) > 0 &&
          (count[a * m + c] ?? 0) > 0 &&
          (count[b * m + c] ?? 0) > 0 &&
          mab > 0 &&
          mac > 0 &&
          mbc > 0
        )
          estimates.push(Math.sqrt((mab * mac) / mbc))
      }
    if (estimates.length === 0) {
      const direct = Array.from({ length: m }, (_, b) => (b === a ? 0 : Math.max(0, moment[a * m + b] ?? 0)))
      return clampC(Math.sqrt(Math.max(...direct, 1e-6)))
    }
    estimates.sort((x, y) => x - y)
    return clampC(estimates[Math.floor(estimates.length / 2)] ?? 0.5)
  }
  return {
    name: 'label-model',
    config: { sources: m, classes: vt.classes.length },
    schema: channels({
      metrics: {
        objective: { description: 'weighted squared error of the pairwise agreement moments', direction: 'lower' },
        change: { description: 'largest change of a source’s centred accuracy this step' },
      },
    }),
    init: () => {
      const c = Array.from({ length: m }, (_, a) => triplet(a))
      return { c, objective: objectiveOf(c), change: Number.POSITIVE_INFINITY, iter: 0 }
    },
    step: (s) => {
      const c = [...s.c]
      let change = 0
      for (let a = 0; a < m; a++) {
        let num = 0
        let den = 0
        for (let b = 0; b < m; b++) {
          if (b === a) continue
          const w = count[a * m + b] ?? 0
          num += w * (moment[a * m + b] ?? 0) * (c[b] ?? 0)
          den += w * (c[b] ?? 0) ** 2
        }
        const next = den === 0 ? (c[a] ?? 0) : clampC(num / den)
        change = Math.max(change, Math.abs(next - (c[a] ?? 0)))
        c[a] = next
      }
      return { c, objective: objectiveOf(c), change, iter: s.iter + 1 }
    },
    done: (s) => (s.change < hyper.tol ? 'converged' : s.iter >= hyper.maxIter ? 'completed' : undefined),
    observe: (s) => ({
      metrics: { objective: s.objective, ...(Number.isFinite(s.change) ? { change: s.change } : {}) },
    }),
    serialise: (s) => stateToJson({ ...s, change: Number.isFinite(s.change) ? s.change : 1e300 }),
    deserialise: (j) => {
      const s = stateFromJson(j) as LabelModelState
      return { ...s, change: s.change >= 1e300 ? Number.POSITIVE_INFINITY : s.change }
    },
  }
}

/** The posterior over classes of rows of votes under per-source accuracies and priors (errors uniform over the others). */
function accuracyPosterior(vt: Votes, accuracies: readonly number[], priors: readonly number[]): Float64Array {
  const k = vt.classes.length
  return expectation(vt, priors, (j, _i, c, l) => {
    const a = Math.min(1 - 1e-9, Math.max(1e-9, accuracies[j] ?? 0.5))
    return c === l ? Math.log(a) : Math.log((1 - a) / (k - 1))
  }).posterior
}

/** What a fitted label model offers. */
export interface LabelModelVerbs {
  readonly predict: (x: TensorLike) => Tensor
  readonly predictProba: (x: TensorLike) => Tensor
  readonly score: (data: LearnData) => number
  readonly classes: readonly number[]
  readonly nFeaturesIn: number
  readonly accuracies: readonly number[]
  readonly priors: readonly number[]
  readonly residuals: readonly (readonly number[])[]
  /** The probabilistic labels of the training items `[n][k]`. */
  readonly posterior: readonly (readonly number[])[]
}

const LM_DEFAULTS: LabelModelHyper = { classes: null, ignore: [], classBalance: null, maxIter: 200, tol: 1e-10 }

/**
 * The data-programming label model as an estimator whose `x` is the votes of labelling functions (`-1` abstains): the
 * accuracy of each function learnt from the agreements alone (see {@link labelModelAlgorithm}), then the probabilistic
 * label of every item from the accuracies and the class balance. The fitted value's `residuals` are the excess agreement
 * of each pair beyond what the other sources predict for it: the dependency structure to look at.
 *
 * @param hyper - `classes`, `ignore` (known correlated pairs), `classBalance`, `maxIter`, `tol`.
 * @returns The estimator.
 * @example
 * ```ts
 * import { labelModel } from 'mdml-engine/learn'
 * const votes = [[0, 0, 0], [1, 1, 1], [0, 0, 1], [1, 1, 0], [0, 0, 0], [1, 1, 1], [0, 1, 0], [1, 0, 1]]
 * const f = labelModel().fit({ x: votes })
 * f.accuracies.length // => 3
 * f.posterior.length // => 8
 * ```
 */
export function labelModel(
  hyper: Partial<LabelModelHyper> = {},
): IterativeEstimator<LabelModelHyper, LabelModelState, LabelModelParams, LabelModelVerbs> {
  const h = { ...LM_DEFAULTS, ...hyper }
  return iterative<LabelModelHyper, LabelModelState, LabelModelParams, LabelModelVerbs>(
    {
      name: 'label-model',
      kind: 'other',
      defaults: h,
      algorithm: (data, c) => labelModelAlgorithm(data.x, c),
      maxSteps: (c) => c.maxIter + 1,
      params: (s, data, c) => {
        const vt = readVotes(data.x, 'labelModel', c.classes ?? undefined)
        const k = vt.classes.length
        const accuracies = s.c.map((ci) => (1 + (k - 1) * ci) / k)
        let priors: number[] = c.classBalance === null ? new Array<number>(k).fill(1 / k) : [...c.classBalance]
        if (c.classBalance === null) {
          for (let it = 0; it < 25; it++) {
            const post = accuracyPosterior(vt, accuracies, priors)
            priors = priors.map((_, a) => {
              let sum = 0
              for (let i = 0; i < vt.n; i++) sum += post[i * k + a] ?? 0
              return sum / vt.n
            })
          }
        }
        const { moment, count } = agreementMoments(vt, [])
        // The excess agreement of a pair over what the other sources predict for it: each centred accuracy by the median
        // of triplet estimates that leave the pair's other member out, so a source that copies another cannot explain
        // the copy away (a plain fit would raise both accuracies to account for it).
        const centred = (a: number, skip: number): number => {
          const est: number[] = []
          for (let k = 0; k < vt.m; k++)
            for (let l = k + 1; l < vt.m; l++) {
              if ([a, skip].includes(k) || [a, skip].includes(l)) continue
              const mak = moment[a * vt.m + k] ?? 0
              const mal = moment[a * vt.m + l] ?? 0
              const mkl = moment[k * vt.m + l] ?? 0
              if (mak > 0 && mal > 0 && mkl > 0) est.push(Math.sqrt((mak * mal) / mkl))
            }
          if (est.length === 0) return s.c[a] ?? 0
          est.sort((x, y) => x - y)
          return Math.min(1, est[Math.floor(est.length / 2)] ?? 0)
        }
        const residuals = Array.from({ length: vt.m }, (_, a) =>
          Array.from({ length: vt.m }, (_, b) =>
            a === b || (count[a * vt.m + b] ?? 0) === 0
              ? 0
              : (moment[a * vt.m + b] ?? 0) - centred(a, b) * centred(b, a),
          ),
        )
        const post = accuracyPosterior(vt, accuracies, priors)
        const posterior = Array.from({ length: vt.n }, (_, i) => Array.from(post.subarray(i * k, (i + 1) * k)))
        return {
          classes: vt.classes,
          accuracies,
          priors,
          residuals,
          posterior,
          objective: s.objective,
          iterations: s.iter,
        }
      },
      verbs: (p) => {
        const k = p.classes.length
        const post = (x: TensorLike): Float64Array =>
          accuracyPosterior(readVotes(x, 'labelModel', p.classes), p.accuracies, p.priors)
        const predict = (x: TensorLike): Tensor => {
          const q = post(x)
          return fromStorage(
            Float64Array.from({ length: q.length / k }, (_, i) => p.classes[argmaxIndex(q, i * k, k)] ?? 0),
            [q.length / k],
            'float64',
          )
        }
        return {
          predict,
          predictProba: (x) => {
            const q = post(x)
            return fromStorage(q, [q.length / k, k], 'float64')
          },
          score: accuracyOf(predict),
          classes: p.classes,
          nFeaturesIn: p.accuracies.length,
          accuracies: p.accuracies,
          priors: p.priors,
          residuals: p.residuals,
          posterior: p.posterior,
        }
      },
      logs: { 'solver/objective': Log(), 'solver/change': Log() },
    },
    h,
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// Noise transition and confident learning

/** Predicted class probabilities `[n, k]` and the given labels, read once. */
function readProba(
  proba: TensorLike,
  labels: TensorLike,
  who: string,
): { n: number; k: number; p: Float64Array; y: Int32Array } {
  const mv = matrixView(proba, who)
  const y = toFlat(asTensor(labels))
  if (y.length !== mv.n) throw new MdmlError('shape', `${who}: ${mv.n} probability rows but ${y.length} labels`)
  const yi = Int32Array.from(y, (c) => {
    if (!Number.isInteger(c) || c < 0 || c >= mv.d)
      throw new MdmlError('value', `${who}: label ${c} is not a class index in [0, ${mv.d})`)
    return c
  })
  return { n: mv.n, k: mv.d, p: mv.data, y: yi }
}

/**
 * The noise-transition matrix `T[i][j] = P(given label j | true class i)` from predicted probabilities of the *given*
 * labels (out of sample). `anchor` (Patrini et al. 2017): the row of class `i` is the predicted distribution at the point
 * most confident of class `i` (the `percentile`-th percentile of the class-`i` column, to resist outliers); `confident`:
 * the calibrated confident joint of Northcutt et al., normalised over the given label for each true class.
 *
 * @param proba - Out-of-sample probabilities `[n, k]` of the given labels.
 * @param labels - The given (noisy) labels as class indices `0..k-1`.
 * @param options - `method` (default `anchor`) and `percentile` (default 97, anchor only).
 * @returns The row-stochastic matrix `[k][k]`.
 * @example
 * ```ts
 * import { estimateNoiseTransition } from 'mdml-engine/learn'
 * const proba = [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]]
 * estimateNoiseTransition(proba, [0, 0, 1, 1], { percentile: 100 }) // => [[0.9, 0.1], [0.1, 0.9]]
 * ```
 */
export function estimateNoiseTransition(
  proba: TensorLike,
  labels: TensorLike,
  options: { readonly method?: 'anchor' | 'confident'; readonly percentile?: number } = {},
): number[][] {
  const { n, k, p } = readProba(proba, labels, 'estimateNoiseTransition')
  if ((options.method ?? 'anchor') === 'confident') {
    // Q[given][true] -> T[true][given] = Q[given][true] / sum over given of Q[given][true].
    const q = confidentJoint(proba, labels).joint
    return Array.from({ length: k }, (_, t) => {
      const z = q.reduce((a, row) => a + (row[t] ?? 0), 0)
      return Array.from({ length: k }, (_, g) => (z > 0 ? (q[g]?.[t] ?? 0) / z : 1 / k))
    })
  }
  const pct = (options.percentile ?? 97) / 100
  return Array.from({ length: k }, (_, c) => {
    const order = Array.from({ length: n }, (_, i) => i).sort(
      (a, b) => (p[a * k + c] ?? 0) - (p[b * k + c] ?? 0) || a - b,
    )
    const anchor = order[Math.min(n - 1, Math.max(0, Math.ceil(pct * n) - 1))] ?? 0
    const row = Array.from({ length: k }, (_, j) => p[anchor * k + j] ?? 0)
    const z = row.reduce((a, b) => a + b, 0)
    return row.map((x) => (z > 0 ? x / z : 1 / k))
  })
}

/** The result of {@link confidentJoint}. */
export interface ConfidentJoint {
  /** The counts `C[given][true]` of items confidently belonging to class `true` among those labelled `given`. */
  readonly counts: number[][]
  /** The per-class thresholds: the mean self-confidence of the items given that label. */
  readonly thresholds: number[]
  /** The calibrated joint distribution `Q[given][true]` of given and true labels (sums to one). */
  readonly joint: number[][]
  /** For every item, the class it is confidently guessed to belong to, or -1 when no class clears its threshold. */
  readonly guess: number[]
}

/**
 * The confident joint of confident learning (Northcutt, Jiang and Chuang 2021): with `t_j` the mean predicted probability
 * of class `j` among the items labelled `j`, an item counts toward `C[given][true]` where `true` is the most probable class
 * among those whose probability reaches their threshold. Calibrated so each row sums to the number of items given that
 * label and the whole to `n`, then divided by `n`, it estimates the joint distribution of given and true labels.
 *
 * @param proba - Out-of-sample class probabilities `[n, k]`.
 * @param labels - The given labels as class indices `0..k-1`.
 * @returns The counts, thresholds, calibrated joint and per-item guesses.
 * @example
 * ```ts
 * import { confidentJoint } from 'mdml-engine/learn'
 * const proba = [[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.2, 0.8], [0.1, 0.9]]
 * const r = confidentJoint(proba, [0, 0, 0, 1, 1])
 * r.counts // => [[2, 0], [0, 1]]
 * r.thresholds // => [0.6666666666666666, 0.85]
 * ```
 */
export function confidentJoint(proba: TensorLike, labels: TensorLike): ConfidentJoint {
  const { n, k, p, y } = readProba(proba, labels, 'confidentJoint')
  const sum = new Array<number>(k).fill(0)
  const size = new Array<number>(k).fill(0)
  for (let i = 0; i < n; i++) {
    sum[y[i] ?? 0] = (sum[y[i] ?? 0] ?? 0) + (p[i * k + (y[i] ?? 0)] ?? 0)
    size[y[i] ?? 0] = (size[y[i] ?? 0] ?? 0) + 1
  }
  const thresholds = sum.map((s, c) => ((size[c] ?? 0) === 0 ? 1 : s / (size[c] ?? 1)))
  const counts = Array.from({ length: k }, () => new Array<number>(k).fill(0))
  const guess: number[] = []
  for (let i = 0; i < n; i++) {
    let best = -1
    for (let c = 0; c < k; c++)
      if (
        (p[i * k + c] ?? 0) >= (thresholds[c] ?? 1) - 1e-6 &&
        (best < 0 || (p[i * k + c] ?? 0) > (p[i * k + best] ?? 0))
      )
        best = c
    guess.push(best)
    const given = counts[y[i] ?? 0]
    if (best >= 0 && given !== undefined) given[best] = (given[best] ?? 0) + 1
  }
  const calibrated = counts.map((row, g) => {
    const z = row.reduce((a, b) => a + b, 0)
    return row.map((x) => (z > 0 ? (x / z) * (size[g] ?? 0) : 0))
  })
  const total = calibrated.reduce((a, row) => a + row.reduce((b, c) => b + c, 0), 0)
  const joint = calibrated.map((row) => row.map((x) => (total > 0 ? x / total : 0)))
  return { counts, thresholds, joint, guess }
}

/** The scores by which {@link labelQuality} ranks items. */
export type LabelQualityMethod = 'self_confidence' | 'normalized_margin'

/**
 * The quality of every given label from the predicted probabilities, low meaning suspect: the probability of the given
 * label (`self_confidence`), or that minus the largest probability of any other class (`normalized_margin`).
 *
 * @param proba - Out-of-sample class probabilities `[n, k]`.
 * @param labels - The given labels as class indices.
 * @param method - The score (default `self_confidence`).
 * @returns One score per item.
 * @example
 * ```ts
 * import { labelQuality } from 'mdml-engine/learn'
 * labelQuality([[0.9, 0.1], [0.2, 0.8]], [0, 0]) // => [0.9, 0.2]
 * labelQuality([[0.9, 0.1], [0.2, 0.8]], [0, 0], 'normalized_margin') // => [0.8, -0.6000000000000001]
 * ```
 */
export function labelQuality(
  proba: TensorLike,
  labels: TensorLike,
  method: LabelQualityMethod = 'self_confidence',
): number[] {
  const { n, k, p, y } = readProba(proba, labels, 'labelQuality')
  return Array.from({ length: n }, (_, i) => {
    const self = p[i * k + (y[i] ?? 0)] ?? 0
    if (method === 'self_confidence') return self
    let other = Number.NEGATIVE_INFINITY
    for (let c = 0; c < k; c++) if (c !== y[i]) other = Math.max(other, p[i * k + c] ?? 0)
    return self - other
  })
}

/** How {@link findLabelIssues} chooses the items to flag. */
export type LabelIssueFilter =
  | 'prune_by_noise_rate'
  | 'prune_by_class'
  | 'both'
  | 'confident_joint'
  | 'low_self_confidence'
  | 'low_normalized_margin'

/**
 * The items whose given label is probably wrong (confident learning), worst first. `prune_by_noise_rate`: for each pair
 * (given `i`, true `j`) remove as many of the items labelled `i` as the calibrated confident joint says, those with the
 * largest margin `p_j - p_i` first; `prune_by_class`: per given class remove its off-diagonal count, lowest self-confidence
 * first; `both`: the intersection of the two; `confident_joint`: the items counted off the diagonal; `low_self_confidence`
 * and `low_normalized_margin`: as many items as the confident joint has off-diagonal counts, by that score.
 *
 * @param proba - Out-of-sample class probabilities `[n, k]`.
 * @param labels - The given labels as class indices.
 * @param filter - The rule (default `prune_by_noise_rate`).
 * @returns The indices of the flagged items, ordered by increasing `self_confidence`.
 * @example
 * ```ts
 * import { findLabelIssues } from 'mdml-engine/learn'
 * const proba = [[0.97, 0.03], [0.95, 0.05], [0.96, 0.04], [0.1, 0.9], [0.05, 0.95], [0.98, 0.02]]
 * findLabelIssues(proba, [0, 0, 0, 1, 1, 1], 'confident_joint') // => [5]
 * ```
 */
export function findLabelIssues(
  proba: TensorLike,
  labels: TensorLike,
  filter: LabelIssueFilter = 'prune_by_noise_rate',
): number[] {
  const { n, k, p, y } = readProba(proba, labels, 'findLabelIssues')
  const cj = confidentJoint(proba, labels)
  const self = (i: number): number => p[i * k + (y[i] ?? 0)] ?? 0
  const given = (c: number): number[] => Array.from({ length: n }, (_, i) => i).filter((i) => y[i] === c)
  const offDiagonal = cj.counts.reduce((a, row, g) => a + row.reduce((b, c, t) => b + (t === g ? 0 : c), 0), 0)
  const calibratedCounts = cj.joint.map((row) => row.map((q) => Math.round(q * n)))
  const byNoiseRate = (): Set<number> => {
    const out = new Set<number>()
    for (let g = 0; g < k; g++)
      for (let t = 0; t < k; t++) {
        if (t === g) continue
        const rows = given(g).sort(
          (a, b) => (p[b * k + t] ?? 0) - (p[b * k + g] ?? 0) - ((p[a * k + t] ?? 0) - (p[a * k + g] ?? 0)) || a - b,
        )
        for (const i of rows.slice(0, calibratedCounts[g]?.[t] ?? 0)) out.add(i)
      }
    return out
  }
  const byClass = (): Set<number> => {
    const out = new Set<number>()
    for (let g = 0; g < k; g++) {
      const drop = (calibratedCounts[g] ?? []).reduce((a, c, t) => a + (t === g ? 0 : c), 0)
      for (const i of given(g)
        .sort((a, b) => self(a) - self(b) || a - b)
        .slice(0, drop))
        out.add(i)
    }
    return out
  }
  let chosen: Set<number>
  switch (filter) {
    case 'prune_by_noise_rate':
      chosen = byNoiseRate()
      break
    case 'prune_by_class':
      chosen = byClass()
      break
    case 'both': {
      const a = byNoiseRate()
      chosen = new Set(Array.from(byClass()).filter((i) => a.has(i)))
      break
    }
    case 'confident_joint':
      chosen = new Set(
        Array.from({ length: n }, (_, i) => i).filter((i) => (cj.guess[i] ?? -1) >= 0 && cj.guess[i] !== y[i]),
      )
      break
    default: {
      const method = filter === 'low_self_confidence' ? 'self_confidence' : 'normalized_margin'
      const q = labelQuality(proba, labels, method)
      chosen = new Set(
        Array.from({ length: n }, (_, i) => i)
          .sort((a, b) => (q[a] ?? 0) - (q[b] ?? 0) || a - b)
          .slice(0, offDiagonal),
      )
    }
  }
  return Array.from(chosen).sort((a, b) => self(a) - self(b) || a - b)
}

// ---------------------------------------------------------------------------------------------------------------------
// Positive-unlabelled learning

/** The surrogate losses {@link puRisk} offers, as functions of the margin `y f(x)`. */
export type PuLoss = 'logistic' | 'sigmoid' | 'squared'

const marginLoss = (loss: PuLoss, z: Value, sign: 1 | -1): Value => {
  const m = sign === 1 ? z : neg(z)
  switch (loss) {
    case 'logistic':
      return neg(logSigmoid(m))
    case 'sigmoid':
      return sigmoid(neg(m))
    case 'squared': {
      const e = sub(1, m)
      return mul(mul(e, e), 0.25)
    }
  }
}

/** The pieces of a positive-unlabelled risk, differentiable in the scores. */
export interface PuRisk {
  /** The risk estimate: unbiased (uPU) or with the negative part clamped at zero (nnPU). */
  readonly risk: Value
  /** `pi E_P[l(f, +1)]`: the risk on the positives labelled as positive. */
  readonly positiveRisk: Value
  /** `E_U[l(f, -1)] - pi E_P[l(f, -1)]` before any clamping: the estimate of the negatives' risk, which can go negative. */
  readonly negativeRisk: Value
}

/**
 * The positive-unlabelled risk of scores `f(x)` (du Plessis et al. 2015, uPU; Kiryo et al. 2017, nnPU): with `pi` the
 * class prior, `R = pi E_P[l(f,+1)] + E_U[l(f,-1)] - pi E_P[l(f,-1)]`, an unbiased estimate of the risk on fully labelled
 * data from positives and an unlabelled mix. The last term can make the estimate negative for flexible models; `nonNegative`
 * clamps the negatives' part at zero (`max(0, E_U[l(f,-1)] - pi E_P[l(f,-1)])`). Differentiable in `scores`.
 *
 * @param scores - The scores `f(x)`, one per row.
 * @param labelledPositive - 1 for a labelled positive row, 0 for an unlabelled one.
 * @param prior - The class prior `pi` of the positives in the unlabelled data.
 * @param options - `loss` (default `logistic`) and `nonNegative` (default false).
 * @returns The risk and its two parts.
 * @example
 * ```ts
 * import { puRisk } from 'mdml-engine/learn'
 * import { item } from 'mdml-engine/tensor'
 * const r = puRisk([2, 2, -2, -2], [1, 0, 0, 0], 0.5, { loss: 'squared' })
 * item(r.risk).toFixed(4) // => '-0.0833'
 * item(puRisk([2, 2, -2, -2], [1, 0, 0, 0], 0.5, { loss: 'squared', nonNegative: true }).risk).toFixed(4) // => '0.1250'
 * ```
 */
export function puRisk(
  scores: TensorLike,
  labelledPositive: TensorLike,
  prior: number,
  options: { readonly loss?: PuLoss; readonly nonNegative?: boolean } = {},
): PuRisk {
  const loss = options.loss ?? 'logistic'
  if (!(prior > 0 && prior < 1)) throw new MdmlError('value', `puRisk: the prior must be in (0, 1), got ${prior}`)
  const s = asTensor(scores)
  const lp = toFlat(asTensor(labelledPositive))
  const nP = lp.reduce((a, b) => a + b, 0)
  const nU = lp.length - nP
  if (nP === 0 || nU === 0) throw new MdmlError('value', 'puRisk: needs both labelled positives and unlabelled rows')
  const maskP = fromStorage(
    Float64Array.from(lp, (v) => v / nP),
    [lp.length],
    'float64',
  )
  const maskU = fromStorage(
    Float64Array.from(lp, (v) => (1 - v) / nU),
    [lp.length],
    'float64',
  )
  const positiveRisk = mul(sum(mul(maskP, marginLoss(loss, s, 1))), prior)
  const negativeRisk = sub(
    sum(mul(maskU, marginLoss(loss, s, -1))),
    mul(sum(mul(maskP, marginLoss(loss, s, -1))), prior),
  )
  const risk = add(positiveRisk, options.nonNegative === true ? maximum(negativeRisk, 0) : negativeRisk)
  return { risk, positiveRisk, negativeRisk }
}

/** The hyperparameters of {@link puClassifier}. */
export interface PuClassifierHyper {
  /** The class prior of the positives in the unlabelled data. */
  readonly prior: number
  readonly loss: PuLoss
  /** Clamp the negatives' risk at zero (nnPU) instead of the unbiased estimate (uPU). */
  readonly nonNegative: boolean
  /** The L2 penalty on the weights. */
  readonly l2: number
  readonly maxIter: number
}

/** The fitted parameters of {@link puClassifier}: a linear score. */
export type PuClassifierParams = {
  readonly weights: number[]
  readonly bias: number
  readonly d: number
  readonly risk: number
  readonly negativeRisk: number
}

/** What a fitted PU classifier offers. */
export interface PuClassifierVerbs {
  readonly predict: (x: TensorLike) => Tensor
  readonly decisionFunction: (x: TensorLike) => Tensor
  readonly score: (data: LearnData) => number
  readonly classes: readonly number[]
  readonly nFeaturesIn: number
}

const augmented = (
  x: TensorLike,
  who: string,
  nFeatures?: number,
): { readonly aug: Tensor; readonly n: number; readonly d: number } => {
  const m = matrixView(x, who, nFeatures === undefined ? {} : { nFeatures })
  const data = new Float64Array(m.n * (m.d + 1))
  for (let i = 0; i < m.n; i++) {
    data.set(m.data.subarray(i * m.d, (i + 1) * m.d), i * (m.d + 1))
    data[i * (m.d + 1) + m.d] = 1
  }
  return { aug: fromStorage(data, [m.n, m.d + 1], 'float64'), n: m.n, d: m.d }
}

/**
 * A linear classifier trained on positive-unlabelled data by minimising {@link puRisk} (uPU, or nnPU with
 * `nonNegative`) with L-BFGS, one iteration per step; the targets are 1 for a labelled positive and 0 for an unlabelled
 * row. Logs `risk`, `negativeRisk` (watch it go below zero for uPU) and `gradNorm`.
 *
 * @param hyper - The prior, the surrogate loss, `nonNegative`, the L2 penalty and the iteration cap.
 * @returns The estimator; `predict` gives 1 for positive and 0 for negative.
 * @example
 * ```ts
 * import { puClassifier } from 'mdml-engine/learn'
 * import { toFlat } from 'mdml-engine/tensor'
 * const x = [[2], [2.5], [3], [-2], [-2.5], [-3], [2.2], [-2.2]]
 * const y = [1, 1, 0, 0, 0, 0, 0, 0]
 * const f = puClassifier({ prior: 0.4, nonNegative: true }).fit({ x, y })
 * toFlat(f.predict([[2.8], [-2.8]])) // => [1, 0]
 * ```
 */
export function puClassifier(
  hyper: Partial<PuClassifierHyper> = {},
): IterativeEstimator<PuClassifierHyper, LbfgsState, PuClassifierParams, PuClassifierVerbs> {
  const h: PuClassifierHyper = { prior: 0.5, loss: 'logistic', nonNegative: true, l2: 1e-3, maxIter: 200, ...hyper }
  const build = (data: LearnData, c: PuClassifierHyper): { alg: Algorithm<LbfgsState>; aug: Tensor; d: number } => {
    const { aug, d } = augmented(data.x, 'puClassifier')
    const y = requireTargets(data, 'puClassifier')
    const objective = (theta: TensorLike): Value => {
      const w = reshape(theta, [d + 1])
      const r = puRisk(matmul(aug, w), y, c.prior, { loss: c.loss, nonNegative: c.nonNegative })
      const penalty = mul(sum(mul(w, w)), c.l2 / 2)
      return add(r.risk, penalty)
    }
    return { alg: lbfgsAlgorithm(objective, new Float64Array(d + 1), { maxIterations: c.maxIter }), aug, d }
  }
  return iterative<PuClassifierHyper, LbfgsState, PuClassifierParams, PuClassifierVerbs>(
    {
      name: 'pu-classifier',
      kind: 'classifier',
      defaults: h,
      hyperSpace: (c) =>
        dict({
          prior: real(0.01, 0.99, { default: c.prior, label: 'class prior' }),
          loss: choice(['logistic', 'sigmoid', 'squared'] as const, { default: c.loss, label: 'surrogate loss' }),
          l2: real(1e-6, 10, { log: true, default: c.l2, label: 'L2 penalty' }),
        }),
      algorithm: (data, c) => build(data, c).alg,
      maxSteps: (c) => c.maxIter + 2,
      params: (s, data, c) => {
        const { aug, d } = build(data, c)
        const scores = matmul(aug, fromStorage(Float64Array.from(s.x), [d + 1], 'float64'))
        const r = puRisk(scores, requireTargets(data, 'puClassifier'), c.prior, {
          loss: c.loss,
          nonNegative: c.nonNegative,
        })
        return {
          weights: Array.from(s.x.subarray(0, d)),
          bias: s.x[d] ?? 0,
          d,
          risk: item(r.risk),
          negativeRisk: item(r.negativeRisk),
        }
      },
      verbs: (p) => {
        const decide = (x: TensorLike): Tensor => {
          const m = matrixView(x, 'puClassifier', { nFeatures: p.d })
          return fromStorage(
            Float64Array.from({ length: m.n }, (_, i) =>
              p.weights.reduce((a, w, j) => a + w * (m.data[i * p.d + j] ?? 0), p.bias),
            ),
            [m.n],
            'float64',
          )
        }
        const predict = (x: TensorLike): Tensor => {
          const z = toFlat(decide(x))
          return fromStorage(
            Float64Array.from(z, (v) => (v > 0 ? 1 : 0)),
            [z.length],
            'float64',
          )
        }
        return { predict, decisionFunction: decide, score: accuracyOf(predict), classes: [0, 1], nFeaturesIn: p.d }
      },
      logs: { 'solver/loss': Log(), 'solver/gradNorm': Log() },
    },
    h,
  )
}

/** The fitted parameters of {@link elkanNoto}. */
export type ElkanNotoParams = {
  /** The parameters of the base classifier of labelled versus unlabelled. */
  readonly model: Json
  /** `c = P(labelled | positive)`, the mean score of the held-out labelled positives. */
  readonly labelFrequency: number
  /** The estimated class prior `P(labelled) / c` over all rows. */
  readonly prior: number
  readonly d: number
}

/** The state of {@link elkanNoto}'s two stages. */
export type ElkanNotoState = {
  readonly stage: 0 | 1 | 2
  /** The rows the base is fitted on, and the labelled positives among the held-out rows that estimate `c`. */
  readonly train: number[]
  readonly held: number[]
  readonly model: Json
  readonly labelFrequency: number
  readonly prior: number
}

/** What a fitted Elkan-Noto classifier offers. */
export interface ElkanNotoVerbs {
  readonly predict: (x: TensorLike) => Tensor
  /** `P(positive | x) = min(1, g(x) / c)` as columns (negative, positive). */
  readonly predictProba: (x: TensorLike) => Tensor
  readonly score: (data: LearnData) => number
  readonly classes: readonly number[]
  readonly nFeaturesIn: number
  readonly labelFrequency: number
  readonly prior: number
}

/**
 * Positive-unlabelled learning by the method of Elkan and Noto (2008): a base classifier `g` is trained to tell labelled
 * rows from unlabelled ones, `c` is its mean score on the labelled positives of a held-out part of the data (the label frequency), and
 * `P(positive | x) = g(x) / c`. The prior follows as `P(labelled) / c`. Two steps: the base fit, then `c` and the prior.
 *
 * @param base - The base classifier (it must offer `predictProba`), e.g. logistic regression.
 * @param hyper - `holdout`: the fraction of all rows kept out of the base fit to estimate `c` on their labelled positives (default 0.2).
 * @returns The estimator; targets are 1 for a labelled positive and 0 for an unlabelled row.
 * @example
 * ```ts
 * import { elkanNoto, gaussianNaiveBayes } from 'mdml-engine/learn'
 * const x = [[2], [2.5], [3], [3.2], [2.8], [-2], [-2.5], [-3], [2.2], [-2.2]]
 * const y = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
 * const f = elkanNoto(gaussianNaiveBayes(), { holdout: 0.25 }).fit({ x, y })
 * f.prior > 0 && f.prior < 1 // => true
 * ```
 */
export function elkanNoto(
  base: Estimator,
  hyper: { readonly holdout?: number } = {},
): IterativeEstimator<
  { readonly base: Estimator; readonly holdout: number },
  ElkanNotoState,
  ElkanNotoParams,
  ElkanNotoVerbs
> {
  const h = { base, holdout: hyper.holdout ?? 0.2 }
  type H = typeof h
  return iterative<H, ElkanNotoState, ElkanNotoParams, ElkanNotoVerbs>(
    {
      name: 'elkan-noto',
      kind: 'classifier',
      defaults: h,
      algorithm: (data, c, key) => {
        const y = Float64Array.from(toFlat(asTensor(requireTargets(data, 'elkanNoto'))))
        const n = y.length
        return {
          name: 'elkan-noto',
          config: { base: c.base.name, holdout: c.holdout },
          schema: channels({
            metrics: {
              labelFrequency: { description: 'c: the mean score of the held-out labelled positives' },
              prior: { description: 'the estimated class prior' },
            },
          }),
          init: () => {
            if (y.reduce((a, v) => a + (v === 1 ? 1 : 0), 0) < 2)
              throw new MdmlError('value', 'elkanNoto: needs at least two labelled positives')
            // A validation subset of all rows (so the labelling rate is the same in both parts), kept out of the base fit.
            const order = Array.from(permutation(foldIn(key, 0), n).data)
            const validation = order.slice(0, Math.max(1, Math.min(n - 1, Math.round(c.holdout * n))))
            const kept = new Set(validation)
            const held = validation.filter((i) => y[i] === 1)
            if (held.length === 0)
              throw new MdmlError('value', 'elkanNoto: the validation rows hold no labelled positive; raise holdout')
            return {
              stage: 0,
              held,
              train: Array.from(y, (_, i) => i).filter((i) => !kept.has(i)),
              model: null,
              labelFrequency: 0,
              prior: 0,
            }
          },
          step: (s, ctx) => {
            if (s.stage === 0) {
              const sub = takeData(data, s.train)
              const labels = fromStorage(
                Float64Array.from(s.train, (i) => y[i] ?? 0),
                [s.train.length],
                'float64',
              )
              return {
                ...s,
                stage: 1,
                model: c.base.fit({ ...sub, y: labels }, { key: ctx.key, explain: false }).params,
              }
            }
            const f = c.base.revive(s.model)
            const pr = Float64Array.from(
              toFlat((f.predictProba as NonNullable<typeof f.predictProba>)(takeData(data, s.held).x)),
            )
            const k = pr.length / s.held.length
            const col = (f.classes ?? []).indexOf(1)
            const freq = s.held.reduce((a, _, i) => a + (pr[i * k + (col < 0 ? k - 1 : col)] ?? 0), 0) / s.held.length
            const labelled = y.reduce((a, v) => a + (v === 1 ? 1 : 0), 0)
            return { ...s, stage: 2, labelFrequency: freq, prior: Math.min(1, labelled / n / Math.max(freq, 1e-12)) }
          },
          done: (s) => (s.stage === 2 ? 'converged' : undefined),
          observe: (s) => ({ metrics: { labelFrequency: s.labelFrequency, prior: s.prior } }),
          serialise: (s) => stateToJson(s),
          deserialise: (j) => stateFromJson(j) as ElkanNotoState,
        }
      },
      maxSteps: () => 3,
      params: (s, data) => ({
        model: s.model,
        labelFrequency: s.labelFrequency,
        prior: s.prior,
        d: matrixView(data.x, 'elkanNoto').d,
      }),
      verbs: (p, c) => {
        const f = c.base.revive(p.model)
        const pos = (f.classes ?? []).indexOf(1)
        const proba = (x: TensorLike): Float64Array => {
          const pr = Float64Array.from(toFlat((f.predictProba as NonNullable<typeof f.predictProba>)(x)))
          const k = (f.classes ?? []).length
          const out = new Float64Array((pr.length / k) * 2)
          for (let i = 0; i < pr.length / k; i++) {
            const q = Math.min(1, (pr[i * k + (pos < 0 ? k - 1 : pos)] ?? 0) / Math.max(p.labelFrequency, 1e-12))
            out[i * 2] = 1 - q
            out[i * 2 + 1] = q
          }
          return out
        }
        const predict = (x: TensorLike): Tensor => {
          const q = proba(x)
          return fromStorage(
            Float64Array.from({ length: q.length / 2 }, (_, i) => ((q[i * 2 + 1] ?? 0) >= 0.5 ? 1 : 0)),
            [q.length / 2],
            'float64',
          )
        }
        return {
          predict,
          predictProba: (x) => {
            const q = proba(x)
            return fromStorage(q, [q.length / 2, 2], 'float64')
          },
          score: accuracyOf(predict),
          classes: [0, 1],
          nFeaturesIn: p.d,
          labelFrequency: p.labelFrequency,
          prior: p.prior,
        }
      },
      logs: { 'solver/labelFrequency': Log(), 'solver/prior': Log() },
    },
    h,
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// Bags: label proportions and multiple instances

/** A dataset of instances grouped into bags: `bags[i]` is the bag of instance `i`. */
export interface BagData extends LearnData {
  readonly bags: readonly number[]
  /** For label proportions: the class proportions of every bag, indexed by bag id. */
  readonly proportions?: readonly (readonly number[])[]
}

/**
 * A dataset of instances in bags.
 *
 * @param x - The instances `[n, d]`.
 * @param bags - The bag id of every instance (0, 1, 2, ...).
 * @param extra - `y`: the bag label of each instance (multiple-instance learning), `proportions`: each bag's class proportions.
 * @returns The data.
 * @example
 * ```ts
 * import { bagData } from 'mdml-engine/learn'
 * bagData([[0], [1], [2]], [0, 0, 1], { y: [1, 1, 0] }).bags // => [0, 0, 1]
 * ```
 */
export function bagData(
  x: TensorLike,
  bags: readonly number[],
  extra: { readonly y?: TensorLike; readonly proportions?: readonly (readonly number[])[] } = {},
): BagData {
  return {
    x,
    bags,
    ...(extra.y === undefined ? {} : { y: extra.y }),
    ...(extra.proportions === undefined ? {} : { proportions: extra.proportions }),
  }
}

const bagsOf = (data: LearnData, who: string): readonly number[] => {
  const b = (data as Partial<BagData>).bags
  if (b === undefined) throw new MdmlError('value', `${who}: needs bag ids (use bagData)`)
  return b
}

/** The hyperparameters of {@link labelProportionClassifier}. */
export interface LabelProportionHyper {
  /** The L2 penalty on the weights. */
  readonly l2: number
  readonly maxIter: number
}

/** The fitted parameters of {@link labelProportionClassifier}: a softmax-linear model. */
export type LabelProportionParams = {
  readonly classes: number[]
  /** The weights `[d + 1][k]` row-major, the last row the bias. */
  readonly weights: number[]
  readonly d: number
  /** The proportion loss `sum_b n_b KL(proportions_b || predicted mean_b) / n` at the solution. */
  readonly loss: number
}

/** What a fitted label-proportion classifier offers. */
export interface LabelProportionVerbs {
  readonly predict: (x: TensorLike) => Tensor
  readonly predictProba: (x: TensorLike) => Tensor
  readonly score: (data: LearnData) => number
  readonly classes: readonly number[]
  readonly nFeaturesIn: number
}

/**
 * Learning from label proportions (Quadrianto et al. 2009; the proportion loss of Ardehaly and Culotta): only the class
 * proportions of bags of instances are known, and a softmax-linear model is trained so that its predicted class
 * distribution averaged over each bag matches the bag's proportions, `sum_b n_b KL(pi_b || mean_{i in b} p(.|x_i))`, by
 * L-BFGS. The smaller the bags, the more it knows about individual rows. Data is {@link bagData} with `proportions`.
 *
 * @param hyper - `l2` and `maxIter`.
 * @returns The estimator.
 * @example
 * ```ts
 * import { bagData, labelProportionClassifier } from 'mdml-engine/learn'
 * import { toFlat } from 'mdml-engine/tensor'
 * const x = [[-2], [-1.8], [2], [1.9], [-2.1], [2.2], [-1.9], [2.1]]
 * const data = bagData(x, [0, 0, 0, 0, 1, 1, 1, 1], { proportions: [[0.5, 0.5], [0.5, 0.5]] })
 * labelProportionClassifier().with({ l2: 0.01 }).fit(data).classes // => [0, 1]
 * ```
 */
export function labelProportionClassifier(
  hyper: Partial<LabelProportionHyper> = {},
): IterativeEstimator<LabelProportionHyper, LbfgsState, LabelProportionParams, LabelProportionVerbs> {
  const h: LabelProportionHyper = { l2: 1e-3, maxIter: 300, ...hyper }
  const read = (
    data: LearnData,
  ): { aug: Tensor; d: number; k: number; membership: Tensor; target: Tensor; sizes: number[]; n: number } => {
    const { aug, n, d } = augmented(data.x, 'labelProportions')
    const bags = bagsOf(data, 'labelProportions')
    const proportions = (data as Partial<BagData>).proportions
    if (proportions === undefined)
      throw new MdmlError('value', 'labelProportions: needs proportions per bag (use bagData)')
    if (bags.length !== n) throw new MdmlError('shape', `labelProportions: ${n} instances but ${bags.length} bag ids`)
    const nb = proportions.length
    const k = proportions[0]?.length ?? 0
    const sizes = new Array<number>(nb).fill(0)
    for (const b of bags) {
      if (!(b >= 0 && b < nb)) throw new MdmlError('value', `labelProportions: bag ${b} has no proportions`)
      sizes[b] = (sizes[b] ?? 0) + 1
    }
    const member = new Float64Array(nb * n)
    bags.forEach((b, i) => {
      member[b * n + i] = 1 / (sizes[b] ?? 1)
    })
    return {
      aug,
      d,
      k,
      n,
      sizes,
      membership: fromStorage(member, [nb, n], 'float64'),
      target: fromStorage(Float64Array.from(proportions.flatMap((r) => [...r])), [nb, k], 'float64'),
    }
  }
  const lossOf = (r: ReturnType<typeof read>, w: Value, l2: number): Value => {
    const q = matmul(r.membership, softmax(matmul(r.aug, w), -1))
    const logq = logSafe(q)
    const weights = fromStorage(
      Float64Array.from(r.sizes, (s) => s / r.n),
      [r.sizes.length, 1],
      'float64',
    )
    const cross = neg(sum(mul(weights, mul(r.target, logq))))
    return add(cross, mul(sum(mul(w, w)), l2 / 2))
  }
  return iterative<LabelProportionHyper, LbfgsState, LabelProportionParams, LabelProportionVerbs>(
    {
      name: 'label-proportions',
      kind: 'classifier',
      defaults: h,
      hyperSpace: (c) => dict({ l2: real(1e-6, 10, { log: true, default: c.l2, label: 'L2 penalty' }) }),
      algorithm: (data, c) => {
        const r = read(data)
        const objective = (theta: TensorLike): Value => lossOf(r, reshape(theta, [r.d + 1, r.k]), c.l2)
        return lbfgsAlgorithm(objective, new Float64Array((r.d + 1) * r.k), { maxIterations: c.maxIter })
      },
      maxSteps: (c) => c.maxIter + 2,
      params: (s, data) => {
        const r = read(data)
        const w = fromStorage(Float64Array.from(s.x), [r.d + 1, r.k], 'float64')
        return {
          classes: Array.from({ length: r.k }, (_, i) => i),
          weights: Array.from(s.x),
          d: r.d,
          loss: item(lossOf(r, w, 0)),
        }
      },
      verbs: (p) => {
        const k = p.classes.length
        const w = fromStorage(Float64Array.from(p.weights), [p.d + 1, k], 'float64')
        const proba = (x: TensorLike): Tensor => {
          const { aug } = augmented(x, 'labelProportions', p.d)
          return asTensor(softmax(matmul(aug, w), -1)) as Tensor
        }
        const predict = (x: TensorLike): Tensor => {
          const pr = toFlat(proba(x))
          return fromStorage(
            Float64Array.from({ length: pr.length / k }, (_, i) => p.classes[argmaxIndex(pr, i * k, k)] ?? 0),
            [pr.length / k],
            'float64',
          )
        }
        return { predict, predictProba: proba, score: accuracyOf(predict), classes: p.classes, nFeaturesIn: p.d }
      },
      logs: { 'solver/loss': Log(), 'solver/gradNorm': Log() },
    },
    h,
  )
}

const logSafe = (x: Value): Value => log(maximum(x, 1e-12))

/** The hyperparameters of {@link multipleInstanceSvm}. */
export interface MultipleInstanceHyper {
  /** The base binary classifier with a decision function (e.g. an SVM of `mdml-engine/svm`). */
  readonly base: Estimator
  /** `instance` (mi-SVM): every instance of a positive bag starts positive and is relabelled by the margin; `bag` (MI-SVM): one representative per positive bag, the most positive instance. */
  readonly mode: 'instance' | 'bag'
  readonly maxIter: number
}

/** The state of {@link multipleInstanceAlgorithm}: the instances currently treated as positive and the model fitted for them. */
export type MultipleInstanceState = {
  /** `instance`: 1 for each instance of a positive bag currently labelled positive; `bag`: 1 for the representatives. */
  readonly selected: number[]
  readonly model: Json
  /** How many instances changed role this step. */
  readonly changes: number
  readonly iter: number
}

/** The fitted parameters of {@link multipleInstanceSvm}. */
export type MultipleInstanceParams = MultipleInstanceState & { readonly d: number }

/** What a fitted multiple-instance classifier offers. */
export interface MultipleInstanceVerbs {
  /** Instance labels (1 positive). */
  readonly predict: (x: TensorLike) => Tensor
  readonly decisionFunction: (x: TensorLike) => Tensor
  /** Bag labels: a bag is positive when its most positive instance is. */
  readonly predictBags: (x: TensorLike, bags: readonly number[]) => Tensor
  readonly score: (data: LearnData) => number
  readonly classes: readonly number[]
  readonly nFeaturesIn: number
}

/**
 * Multiple-instance SVM as an algorithm (Andrews, Tsochantaridis and Hofmann 2002): a negative bag has only negative
 * instances, a positive bag at least one positive. `instance` (mi-SVM): fit the base with every instance of a positive bag
 * labelled positive, then relabel those instances by the sign of the decision function (a positive bag keeps at least its
 * most positive instance), and repeat until no label changes. `bag` (MI-SVM): the positive bags contribute one
 * representative each, starting at the instance nearest the bag's mean and moving to the bag's highest-scoring instance.
 * One refit per step; logs `changes` and `positives`.
 *
 * @param data - {@link bagData} with `y` the bag label (1 positive) of each instance.
 * @param hyper - The base, the mode and the cap.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { bagData, gaussianNaiveBayes, multipleInstanceAlgorithm } from 'mdml-engine/learn'
 * const data = bagData([[0], [0.1], [3], [0.2]], [0, 0, 1, 1], { y: [0, 0, 1, 1] })
 * multipleInstanceAlgorithm(data, { base: gaussianNaiveBayes(), mode: 'instance', maxIter: 10 }).name // => 'multiple-instance-instance'
 * ```
 */
export function multipleInstanceAlgorithm(
  data: LearnData,
  hyper: MultipleInstanceHyper,
): Algorithm<MultipleInstanceState> {
  const who = 'multipleInstanceSvm'
  const m = matrixView(data.x, who)
  const bags = bagsOf(data, who)
  const y = Float64Array.from(toFlat(asTensor(requireTargets(data, who))))
  if (bags.length !== m.n) throw new MdmlError('shape', `${who}: ${m.n} instances but ${bags.length} bag ids`)
  const members = new Map<number, number[]>()
  bags.forEach((b, i) => {
    members.set(b, [...(members.get(b) ?? []), i])
  })
  const positiveBags = Array.from(members.entries()).filter(([, rows]) => (y[rows[0] ?? 0] ?? 0) === 1)
  for (const [b, rows] of members) {
    if (new Set(rows.map((i) => y[i])).size !== 1)
      throw new MdmlError('value', `${who}: bag ${b} has instances with different bag labels`)
  }
  const negativeRows = Array.from({ length: m.n }, (_, i) => i).filter((i) => y[i] !== 1)
  const trainingRows = (selected: readonly number[]): number[] =>
    hyper.mode === 'instance'
      ? Array.from({ length: m.n }, (_, i) => i)
      : [...negativeRows, ...Array.from({ length: m.n }, (_, i) => i).filter((i) => selected[i] === 1)]
  const fitOn = (selected: readonly number[], key: Key): Json => {
    const rows = trainingRows(selected)
    const labels = rows.map((i) => (y[i] === 1 && selected[i] === 1 ? 1 : 0))
    return hyper.base.fit(
      { ...takeData(data, rows), y: fromStorage(Float64Array.from(labels), [rows.length], 'float64') },
      { key, explain: false },
    ).params
  }
  const x = fromStorageView(m)
  return {
    name: `multiple-instance-${hyper.mode}`,
    config: { base: hyper.base.name, mode: hyper.mode, instances: m.n, bags: members.size },
    schema: channels({
      metrics: {
        changes: { description: 'instances that changed role this step', direction: 'lower' },
        positives: { description: 'instances treated as positive' },
      },
    }),
    init: (key) => {
      const selected = new Array<number>(m.n).fill(0)
      for (const [, rows] of positiveBags) {
        if (hyper.mode === 'instance') for (const i of rows) selected[i] = 1
        else {
          const mean = Array.from(
            { length: m.d },
            (_, j) => rows.reduce((a, i) => a + (m.data[i * m.d + j] ?? 0), 0) / rows.length,
          )
          const near = rows.reduce((best, i) => {
            const dist = (r: number): number => mean.reduce((a, mj, j) => a + ((m.data[r * m.d + j] ?? 0) - mj) ** 2, 0)
            return dist(i) < dist(best) ? i : best
          }, rows[0] ?? 0)
          selected[near] = 1
        }
      }
      return { selected, model: fitOn(selected, foldIn(key, 0)), changes: 0, iter: 0 }
    },
    step: (s, ctx) => {
      const f = confidence(hyper.base.revive(s.model), x)
      const selected = [...s.selected]
      let changes = 0
      for (const [, rows] of positiveBags) {
        const best = rows.reduce((b, i) => ((f[i] ?? 0) > (f[b] ?? 0) ? i : b), rows[0] ?? 0)
        const next = new Map<number, number>()
        if (hyper.mode === 'instance') {
          for (const i of rows) next.set(i, (f[i] ?? 0) > 0 || i === best ? 1 : 0)
        } else for (const i of rows) next.set(i, i === best ? 1 : 0)
        for (const [i, v] of next) {
          if (selected[i] !== v) changes++
          selected[i] = v
        }
      }
      return {
        selected,
        model: changes === 0 ? s.model : fitOn(selected, foldIn(ctx.key, s.iter + 1)),
        changes,
        iter: s.iter + 1,
      }
    },
    done: (s) => (s.iter > 0 && s.changes === 0 ? 'converged' : s.iter >= hyper.maxIter ? 'completed' : undefined),
    observe: (s) => ({ metrics: { changes: s.changes, positives: s.selected.reduce((a, b) => a + b, 0) } }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as MultipleInstanceState,
  }
}

const fromStorageView = (m: { n: number; d: number; data: Float64Array }): Tensor =>
  fromStorage(m.data, [m.n, m.d], 'float64')

/**
 * Multiple-instance learning with an SVM (or any binary classifier with a decision function) as the base: the mi-SVM and
 * MI-SVM of Andrews et al. 2002 (see {@link multipleInstanceAlgorithm}). Instances are classified by the base, bags by
 * their most positive instance.
 *
 * @param base - The binary base classifier.
 * @param hyper - `mode` (default `instance`) and `maxIter`.
 * @returns The estimator; fit it on {@link bagData} with the bag labels as `y`.
 * @example
 * ```ts
 * import { bagData, gaussianNaiveBayes, multipleInstanceSvm } from 'mdml-engine/learn'
 * const x = [[-2], [-2.2], [-1.8], [2], [-1.9], [-2.1], [2.2], [-2]]
 * const data = bagData(x, [0, 0, 1, 1, 2, 2, 3, 3], { y: [0, 0, 1, 1, 0, 0, 1, 1] })
 * const f = multipleInstanceSvm(gaussianNaiveBayes()).fit(data)
 * f.classes // => [0, 1]
 * ```
 */
export function multipleInstanceSvm(
  base: Estimator,
  hyper: Partial<Omit<MultipleInstanceHyper, 'base'>> = {},
): IterativeEstimator<MultipleInstanceHyper, MultipleInstanceState, MultipleInstanceParams, MultipleInstanceVerbs> {
  const h: MultipleInstanceHyper = { base, mode: 'instance', maxIter: 30, ...hyper }
  return iterative<MultipleInstanceHyper, MultipleInstanceState, MultipleInstanceParams, MultipleInstanceVerbs>(
    {
      name: 'multiple-instance-svm',
      kind: 'classifier',
      defaults: h,
      hyperSpace: (c) =>
        dict({
          mode: choice(['instance', 'bag'] as const, { default: c.mode, label: 'mode' }),
          maxIter: int(1, 100, { default: c.maxIter, label: 'rounds' }),
        }),
      algorithm: (data, c) => multipleInstanceAlgorithm(data, c),
      maxSteps: (c) => c.maxIter + 2,
      params: (s, data) => ({ ...s, d: matrixView(data.x, 'multipleInstanceSvm').d }),
      verbs: (p, c) => {
        const f: Fitted = c.base.revive(p.model)
        const decide = (x: TensorLike): Float64Array => confidence(f, x)
        const predict = (x: TensorLike): Tensor => {
          const z = decide(x)
          return fromStorage(
            Float64Array.from(z, (v) => (v > 0 ? 1 : 0)),
            [z.length],
            'float64',
          )
        }
        return {
          predict,
          decisionFunction: (x) => {
            const z = decide(x)
            return fromStorage(z, [z.length], 'float64')
          },
          predictBags: (x, bags) => {
            const z = decide(x)
            const ids = Array.from(new Set(bags)).sort((a, b) => a - b)
            return fromStorage(
              Float64Array.from(ids, (b) =>
                bags.reduce(
                  (best, bag, i) => (bag === b ? Math.max(best, z[i] ?? 0) : best),
                  Number.NEGATIVE_INFINITY,
                ) > 0
                  ? 1
                  : 0,
              ),
              [ids.length],
              'float64',
            )
          },
          score: accuracyOf(predict),
          classes: [0, 1],
          nFeaturesIn: p.d,
        }
      },
      logs: { 'solver/changes': Log(), 'solver/positives': Log() },
    },
    h,
  )
}
