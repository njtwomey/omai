/**
 * Dictionary learning (scikit-learn's `DictionaryLearning` and `dict_learning`): atoms `D` and sparse codes `C` found
 * for a set of signals by alternating, one iteration per step, a sparse coding of every signal over the current atoms
 * with an update of the atoms for those codes. The coding is any coder of `sparse-coding.ts` (the lasso by least angle
 * regression as scikit-learn, coordinate descent, OMP with a fixed number of atoms per signal); the update of the
 * atoms is a slot: `cd`, one pass of block coordinate descent on the atoms inside the unit ball (scikit-learn's), `mod`,
 * the method of optimal directions (the least-squares atoms for the codes, renormalised), or `ksvd`, K-SVD (each atom
 * and the codes that use it replaced by the rank-one SVD approximation of what the other atoms leave unexplained).
 * Atoms nobody uses are replaced by the worst-represented signals. The learner is a traced `Algorithm` observing the
 * objective, the reconstruction error, the sparsity of the codes and the coherence of the atoms, with the atoms
 * themselves as a tensor, and the estimator exposes them and their usage as records any step can be asked for.
 */

import { OmaiError } from '../base/index.ts'
import { componentsRecord } from '../exposition/index.ts'
import { type IterativeEstimator, iterative } from '../learn/index.ts'
import { dnrm2 } from '../numerics/linalg/index.ts'
import { foldIn, type Key } from '../random/index.ts'
import { choice, dict, int, real } from '../space/index.ts'
import { type Algorithm, channels, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import {
  asRows,
  coherenceOf,
  type DictionaryVerbs,
  dictionaryContract,
  dictionaryVerbs,
  encode,
  type FitCoding,
  freshAtom,
  meanNonzeros,
  objectiveOf,
  penaltyWeight,
  type Rows,
  randomDictionary,
  sampleDictionary,
  svdDictionary,
  updateAtoms,
  worstRepresented,
} from './_dictionary.ts'
import { mm, pinvRows, thinSvd, tr } from './_linear.ts'
import type { SparseCodeAlgorithm } from './sparse-coding.ts'

/** The update of the atoms: block coordinate descent (scikit-learn's), the method of optimal directions, or K-SVD. */
export type DictionaryUpdate = 'cd' | 'mod' | 'ksvd'

/** The state of dictionary learning: the atoms and codes after an iteration, and what was measured of them. */
export interface DictionaryLearningState {
  /** The atoms `[k, d]`. */
  readonly dictionary: number[][]
  /** The codes `[n, k]` of the iteration (as scikit-learn returns them: coded over the atoms before their update, for `cd`). */
  readonly code: number[][]
  /** Half the squared residual plus the penalty times the L1 norm of the codes (the penalty is zero for `omp` and `lars`). */
  readonly objective: number
  /** The Frobenius norm of the residual `X - C D`. */
  readonly error: number
  /** The mean number of nonzero codes per signal. */
  readonly sparsity: number
  /** The largest absolute cosine between two atoms. */
  readonly coherence: number
  /** Atoms replaced in this iteration because no code used them. */
  readonly unused: number
  /** The relative decrease of the objective in this iteration (0 in the first). */
  readonly change: number
  /** Iterations done. */
  readonly iteration: number
  /** Whether the relative decrease fell below the tolerance. */
  readonly stalled: boolean
}

/** Options of {@link dictionaryLearningAlgorithm}. */
export interface DictionaryLearningOptions {
  /** The penalty of the lasso coders on `|x - c D|^2 / 2 + alpha |c|_1` (default 1). */
  readonly alpha?: number
  /** The coder of the coding step (default `lassoLars`, scikit-learn's `fit_algorithm='lars'`). */
  readonly fitAlgorithm?: SparseCodeAlgorithm
  /** Atoms per signal for the `omp` and `lars` coders (default scikit-learn's). */
  readonly fitNonzeroCoefs?: number | null
  /** The update of the atoms (default `cd`). */
  readonly dictUpdate?: DictionaryUpdate
  /** Stop once the relative decrease of the objective falls below this, from the second iteration on (default 1e-8). */
  readonly tol?: number
  readonly positiveCode?: boolean
  readonly positiveDict?: boolean
  /** The sweep cap of the coordinate descent coder (default 1000). */
  readonly fitMaxIter?: number
}

const SCHEMA = channels({
  metrics: {
    objective: {
      description: 'half the squared residual plus the penalty times the L1 norm of the codes',
      direction: 'lower',
    },
    error: { description: 'Frobenius norm of the residual X - C D', direction: 'lower' },
    sparsity: { description: 'mean number of nonzero codes per signal' },
    coherence: { description: 'largest absolute cosine between two atoms' },
    unused: { description: 'atoms replaced because no code used them' },
    change: { description: 'relative decrease of the objective in the iteration' },
  },
  tensors: {
    dictionary: { role: 'param', description: 'the atoms, one per row', shape: ['k', 'd'] },
    codes: { role: 'state', description: 'the sparse codes of the signals', shape: ['n', 'k'] },
  },
})

type Update = { dictionary: number[][]; code: number[][]; unused: number }

const normalised = (v: readonly number[]): number[] => {
  const n = dnrm2(v) || 1
  return v.map((a) => a / n)
}

/** The method of optimal directions: the least-squares atoms for the codes, unit norm with the codes taking the scale. */
function modUpdate(dictionary: Rows, x: Rows, code: Rows, key: Key): Update {
  const out = mm(pinvRows(code), x).map((r) => [...r])
  const newCode = code.map((r) => [...r])
  let unused = 0
  let order: number[] | undefined
  for (const [j, atom] of out.entries()) {
    const norm = dnrm2(atom)
    if (norm > 1e-6) {
      for (const [q, v] of atom.entries()) atom[q] = v / norm
      for (const row of newCode) row[j] = (row[j] ?? 0) * norm
    } else {
      order ??= worstRepresented(x, dictionary, code)
      out[j] = normalised(freshAtom(x[order[unused % order.length] ?? 0] ?? [], foldIn(key, j)))
      for (const row of newCode) row[j] = 0
      unused++
    }
  }
  return { dictionary: out, code: newCode, unused }
}

/** K-SVD: each atom in turn, with the codes of the signals that use it, by the rank-one SVD of the residual they leave. */
function ksvdUpdate(dictionary: Rows, x: Rows, code: Rows, key: Key): Update {
  const k = dictionary.length
  const d = x[0]?.length ?? 0
  const out = dictionary.map((r) => [...r])
  const newCode = code.map((r) => [...r])
  const fit = mm(newCode, out)
  // The residual of every signal, kept up to date as the atoms change one by one.
  const residual = x.map((row, i) => row.map((v, q) => v - (fit[i]?.[q] ?? 0)))
  const order = worstRepresented(x, dictionary, code)
  let unused = 0
  for (let j = 0; j < k; j++) {
    const users: number[] = []
    for (const [i, row] of newCode.entries()) if ((row[j] ?? 0) !== 0) users.push(i)
    if (users.length === 0) {
      out[j] = normalised(freshAtom(x[order[unused % order.length] ?? 0] ?? [], foldIn(key, j)))
      unused++
      continue
    }
    // E = what the other atoms leave of the users' signals: the residual with atom j's part added back.
    const e = users.map((i) => (residual[i] ?? []).map((v, q) => v + (newCode[i]?.[j] ?? 0) * (out[j]?.[q] ?? 0)))
    const { u, s, vt } = thinSvd(Float64Array.from(e.flat()), users.length, d)
    const atom = vt[0] ?? new Array<number>(d).fill(0)
    out[j] = [...atom]
    for (const [r, i] of users.entries()) {
      const c = (s[0] ?? 0) * (u[r]?.[0] ?? 0)
      newCode[i] = (newCode[i] ?? []).map((v, q) => (q === j ? c : v))
      residual[i] = (e[r] ?? []).map((v, q) => v - c * (atom[q] ?? 0))
    }
  }
  return { dictionary: out, code: newCode, unused }
}

/**
 * Dictionary learning as a traced {@link Algorithm}, one iteration per step: code every signal over the atoms
 * (`fitAlgorithm`), then update the atoms (`dictUpdate`) and replace the ones nobody uses. The objective is measured
 * after the update with the codes of the iteration, as scikit-learn's `errors_`; the run stops once its relative
 * decrease falls below `tol` (from the second iteration on, as scikit-learn). It observes the objective, the error,
 * the sparsity, the coherence, the atoms and the codes.
 *
 * With `cd` and an exact coder (`lassoLars`, or `lassoCd` at a small tolerance) every iteration lowers the objective;
 * `mod` and `ksvd` choose the atoms for the codes and then give the codes the scale of the unit atoms, which suits
 * the coders that fix the number of atoms (`omp`, `lars`) and leaves the penalised ones with no such guarantee.
 *
 * @param x - Signals `[n, d]`.
 * @param dictionary0 - The starting atoms `[k, d]`.
 * @param options - Penalty, coder, update, tolerance and positivity.
 * @returns The algorithm.
 * @throws OmaiError `value` when `positiveDict` is asked of `mod` or `ksvd`, or `positiveCode` of `ksvd`.
 * @concept dictionary-learning
 * @example
 * ```ts
 * import { dictionaryLearningAlgorithm } from 'omai-engine/decomposition'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const x = [[2, 0], [0, 3], [4, 0], [0, -1]]
 * const alg = dictionaryLearningAlgorithm(x, [[1, 0.1], [0.1, 1]], { dictUpdate: 'ksvd', fitAlgorithm: 'omp', fitNonzeroCoefs: 1 })
 * runAlgorithm(alg, { maxSteps: 5 }).state.error < 1e-9 // => true
 * ```
 */
export function dictionaryLearningAlgorithm(
  x: Rows,
  dictionary0: Rows,
  options: DictionaryLearningOptions = {},
): Algorithm<DictionaryLearningState> {
  const update = options.dictUpdate ?? 'cd'
  const tol = options.tol ?? 1e-8
  const positiveDict = options.positiveDict ?? false
  const positiveCode = options.positiveCode ?? false
  if (update !== 'cd' && positiveDict)
    throw new OmaiError('value', `dictionaryLearning: ${update} has no positive dictionary option; use cd`)
  if (update === 'ksvd' && positiveCode)
    throw new OmaiError('value', 'dictionaryLearning: ksvd rewrites the codes and cannot keep them positive')
  const coding: FitCoding = {
    fitAlgorithm: options.fitAlgorithm ?? 'lassoLars',
    alpha: options.alpha ?? 1,
    fitNonzeroCoefs: options.fitNonzeroCoefs ?? null,
    positiveCode,
    codeMaxIter: options.fitMaxIter ?? 1000,
  }
  const penalty = penaltyWeight(coding)
  const n = x.length
  const d = x[0]?.length ?? 0
  const k = dictionary0.length
  const measured = (dictionary: number[][], code: number[][]) => {
    const { residual, objective } = objectiveOf(x, dictionary, code, penalty)
    return { objective, error: Math.sqrt(residual), sparsity: meanNonzeros(code), coherence: coherenceOf(dictionary) }
  }
  return {
    name: `dictionary-learning-${update}`,
    config: { ...coding, dictUpdate: update, tol, positiveDict },
    schema: SCHEMA,
    init: () => {
      const dictionary = dictionary0.map((r) => [...r])
      const code = x.map(() => new Array<number>(k).fill(0))
      return { dictionary, code, ...measured(dictionary, code), unused: 0, change: 0, iteration: 0, stalled: false }
    },
    step: (s, ctx) => {
      const coded = encode(x, s.dictionary, coding)
      let next: Update
      if (update === 'cd')
        next = updateAtoms(s.dictionary, x, coded, mm(tr(coded), coded), mm(tr(x), coded), ctx.key, positiveDict)
      else if (update === 'mod') next = modUpdate(s.dictionary, x, coded, ctx.key)
      else next = ksvdUpdate(s.dictionary, x, coded, ctx.key)
      const m = measured(next.dictionary, next.code)
      const decrease = s.objective - m.objective
      return {
        dictionary: next.dictionary,
        code: next.code,
        ...m,
        unused: next.unused,
        change: ctx.t > 0 ? decrease / (m.objective || 1) : 0,
        iteration: s.iteration + 1,
        stalled: ctx.t > 0 && decrease < tol * m.objective,
      }
    },
    done: (s) => (s.stalled ? 'converged' : undefined),
    observe: (s) => ({
      metrics: {
        objective: s.objective,
        error: s.error,
        sparsity: s.sparsity,
        coherence: s.coherence,
        unused: s.unused,
        change: s.change,
      },
      tensors: {
        dictionary: { role: 'param', value: s.dictionary.flat(), shape: [k, d] },
        codes: { role: 'state', value: s.code.flat(), shape: [n, k] },
      },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as DictionaryLearningState,
  }
}

/** How the atoms start: the SVD of scikit-learn, random signals (K-SVD's classical start), random directions, or given. */
export type DictionaryInit = 'svd' | 'samples' | 'random' | 'custom'

/** Hyperparameters of {@link dictionaryLearning}. */
export interface DictionaryLearningHyper {
  /** The number of atoms (`null`: the number of features, scikit-learn's default). */
  readonly nComponents: number | null
  readonly alpha: number
  readonly maxIter: number
  readonly tol: number
  /** The coder of the fit; scikit-learn's `'lars'` is `'lassoLars'` and its `'cd'` is `'lassoCd'`. */
  readonly fitAlgorithm: SparseCodeAlgorithm
  readonly fitNonzeroCoefs: number | null
  readonly fitMaxIter: number
  /** The update of the atoms: the slot. */
  readonly dictUpdate: DictionaryUpdate
  readonly init: DictionaryInit
  /** The atoms for `init: 'custom'`. */
  readonly dictInit: number[][] | null
  /** The coder of `transform` (default `omp`, scikit-learn's). */
  readonly transformAlgorithm: SparseCodeAlgorithm
  readonly transformNonzeroCoefs: number | null
  /** The penalty of the `transform` coder (`null`: `alpha`). */
  readonly transformAlpha: number | null
  readonly transformMaxIter: number
  readonly splitSign: boolean
  readonly positiveCode: boolean
  readonly positiveDict: boolean
}

/** The fitted parameters of {@link dictionaryLearning}. */
export type DictionaryLearningParams = {
  /** The atoms `[k, d]`. */
  readonly components: number[][]
  /** The codes of the training signals. */
  readonly codes: number[][]
  readonly objective: number
  readonly reconstructionErr: number
  readonly nIter: number
  readonly dictUpdate: DictionaryUpdate
}

/** The starting atoms of a fit. */
function startingAtoms(
  x: Rows,
  h: Pick<DictionaryLearningHyper, 'nComponents' | 'init' | 'dictInit'>,
  key: Key,
): number[][] {
  const k = h.nComponents ?? x[0]?.length ?? 0
  const d = x[0]?.length ?? 0
  if (h.init === 'custom') {
    if (h.dictInit === null) throw new OmaiError('value', "dictionaryLearning: init 'custom' needs dictInit")
    if (h.dictInit.length !== k || h.dictInit.some((r) => r.length !== d))
      throw new OmaiError('shape', `dictionaryLearning: dictInit must be [${k}, ${d}]`)
    return h.dictInit
  }
  if (h.init === 'random') return randomDictionary(k, d, key)
  if (h.init === 'samples') return sampleDictionary(x, k, key)
  return svdDictionary(x, k)
}

/**
 * Dictionary learning (sklearn.decomposition.DictionaryLearning) on the iterative base: atoms and sparse codes found
 * by alternating coding and an update of the atoms (`dictUpdate`: block coordinate descent, MOD or K-SVD), one
 * iteration per step. `transform` codes new signals with `transformAlgorithm` (OMP by default), `inverseTransform`
 * writes signals from codes. The fit's exposition is a components record of the atoms with the codes of the training
 * signals as scores; a session offers `dict/atoms` and `dict/usage` at any step.
 *
 * @index estimator Dictionary learning by alternating sparse coding and a cd, MOD or K-SVD atom update.
 * @concept dictionary-learning
 * @example
 * ```ts
 * import { dictionaryLearning } from 'omai-engine/decomposition'
 * const x = [[2, 0, 1], [0, 3, 0], [4, 0, 2], [0, -1, 0], [1, 1, 0.5], [2, 3, 1]]
 * const f = dictionaryLearning.with({ nComponents: 2, alpha: 0.1, maxIter: 30 }).fit({ x })
 * [f.params.components.length, f.transform(x).shape] // => [2, [6, 2]]
 * ```
 */
export const dictionaryLearning: IterativeEstimator<
  DictionaryLearningHyper,
  DictionaryLearningState,
  DictionaryLearningParams,
  DictionaryVerbs
> = iterative<DictionaryLearningHyper, DictionaryLearningState, DictionaryLearningParams, DictionaryVerbs>({
  name: 'decomposition/dictionary-learning',
  kind: 'transformer',
  defaults: {
    nComponents: null,
    alpha: 1,
    maxIter: 1000,
    tol: 1e-8,
    fitAlgorithm: 'lassoLars',
    fitNonzeroCoefs: null,
    fitMaxIter: 1000,
    dictUpdate: 'cd',
    init: 'svd',
    dictInit: null,
    transformAlgorithm: 'omp',
    transformNonzeroCoefs: null,
    transformAlpha: null,
    transformMaxIter: 1000,
    splitSign: false,
    positiveCode: false,
    positiveDict: false,
  },
  hyperSpace: (h) =>
    dict({
      ...(h.nComponents === null ? {} : { nComponents: int(1, 100, { default: h.nComponents, label: 'atoms' }) }),
      alpha: real(0.001, 100, { log: true, default: h.alpha, label: 'penalty' }),
      fitAlgorithm: choice(['lassoLars', 'lassoCd', 'omp', 'lars', 'threshold'] as const, {
        default: h.fitAlgorithm,
        label: 'coder',
      }),
      dictUpdate: choice(['cd', 'mod', 'ksvd'] as const, { default: h.dictUpdate, label: 'atom update' }),
      maxIter: int(1, 1000, { default: h.maxIter, label: 'iterations' }),
    }),
  algorithm: (data, h, key) => {
    const x = asRows(data.x, 'dictionaryLearning')
    return dictionaryLearningAlgorithm(x, startingAtoms(x, h, key), h)
  },
  contract: () => dictionaryContract<DictionaryLearningState>(),
  maxSteps: (h) => h.maxIter,
  params: (s, _data, h) => ({
    components: s.dictionary,
    codes: s.code,
    objective: s.objective,
    reconstructionErr: s.error,
    nIter: s.iteration,
    dictUpdate: h.dictUpdate,
  }),
  records: (s) => [
    componentsRecord({
      method: 'dictionary learning',
      features: Array.from({ length: s.dictionary[0]?.length ?? 0 }, (_, q) => `x${q}`),
      components: s.dictionary,
      scores: s.code,
    }),
  ],
  verbs: (p, h) => dictionaryVerbs(p.components, h),
})

/** The result of {@link dictLearning}. */
export interface DictLearningResult {
  /** The codes `[n, k]` of the last iteration. */
  readonly code: number[][]
  /** The atoms `[k, d]`. */
  readonly dictionary: number[][]
  /** The objective after each iteration (scikit-learn's `errors`). */
  readonly errors: readonly number[]
  readonly nIter: number
}

/**
 * Dictionary learning as a function (sklearn.decomposition.dict_learning): the codes, the atoms and the objective after
 * each iteration, from the SVD start (or `dictInit`).
 *
 * @param x - Signals `[n, d]`.
 * @param nComponents - The number of atoms.
 * @param alpha - The penalty of the lasso coders.
 * @param options - Everything {@link DictionaryLearningOptions} has, `maxIter` (default 100), `dictInit`, and the key of any resampling.
 * @returns The codes, atoms, objectives and the number of iterations.
 * @concept dictionary-learning
 * @index function Dictionary learning by alternating sparse coding and an atom update, as a function.
 * @example
 * ```ts
 * import { dictLearning } from 'omai-engine/decomposition'
 * const r = dictLearning([[2, 0], [0, 3], [4, 0], [0, 1]], 2, 0.1, { maxIter: 20 })
 * [r.dictionary.length, r.errors.length > 0] // => [2, true]
 * ```
 */
export function dictLearning(
  x: Rows,
  nComponents: number,
  alpha: number,
  options: DictionaryLearningOptions & { readonly maxIter?: number; readonly dictInit?: Rows; readonly key?: Key } = {},
): DictLearningResult {
  const start = options.dictInit?.map((r) => [...r]) ?? svdDictionary(x, nComponents)
  const base = dictionaryLearningAlgorithm(x, start, { ...options, alpha })
  const errors: number[] = []
  const algorithm: Algorithm<DictionaryLearningState> = {
    ...base,
    step: (s, ctx) => {
      const next = base.step(s, ctx)
      errors.push(next.objective)
      return next
    },
  }
  const out = runAlgorithm(algorithm, {
    maxSteps: options.maxIter ?? 100,
    ...(options.key === undefined ? {} : { key: options.key }),
  })
  return { code: out.state.code, dictionary: out.state.dictionary, errors, nIter: out.state.iteration }
}
