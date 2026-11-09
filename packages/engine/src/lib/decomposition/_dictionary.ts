/**
 * What the batch and the online dictionary learners share and nobody else needs: coding a set of signals over a
 * dictionary as rows, the objective and its measures (sparsity, coherence), the initial dictionaries (the SVD one of
 * scikit-learn, random signals, random directions), and the block coordinate descent update of the atoms with its
 * handling of atoms nobody uses (replaced by the worst-represented signals).
 */

import { toRows } from '../base/index.ts'
import { curveSet } from '../exposition/index.ts'
import { matrixView } from '../learn/index.ts'
import { dnrm2 } from '../numerics/linalg/index.ts'
import { mutualCoherence } from '../optim/index.ts'
import { foldIn, type Key, normal, permutation } from '../random/index.ts'
import type { Tensor, TensorLike } from '../tensor/index.ts'
import { type Contract, defineContract, derived } from '../trace/index.ts'
import { mm, rowsOf, thinSvd, tr } from './_linear.ts'
import {
  coderOptions,
  type SparseCodeAlgorithm,
  type SparseCodeOptions,
  sparseCode,
  splitSign,
  synthesise,
} from './sparse-coding.ts'

/** Row arrays. */
export type Rows = readonly (readonly number[])[]

/** How a learner's coding step is configured: the coder, its penalty or atoms, and positivity. */
export interface FitCoding {
  readonly fitAlgorithm: SparseCodeAlgorithm
  readonly alpha: number
  readonly fitNonzeroCoefs: number | null
  readonly positiveCode: boolean
  readonly codeMaxIter: number
}

/** The codes of signals over a dictionary under a learner's coding configuration. */
export function encode(x: Rows, dictionary: Rows, c: FitCoding): number[][] {
  const options: SparseCodeOptions = {
    algorithm: c.fitAlgorithm,
    alpha: c.alpha,
    positive: c.positiveCode,
    maxIter: c.codeMaxIter,
    ...(c.fitNonzeroCoefs === null ? {} : { nNonzeroCoefs: c.fitNonzeroCoefs }),
  }
  return rowsOf(sparseCode(x as number[][], dictionary as number[][], options))
}

/** The weight on |code|_1 in the objective: the penalty of the penalised coders, none for the ones that fix the atoms. */
export function penaltyWeight(c: Pick<FitCoding, 'fitAlgorithm' | 'alpha'>): number {
  return c.fitAlgorithm === 'omp' || c.fitAlgorithm === 'lars' ? 0 : c.alpha
}

/** The squared Frobenius residual |X - C D|^2 and the objective half of it plus `penalty` |C|_1. */
export function objectiveOf(
  x: Rows,
  dictionary: Rows,
  code: Rows,
  penalty: number,
): { readonly residual: number; readonly objective: number } {
  const fit = mm(code, dictionary)
  let residual = 0
  for (const [i, row] of x.entries()) for (const [q, v] of row.entries()) residual += (v - (fit[i]?.[q] ?? 0)) ** 2
  let l1 = 0
  for (const row of code) for (const v of row) l1 += Math.abs(v)
  return { residual, objective: 0.5 * residual + penalty * l1 }
}

/** The mean number of nonzero entries per row. */
export function meanNonzeros(code: Rows): number {
  let count = 0
  for (const row of code) for (const v of row) if (v !== 0) count++
  return count / Math.max(code.length, 1)
}

/** The mutual coherence of the atoms (rows). */
export function coherenceOf(dictionary: Rows): number {
  return dictionary.length < 2 ? 0 : mutualCoherence(tr(dictionary))
}

/** Row `i` of x. */
const rowAt = (x: Rows, i: number): readonly number[] => x[i] ?? []

/**
 * scikit-learn's initial dictionary: the top singular directions scaled by their singular values (signs fixed so each
 * left vector's largest entry is positive), zero atoms for components beyond the rank.
 */
export function svdDictionary(x: Rows, k: number): number[][] {
  const n = x.length
  const d = x[0]?.length ?? 0
  const { u, s, vt } = thinSvd(Float64Array.from(x.flat()), n, d)
  const out: number[][] = []
  for (let c = 0; c < Math.min(k, s.length); c++) {
    let best = 0
    for (let i = 1; i < n; i++) if (Math.abs(u[i]?.[c] ?? 0) > Math.abs(u[best]?.[c] ?? 0)) best = i
    const sign = (u[best]?.[c] ?? 0) < 0 ? -1 : 1
    out.push((vt[c] ?? []).map((v) => sign * (s[c] ?? 0) * v))
  }
  while (out.length < k) out.push(new Array<number>(d).fill(0))
  return out
}

const unit = (v: readonly number[]): number[] => {
  const n = dnrm2(v) || 1
  return v.map((a) => a / n)
}

/** `k` distinct random signals, normalised: the classical initialisation of K-SVD. */
export function sampleDictionary(x: Rows, k: number, key: Key): number[][] {
  const order = permutation(key, x.length).data
  return Array.from({ length: k }, (_, j) => unit(rowAt(x, Number(order[j % x.length] ?? 0))))
}

/** `k` random unit directions of dimension `d`. */
export function randomDictionary(k: number, d: number, key: Key): number[][] {
  const z = normal(key, [k * d]).data
  return Array.from({ length: k }, (_, j) => unit(Array.from({ length: d }, (_, q) => z[j * d + q] ?? 0)))
}

/** The signals ordered by how badly the code and dictionary represent them, worst first. */
export function worstRepresented(x: Rows, dictionary: Rows, code: Rows): number[] {
  const fit = mm(code, dictionary)
  const err = x.map((row, i) => row.reduce((a, v, q) => a + (v - (fit[i]?.[q] ?? 0)) ** 2, 0))
  return err.map((_, i) => i).sort((a, b) => (err[b] ?? 0) - (err[a] ?? 0) || a - b)
}

/**
 * A replacement for an unused atom: a badly represented signal plus a little noise (a hundredth of the signal's
 * standard deviation, as scikit-learn), so that the sparse coding stays well conditioned.
 */
export function freshAtom(signal: readonly number[], key: Key): number[] {
  const m = signal.reduce((a, v) => a + v, 0) / Math.max(signal.length, 1)
  const sd = Math.sqrt(signal.reduce((a, v) => a + (v - m) ** 2, 0) / Math.max(signal.length, 1)) || 1
  const z = normal(key, [signal.length]).data
  return signal.map((v, q) => v + 0.01 * sd * (z[q] ?? 0))
}

/**
 * One pass of block coordinate descent on the atoms (Mairal et al. 2009, Algorithm 2; scikit-learn's `_update_dict`):
 * `d_j <- d_j + (b_j - D a_j) / A_jj`, projected on the unit ball (and the positive orthant when asked), the
 * statistics `A = C^T C` and `B = X^T C` given. An atom whose `A_jj` is nearly zero is unused and replaced by the
 * signal represented worst (the next worst for the next one), its code column zeroed.
 *
 * @returns The new dictionary, the code with the replaced atoms' columns zeroed, and how many atoms were replaced.
 */
export function updateAtoms(
  dictionary: Rows,
  x: Rows,
  code: Rows,
  a: Rows,
  b: Rows,
  key: Key,
  positive: boolean,
): { readonly dictionary: number[][]; readonly code: number[][]; readonly unused: number } {
  const k = dictionary.length
  const d = dictionary[0]?.length ?? 0
  const out = dictionary.map((r) => [...r])
  const newCode = code.map((r) => [...r])
  let unused = 0
  let order: number[] | undefined
  for (let j = 0; j < k; j++) {
    const ajj = a[j]?.[j] ?? 0
    const atom = out[j] as number[]
    if (ajj > 1e-6) {
      for (let q = 0; q < d; q++) {
        let da = 0
        for (let r = 0; r < k; r++) da += (a[j]?.[r] ?? 0) * (out[r]?.[q] ?? 0)
        atom[q] = (atom[q] ?? 0) + ((b[q]?.[j] ?? 0) - da) / ajj
      }
    } else {
      order ??= worstRepresented(x, dictionary, code)
      const pick = order[unused % order.length] ?? 0
      out[j] = freshAtom(rowAt(x, pick), foldIn(key, j))
      for (const row of newCode) row[j] = 0
      unused++
    }
    const now = out[j] as number[]
    if (positive) for (let q = 0; q < d; q++) now[q] = Math.max(now[q] ?? 0, 0)
    const norm = Math.max(dnrm2(now), 1)
    for (let q = 0; q < d; q++) now[q] = (now[q] ?? 0) / norm
  }
  return { dictionary: out, code: newCode, unused }
}

/** Rows of a matrix-like input, checking the feature count when it is known. */
export const asRows = (x: TensorLike, who: string, d?: number): number[][] => {
  const m = matrixView(x, who, d === undefined ? {} : { nFeatures: d })
  return toRows(m.data, m.n, m.d)
}

/** The hyperparameters of a dictionary learner that `transform` reads (scikit-learn's `transform_*`, `split_sign`). */
export interface TransformHyper {
  readonly alpha: number
  readonly transformAlgorithm: SparseCodeAlgorithm
  readonly transformNonzeroCoefs: number | null
  /** The penalty of the transform coder (`null`: `alpha`). */
  readonly transformAlpha: number | null
  readonly transformMaxIter: number
  readonly splitSign: boolean
  readonly positiveCode: boolean
}

/** The verbs of a fitted dictionary: sparse codes of new signals and the signals codes write. */
export interface DictionaryVerbs {
  readonly transform: (x: TensorLike) => Tensor
  readonly inverseTransform: (code: TensorLike) => Tensor
  readonly nFeaturesIn: number
}

/** The verbs of a dictionary under the hyperparameters of its transform. */
export function dictionaryVerbs(components: number[][], h: TransformHyper): DictionaryVerbs {
  const d = components[0]?.length ?? 0
  return {
    transform: (x) => {
      const code = sparseCode(asRows(x, 'dictionary transform', d), components, {
        ...coderOptions({ ...h, transformAlpha: h.transformAlpha ?? h.alpha }),
      })
      return h.splitSign ? splitSign(code) : code
    },
    inverseTransform: (code) => synthesise(code, components, h.splitSign),
    nFeaturesIn: d,
  }
}

/** What the probes of a dictionary learner read of a state. */
export interface AtomsState {
  readonly dictionary: number[][]
  readonly code: number[][]
}

/** The curves of the atoms, one per atom against the feature, and how many signals use each, as probes of a step. */
export function dictionaryContract<S extends AtomsState>(): Contract<S, 'dict/atoms' | 'dict/usage'> {
  return defineContract({
    'dict/atoms': derived<S>({
      description: 'the atoms against the feature, one curve each',
      compute: (s) =>
        curveSet({
          x: Array.from({ length: s.dictionary[0]?.length ?? 0 }, (_, q) => q),
          xLabel: 'feature',
          series: s.dictionary.map((a, j) => ({ name: `atom ${j + 1}`, y: a })),
        }),
    }),
    'dict/usage': derived<S>({
      description: 'the fraction of signals whose code uses each atom',
      compute: (s) =>
        curveSet({
          x: s.dictionary.map((_, j) => j),
          xLabel: 'atom',
          series: [
            {
              name: 'usage',
              y: s.dictionary.map(
                (_, j) => s.code.filter((r) => (r[j] ?? 0) !== 0).length / Math.max(s.code.length, 1),
              ),
            },
          ],
        }),
    }),
  })
}
