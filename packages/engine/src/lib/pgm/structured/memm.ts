/**
 * The maximum-entropy Markov model (McCallum, Freitag and Pereira 2000), the locally normalised cousin of the
 * linear-chain CRF. It has the CRF's parameters (state weights per attribute and label, transition weights, bigram
 * weights: the layout of `crfParameters`) but a different probability model: each label is a softmax over the next
 * labels given the previous label and the observation at that position,
 * p(y | x) = prod_t softmax_k(emission_t(k) + transition(y_{t-1}, k))[y_t], so every factor is a proper distribution.
 *
 * That makes training easy, a sum of independent softmax regressions with no global partition function (so a MEMM is
 * a `SmoothProblem` for the solver slot, `memmSmoothProblem`, and not a `StructuredOracle`: there is no log Z to
 * differentiate and the gradient needs no forward-backward), and it makes the model biased. A state with few
 * successors passes on all its probability mass whatever it observes: the label-bias problem (Lafferty, McCallum and
 * Pereira 2001), and the reason the CRF exists. The example of that paper is in `memm.test.ts`: two paths through six
 * states, "rib" and "rob", where the states after the shared first letter each have one successor, so the observation
 * of the second letter cannot matter and the model cannot prefer the right path, for any weights. A CRF normalises over
 * whole paths and can.
 */

import { logSumExpOf, OmaiError } from '../../base/index.ts'
import { analyticObjective, type SmoothProblem } from '../../optim/index.ts'
import type { Value } from '../../tensor/index.ts'
import { type ChainModel, viterbi } from '../inference/chain.ts'
import {
  type CrfInput,
  crfBigrams,
  crfChain,
  crfParameters,
  crfUnigrams,
  type LabelledSequence,
  type LinearChainCrf,
  withCrfParameters,
} from './crf.ts'

/**
 * Which labels a MEMM may take: the labels that may start a sequence and, for each previous label, the labels that may
 * follow it (a transition structure, as in a finite-state MEMM). A previous label with no allowed successor can only end
 * a sequence. The default allows everything.
 */
export interface MemmStructure {
  readonly start?: readonly boolean[]
  readonly transition?: readonly (readonly boolean[])[]
}

/** Row-wise log-softmax of `scores` restricted to the allowed entries (disallowed entries become minus infinity). */
function logSoftmaxRow(scores: readonly number[], allowed: readonly boolean[] | undefined): number[] {
  const masked =
    allowed === undefined ? scores : scores.map((s, k) => (allowed[k] === false ? Number.NEGATIVE_INFINITY : s))
  const z = logSumExpOf(masked)
  return z === Number.NEGATIVE_INFINITY ? masked.map(() => Number.NEGATIVE_INFINITY) : masked.map((s) => s - z)
}

const transitionOf = (chain: ChainModel, t: number): readonly (readonly number[])[] => {
  const tr = chain.logTransition
  return (
    Array.isArray(tr[0]?.[0]) ? (tr as readonly (readonly (readonly number[])[])[])[t - 1] : tr
  ) as readonly (readonly number[])[]
}

/**
 * The chain of a MEMM on one sequence: the log of its local distributions, as a chain whose total is one. The initial
 * vector is the log-softmax of the first item's scores, step t carries log p(y_t = j | y_{t-1} = i, x_t) as its
 * transition matrix, and the emissions are zero, so `forwardBackward` (marginals), `viterbi` and `backwardSample` of
 * the chain module apply directly.
 *
 * @param crf - The model (weights are the parameters).
 * @param x - The item (and edge) attributes.
 * @param structure - Allowed first labels and transitions.
 * @returns The locally normalised chain.
 * @concept maximum-entropy-markov-models
 * @example
 * ```ts
 * import { linearChainCrf, memmChain } from 'omai-engine/pgm'
 * memmChain(linearChainCrf(['A', 'B'], ['f'], { state: [Math.log(3), 0] }), [{ f: 1 }]).logInitial.map(Math.exp) // => [0.75, 0.25]
 * ```
 */
export function memmChain(crf: LinearChainCrf, x: CrfInput, structure: MemmStructure = {}): ChainModel {
  const K = crf.labels.length
  const potentials = crfChain(crf, x)
  const T = potentials.logEmission.length
  const zero = potentials.logEmission.map(() => new Array<number>(K).fill(0))
  const logInitial =
    T === 0 ? new Array<number>(K).fill(0) : logSoftmaxRow(potentials.logEmission[0] ?? [], structure.start)
  const logTransition = Array.from({ length: Math.max(T - 1, 0) }, (_, s) => {
    const A = transitionOf(potentials, s + 1)
    const emission = potentials.logEmission[s + 1] ?? []
    return Array.from({ length: K }, (_, i) =>
      logSoftmaxRow(
        emission.map((e, k) => e + (A[i]?.[k] ?? 0)),
        structure.transition?.[i],
      ),
    )
  })
  // a chain of one step has no transition; keep the homogeneous matrix shape for forwardBackward's check
  return {
    logInitial,
    logTransition: T <= 1 ? Array.from({ length: K }, () => new Array<number>(K).fill(0)) : logTransition,
    logEmission: zero,
  }
}

function indices(crf: LinearChainCrf, y: readonly (string | number)[]): number[] {
  return y.map((l) => {
    const i = typeof l === 'number' ? l : crf.labels.indexOf(l)
    if (!(i >= 0 && i < crf.labels.length)) throw new OmaiError('value', `memm: unknown label ${String(l)}`)
    return i
  })
}

/**
 * The log-likelihood log p(y | x) of a labelling under a MEMM: the sum of the log local probabilities of its labels.
 *
 * @param crf - The model (weights are the parameters).
 * @param x - The item (and edge) attributes.
 * @param y - The labels.
 * @param structure - Allowed first labels and transitions.
 * @returns log p(y | x), minus infinity for a labelling the structure forbids.
 * @throws OmaiError with code `shape` for a different number of labels and items.
 * @concept maximum-entropy-markov-models
 * @example
 * ```ts
 * import { linearChainCrf, memmLogLikelihood } from 'omai-engine/pgm'
 * memmLogLikelihood(linearChainCrf(['A', 'B'], ['f']), [{ f: 1 }, { f: 1 }], ['A', 'B']) === 2 * Math.log(0.5) // => true
 * ```
 */
export function memmLogLikelihood(
  crf: LinearChainCrf,
  x: CrfInput,
  y: readonly (string | number)[],
  structure: MemmStructure = {},
): number {
  const idx = indices(crf, y)
  if (idx.length !== crfUnigrams(x).length) throw new OmaiError('shape', 'memmLogLikelihood: one label per item')
  const chain = memmChain(crf, x, structure)
  let s = 0
  for (const [t, k] of idx.entries())
    s += t === 0 ? (chain.logInitial[k] ?? 0) : (transitionOf(chain, t)[idx[t - 1] ?? 0]?.[k] ?? 0)
  return s
}

/**
 * The most probable labelling of a MEMM (Viterbi over the local distributions) and its log probability.
 *
 * @param crf - The model (weights are the parameters).
 * @param x - The item (and edge) attributes.
 * @param structure - Allowed first labels and transitions.
 * @returns The labels and log p(y | x).
 * @concept maximum-entropy-markov-models
 * @example
 * ```ts
 * import { linearChainCrf, memmDecode } from 'omai-engine/pgm'
 * memmDecode(linearChainCrf(['A', 'B'], ['f'], { state: [1, 0], transition: [0, 5, 0, 0] }), [{ f: 1 }, { f: 1 }]).labels // => ['A', 'B']
 * ```
 */
export function memmDecode(
  crf: LinearChainCrf,
  x: CrfInput,
  structure: MemmStructure = {},
): { readonly labels: string[]; readonly logProbability: number } {
  if (crfUnigrams(x).length === 0) return { labels: [], logProbability: 0 }
  const v = viterbi(memmChain(crf, x, structure))
  return { labels: v.path.map((k) => crf.labels[k] ?? ''), logProbability: v.logProbability }
}

/**
 * The negative conditional log-likelihood of labelled sequences under a MEMM, plus `c2 ||w||^2`, and its gradient: for
 * each position the local softmax's expected minus observed feature counts, with the previous label held at its
 * observed value (teacher forcing, which is exactly maximum likelihood for a locally normalised model). No forward-backward
 * is needed. The gradient is in {@link crfParameters} order.
 *
 * @param crf - The model (its weights are the point of evaluation).
 * @param data - Labelled sequences.
 * @param options - The L2 coefficient and the transition structure.
 * @returns The objective and its gradient.
 * @throws OmaiError with code `value` when a training labelling violates the structure.
 * @concept maximum-entropy-markov-models
 * @index function MEMM negative log-likelihood and gradient, a sum of local softmax regressions.
 * @example
 * ```ts
 * import { linearChainCrf, memmObjective } from 'omai-engine/pgm'
 * const r = memmObjective(linearChainCrf(['A', 'B'], ['f']), [{ x: [{ f: 1 }], y: ['A'] }])
 * [r.value === Math.LN2, Array.from(r.gradient)] // => [true, [-0.5, 0.5, 0, 0, 0, 0]]
 * ```
 */
export function memmObjective(
  crf: LinearChainCrf,
  data: readonly LabelledSequence[],
  options: { readonly c2?: number; readonly structure?: MemmStructure } = {},
): { readonly value: number; readonly gradient: Float64Array } {
  const K = crf.labels.length
  const nState = crf.stateWeights.length
  const nTrans = crf.transitionWeights.length
  const bigramAt = nState + nTrans
  const structure = options.structure ?? {}
  const gradient = new Float64Array(bigramAt + crf.bigramWeights.length)
  const attribute = new Map(crf.attributes.map((a, i) => [a, i]))
  const edgeAttribute = new Map(crf.bigramAttributes.map((a, i) => [a, i]))
  const add = (i: number, v: number): void => {
    gradient[i] = (gradient[i] ?? 0) + v
  }
  let value = 0
  for (const { x, y } of data) {
    const items = crfUnigrams(x)
    if (items.length === 0) continue
    const idx = indices(crf, y)
    if (idx.length !== items.length) throw new OmaiError('shape', 'memmObjective: one label per item')
    const edges = crfBigrams(x)
    const potentials = crfChain(crf, x)
    for (const [t, item] of items.entries()) {
      const yt = idx[t] ?? 0
      const prev = idx[t - 1] ?? 0
      const A = t === 0 ? undefined : transitionOf(potentials, t)
      const logP = logSoftmaxRow(
        (potentials.logEmission[t] ?? []).map((e, k) => e + (A?.[prev]?.[k] ?? 0)),
        t === 0 ? structure.start : structure.transition?.[prev],
      )
      const lp = logP[yt] ?? Number.NEGATIVE_INFINITY
      if (lp === Number.NEGATIVE_INFINITY)
        throw new OmaiError('value', `memmObjective: the labelling has zero probability at position ${t}`)
      value -= lp
      const resid = logP.map((l, k) => Math.exp(l) - (k === yt ? 1 : 0))
      for (const [a, v] of Object.entries(item)) {
        const i = attribute.get(a)
        if (i !== undefined) for (let k = 0; k < K; k++) add(i * K + k, v * (resid[k] ?? 0))
      }
      if (t === 0) continue
      if (nTrans > 0) for (let k = 0; k < K; k++) add(nState + prev * K + k, resid[k] ?? 0)
      for (const [b, v] of Object.entries(edges?.[t] ?? {})) {
        const a = edgeAttribute.get(b)
        if (a !== undefined) for (let k = 0; k < K; k++) add(bigramAt + (a * K + prev) * K + k, v * (resid[k] ?? 0))
      }
    }
  }
  const c2 = options.c2 ?? 0
  const w = crfParameters(crf)
  for (let i = 0; i < w.length; i++) {
    value += c2 * (w[i] ?? 0) ** 2
    gradient[i] = (gradient[i] ?? 0) + 2 * c2 * (w[i] ?? 0)
  }
  return { value, gradient }
}

/** What a MEMM is trained on: the starting model (labels and vocabularies), the labelled sequences and the penalty. */
export interface MemmTrainingProblem {
  readonly crf: LinearChainCrf
  readonly data: readonly LabelledSequence[]
  readonly c2?: number
  readonly structure?: MemmStructure
}

/**
 * The MEMM training objective as a smooth problem for the solver slot (`smoothSolvers`: L-BFGS, Newton, stochastic
 * gradient, ...): {@link memmObjective} over the flat weight vector, with batches over the sequences (the likelihood part
 * rescaled by n / |batch|, an unbiased estimate of the full objective). Because a MEMM is locally normalised there is no
 * partition function and no `StructuredOracle` for it; the solver slot takes it as it takes any other smooth model.
 *
 * @param problem - The model, data, L2 coefficient and structure.
 * @returns The problem: objective (with batches over the sequences) and the starting weights.
 * @concept maximum-entropy-markov-models
 * @example
 * ```ts
 * import { linearChainCrf, memmSmoothProblem } from 'omai-engine/pgm'
 * memmSmoothProblem({ crf: linearChainCrf(['A', 'B'], ['f']), data: [{ x: [{ f: 1 }], y: ['A'] }] }).x0.length // => 6
 * ```
 */
export function memmSmoothProblem(problem: MemmTrainingProblem): SmoothProblem {
  const { crf, data } = problem
  const c2 = problem.c2 ?? 0
  const n = data.length
  const structure = problem.structure ?? {}
  const at = (w: ArrayLike<number>, batch?: readonly number[]): { value: number; gradient: Float64Array } => {
    const model = withCrfParameters(crf, w)
    const rows = batch === undefined ? data : batch.map((i) => data[i] as LabelledSequence)
    const scale = batch === undefined ? 1 : n / Math.max(1, batch.length)
    const r = memmObjective(model, rows, { structure })
    const value = scale * r.value
    const gradient = r.gradient.map((g, i) => scale * g + 2 * c2 * (w[i] ?? 0))
    return { value: value + c2 * Array.from(w).reduce((s, v) => s + v * v, 0), gradient }
  }
  const objectiveOf = (batch?: readonly number[]): ((x: Value) => Value) =>
    analyticObjective({ value: (w) => at(w, batch).value, gradient: (w) => at(w, batch).gradient })
  const full = objectiveOf()
  return {
    objective: ((x: unknown, batch?: readonly number[]) =>
      (batch === undefined ? full : objectiveOf(batch))(x as Value)) as SmoothProblem['objective'],
    x0: Array.from(crfParameters(crf)),
    batches: { n },
  }
}
