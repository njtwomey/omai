/**
 * Linear-chain CRFs trained on labels that are not all there. Two models share one mechanism, the clamped oracle: a
 * sum over only those labellings (or hidden paths) that agree with what is known of a training example. The
 * partial-label CRF learns from incomplete annotation, each position carrying one label, a set of allowed labels or
 * nothing at all (Tsuboi et al. 2008; Bellare and McCallum 2007); the hidden-state CRF and its latent-dynamic variant
 * give every observed label several hidden substates whose assignment is never annotated, so a label can have
 * sub-structure (Quattoni et al. 2007; Morency et al. 2007). In both the negative log-likelihood is log Z (free) minus
 * log Z (clamped), and its gradient is the free minus the clamped expected feature counts, which is what the `clamped`
 * member of `StructuredOracle` returns; the generic objective of the structured solvers then needs no change. The
 * clamped sums are forward-backward on the chain with the excluded labels masked out (`maskChain`).
 */

import { argmaxOf, OmaiError } from '../../base/index.ts'
import { forwardBackward, viterbi } from '../inference/chain.ts'
import {
  type CrfInput,
  crfChain,
  crfExpectedFeatureCounts,
  crfFeatureCounts,
  crfParameters,
  crfUnigrams,
  type LinearChainCrf,
  linearChainCrf,
  maskChain,
  withCrfParameters,
} from './crf.ts'
import type { StructuredOracle } from './oracle.ts'

/**
 * What is known of the label at one position of a partially labelled sequence: a label (a name or an index), a set of
 * labels it may be, or `null` for unknown.
 */
export type LabelSet = number | string | readonly (number | string)[] | null

/** A partially labelled sequence: what is known of the label at each position. */
export type PartialLabelling = readonly LabelSet[]

function labelIndex(crf: LinearChainCrf, l: number | string): number {
  const i = typeof l === 'number' ? l : crf.labels.indexOf(l)
  if (!(i >= 0 && i < crf.labels.length)) throw new OmaiError('value', `partial-label CRF: unknown label ${String(l)}`)
  return i
}

/** The allowed-label flags of a partial labelling (undefined for an unknown position). */
function allowedOf(crf: LinearChainCrf, y: PartialLabelling): (boolean[] | undefined)[] {
  return y.map((set) => {
    if (set === null) return undefined
    const mask = new Array<boolean>(crf.labels.length).fill(false)
    for (const l of typeof set === 'object' ? set : [set]) mask[labelIndex(crf, l)] = true
    return mask
  })
}

/** The Hamming cost of the labels against a partial gold labelling: each label outside its position's set costs 1. */
function addCost(
  chain: ReturnType<typeof crfChain>,
  allowed: readonly (boolean[] | undefined)[],
): ReturnType<typeof crfChain> {
  return {
    ...chain,
    logEmission: chain.logEmission.map((row, t) => {
      const mask = allowed[t]
      return mask === undefined ? row : row.map((v, k) => v + (mask[k] === true ? 0 : 1))
    }),
  }
}

/**
 * The structured oracle of a linear-chain CRF that learns from partial labels. A position of a training example is a
 * label, a set of allowed labels or `null` (unknown); `clamped` runs forward-backward over the labellings that stay in
 * the sets, so with every set a singleton it equals the supervised model, with every position unknown it equals the
 * free expectations (the likelihood of any labelling is one), and in between it marginalises the missing labels
 * exactly. The negative log-likelihood of an example is the free log Z minus the clamped log Z; the free and the clamped
 * counts give its gradient. The parameter vector is that of {@link crfParameters}: state, transition, then bigram
 * weights.
 *
 * `features` needs a complete labelling (a singleton at every position); a partial one has no single count vector, which
 * is why the likelihood learners use `clamped`. `decode` is Viterbi over all labellings, with `against` adding one to
 * the score of each label outside its position's set (unknown positions cost nothing), and `loss` counts the positions
 * whose predicted label lies outside the gold set.
 *
 * @param crf - The model (labels, attributes, bigram attributes; its weights are not used).
 * @returns The oracle over input attributes and partial labellings.
 * @concept conditional-random-fields
 * @index estimator Linear-chain CRF oracle for partial labels: per-position label sets, clamped forward-backward.
 * @example
 * ```ts
 * import { linearChainCrf, partialLabelOracle } from 'omai-engine/pgm'
 * const oracle = partialLabelOracle(linearChainCrf(['A', 'B'], ['f']))
 * const w = new Float64Array(oracle.size)
 * const x = [{ f: 1 }, { f: 1 }]
 * Math.exp(oracle.expectations(w, x).logPartition - (oracle.clamped?.(w, x, [null, 'A']).logPartition ?? 0)) // => 2
 * ```
 */
export function partialLabelOracle(crf: LinearChainCrf): StructuredOracle<CrfInput, PartialLabelling> {
  const size = crfParameters(crf).length
  const check = (x: CrfInput, y: PartialLabelling, name: string): void => {
    if (y.length !== crfUnigrams(x).length) throw new OmaiError('shape', `${name}: one label set per item`)
  }
  return {
    size,
    features: (x, y) => {
      check(x, y, 'partialLabelOracle.features')
      const single = y.map((set) => {
        const flat = set === null ? [] : typeof set === 'object' ? set : [set]
        if (flat.length !== 1)
          throw new OmaiError('value', 'partialLabelOracle.features: needs one label at every position; use clamped')
        return labelIndex(crf, flat[0] as number | string)
      })
      return crfFeatureCounts(crf, x, single)
    },
    expectations: (w, x) => crfExpectedFeatureCounts(withCrfParameters(crf, w), x),
    clamped: (w, x, y) => {
      check(x, y, 'partialLabelOracle.clamped')
      return crfExpectedFeatureCounts(withCrfParameters(crf, w), x, allowedOf(crf, y))
    },
    decode: (w, x, options = {}) => {
      if (crfUnigrams(x).length === 0) return { y: [], score: 0 }
      const chain = crfChain(withCrfParameters(crf, w), x)
      const against = options.against
      if (against !== undefined) check(x, against, 'partialLabelOracle.decode')
      const v = viterbi(against === undefined ? chain : addCost(chain, allowedOf(crf, against)))
      return { y: v.path, score: v.logProbability }
    },
    loss: (gold, predicted) => {
      const allowed = allowedOf(crf, gold)
      return predicted.reduce<number>(
        (s, l, t) => s + (allowed[t]?.[labelIndex(crf, l as number | string)] === false ? 1 : 0),
        0,
      )
    },
  }
}

/**
 * A hidden-state CRF: a linear-chain CRF over `labels.length * substates` hidden states, where hidden state s belongs
 * to label floor(s / substates). `crf` is the chain over the hidden states (state `k * substates + h` is substate h of
 * label k, named `label#h`), with the usual parameter layout.
 */
export interface HiddenStateCrf {
  readonly kind: 'hiddenStateCrf'
  /** The observed labels. */
  readonly labels: readonly string[]
  /** Hidden substates per label. */
  readonly substates: number
  /** The linear-chain CRF over the hidden states. */
  readonly crf: LinearChainCrf
}

/**
 * Build a hidden-state CRF (Quattoni, Wang, Morency, Collins and Darrell 2007; with a chain of hidden states per label
 * it is the latent-dynamic CRF of Morency, Quattoni and Darrell 2007): every observed label owns `substates` hidden
 * states, the observed label of a hidden state is a deterministic map, the clamped sum covers the hidden paths that
 * map onto the gold labels and the free sum covers all of them. Hidden transitions let substates specialise (a label
 * "walk" with substates for the start, the middle and the end of a step). With zero weights every substate of a label
 * is identical and a gradient step cannot tell them apart, so start from small random weights.
 *
 * @param labels - The observed labels.
 * @param attributes - The attribute vocabulary.
 * @param substates - Hidden substates per label (1 gives the plain linear-chain CRF).
 * @param options - Bigram attributes and whether to have transitions, as {@link linearChainCrf}.
 * @returns The model with zero weights.
 * @throws OmaiError with code `value` for fewer than one substate.
 * @concept conditional-random-fields
 * @example
 * ```ts
 * import { hiddenStateCrf } from 'omai-engine/pgm'
 * hiddenStateCrf(['A', 'B'], ['f'], 2).crf.labels // => ['A#0', 'A#1', 'B#0', 'B#1']
 * ```
 */
export function hiddenStateCrf(
  labels: readonly string[],
  attributes: readonly string[],
  substates: number,
  options: { readonly bigramAttributes?: readonly string[]; readonly transitions?: boolean } = {},
): HiddenStateCrf {
  if (!(Number.isInteger(substates) && substates >= 1))
    throw new OmaiError('value', 'hiddenStateCrf: substates must be a positive integer')
  const hidden = labels.flatMap((l) => Array.from({ length: substates }, (_, h) => `${l}#${h}`))
  return { kind: 'hiddenStateCrf', labels: [...labels], substates, crf: linearChainCrf(hidden, attributes, options) }
}

/** The structured oracle of a hidden-state CRF, with the pieces a latent learner needs beside the contract. */
export interface HiddenStateOracle extends StructuredOracle<CrfInput, readonly (string | number)[]> {
  /**
   * The best hidden path that maps onto the labels `y` and its score: what a latent structured perceptron or the
   * concave-convex procedure of a latent structured SVM completes the example with. `counts` are the feature counts of
   * that path and `path` its hidden states.
   */
  completion(
    w: ArrayLike<number>,
    x: CrfInput,
    y: readonly (string | number)[],
  ): { readonly path: number[]; readonly counts: Float64Array; readonly score: number }
  /** The marginal p(label_t = k | x) at every position, summing the substates of a label. */
  labelMarginals(w: ArrayLike<number>, x: CrfInput): number[][]
}

/**
 * The structured oracle of a hidden-state CRF over observed label sequences. `clamped` sums over the hidden paths whose
 * labels equal the gold ones and `expectations` over all hidden paths, so the generic objective (free minus clamped) is
 * the hidden-state CRF's negative conditional log-likelihood; the parameters are those of the hidden chain
 * ({@link crfParameters} of `model.crf`).
 *
 * Decoding is where the latent-dynamic and the hidden-state CRF differ: with `decoding: 'marginal'` (default, Morency's
 * latent-dynamic inference) each position takes the label of largest marginal summed over its substates, and the score is
 * the log of the total score of the hidden paths that agree with the decoded labels (the clamped log-partition
 * function); with `'viterbi'` the labels of the best hidden path are taken (the exact MAP over hidden paths, not over
 * labellings, which is intractable). `decode` with `against` always uses Viterbi with one added to each hidden state
 * whose label differs from the gold one. `features` throws: the counts of a latent labelling depend on the weights, so
 * use `clamped` or `completion`.
 *
 * @param model - The hidden-state CRF.
 * @param options - The decoding rule.
 * @returns The oracle over input attributes and observed label sequences.
 * @concept conditional-random-fields
 * @index estimator Hidden-state and latent-dynamic CRF oracle: substates per label, clamped sums over hidden paths.
 * @example
 * ```ts
 * import { hiddenStateCrf, hiddenStateOracle } from 'omai-engine/pgm'
 * const oracle = hiddenStateOracle(hiddenStateCrf(['A', 'B'], ['f'], 2))
 * const w = new Float64Array(oracle.size)
 * oracle.clamped?.(w, [{ f: 1 }], ['A']).logPartition === Math.log(2) // => true
 * ```
 */
export function hiddenStateOracle(
  model: HiddenStateCrf,
  options: { readonly decoding?: 'marginal' | 'viterbi' } = {},
): HiddenStateOracle {
  const { crf, substates: H } = model
  const K = model.labels.length
  const decoding = options.decoding ?? 'marginal'
  const size = crfParameters(crf).length
  const resolve = (l: number | string): number => {
    const i = typeof l === 'number' ? l : model.labels.indexOf(l)
    if (!(i >= 0 && i < K)) throw new OmaiError('value', `hidden-state CRF: unknown label ${String(l)}`)
    return i
  }
  const mask = (y: readonly (number | string)[]): boolean[][] =>
    y.map((l) => {
      const k = resolve(l)
      return Array.from({ length: K * H }, (_, s) => Math.floor(s / H) === k)
    })
  const check = (x: CrfInput, y: readonly (number | string)[], name: string): void => {
    if (y.length !== crfUnigrams(x).length) throw new OmaiError('shape', `${name}: one label per item`)
  }
  const labelMarginals = (w: ArrayLike<number>, x: CrfInput): number[][] => {
    if (crfUnigrams(x).length === 0) return []
    const post = forwardBackward(crfChain(withCrfParameters(crf, w), x)).posteriors
    return post.map((row) =>
      Array.from({ length: K }, (_, k) => row.slice(k * H, (k + 1) * H).reduce((a, p) => a + p, 0)),
    )
  }
  const clampedAt = (
    w: ArrayLike<number>,
    x: CrfInput,
    y: readonly (number | string)[],
  ): { readonly counts: Float64Array; readonly logPartition: number } =>
    crfExpectedFeatureCounts(withCrfParameters(crf, w), x, mask(y))
  return {
    size,
    features: () => {
      throw new OmaiError('value', 'hiddenStateOracle.features: the hidden path is latent; use clamped or completion')
    },
    expectations: (w, x) => crfExpectedFeatureCounts(withCrfParameters(crf, w), x),
    clamped: (w, x, y) => {
      check(x, y, 'hiddenStateOracle.clamped')
      return clampedAt(w, x, y)
    },
    completion: (w, x, y) => {
      check(x, y, 'hiddenStateOracle.completion')
      if (y.length === 0) return { path: [], counts: new Float64Array(size), score: 0 }
      const v = viterbi(maskChain(crfChain(withCrfParameters(crf, w), x), mask(y)))
      return { path: v.path, counts: crfFeatureCounts(crf, x, v.path), score: v.logProbability }
    },
    labelMarginals,
    decode: (w, x, decodeOptions = {}) => {
      if (crfUnigrams(x).length === 0) return { y: [], score: 0 }
      const against = decodeOptions.against
      if (against === undefined && decoding === 'marginal') {
        const y = labelMarginals(w, x).map((row) => argmaxOf(row))
        return { y, score: clampedAt(w, x, y).logPartition }
      }
      let chain = crfChain(withCrfParameters(crf, w), x)
      if (against !== undefined) {
        check(x, against, 'hiddenStateOracle.decode')
        const gold = against.map(resolve)
        chain = {
          ...chain,
          logEmission: chain.logEmission.map((row, t) =>
            row.map((v, s) => v + (Math.floor(s / H) === gold[t] ? 0 : 1)),
          ),
        }
      }
      const v = viterbi(chain)
      return { y: v.path.map((s) => Math.floor(s / H)), score: v.logProbability }
    },
    loss: (gold, predicted) =>
      gold.reduce<number>((s, g, t) => s + (resolve(g) === resolve(predicted[t] ?? -1) ? 0 : 1), 0),
  }
}
