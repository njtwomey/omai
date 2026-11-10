/**
 * Decoding a linear-chain CRF under constraints and beyond the single best path: the moves a tagging scheme allows
 * (`allowedTransitions` for BIO, IOB1, BIOES, BILOU and BMES label sets, with the allowed first and last labels), the
 * best labelling that obeys them (constrained Viterbi), the n best labellings with their probabilities, maximum
 * expected accuracy under the constraints (constrained posterior decoding) and samples from the posterior. All of them
 * are thin wrappers of the generic chain algorithms of `inference/chain.ts` over the potentials of the CRF, so a decoded
 * sequence can never contain an invalid transition such as `I-PER` after `O`.
 */

import { OmaiError } from '../../base/index.ts'
import type { Key } from '../../random/index.ts'
import {
  backwardSample,
  type ChainConstraints,
  constrainChain,
  constrainedPosteriorDecode,
  constrainedViterbi,
  forwardBackward,
  viterbiKBest,
} from '../inference/chain.ts'
import { type CrfInput, crfChain, crfUnigrams, type LinearChainCrf } from './crf.ts'

/**
 * A tagging scheme of entity labels like `B-PER`: `BIO` (also called IOB2: every entity starts with `B`), `IOB1` (`I`
 * starts an entity, `B` only separates two adjacent entities of one type), `BIOES` and `BILOU` (entities end with `E`
 * or `L`; single-token ones are `S` or `U`) and `BMES` (no `O`; the middle is `M`).
 */
export type TaggingScheme = 'BIO' | 'IOB2' | 'IOB1' | 'BIOES' | 'BILOU' | 'BMES'

/** The constraints a tagging scheme puts on a label set: the allowed moves and the allowed first and last labels. */
export interface SchemeConstraints extends ChainConstraints {
  readonly transitions: readonly (readonly boolean[])[]
  readonly start: readonly boolean[]
  readonly end: readonly boolean[]
}

type Role = 'B' | 'I' | 'E' | 'S' | 'O'

const letters: Record<TaggingScheme, Readonly<Record<string, Role>>> = {
  BIO: { B: 'B', I: 'I', O: 'O' },
  IOB2: { B: 'B', I: 'I', O: 'O' },
  IOB1: { B: 'B', I: 'I', O: 'O' },
  BIOES: { B: 'B', I: 'I', E: 'E', S: 'S', O: 'O' },
  BILOU: { B: 'B', I: 'I', L: 'E', U: 'S', O: 'O' },
  BMES: { B: 'B', M: 'I', E: 'E', S: 'S' },
}

interface Parsed {
  readonly role: Role
  readonly type: string
}

function parse(label: string, scheme: TaggingScheme): Parsed {
  const cut = label.indexOf('-')
  const prefix = cut < 0 ? label : label.slice(0, cut)
  const type = cut < 0 ? '' : label.slice(cut + 1)
  const role = letters[scheme][prefix]
  if (role === undefined || (role === 'O') !== cut < 0 || (role !== 'O' && type === ''))
    throw new OmaiError('value', `allowedTransitions: "${label}" is not a ${scheme} label`)
  return { role, type }
}

/**
 * The moves a tagging scheme allows between labels (AllenNLP's `allowed_transitions`, with the start and end of the
 * sequence): `transitions[i][j]` says whether label j may follow label i, `start` which labels may begin a sequence and
 * `end` which may end one. In BIO, `I-X` may follow only `B-X` or `I-X`; in BIOES and BILOU an entity once begun runs
 * through `I-X` to `E-X` (`L-X`) and a new one starts only after `O`, an end or a single; BMES is BIOES without `O`;
 * IOB1 lets `I-X` follow anything and `B-X` only `B-X` or `I-X`. Feed the result to `crfDecodeConstrained`, `crfNBest`,
 * `crfPosteriorDecode` or `crfSample`, or to `constrainedViterbi` on any chain.
 *
 * @param labels - The label names, `O` and `<prefix>-<type>`, in the order of the CRF's labels.
 * @param scheme - The tagging scheme.
 * @returns The allowed moves, start and end labels.
 * @throws OmaiError with code `value` for a label that is not in the scheme (a wrong prefix, `O` in BMES).
 * @concept sequence-labelling-metrics
 * @index function Allowed transitions, starts and ends of a tagging scheme (BIO, IOB1, BIOES, BILOU, BMES) over labels.
 * @example
 * ```ts
 * import { allowedTransitions } from 'omai-engine/pgm'
 * const c = allowedTransitions(['O', 'B-PER', 'I-PER'], 'BIO')
 * [c.transitions, c.start, c.end] // => [[[true, true, false], [true, true, true], [true, true, true]], [true, true, false], [true, true, true]]
 * ```
 */
export function allowedTransitions(labels: readonly string[], scheme: TaggingScheme): SchemeConstraints {
  const parsed = labels.map((l) => parse(l, scheme))
  const io = scheme === 'BIO' || scheme === 'IOB2' || scheme === 'IOB1'
  const allowed = (from: Parsed, to: Parsed): boolean => {
    const same = from.type === to.type
    const inEntity = from.role === 'B' || from.role === 'I'
    if (scheme === 'IOB1') return to.role === 'O' || to.role === 'I' || (to.role === 'B' && inEntity && same)
    if (io) return to.role === 'O' || to.role === 'B' || (to.role === 'I' && inEntity && same)
    if (inEntity) return (to.role === 'I' || to.role === 'E') && same
    return to.role === 'O' || to.role === 'B' || to.role === 'S'
  }
  const starts = (p: Parsed): boolean =>
    scheme === 'IOB1' ? p.role === 'O' || p.role === 'I' : p.role === 'O' || p.role === 'B' || (!io && p.role === 'S')
  const ends = (p: Parsed): boolean => io || p.role === 'O' || p.role === 'E' || p.role === 'S'
  return {
    transitions: parsed.map((from) => parsed.map((to) => allowed(from, to))),
    start: parsed.map(starts),
    end: parsed.map(ends),
  }
}

/** Constraints given directly, or as a tagging scheme read over the CRF's own labels. */
export type DecodingConstraints = ChainConstraints | TaggingScheme

const resolve = (crf: LinearChainCrf, constraints: DecodingConstraints | undefined): ChainConstraints =>
  constraints === undefined
    ? {}
    : typeof constraints === 'string'
      ? allowedTransitions(crf.labels, constraints)
      : constraints

const names = (crf: LinearChainCrf, path: readonly number[]): string[] => path.map((k) => crf.labels[k] ?? '')

/**
 * The most probable labelling that obeys the constraints (constrained Viterbi on the CRF): with a tagging scheme it is
 * always a valid tag sequence, which plain Viterbi decoding need not be.
 *
 * @param crf - The model.
 * @param x - The item (and edge) attributes.
 * @param constraints - Allowed moves over the CRF's labels, or a tagging scheme read over its label names.
 * @returns The labels and their unnormalised score.
 * @throws OmaiError with code `value` when no labelling satisfies the constraints.
 * @concept conditional-random-fields
 * @index function Viterbi decoding of a CRF under constraints (e.g. a tagging scheme): never an invalid sequence.
 * @example
 * ```ts
 * import { crfDecodeConstrained, linearChainCrf } from 'omai-engine/pgm'
 * const crf = linearChainCrf(['O', 'B-X', 'I-X'], ['f'], { state: [0, -2, 1] })
 * crfDecodeConstrained(crf, [{ f: 1 }, { f: 1 }], 'BIO').labels // => ['O', 'O']
 * ```
 */
export function crfDecodeConstrained(
  crf: LinearChainCrf,
  x: CrfInput,
  constraints: DecodingConstraints,
): { readonly labels: string[]; readonly score: number } {
  if (crfUnigrams(x).length === 0) return { labels: [], score: 0 }
  const v = constrainedViterbi(crfChain(crf, x), resolve(crf, constraints))
  return { labels: names(crf, v.path), score: v.logProbability }
}

/** One of the n best labellings: the labels, the unnormalised score and the log probability p(y | x). */
export interface CrfCandidate {
  readonly labels: string[]
  readonly score: number
  /** log p(y | x); under constraints, conditioned on the labelling satisfying them. */
  readonly logProbability: number
}

/**
 * The n most probable labellings (CRF++'s `-n` and AllenNLP's `viterbi_tags(top_k)`: list Viterbi), best first, with
 * their scores and probabilities, optionally only among those that obey the constraints. Fewer are returned when fewer
 * labellings are feasible. The first is `crfDecode` (`crfDecodeConstrained`).
 *
 * @param crf - The model.
 * @param x - The item (and edge) attributes.
 * @param n - How many labellings.
 * @param constraints - Allowed moves, or a tagging scheme (default none).
 * @returns The labellings, best first.
 * @throws OmaiError with code `value` for n below 1.
 * @concept conditional-random-fields
 * @index function The n best labellings of a CRF with scores and probabilities, optionally under constraints.
 * @example
 * ```ts
 * import { crfNBest, linearChainCrf } from 'omai-engine/pgm'
 * const crf = linearChainCrf(['A', 'B'], ['f'], { state: [Math.log(3), 0] })
 * crfNBest(crf, [{ f: 1 }], 2).map((c) => [c.labels, Math.round(c.logProbability * 1000) / 1000]) // => [[['A'], -0.288], [['B'], -1.386]]
 * ```
 */
export function crfNBest(
  crf: LinearChainCrf,
  x: CrfInput,
  n: number,
  constraints?: DecodingConstraints,
): CrfCandidate[] {
  if (!Number.isInteger(n) || n < 1) throw new OmaiError('value', 'crfNBest: n must be a positive integer')
  if (crfUnigrams(x).length === 0) return [{ labels: [], score: 0, logProbability: 0 }]
  const chain = constrainChain(crfChain(crf, x), resolve(crf, constraints))
  const logZ = forwardBackward(chain, { method: 'log' }).logLikelihood
  return viterbiKBest(chain, n).map((r) => ({
    labels: names(crf, r.path),
    score: r.logProbability,
    logProbability: r.logProbability - logZ,
  }))
}

/**
 * Maximum-expected-accuracy decoding of a CRF under constraints (constrained posterior decoding): the labelling that
 * obeys the constraints and maximises the sum of the posterior marginals of its labels, those marginals being the CRF's
 * conditioned on the constraints. It is the marginal decoder that cannot emit an invalid sequence (the plain
 * `posteriorDecode` picks each label alone and can); with no constraints it is the per-position argmax of
 * `crfMarginals`.
 *
 * @param crf - The model.
 * @param x - The item (and edge) attributes.
 * @param constraints - Allowed moves, or a tagging scheme (default none).
 * @returns The labels, the (constrained) marginals per position and the expected number of correct labels.
 * @throws OmaiError with code `value` when no labelling satisfies the constraints.
 * @concept conditional-random-fields
 * @index function Marginal (maximum expected accuracy) CRF decoding under constraints; never an invalid sequence.
 * @example
 * ```ts
 * import { crfPosteriorDecode, linearChainCrf } from 'omai-engine/pgm'
 * const crf = linearChainCrf(['A', 'B'], ['f'], { state: [Math.log(3), 0] })
 * crfPosteriorDecode(crf, [{ f: 1 }]).labels // => ['A']
 * ```
 */
export function crfPosteriorDecode(
  crf: LinearChainCrf,
  x: CrfInput,
  constraints?: DecodingConstraints,
): { readonly labels: string[]; readonly marginals: number[][]; readonly expectedCorrect: number } {
  if (crfUnigrams(x).length === 0) return { labels: [], marginals: [], expectedCorrect: 0 }
  const r = constrainedPosteriorDecode(crfChain(crf, x), resolve(crf, constraints))
  return { labels: names(crf, r.path), marginals: r.posteriors, expectedCorrect: r.expectedCorrect }
}

/**
 * Exact samples of labellings from the CRF's posterior p(y | x), by forward filtering and backward sampling, optionally
 * from the posterior conditioned on the constraints (so every sample is a valid tag sequence).
 *
 * @param crf - The model.
 * @param x - The item (and edge) attributes.
 * @param key - The random key.
 * @param n - How many samples.
 * @param constraints - Allowed moves, or a tagging scheme (default none).
 * @returns The sampled labellings, `[n][T]`.
 * @throws OmaiError with code `value` when no labelling satisfies the constraints.
 * @concept conditional-random-fields
 * @index function Exact samples of label sequences from a CRF's posterior, optionally under constraints.
 * @example
 * ```ts
 * import { crfSample, linearChainCrf } from 'omai-engine/pgm'
 * import { key } from 'omai-engine/random'
 * const crf = linearChainCrf(['A', 'B'], ['f'], { state: [50, 0] })
 * crfSample(crf, [{ f: 1 }, { f: 1 }], key(0), 2) // => [['A', 'A'], ['A', 'A']]
 * ```
 */
export function crfSample(
  crf: LinearChainCrf,
  x: CrfInput,
  key: Key,
  n: number,
  constraints?: DecodingConstraints,
): string[][] {
  if (crfUnigrams(x).length === 0) return Array.from({ length: n }, () => [])
  const chain = constrainChain(crfChain(crf, x), resolve(crf, constraints))
  return backwardSample(chain, key, n).map((path) => names(crf, path))
}
