/**
 * Alignment-free sequence losses: connectionist temporal classification (CTC, Graves et al., 2006), which trains a
 * frame-wise classifier to emit a label sequence without knowing which frame carries which label by summing over every
 * alignment, in the log semiring, with the blank symbol and repeats collapsed. `ctc` is the differentiable loss (torch's
 * CTCLoss conventions); `ctcTrellis` runs the forward-backward algorithm on plain numbers and exposes the alignment
 * posterior the loss marginalises over; `ctcForcedAlignment` and `ctcGreedyDecode` read an alignment and a labelling
 * off a classifier's outputs.
 */

import { OmaiError, type Reduction } from '../base/index.ts'
import {
  add,
  asTensor,
  div,
  greater,
  logsumexp,
  neg,
  pad,
  promoteFloat,
  sliceAxis,
  squeeze,
  stack,
  type TensorLike,
  take,
  tensor,
  toFlat,
  type Value,
  where,
} from '../tensor/index.ts'
import { reduce } from './_reduce.ts'

/** A log-probability low enough to vanish from every sum and high enough not to overflow when added to. */
const NEG = -1e30

/** The blank-extended label sequence: blank, l1, blank, l2, ..., lS, blank. */
function extend(target: readonly number[], blank: number): number[] {
  const out = [blank]
  for (const label of target) out.push(label, blank)
  return out
}

/** Whether the extended state may be reached from two states back: a label that differs from the one before its blank. */
function skipAllowed(extended: readonly number[], blank: number): boolean[] {
  return extended.map((label, s) => s >= 2 && label !== blank && label !== extended[s - 2])
}

/** The fewest frames that can emit a target: one per label, plus a blank between equal neighbours. */
function minimumFrames(target: readonly number[]): number {
  return target.length + target.filter((label, i) => i > 0 && label === target[i - 1]).length
}

/** Options of {@link ctc}. */
export interface CtcOptions {
  /** The blank class index (default 0, as torch). */
  readonly blank?: number
  /**
   * The number of valid frames of every sequence in the batch (default all of them), as torch's `input_lengths`.
   */
  readonly inputLengths?: readonly number[]
  /** The number of valid labels of every row of a padded target matrix (default all columns). */
  readonly targetLengths?: readonly number[]
  /**
   * 'mean' (default) divides every sequence's loss by its target length and averages over the batch (torch's
   * convention); 'sum' adds them; 'none' returns one loss per sequence.
   */
  readonly reduction?: Reduction
  /** Replace infinite losses (inputs too short for the target) by zero, as torch's `zero_infinity`. */
  readonly zeroInfinity?: boolean
}

/** The negative log-likelihood of one target under log-probabilities `[T, C]`, differentiable in them. */
function ctcOne(logProbs: Value, target: readonly number[], blank: number): Value {
  const frames = logProbs.shape[0] ?? 0
  if (frames < minimumFrames(target)) return tensor(Number.POSITIVE_INFINITY)
  const extended = extend(target, blank)
  const states = extended.length
  // The log-probability every frame gives to every extended state: [T, L].
  const emissions = take(logProbs, tensor(extended, { dtype: 'int32' }), 1)
  const mayJump = greater(tensor(skipAllowed(extended, blank).map((b) => (b ? 1 : 0))), 0)
  const row = (t: number): Value => squeeze(sliceAxis(emissions, 0, t, t + 1), 0)
  // Frame 0 starts in the first blank or the first label.
  let alpha = add(row(0), tensor(extended.map((_, s) => (s < 2 ? 0 : NEG))))
  for (let t = 1; t < frames; t++) {
    const moves = [alpha, pad(sliceAxis(alpha, 0, 0, states - 1), [[1, 0]], NEG)]
    if (states >= 3) moves.push(where(mayJump, pad(sliceAxis(alpha, 0, 0, states - 2), [[2, 0]], NEG), NEG))
    alpha = add(logsumexp(stack(moves, 0), 0), row(t))
  }
  const ends = sliceAxis(alpha, 0, Math.max(states - 2, 0), states)
  return neg(logsumexp(ends, 0))
}

/** The rows of a padded target matrix, trimmed to their lengths. */
function targetRows(targets: TensorLike, lengths: readonly number[] | undefined, n: number): number[][] {
  const t = asTensor(targets)
  if (t.shape.length === 1) return [toFlat(t).slice(0, lengths?.[0] ?? t.shape[0])]
  const width = t.shape[1] ?? 0
  const flat = toFlat(t)
  return Array.from({ length: n }, (_, i) => flat.slice(i * width, i * width + (lengths?.[i] ?? width)))
}

/**
 * Connectionist temporal classification loss (Graves et al., 2006): the negative log of the total probability of a
 * target label sequence over all frame-level alignments that collapse to it (repeats merged, then blanks removed),
 * computed by the forward algorithm in the log semiring over the blank-extended target and differentiable in the
 * log-probabilities. It follows torch's CTCLoss: a mean reduction divides each loss by its target length before
 * averaging, and a target too long for the frames has infinite loss (or zero with `zeroInfinity`).
 *
 * @param logProbs - Log-probabilities `[T, N, C]` (time first, as torch) or `[T, C]` for one sequence; the caller
 *   applies the log-softmax.
 * @param targets - Label sequences without blanks: `[N, S]` padded (use `targetLengths`) or `[S]`.
 * @param options - blank, inputLengths, targetLengths, reduction and zeroInfinity.
 * @returns The loss, differentiable in the log-probabilities.
 * @throws OmaiError with code `shape` when the targets and the batch disagree.
 * @concept ctc
 * @example
 * ```ts
 * import { ctc } from 'omai-engine/losses'
 * import { item } from 'omai-engine/tensor'
 * // Two frames of uniform probabilities over {blank, a}: the target "a" has alignments aa, a-, -a, so p = 3/4.
 * const logp = [[Math.log(0.5), Math.log(0.5)], [Math.log(0.5), Math.log(0.5)]]
 * Math.abs(item(ctc(logp, [1], { reduction: 'sum' })) + Math.log(0.75)) < 1e-14 // => true
 * ```
 */
export function ctc(logProbs: TensorLike, targets: TensorLike, options: CtcOptions = {}): Value {
  const [lp] = promoteFloat(logProbs) as [Value]
  const batched = lp.shape.length === 3
  if (!batched && lp.shape.length !== 2)
    throw new OmaiError('shape', 'ctc: log-probabilities must be [T, N, C] or [T, C]')
  const frames = lp.shape[0] ?? 0
  const n = batched ? (lp.shape[1] ?? 1) : 1
  const blank = options.blank ?? 0
  const rows = targetRows(targets, options.targetLengths, n)
  if (rows.length !== n) throw new OmaiError('shape', `ctc: ${rows.length} targets for a batch of ${n}`)
  const losses = rows.map((target, i) => {
    const sequence = batched ? squeeze(sliceAxis(lp, 1, i, i + 1), 1) : lp
    const valid = options.inputLengths?.[i] ?? frames
    const loss = ctcOne(valid < frames ? sliceAxis(sequence, 0, 0, valid) : sequence, target, blank)
    return options.zeroInfinity === true && !Number.isFinite(toFlat(loss)[0] ?? 0) ? tensor(0) : loss
  })
  const perSequence = stack(losses, 0)
  const reduction = options.reduction ?? 'mean'
  if (reduction !== 'mean') return reduce(perSequence, { reduction })
  const lengths = tensor(rows.map((r) => Math.max(r.length, 1)))
  return reduce(div(perSequence, lengths), { reduction: 'mean' })
}

/** The forward-backward quantities of CTC for one sequence. */
export interface CtcTrellis {
  /** The blank-extended target (blank, l1, blank, ..., lS, blank), length 2S + 1. */
  readonly extended: readonly number[]
  /** log p(target | input), minus infinity when the input is too short. */
  readonly logLikelihood: number
  /** Forward log-probabilities `[T][2S + 1]`: the paths ending in each state at each frame, that frame included. */
  readonly alpha: readonly (readonly number[])[]
  /** Backward log-probabilities `[T][2S + 1]`: the continuations from each state after each frame. */
  readonly beta: readonly (readonly number[])[]
  /** The posterior of each extended state at each frame `[T][2S + 1]` (each row sums to one). */
  readonly statePosterior: readonly (readonly number[])[]
  /** The posterior of each class at each frame `[T][C]`, blank included: the expected alignment (rows sum to one). */
  readonly labelPosterior: readonly (readonly number[])[]
}

const logAdd = (a: number, b: number): number => {
  const m = Math.max(a, b)
  return m === Number.NEGATIVE_INFINITY ? m : m + Math.log(Math.exp(a - m) + Math.exp(b - m))
}

/** A dense frames-by-states table of numbers with checked access. */
class Lattice {
  readonly rows: number
  readonly cols: number
  private readonly cells: Float64Array
  constructor(rows: number, cols: number, fill: number) {
    this.rows = rows
    this.cols = cols
    this.cells = new Float64Array(rows * cols).fill(fill)
  }
  get(t: number, s: number): number {
    return this.cells[t * this.cols + s] ?? Number.NEGATIVE_INFINITY
  }
  set(t: number, s: number, value: number): void {
    this.cells[t * this.cols + s] = value
  }
  toRows(): number[][] {
    return Array.from({ length: this.rows }, (_, t) => Array.from({ length: this.cols }, (_, s) => this.get(t, s)))
  }
}

/**
 * The CTC trellis of one sequence: the forward and backward log-probabilities over the blank-extended target and the
 * alignment posterior they give, the probability that each frame sits in each state (and so on each class) given the
 * target. Training with the loss moves the classifier's outputs toward this posterior: the gradient of
 * {@link ctc} with respect to the log-probabilities is minus the label posterior.
 *
 * @param logProbs - Log-probabilities of one sequence `[T, C]`.
 * @param target - The label sequence, without blanks.
 * @param options - blank (default 0).
 * @returns The extended target, the log-likelihood, alpha, beta and the state and label posteriors.
 * @example
 * ```ts
 * import { ctcTrellis } from 'omai-engine/losses'
 * const half = Math.log(0.5)
 * const t = ctcTrellis([[half, half], [half, half]], [1])
 * t.labelPosterior[0] // => [0.3333333333333333, 0.6666666666666666]
 * ```
 */
export function ctcTrellis(
  logProbs: TensorLike,
  target: readonly number[],
  options: { readonly blank?: number } = {},
): CtcTrellis {
  const lp = asTensor(logProbs)
  const [frames, classes] = [lp.shape[0] ?? 0, lp.shape[1] ?? 0]
  const flat = toFlat(lp)
  const blank = options.blank ?? 0
  const extended = extend(target, blank)
  const states = extended.length
  const skip = skipAllowed(extended, blank)
  const emit = (t: number, s: number): number => flat[t * classes + (extended[s] ?? blank)] ?? Number.NEGATIVE_INFINITY
  const alpha = new Lattice(frames, states, Number.NEGATIVE_INFINITY)
  const beta = new Lattice(frames, states, Number.NEGATIVE_INFINITY)
  if (frames > 0) {
    alpha.set(0, 0, emit(0, 0))
    if (states > 1) alpha.set(0, 1, emit(0, 1))
    beta.set(frames - 1, states - 1, 0)
    if (states > 1) beta.set(frames - 1, states - 2, 0)
  }
  for (let t = 1; t < frames; t++) {
    for (let s = 0; s < states; s++) {
      let incoming = alpha.get(t - 1, s)
      if (s >= 1) incoming = logAdd(incoming, alpha.get(t - 1, s - 1))
      if (skip[s]) incoming = logAdd(incoming, alpha.get(t - 1, s - 2))
      alpha.set(t, s, incoming + emit(t, s))
    }
  }
  for (let t = frames - 2; t >= 0; t--) {
    for (let s = 0; s < states; s++) {
      let outgoing = beta.get(t + 1, s) + emit(t + 1, s)
      if (s + 1 < states) outgoing = logAdd(outgoing, beta.get(t + 1, s + 1) + emit(t + 1, s + 1))
      if (s + 2 < states && skip[s + 2]) outgoing = logAdd(outgoing, beta.get(t + 1, s + 2) + emit(t + 1, s + 2))
      beta.set(t, s, outgoing)
    }
  }
  const logLikelihood =
    frames === 0
      ? Number.NEGATIVE_INFINITY
      : logAdd(
          alpha.get(frames - 1, states - 1),
          states > 1 ? alpha.get(frames - 1, states - 2) : Number.NEGATIVE_INFINITY,
        )
  const feasible = Number.isFinite(logLikelihood)
  const statePosterior = alpha
    .toRows()
    .map((row, t) => row.map((v, s) => (feasible ? Math.exp(v + beta.get(t, s) - logLikelihood) : 0)))
  const labelPosterior = statePosterior.map((row) => {
    const out = new Array<number>(classes).fill(0)
    for (const [s, p] of row.entries()) {
      const c = extended[s] ?? blank
      out[c] = (out[c] ?? 0) + p
    }
    return out
  })
  return { extended, logLikelihood, alpha: alpha.toRows(), beta: beta.toRows(), statePosterior, labelPosterior }
}

/**
 * The best single alignment of a target to the frames (Viterbi over the CTC trellis): the most probable path through
 * the blank-extended target, as the class at every frame.
 *
 * @param logProbs - Log-probabilities of one sequence `[T, C]`.
 * @param target - The label sequence, without blanks.
 * @param options - blank (default 0).
 * @returns The class at every frame (blank included) and the log-probability of that path.
 * @throws OmaiError with code `value` when the frames cannot emit the target.
 * @example
 * ```ts
 * import { ctcForcedAlignment } from 'omai-engine/losses'
 * const lp = [[-0.1, -3], [-3, -0.1], [-3, -0.1], [-0.1, -3]].map((r) => r.map((v) => v))
 * ctcForcedAlignment(lp, [1]).path // => [0, 1, 1, 0]
 * ```
 */
export function ctcForcedAlignment(
  logProbs: TensorLike,
  target: readonly number[],
  options: { readonly blank?: number } = {},
): { readonly path: number[]; readonly logProbability: number } {
  const lp = asTensor(logProbs)
  const [frames, classes] = [lp.shape[0] ?? 0, lp.shape[1] ?? 0]
  const flat = toFlat(lp)
  const blank = options.blank ?? 0
  if (frames < minimumFrames(target)) throw new OmaiError('value', 'ctcForcedAlignment: too few frames for the target')
  const extended = extend(target, blank)
  const states = extended.length
  const skip = skipAllowed(extended, blank)
  const emit = (t: number, s: number): number => flat[t * classes + (extended[s] ?? blank)] ?? Number.NEGATIVE_INFINITY
  const score = new Lattice(frames, states, Number.NEGATIVE_INFINITY)
  const from = new Lattice(frames, states, 0)
  score.set(0, 0, emit(0, 0))
  if (states > 1) score.set(0, 1, emit(0, 1))
  for (let t = 1; t < frames; t++) {
    for (let s = 0; s < states; s++) {
      let best = score.get(t - 1, s)
      let arg = s
      if (s >= 1 && score.get(t - 1, s - 1) > best) [best, arg] = [score.get(t - 1, s - 1), s - 1]
      if (skip[s] && score.get(t - 1, s - 2) > best) [best, arg] = [score.get(t - 1, s - 2), s - 2]
      score.set(t, s, best + emit(t, s))
      from.set(t, s, arg)
    }
  }
  let state = states - 1
  if (states > 1 && score.get(frames - 1, states - 2) > score.get(frames - 1, states - 1)) state = states - 2
  const logProbability = score.get(frames - 1, state)
  const path = new Array<number>(frames)
  for (let t = frames - 1; t >= 0; t--) {
    path[t] = extended[state] ?? blank
    state = from.get(t, state)
  }
  return { path, logProbability }
}

/**
 * Greedy (best-path) CTC decoding: the most probable class at every frame, repeats merged, blanks removed.
 *
 * @param logProbs - Scores of one sequence `[T, C]` (log-probabilities or logits).
 * @param options - blank (default 0).
 * @returns The decoded label sequence.
 * @example
 * ```ts
 * import { ctcGreedyDecode } from 'omai-engine/losses'
 * ctcGreedyDecode([[0, 5, 0], [0, 5, 0], [5, 0, 0], [0, 5, 0], [0, 0, 5]]) // => [1, 1, 2]
 * ```
 */
export function ctcGreedyDecode(logProbs: TensorLike, options: { readonly blank?: number } = {}): number[] {
  const lp = asTensor(logProbs)
  const [frames, classes] = [lp.shape[0] ?? 0, lp.shape[1] ?? 0]
  const flat = toFlat(lp)
  const blank = options.blank ?? 0
  const out: number[] = []
  let previous = -1
  for (let t = 0; t < frames; t++) {
    let best = 0
    for (let c = 1; c < classes; c++) if ((flat[t * classes + c] ?? 0) > (flat[t * classes + best] ?? 0)) best = c
    if (best !== previous && best !== blank) out.push(best)
    previous = best
  }
  return out
}
