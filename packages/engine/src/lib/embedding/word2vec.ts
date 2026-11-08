/**
 * Word2vec (Mikolov et al. 2013) and fastText (Bojanowski et al. 2017) as an objective and a slot. The examples are
 * built once from a corpus (skip-gram pairs or CBOW windows with a dynamic window and frequent-word subsampling, the
 * unigram-to-the-0.75 negatives drawn up front, or the Huffman paths of the hierarchical softmax; fastText adds the
 * hashed character n-gram rows to a word's input), so the loss is a smooth function of the two embedding matrices: it
 * is exposed as a `SmoothProblem` and minimised by the solver slot, of which word2vec's own training (plain SGD, one
 * example at a time, learning rate decaying linearly to a floor, touching only the rows involved) is the named mode
 * `word2vecSgd` beside the library's generic solvers.
 */

import { type Json, maxOf, OmaiError, sigmoidOf } from '../base/index.ts'
import { type HuffmanCoding, hierarchicalSoftmax, huffmanCoding } from '../losses/index.ts'
import { logSigmoid } from '../numerics/special/index.ts'
import { type BatchObjective, type SmoothProblem, smoothSolvers, type VectorParams } from '../optim/index.ts'
import { categorical, foldIn, type Key, permutation, uniform } from '../random/index.ts'
import { bool, dict, int, real, type SpaceValue } from '../space/index.ts'
import { add, expandDims, fromStorage, item, mul, reshape, sliceAxis, sum, take, type Value } from '../tensor/index.ts'
import { fastTextBuckets, type WordMatrix } from '../text/index.ts'
import { type Algorithm, type ModeSpec, type ModeTable, modes, type RunOptions, runAlgorithm } from '../trace/index.ts'

/** A vocabulary for training: words by decreasing frequency (ties by code point), their counts and the index. */
export interface Word2vecVocabulary {
  readonly words: readonly string[]
  readonly counts: readonly number[]
  readonly index: ReadonlyMap<string, number>
}

/**
 * The training vocabulary of tokenised sentences: words seen at least `minCount` times, most frequent first.
 *
 * @param sentences - Token lists.
 * @param minCount - The smallest count kept (default 1).
 * @returns The vocabulary.
 * @concept word-embeddings
 * @example
 * ```ts
 * import { word2vecVocabulary } from 'omai-engine/embedding'
 * word2vecVocabulary([['b', 'a', 'b'], ['c', 'b', 'a']], 2).words // => ['b', 'a']
 * ```
 */
export function word2vecVocabulary(sentences: readonly (readonly string[])[], minCount = 1): Word2vecVocabulary {
  const counts = new Map<string, number>()
  for (const s of sentences) for (const w of s) counts.set(w, (counts.get(w) ?? 0) + 1)
  const kept = [...counts].filter(([, c]) => c >= minCount).sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1))
  return {
    words: kept.map(([w]) => w),
    counts: kept.map(([, c]) => c),
    index: new Map(kept.map(([w], i) => [w, i])),
  }
}

/** The knobs that decide which examples a corpus yields and how the model reads them. */
export interface Word2vecOptions {
  /** Skip-gram (the centre word predicts each context word) or CBOW (the mean of the context predicts the centre). */
  readonly architecture?: 'skip-gram' | 'cbow'
  /** Negative sampling (default) or the hierarchical softmax over a Huffman tree. */
  readonly output?: 'negative-sampling' | 'hierarchical-softmax'
  /** Context words on each side (default 5). */
  readonly window?: number
  /** Negatives per example (default 5). */
  readonly negatives?: number
  /** Frequent-word subsampling threshold t (original word2vec: 1e-3); 0 keeps every word (default). */
  readonly subsample?: number
  /** Draw each position's window uniformly from 1 to `window`, as word2vec does (default true). */
  readonly dynamicWindow?: boolean
  /** The embedding dimension (default 50). */
  readonly dimension?: number
  /** fastText: add the hashed character n-grams of each input word (skip-gram only). */
  readonly subwords?: { readonly minn: number; readonly maxn: number; readonly buckets: number }
}

/** The training examples of a corpus: inputs (rows of the input matrix, -1 padded), targets and negatives. */
export interface Word2vecExamples {
  readonly n: number
  /** The width of `inputs`: rows per example. */
  readonly width: number
  readonly inputs: Int32Array
  readonly targets: Int32Array
  /** `n x negatives` word ids (empty for the hierarchical softmax). */
  readonly negatives: Int32Array
  readonly negativesPerExample: number
}

/** A word2vec training problem: the smooth loss plus the data the named SGD mode reads. */
export interface Word2vecProblem extends SmoothProblem {
  readonly vocabulary: Word2vecVocabulary
  readonly examples: Word2vecExamples
  readonly options: Required<Pick<Word2vecOptions, 'architecture' | 'output' | 'dimension'>> & Word2vecOptions
  /** Rows of the input matrix: the words, then the n-gram buckets of fastText. */
  readonly inputRows: number
  /** Rows of the output matrix: the words (negative sampling) or the inner nodes (hierarchical softmax). */
  readonly outputRows: number
  /** The Huffman coding of the hierarchical softmax. */
  readonly coding?: HuffmanCoding
  /** The input rows of each word: itself, then its n-gram buckets (offset by the vocabulary size). */
  readonly rowsOfWord: readonly (readonly number[])[]
  /** The loss and its gradient at the flat parameter vector `[input matrix, output matrix]`. */
  readonly unravel: (x: ArrayLike<number>) => { readonly input: Float64Array[]; readonly output: Float64Array[] }
}

/**
 * Build the examples of a corpus: for each kept word the context words within a (dynamic) window, the frequent-word
 * subsampling of Mikolov et al., and negatives drawn from the unigram distribution to the power 0.75.
 *
 * @param sentences - Token lists.
 * @param vocabulary - The training vocabulary ({@link word2vecVocabulary}).
 * @param options - Architecture, window, negatives, subsampling and subwords.
 * @param key - The key for the window sizes, subsampling and negatives.
 * @returns The examples.
 * @throws OmaiError with code `value` for a window or negatives below 1, or subwords with CBOW.
 * @concept word-embeddings
 * @example
 * ```ts
 * import { word2vecExamples, word2vecVocabulary } from 'omai-engine/embedding'
 * import { key } from 'omai-engine/random'
 * const s = [['a', 'b', 'c']]
 * word2vecExamples(s, word2vecVocabulary(s), { window: 1, dynamicWindow: false, negatives: 2 }, key(0)).n // => 4
 * ```
 */
export function word2vecExamples(
  sentences: readonly (readonly string[])[],
  vocabulary: Word2vecVocabulary,
  options: Word2vecOptions,
  key: Key,
): Word2vecExamples {
  const window = options.window ?? 5
  const k = options.output === 'hierarchical-softmax' ? 0 : (options.negatives ?? 5)
  const architecture = options.architecture ?? 'skip-gram'
  if (!Number.isInteger(window) || window < 1) throw new OmaiError('value', 'word2vec: window must be at least 1')
  if (options.output !== 'hierarchical-softmax' && (!Number.isInteger(k) || k < 1))
    throw new OmaiError('value', 'word2vec: negatives must be at least 1')
  if (options.subwords !== undefined && architecture === 'cbow')
    throw new OmaiError('value', 'word2vec: subwords are for skip-gram')
  const V = vocabulary.words.length
  const total = vocabulary.counts.reduce((a, b) => a + b, 0)
  const rows = rowsOfWords(vocabulary, options)
  const width = architecture === 'skip-gram' ? Math.max(maxOf(rows.map((r) => r.length)), 1) : 2 * window
  const t = options.subsample ?? 0
  const pairs: { input: number[]; target: number }[] = []
  for (const [si, sentence] of sentences.entries()) {
    const ids: number[] = []
    const kk = foldIn(key, si)
    const u = uniform(foldIn(kk, 0), [sentence.length]).data
    for (const [i, w] of sentence.entries()) {
      const id = vocabulary.index.get(w)
      if (id === undefined) continue
      if (t > 0) {
        const c = vocabulary.counts[id] ?? 1
        const keep = (Math.sqrt(c / (t * total)) + 1) * ((t * total) / c)
        if ((u[i] ?? 0) >= keep) continue
      }
      ids.push(id)
    }
    const b =
      options.dynamicWindow === false
        ? new Array<number>(ids.length).fill(window)
        : Array.from(uniform(foldIn(kk, 1), [ids.length]).data, (x) => 1 + Math.floor(x * window))
    for (let i = 0; i < ids.length; i++) {
      const lo = Math.max(0, i - (b[i] ?? window))
      const hi = Math.min(ids.length - 1, i + (b[i] ?? window))
      const centre = ids[i] as number
      if (architecture === 'skip-gram') {
        for (let j = lo; j <= hi; j++)
          if (j !== i) pairs.push({ input: rows[centre] as number[], target: ids[j] as number })
      } else {
        const ctx: number[] = []
        for (let j = lo; j <= hi; j++) if (j !== i) ctx.push(ids[j] as number)
        if (ctx.length > 0) pairs.push({ input: ctx, target: centre })
      }
    }
  }
  const n = pairs.length
  const inputs = new Int32Array(n * width).fill(-1)
  const targets = new Int32Array(n)
  for (const [e, p] of pairs.entries()) {
    targets[e] = p.target
    for (const [j, r] of p.input.entries()) inputs[e * width + j] = r
  }
  let negatives = new Int32Array(0)
  if (k > 0 && n > 0) {
    const weights = vocabulary.counts.map((c) => c ** 0.75)
    negatives = Int32Array.from(categorical(foldIn(key, 1_000_003), weights, [n * k]).data)
    for (let e = 0; e < n; e++)
      for (let j = 0; j < k; j++)
        if (negatives[e * k + j] === targets[e]) negatives[e * k + j] = ((targets[e] ?? 0) + 1) % V
  }
  return { n, width, inputs, targets, negatives, negativesPerExample: k }
}

/** The input rows of every word: itself, then (fastText) the vocabulary-offset buckets of its character n-grams. */
function rowsOfWords(vocabulary: Word2vecVocabulary, options: Word2vecOptions): number[][] {
  const V = vocabulary.words.length
  return vocabulary.words.map((w, i) => {
    if (options.subwords === undefined) return [i]
    const { minn, maxn, buckets } = options.subwords
    return [i, ...fastTextBuckets(w, minn, maxn, buckets).map((b) => V + b)]
  })
}

/**
 * The word2vec loss as a {@link SmoothProblem} over the flat vector `[input matrix, output matrix]` (row-major,
 * input rows first): the mean over examples of `-log sigmoidOf(h . o_target) - sum_j log sigmoidOf(-h . o_j)` for
 * negative sampling, or the hierarchical softmax's path loss, with `h` the mean of the input rows (the centre word and,
 * for fastText, its n-gram buckets; the context words for CBOW). The start is word2vec's: input rows uniform in
 * `[-0.5 / d, 0.5 / d]`, output rows zero. The examples are the batches the stochastic `gradient` mode draws.
 *
 * @param sentences - Token lists.
 * @param options - The model: architecture, output, window, negatives, subsampling, dimension and subwords.
 * @param key - The key for the examples and the start.
 * @param minCount - The vocabulary cut-off (default 1).
 * @returns The problem.
 * @throws OmaiError with code `value` for fewer than two words or no examples.
 * @concept word-embeddings
 * @example
 * ```ts
 * import { word2vecProblem } from 'omai-engine/embedding'
 * import { key } from 'omai-engine/random'
 * const p = word2vecProblem([['a', 'b', 'c', 'a', 'b']], { dimension: 2, window: 1, negatives: 1 }, key(0))
 * [p.inputRows, p.outputRows, p.batches?.n] // => [3, 3, 8]
 * ```
 */
export function word2vecProblem(
  sentences: readonly (readonly string[])[],
  options: Word2vecOptions,
  key: Key,
  minCount = 1,
): Word2vecProblem {
  const vocabulary = word2vecVocabulary(sentences, minCount)
  const V = vocabulary.words.length
  if (V < 2) throw new OmaiError('value', 'word2vec: needs at least two distinct words')
  const opts = {
    ...options,
    architecture: options.architecture ?? 'skip-gram',
    output: options.output ?? 'negative-sampling',
    dimension: options.dimension ?? 50,
  } as Word2vecProblem['options']
  const d = opts.dimension
  const examples = word2vecExamples(sentences, vocabulary, opts, foldIn(key, 0))
  if (examples.n === 0) throw new OmaiError('value', 'word2vec: the corpus gives no training examples')
  const hs = opts.output === 'hierarchical-softmax'
  const coding = hs ? huffmanCoding(vocabulary.counts) : undefined
  const inputRows = V + (opts.subwords?.buckets ?? 0)
  const outputRows = hs ? V - 1 : V
  const rowsOfWord = rowsOfWords(vocabulary, opts)
  const x0 = new Float64Array((inputRows + outputRows) * d)
  const init = uniform(foldIn(key, 2), [inputRows * d]).data
  for (let i = 0; i < inputRows * d; i++) x0[i] = ((init[i] ?? 0.5) - 0.5) / d
  const { width, negativesPerExample: kn } = examples
  const objective: BatchObjective = (params, batch) => {
    const x = params as Value
    const rows = batch ?? undefined
    const n = rows === undefined ? examples.n : rows.length
    const pick = (i: number): number => (rows === undefined ? i : (rows[i] as number))
    const idx = new Int32Array(n * width)
    const mask = new Float64Array(n * width)
    const count = new Float64Array(n)
    const target = new Int32Array(n)
    const neg = new Int32Array(n * kn)
    for (let i = 0; i < n; i++) {
      const e = pick(i)
      target[i] = examples.targets[e] as number
      for (let j = 0; j < width; j++) {
        const r = examples.inputs[e * width + j] as number
        if (r >= 0) {
          idx[i * width + j] = r
          mask[i * width + j] = 1
          count[i] = (count[i] ?? 0) + 1
        }
      }
      for (let j = 0; j < kn; j++) neg[i * kn + j] = examples.negatives[e * kn + j] as number
    }
    const win = reshape(sliceAxis(x, 0, 0, inputRows * d), [inputRows, d])
    const wout = reshape(sliceAxis(x, 0, inputRows * d, (inputRows + outputRows) * d), [outputRows, d])
    const e = reshape(take(win, fromStorage(idx, [n * width], 'int32'), 0), [n, width, d])
    const w = fromStorage(
      Float64Array.from(mask, (m, i) => m / (count[Math.floor(i / width)] || 1)),
      [n, width, 1],
      'float64',
    )
    const h = sum(mul(e, w), 1)
    if (coding !== undefined) return hierarchicalSoftmax(h, wout, fromStorage(target, [n], 'int32'), coding)
    const o = take(wout, fromStorage(target, [n], 'int32'), 0)
    const pos = sum(mul(h, o), -1)
    const nOut = reshape(take(wout, fromStorage(neg, [n * kn], 'int32'), 0), [n, kn, d])
    const negScores = sum(mul(expandDims(h, 1), nOut), -1)
    const perExample = add(logSigmoid(pos), sum(logSigmoid(mul(negScores, -1)), -1))
    return mul(sum(perExample), -1 / n)
  }
  return {
    objective,
    x0,
    batches: { n: examples.n },
    vocabulary,
    examples,
    options: opts,
    inputRows,
    outputRows,
    ...(coding === undefined ? {} : { coding }),
    rowsOfWord,
    unravel: (x) => ({
      input: Array.from({ length: inputRows }, (_, r) => Float64Array.from({ length: d }, (_, c) => x[r * d + c] ?? 0)),
      output: Array.from({ length: outputRows }, (_, r) =>
        Float64Array.from({ length: d }, (_, c) => x[(inputRows + r) * d + c] ?? 0),
      ),
    }),
  }
}

const logSigmoidNumber = (z: number): number => (z >= 0 ? -Math.log1p(Math.exp(-z)) : z - Math.log1p(Math.exp(z)))

/** The state of word2vec's own SGD: the flat parameters (updated in place), the step and what the step saw. */
export interface Word2vecSgdState {
  /** `[input matrix, output matrix]`, row-major. */
  readonly x: Float64Array
  /** The mean loss of the examples of the last step, before their updates. */
  readonly value: number
  /** One entry: the root mean square norm of the gradient steps of the last step. */
  readonly grad: Float64Array
  /** The length of the last step's largest single update. */
  readonly step: number
  /** The learning rate used in the last step. */
  readonly lr: number
  readonly t: number
  /** The order the examples are visited in this epoch. */
  readonly order: Int32Array
}

/** Options of {@link word2vecSgdAlgorithm}. */
export interface Word2vecSgdOptions {
  /** The starting learning rate (default 0.025). */
  readonly learningRate?: number
  /** The floor of the linear decay, as a share of the starting rate (default 1e-4). */
  readonly minLearningRate?: number
  /** Passes over the examples (default 5). */
  readonly epochs?: number
  /** Examples per step, each updated in turn (default 1: the original). */
  readonly batch?: number
  /** Visit the examples in a fresh random order each epoch instead of corpus order (default false). */
  readonly shuffle?: boolean
}

/**
 * Word2vec's training as an {@link Algorithm}: plain SGD one example at a time, the learning rate decaying linearly
 * from its start to a floor over all the passes, and only the rows an example touches updated (the input rows, the
 * target and negative output rows or the nodes of the target's Huffman path). Each step takes `batch` examples in
 * turn. The update is the exact gradient of {@link word2vecProblem}'s loss for the example (a CBOW input mean divides
 * its gradient among the context words), which the tests check against autodiff.
 *
 * @param problem - The problem ({@link word2vecProblem}).
 * @param options - Learning rate and its floor, epochs, batch and shuffling.
 * @returns The algorithm; its state's arrays are updated in place.
 * @concept word-embeddings
 * @example
 * ```ts
 * import { word2vecProblem, word2vecSgdAlgorithm } from 'omai-engine/embedding'
 * import { key } from 'omai-engine/random'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const p = word2vecProblem([['a', 'b', 'c', 'a', 'b']], { dimension: 2, window: 1, negatives: 1 }, key(0))
 * runAlgorithm(word2vecSgdAlgorithm(p, { epochs: 2 }), { seed: 0, maxSteps: 1000 }).steps // => 16
 * ```
 */
export function word2vecSgdAlgorithm(
  problem: Word2vecProblem,
  options: Word2vecSgdOptions = {},
): Algorithm<Word2vecSgdState> {
  const lr0 = options.learningRate ?? 0.025
  const floor = lr0 * (options.minLearningRate ?? 1e-4)
  const epochs = options.epochs ?? 5
  const batch = options.batch ?? 1
  const { examples, inputRows, outputRows, coding } = problem
  const d = problem.options.dimension
  const { n, width, inputs, targets, negatives, negativesPerExample: kn } = examples
  const stepsPerEpoch = Math.ceil(n / batch)
  const total = stepsPerEpoch * epochs
  if (!(lr0 > 0) || !Number.isInteger(epochs) || epochs < 1 || !Number.isInteger(batch) || batch < 1)
    throw new OmaiError('value', 'word2vecSgd: need a positive learning rate, epochs and batch')
  const inOff = 0
  const outOff = inputRows * d
  const order = (k: Key): Int32Array =>
    options.shuffle === true ? Int32Array.from(permutation(k, n).data) : Int32Array.from({ length: n }, (_, i) => i)
  const hs = new Float64Array(d)
  const grad = new Float64Array(d)
  const signs = coding === undefined ? [] : coding.codes.map((c) => c.map((b) => 1 - 2 * b))
  void outputRows
  return {
    name: 'word2vecSgd',
    config: { learningRate: lr0, epochs, batch, examples: n } as Json,
    init: (key) => ({
      x: Float64Array.from(problem.x0 as Float64Array),
      value: Number.NaN,
      grad: new Float64Array(1),
      step: 0,
      lr: lr0,
      t: 0,
      order: order(key),
    }),
    step: (s, ctx) => {
      const t = s.t
      const lr = Math.max(floor, lr0 * (1 - t / total))
      const epochStart = (t % stepsPerEpoch) * batch
      const ord = t > 0 && t % stepsPerEpoch === 0 && options.shuffle === true ? order(ctx.key) : s.order
      const x = s.x
      let loss = 0
      let gsq = 0
      let biggest = 0
      let seen = 0
      for (let e0 = epochStart; e0 < Math.min(n, epochStart + batch); e0++) {
        const e = ord[e0] as number
        seen++
        hs.fill(0)
        let cnt = 0
        for (let j = 0; j < width; j++) {
          const r = inputs[e * width + j] as number
          if (r < 0) continue
          cnt++
          for (let c = 0; c < d; c++) hs[c] = (hs[c] ?? 0) + (x[inOff + r * d + c] ?? 0)
        }
        for (let c = 0; c < d; c++) hs[c] = (hs[c] ?? 0) / cnt
        grad.fill(0)
        const target = targets[e] as number
        const touch = (row: number, label: number, sign: number): void => {
          // The logistic unit on output row `row`: z = h . o, loss -log sigmoidOf(sign z), gradient factor sign (sigmoidOf(sign z) - 1).
          let z = 0
          const base = outOff + row * d
          for (let c = 0; c < d; c++) z += (hs[c] ?? 0) * (x[base + c] ?? 0)
          void label
          const sz = sign * z
          loss -= logSigmoidNumber(sz)
          const g = sign * (sigmoidOf(sz) - 1)
          for (let c = 0; c < d; c++) {
            grad[c] = (grad[c] ?? 0) + g * (x[base + c] ?? 0)
            const u = lr * g * (hs[c] ?? 0)
            x[base + c] = (x[base + c] ?? 0) - u
            biggest = Math.max(biggest, Math.abs(u))
            gsq += (g * (hs[c] ?? 0)) ** 2
          }
        }
        if (coding === undefined) {
          touch(target, 1, 1)
          for (let j = 0; j < kn; j++) touch(negatives[e * kn + j] as number, 0, -1)
        } else {
          const path = coding.paths[target] as readonly number[]
          const sg = signs[target] as number[]
          for (const [j, node] of path.entries()) touch(node, 0, sg[j] as number)
        }
        for (let j = 0; j < width; j++) {
          const r = inputs[e * width + j] as number
          if (r < 0) continue
          for (let c = 0; c < d; c++) {
            const u = (lr * (grad[c] ?? 0)) / cnt
            x[inOff + r * d + c] = (x[inOff + r * d + c] ?? 0) - u
            biggest = Math.max(biggest, Math.abs(u))
          }
        }
      }
      void ctx
      return {
        x,
        value: loss / Math.max(1, seen),
        grad: Float64Array.of(Math.sqrt(gsq / Math.max(1, seen))),
        step: biggest,
        lr,
        t: t + 1,
        order: ord,
      }
    },
    done: (s) => (s.t >= total ? 'converged' : undefined),
    observe: (s) => ({ metrics: { loss: s.value, lr: s.lr } }),
    serialise: (s) =>
      ({
        x: Array.from(s.x),
        value: s.value,
        step: s.step,
        lr: s.lr,
        t: s.t,
        order: Array.from(s.order),
      }) as unknown as Json,
    deserialise: (j) => {
      const o = j as unknown as {
        x: number[]
        value: number | null
        step: number
        lr: number
        t: number
        order: number[]
      }
      return {
        x: Float64Array.from(o.x),
        value: o.value ?? Number.NaN,
        grad: new Float64Array(1),
        step: o.step,
        lr: o.lr,
        t: o.t,
        order: Int32Array.from(o.order),
      }
    },
  }
}

/** The named mode of word2vec's own SGD, for a solver table. */
const word2vecSgd: ModeSpec<Word2vecProblem, Word2vecSgdState, VectorParams> = {
  description:
    "word2vec's own training: SGD one example at a time, the rate decaying linearly, only the rows involved updated",
  space: dict({
    learningRate: real(1e-4, 1, { log: true, default: 0.025, label: 'starting learning rate' }),
    minLearningRate: real(1e-8, 1, { log: true, default: 1e-4, label: 'floor, as a share of the start' }),
    epochs: int(1, 10_000, { default: 5, label: 'passes over the examples' }),
    batch: int(1, 100_000, { default: 1, label: 'examples per step' }),
    shuffle: bool({ default: false, label: 'new random order each pass' }),
  }),
  make: (p, k) =>
    word2vecSgdAlgorithm(p, {
      learningRate: k['learningRate'] as number,
      minLearningRate: k['minLearningRate'] as number,
      epochs: k['epochs'] as number,
      batch: k['batch'] as number,
      shuffle: k['shuffle'] as boolean,
    }),
  params: (s) => ({ x: Array.from(s.x) }),
}

/**
 * Extend the library's smooth solvers with named modes of one model's own training: the generic solvers (gradient
 * descent, conjugate gradient, Newton, BFGS, L-BFGS, coordinate descent and the stochastic `gradient` mode with its
 * optimiser, schedule and batch) stay in the table, the model's specific algorithm sits beside them.
 *
 * @param extra - The named modes to add, built over the model's problem.
 * @param defaultMode - The default solver (default: the first named mode, else gradient descent).
 * @returns The table.
 * @concept word-embeddings
 * @example
 * ```ts
 * import { solversWith } from 'omai-engine/embedding'
 * solversWith({}).names.includes('lbfgs') // => true
 * ```
 */
export function solversWith<P extends SmoothProblem>(
  // biome-ignore lint/suspicious/noExplicitAny: each mode has a state of its own; the table erases them
  extra: Readonly<Record<string, ModeSpec<P, any, VectorParams>>>,
  defaultMode?: string,
): ModeTable<P, VectorParams> {
  // biome-ignore lint/suspicious/noExplicitAny: as above
  const generic: Record<string, ModeSpec<P, any, VectorParams>> = {}
  for (const name of smoothSolvers.names) generic[name] = smoothSolvers.modeOf({ kind: name }) as never
  return modes(
    { ...extra, ...generic },
    {
      label: 'solver',
      description: 'how the objective is minimised',
      default: defaultMode ?? Object.keys(extra)[0] ?? 'gradientDescent',
    },
  )
}

/** The solvers of the word2vec loss: its own SGD (`word2vecSgd`, the default) and the generic smooth solvers. */
export const word2vecSolvers: ModeTable<Word2vecProblem, VectorParams> = solversWith({ word2vecSgd })

/** A trained word-embedding model. */
export interface Word2vecModel {
  /** The words, most frequent first. */
  readonly words: readonly string[]
  /** The word vectors (the mean of a word's input rows, so fastText's include the n-gram vectors). */
  readonly vectors: WordMatrix
  /** The output (context) vectors, one per output row. */
  readonly outputVectors: readonly Float64Array[]
  /** The input rows (words, then n-gram buckets) before averaging. */
  readonly inputVectors: readonly Float64Array[]
  /** The objective at the end, over all examples. */
  readonly loss: number
  readonly steps: number
  /** The subword settings (fastText), kept to embed unseen words. */
  readonly subwords?: { readonly minn: number; readonly maxn: number; readonly buckets: number }
}

/**
 * The model that a flat parameter vector of a problem stands for.
 *
 * @param problem - The problem.
 * @param x - The flat parameters.
 * @param extra - The loss and steps to record.
 * @returns The model.
 * @concept word-embeddings
 * @example
 * ```ts
 * import { word2vecModel, word2vecProblem } from 'omai-engine/embedding'
 * import { key } from 'omai-engine/random'
 * const p = word2vecProblem([['a', 'b', 'c']], { dimension: 2, window: 1, negatives: 1 }, key(0))
 * word2vecModel(p, p.x0 as Float64Array).words // => ['a', 'b', 'c']
 * ```
 */
export function word2vecModel(
  problem: Word2vecProblem,
  x: ArrayLike<number>,
  extra: { readonly loss?: number; readonly steps?: number } = {},
): Word2vecModel {
  const { input, output } = problem.unravel(x)
  const d = problem.options.dimension
  const rows = problem.rowsOfWord.map((rs) => {
    const v = new Float64Array(d)
    for (const r of rs)
      for (let c = 0; c < d; c++) v[c] = (v[c] ?? 0) + ((input[r] as Float64Array)[c] ?? 0) / rs.length
    return v
  })
  return {
    words: problem.vocabulary.words,
    vectors: { words: problem.vocabulary.words, rows },
    outputVectors: output,
    inputVectors: input,
    loss: extra.loss ?? Number.NaN,
    steps: extra.steps ?? 0,
    ...(problem.options.subwords === undefined ? {} : { subwords: problem.options.subwords }),
  }
}

/** The hyperparameters of {@link trainWord2vec}. */
export interface Word2vecHyper extends Word2vecOptions {
  /** The solver, a value of {@link word2vecSolvers}'s space (default word2vec's own SGD). */
  readonly solver?: SpaceValue
  /** The step budget (default 100000; word2vec's own SGD stops by itself after its epochs). */
  readonly maxSteps?: number
  /** Words rarer than this are dropped (default 1). */
  readonly minCount?: number
}

/**
 * Train word2vec or fastText: build the problem for the corpus and run the solver of the slot on it.
 *
 * @param sentences - Token lists.
 * @param hyper - The model, the solver and the budget.
 * @param key - The key for examples, start and any random solver steps.
 * @param run - Run options (`tracer`, `policy`) to record the solver's steps.
 * @returns The model.
 * @concept word-embeddings
 * @index function Train skip-gram or CBOW word vectors with negative sampling.
 * @example
 * ```ts
 * import { trainWord2vec } from 'omai-engine/embedding'
 * import { key } from 'omai-engine/random'
 * const m = trainWord2vec([['a', 'b', 'c', 'a', 'b', 'c']], { dimension: 4, window: 1, solver: { kind: 'word2vecSgd', epochs: 2 } }, key(0))
 * m.vectors.rows.length // => 3
 * ```
 */
export function trainWord2vec(
  sentences: readonly (readonly string[])[],
  hyper: Word2vecHyper,
  key: Key,
  run: Partial<Omit<RunOptions, 'maxSteps' | 'seed' | 'key'>> = {},
): Word2vecModel {
  const problem = word2vecProblem(sentences, hyper, foldIn(key, 0), hyper.minCount ?? 1)
  const solver = hyper.solver ?? { kind: 'word2vecSgd' }
  const algorithm = word2vecSolvers.make(problem, solver, foldIn(key, 1))
  const result = runAlgorithm(algorithm, { ...run, key: foldIn(key, 2), maxSteps: hyper.maxSteps ?? 100_000 })
  const x = word2vecSolvers.params(solver, result.state).x
  const loss = item(problem.objective(fromStorage(Float64Array.from(x), [x.length], 'float64')) as Value)
  return word2vecModel(problem, x, { loss, steps: result.steps })
}

/**
 * The vector of a word that may not be in the vocabulary: the mean of its input rows for a known word, and for fastText
 * the mean of the character n-gram rows of an unseen one (zeros when it shares no n-gram bucket's training signal).
 *
 * @param model - A trained model.
 * @param word - Any word.
 * @returns The vector, or undefined for an unknown word in a model without subwords.
 * @concept word-embeddings
 * @example
 * ```ts
 * import { subwordVector, trainWord2vec } from 'omai-engine/embedding'
 * import { key } from 'omai-engine/random'
 * const m = trainWord2vec([['walking', 'walked', 'walks']], { dimension: 3, window: 2, subwords: { minn: 3, maxn: 4, buckets: 50 }, solver: { kind: 'word2vecSgd', epochs: 1 } }, key(1))
 * subwordVector(m, 'walker')?.length // => 3
 * ```
 */
export function subwordVector(model: Word2vecModel, word: string): Float64Array | undefined {
  const i = model.words.indexOf(word)
  if (i >= 0) return model.vectors.rows[i]
  if (model.subwords === undefined) return undefined
  const { minn, maxn, buckets } = model.subwords
  const rows = fastTextBuckets(word, minn, maxn, buckets).map((b) => model.words.length + b)
  const d = model.vectors.rows[0]?.length ?? 0
  const v = new Float64Array(d)
  for (const r of rows)
    for (let c = 0; c < d; c++) v[c] = (v[c] ?? 0) + ((model.inputVectors[r] as Float64Array)[c] ?? 0) / rows.length
  return v
}
