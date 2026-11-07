/**
 * Losses for classifiers with very many classes, such as word vocabularies, where a full softmax is too costly: the
 * hierarchical softmax over a Huffman code of the classes (word2vec's), which costs O(log C) per example, and the
 * sampled softmax, which scores the true class against a few sampled candidates with a correction for the sampling.
 */

import { MdmlError, type Reduction } from '../base/index.ts'
import { huffmanTree } from '../information/index.ts'
import { logSigmoid } from '../numerics/special/index.ts'
import { type Key, randint, uniform } from '../random/index.ts'
import {
  add,
  asTensor,
  concat,
  equal,
  expandDims,
  logSoftmax,
  matmul,
  mul,
  neg,
  promoteFloat,
  reshape,
  sliceAxis,
  squeeze,
  sub,
  sum,
  type TensorLike,
  take,
  takeAlongAxis,
  tensor,
  toFlat,
  transpose,
  type Value,
  where,
} from '../tensor/index.ts'
import { reduce } from './_reduce.ts'

/** A Huffman code over classes: for each class, the inner nodes on its root-to-leaf path and the branch bits. */
export interface HuffmanCoding {
  readonly nClasses: number
  /** Inner-node indices (rows of the node-weight matrix, `0..C-2`) from the root down, per class. */
  readonly paths: readonly (readonly number[])[]
  /** The branch taken at each node of the path (0 or 1), per class. */
  readonly codes: readonly (readonly number[])[]
}

/**
 * Build a binary Huffman code over classes from their frequencies (Mikolov et al., 2013): frequent classes get short
 * paths, so the hierarchical softmax costs O(log C) on average. The tree is information's `huffmanTree` (ties break
 * towards the lower node id, so the code is deterministic).
 *
 * @param counts - Class frequencies (positive), one per class; at least two classes.
 * @returns The coding: inner-node paths and branch bits per class.
 * @throws MdmlError with code `value` for fewer than two classes or a non-positive count.
 * @example
 * ```ts
 * import { huffmanCoding } from 'mdml-engine/losses'
 * huffmanCoding([5, 1, 1]).codes // => [[1], [0, 0], [0, 1]]
 * ```
 */
export function huffmanCoding(counts: readonly number[]): HuffmanCoding {
  const c = counts.length
  if (c < 2) throw new MdmlError('value', 'huffmanCoding: needs at least two classes')
  if (counts.some((w) => !(w > 0))) throw new MdmlError('value', 'huffmanCoding: counts must be positive')
  // Leaves are nodes 0..C-1 and inner node k is node C + k (a binary tree over C >= 2 leaves needs no padding).
  const { nodes } = huffmanTree(counts)
  const parent = new Int32Array(nodes.length).fill(-1)
  const bit = new Int8Array(nodes.length)
  for (const n of nodes) {
    for (const [digit, child] of n.children.entries()) {
      parent[child] = n.id
      bit[child] = digit
    }
  }
  const paths: number[][] = []
  const codes: number[][] = []
  for (let leaf = 0; leaf < c; leaf++) {
    const path: number[] = []
    const code: number[] = []
    for (let node = leaf; (parent[node] ?? -1) >= 0; node = parent[node] ?? -1) {
      path.push((parent[node] ?? 0) - c)
      code.push(bit[node] ?? 0)
    }
    paths.push(path.reverse())
    codes.push(code.reverse())
  }
  return { nClasses: c, paths, codes }
}

/** Padded `[C, L]` node indices, signs (+1 for bit 0, -1 for bit 1, 0 for padding) of a coding. */
function padded(coding: HuffmanCoding): {
  readonly nodes: number[][]
  readonly signs: number[][]
  readonly depth: number
} {
  let depth = 0
  for (const p of coding.paths) depth = Math.max(depth, p.length)
  const nodes = coding.paths.map((p) => Array.from({ length: depth }, (_, l) => p[l] ?? 0))
  const signs = coding.codes.map((code) =>
    Array.from({ length: depth }, (_, l) => (l < code.length ? 1 - 2 * (code[l] ?? 0) : 0)),
  )
  return { nodes, signs, depth }
}

/** Options of {@link hierarchicalSoftmax}. */
export interface HierarchicalSoftmaxOptions {
  /** Per-node biases `[C - 1]`. */
  readonly nodeBias?: TensorLike
  readonly reduction?: Reduction
}

/**
 * Hierarchical softmax loss (Morin and Bengio, 2005; Mikolov et al., 2013): the negative log-probability of each
 * target class as a product of binary decisions sigma(+-(h . w_n + b_n)) along its path in a code tree. Only the
 * nodes on the target's path are touched, so the cost is O(depth D) per sample rather than O(C D).
 *
 * @param hidden - Hidden vectors `[N, D]`.
 * @param nodeWeights - Inner-node vectors `[C - 1, D]`.
 * @param target - Class indices `[N]`.
 * @param coding - The class code ({@link huffmanCoding}).
 * @param options - Node biases and reduction.
 * @returns The loss, differentiable in the hidden vectors, node weights and biases.
 * @concept losses
 * @example
 * ```ts
 * import { hierarchicalSoftmax, huffmanCoding } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * item(hierarchicalSoftmax([[0, 0]], [[1, 1]], [0], huffmanCoding([1, 1]))) === Math.LN2 // => true
 * ```
 */
export function hierarchicalSoftmax(
  hidden: TensorLike,
  nodeWeights: TensorLike,
  target: TensorLike,
  coding: HuffmanCoding,
  options: HierarchicalSoftmaxOptions = {},
): Value {
  const [h, w] = promoteFloat(hidden, nodeWeights) as [Value, Value]
  const n = h.shape[0] ?? 0
  const d = h.shape[1] ?? 0
  const { nodes, signs, depth } = padded(coding)
  const labels = toFlat(asTensor(target))
  const idx = tensor(labels.map((y) => nodes[y] ?? []))
  const sign = tensor(labels.map((y) => signs[y] ?? []))
  const rows = reshape(take(w, reshape(idx, [n * depth]), 0), [n, depth, d])
  let scores = sum(mul(rows, expandDims(h, 1)), -1)
  if (options.nodeBias !== undefined) {
    const bias = take(promoteFloat(options.nodeBias)[0] as Value, reshape(idx, [n * depth]))
    scores = add(scores, reshape(bias, [n, depth]))
  }
  // Padding has sign 0: logSigmoid(0) is masked out by multiplying by |sign|.
  const terms = neg(sum(mul(logSigmoid(mul(scores, sign)), mul(sign, sign)), -1))
  return reduce(terms, { reduction: options.reduction ?? 'mean' })
}

/**
 * The full log-probabilities `[N, C]` of a hierarchical softmax, for evaluation (each row's probabilities sum to 1).
 *
 * @param hidden - Hidden vectors `[N, D]`.
 * @param nodeWeights - Inner-node vectors `[C - 1, D]`.
 * @param coding - The class code.
 * @param options - Node biases.
 * @returns Log-probabilities `[N, C]`, differentiable.
 * @example
 * ```ts
 * import { hierarchicalSoftmaxLogProbs, huffmanCoding } from 'mdml-engine/losses'
 * import { exp, sum, item } from 'mdml-engine/tensor'
 * const lp = hierarchicalSoftmaxLogProbs([[0.3, -1]], [[1, 2], [0.5, 0]], huffmanCoding([3, 1, 1]))
 * Math.abs(item(sum(exp(lp))) - 1) < 1e-15 // => true
 * ```
 */
export function hierarchicalSoftmaxLogProbs(
  hidden: TensorLike,
  nodeWeights: TensorLike,
  coding: HuffmanCoding,
  options: { readonly nodeBias?: TensorLike } = {},
): Value {
  const [h, w] = promoteFloat(hidden, nodeWeights) as [Value, Value]
  const n = h.shape[0] ?? 0
  const c = coding.nClasses
  const { nodes, signs, depth } = padded(coding)
  let all = matmul(h, transpose(w))
  if (options.nodeBias !== undefined) all = add(all, promoteFloat(options.nodeBias)[0] as Value)
  const flatNodes = nodes.flat()
  const idx = tensor(Array.from({ length: n }, () => flatNodes))
  const sign = tensor(signs)
  const scores = reshape(takeAlongAxis(all, idx, 1), [n, c, depth])
  return sum(mul(logSigmoid(mul(scores, sign)), mul(sign, sign)), -1)
}

/** How {@link sampledSoftmax} proposes negative classes. */
export type CandidateSampler = 'uniform' | 'log-uniform'

/**
 * Draw candidate classes (with replacement) and their log expected counts log(S q(c)), as TensorFlow's candidate
 * samplers: 'uniform' over the classes, or 'log-uniform' (Zipfian, q(c) = log((c + 2) / (c + 1)) / log(C + 1), for
 * classes sorted by decreasing frequency).
 *
 * @param key - The random key.
 * @param nClasses - The number of classes C.
 * @param nSampled - The number of candidates S.
 * @param sampler - The proposal (default 'log-uniform').
 * @returns The sampled class indices.
 * @example
 * ```ts
 * import { sampleClasses } from 'mdml-engine/losses'
 * import { key } from 'mdml-engine/random'
 * sampleClasses(key(0), 10, 5).every((c) => c >= 0 && c < 10) // => true
 * ```
 */
export function sampleClasses(
  key: Key,
  nClasses: number,
  nSampled: number,
  sampler: CandidateSampler = 'log-uniform',
): number[] {
  if (sampler === 'uniform') return toFlat(randint(key, [nSampled], 0, nClasses))
  const logRange = Math.log(nClasses + 1)
  return Array.from(uniform(key, [nSampled]).data, (u) =>
    Math.min(nClasses - 1, Math.floor(Math.exp(u * logRange)) - 1),
  )
}

function proposalProbability(c: number, nClasses: number, sampler: CandidateSampler): number {
  return sampler === 'uniform' ? 1 / nClasses : Math.log((c + 2) / (c + 1)) / Math.log(nClasses + 1)
}

/** Options of {@link sampledSoftmax}. */
export interface SampledSoftmaxOptions {
  /** Output biases `[C]`. */
  readonly biases?: TensorLike
  /** Explicit candidate classes `[S]` (shared by the batch); otherwise `nSampled` are drawn with `key`. */
  readonly sampled?: readonly number[]
  readonly nSampled?: number
  readonly key?: Key
  /** The proposal the candidates come from (default 'log-uniform'). */
  readonly sampler?: CandidateSampler
  /** Mask candidates equal to a sample's own target (default true). */
  readonly removeAccidentalHits?: boolean
  readonly reduction?: Reduction
}

/**
 * Sampled softmax loss (Jean et al., 2015; tf.nn.sampled_softmax_loss): the cross-entropy of the true class against a
 * shared set of S sampled candidates, with every logit corrected by -log(S q(c)) so that the loss is an unbiased-in-the-
 * limit estimate of the full softmax cross-entropy. Candidates equal to the target are masked out.
 *
 * @param hidden - Hidden vectors `[N, D]`.
 * @param weights - Output-class vectors `[C, D]`.
 * @param target - Class indices `[N]`.
 * @param options - Candidates (explicit, or drawn with a key), proposal, biases, accidental-hit removal and reduction.
 * @returns The loss, differentiable in the hidden vectors, weights and biases.
 * @throws MdmlError with code `value` when neither `sampled` nor `key` with `nSampled` is given.
 * @example
 * ```ts
 * import { sampledSoftmax } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * const loss = sampledSoftmax([[0, 0]], [[1, 0], [0, 1], [1, 1]], [0], { sampled: [1, 2], sampler: 'uniform' })
 * Math.abs(item(loss) - Math.log(3)) < 1e-15 // => true
 * ```
 */
export function sampledSoftmax(
  hidden: TensorLike,
  weights: TensorLike,
  target: TensorLike,
  options: SampledSoftmaxOptions = {},
): Value {
  const [h, w] = promoteFloat(hidden, weights) as [Value, Value]
  const nClasses = w.shape[0] ?? 0
  const sampler = options.sampler ?? 'log-uniform'
  let sampled = options.sampled
  if (sampled === undefined) {
    if (options.key === undefined || options.nSampled === undefined)
      throw new MdmlError('value', 'sampledSoftmax: pass `sampled`, or `key` and `nSampled`')
    sampled = sampleClasses(options.key, nClasses, options.nSampled, sampler)
  }
  const s = sampled.length
  const labels = toFlat(asTensor(target))
  const logQ = (c: number): number => Math.log(s * proposalProbability(c, nClasses, sampler))
  const bias = options.biases === undefined ? undefined : (promoteFloat(options.biases)[0] as Value)
  const labelIdx = tensor(labels, { dtype: 'int32' })
  const sampledIdx = tensor([...sampled], { dtype: 'int32' })
  let trueLogits = sum(mul(take(w, labelIdx, 0), h), -1)
  let sampledLogits = matmul(h, transpose(take(w, sampledIdx, 0)))
  if (bias !== undefined) {
    trueLogits = add(trueLogits, take(bias, labelIdx))
    sampledLogits = add(sampledLogits, take(bias, sampledIdx))
  }
  trueLogits = sub(trueLogits, tensor(labels.map(logQ)))
  sampledLogits = sub(sampledLogits, tensor(sampled.map(logQ)))
  if (options.removeAccidentalHits ?? true) {
    const hits = equal(expandDims(labelIdx, 1), expandDims(sampledIdx, 0))
    sampledLogits = where(hits, Number.NEGATIVE_INFINITY, sampledLogits)
  }
  const logits = concat([expandDims(trueLogits, 1), sampledLogits], 1)
  const terms = neg(squeeze(sliceAxis(logSoftmax(logits, -1), 1, 0, 1), 1))
  return reduce(terms, { reduction: options.reduction ?? 'mean' })
}
