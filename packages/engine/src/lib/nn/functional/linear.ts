/**
 * The linear layers as functions, torch.nn.functional style: the affine map `linear` with PyTorch's `[out, in]` weight
 * layout, the `bilinear` map, `embedding` lookup and `embeddingBag`, which reduces bags of embedding rows by sum, mean
 * or max without materialising them.
 */

import { OmaiError } from '../../base/index.ts'
import {
  add,
  asTensor,
  concat,
  div,
  equal,
  expandDims,
  fromStorage,
  matmul,
  max,
  minimum,
  mul,
  norm,
  primalOf,
  promoteFloat,
  reshape,
  segmentSum,
  sliceAxis,
  stopGradient,
  sum,
  swapaxes,
  type Tensor,
  type TensorLike,
  take,
  toFlat,
  type Value,
  where,
  zeros,
} from '../../tensor/index.ts'

/**
 * Affine map y = x W^T + b with PyTorch's weight layout `[out, in]` (torch.nn.functional.linear).
 *
 * @param x - Input `[..., in]`.
 * @param weight - Weight `[out, in]`.
 * @param bias - Bias `[out]` (optional).
 * @returns Output `[..., out]`, differentiable in all arguments.
 * @concept layers
 * @example
 * ```ts
 * import { linear } from 'omai-engine/nn'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(linear([[1, 2]], [[1, 1], [0, 1]], [0.5, 0])) // => [[3.5, 2]]
 * ```
 */
export function linear(x: TensorLike, weight: TensorLike, bias?: TensorLike): Value {
  const [xv, w] = promoteFloat(x, weight) as [Value, Value]
  const y = matmul(xv, swapaxes(w, -1, -2))
  return bias === undefined ? y : add(y, promoteFloat(bias)[0] as Value)
}

/**
 * Bilinear map y_k = x1^T W_k x2 + b_k (torch.nn.functional.bilinear).
 *
 * @param x1 - First input `[..., in1]`.
 * @param x2 - Second input `[..., in2]` (same leading shape).
 * @param weight - Weight `[out, in1, in2]`.
 * @param bias - Bias `[out]` (optional).
 * @returns Output `[..., out]`.
 * @example
 * ```ts
 * import { bilinear } from 'omai-engine/nn'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(bilinear([[1, 2]], [[3, 4]], [[[1, 0], [0, 1]]])) // => [[11]]
 * ```
 */
export function bilinear(x1: TensorLike, x2: TensorLike, weight: TensorLike, bias?: TensorLike): Value {
  const [a, b, w] = promoteFloat(x1, x2, weight) as [Value, Value, Value]
  const [out = 0, in1 = 0, in2 = 0] = w.shape
  if (a.shape.at(-1) !== in1 || b.shape.at(-1) !== in2)
    throw new OmaiError('shape', 'bilinear: inputs must end in the weight sizes in1 and in2')
  const lead = a.shape.slice(0, -1)
  const af = reshape(a, [-1, in1])
  const bf = reshape(b, [-1, 1, in2])
  // (x1 W) as [N, out, in2], then contracted with x2.
  const t = reshape(matmul(af, reshape(swapaxes(w, 0, 1), [in1, out * in2])), [-1, out, in2])
  let y = reshape(sum(mul(t, bf), 2), [...lead, out])
  if (bias !== undefined) y = add(y, promoteFloat(bias)[0] as Value)
  return y
}

/** Options of {@link embedding}. */
export interface EmbeddingOptions {
  /** An index whose row gets no gradient (a padding token). */
  readonly paddingIdx?: number
  /** Rescale looked-up rows whose norm exceeds this. */
  readonly maxNorm?: number
  /** The norm for `maxNorm` (default 2). */
  readonly normType?: number
}

const indices = (x: TensorLike): Tensor<'int32'> => primalOf(asTensor(x, 'int32')) as Tensor<'int32'>

/**
 * Embedding lookup (torch.nn.functional.embedding): row i of the table for each index i. The padding index's row
 * receives no gradient; with `maxNorm`, rows longer than it are rescaled to that norm.
 *
 * @param input - Integer indices of any shape.
 * @param weight - The table `[numEmbeddings, dim]`.
 * @param options - Padding index, maximum norm and its order.
 * @returns `[...input.shape, dim]`.
 * @concept embeddings
 * @example
 * ```ts
 * import { embedding } from 'omai-engine/nn'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(embedding([2, 0], [[0, 0], [1, 1], [2, 2]])) // => [[2, 2], [0, 0]]
 * ```
 */
export function embedding(input: TensorLike, weight: TensorLike, options: EmbeddingOptions = {}): Value {
  const [w] = promoteFloat(weight) as [Value]
  const idx = indices(input)
  const n = w.shape[0] ?? 0
  if (toFlat(idx).some((i) => i < 0 || i >= n)) throw new OmaiError('value', `embedding: indices must be in [0, ${n})`)
  let rows = take(w, reshape(idx, [-1]), 0)
  if (options.maxNorm !== undefined) {
    const lengths = norm(rows, options.normType ?? 2, 1, true)
    rows = mul(rows, stopGradient(minimum(div(options.maxNorm, add(lengths, 1e-7)), 1)))
  }
  if (options.paddingIdx !== undefined) {
    const pad = expandDims(equal(reshape(idx, [-1]), options.paddingIdx), 1)
    rows = where(pad, stopGradient(rows), rows)
  }
  return reshape(rows, [...idx.shape, w.shape[1] ?? 0])
}

/** Options of {@link embeddingBag}. */
export interface EmbeddingBagOptions {
  /** Start of each bag in a 1-D input (required then; a 2-D input has one bag per row). */
  readonly offsets?: TensorLike
  /** How to reduce a bag (default `mean`). */
  readonly mode?: 'sum' | 'mean' | 'max'
  /** A weight per index, multiplied into its row (`sum` mode only). */
  readonly perSampleWeights?: TensorLike
  /** An index left out of every bag. */
  readonly paddingIdx?: number
  /** The last offset marks the end of the last bag (CSR style) rather than the start of one. */
  readonly includeLastOffset?: boolean
  readonly maxNorm?: number
  readonly normType?: number
}

/**
 * Bags of embeddings reduced without materialising the rows (torch.nn.functional.embedding_bag): sum, mean or max of
 * the rows of each bag, bags given by offsets into a flat index list or by the rows of a 2-D input. Empty bags give
 * zeros; the padding index is skipped (and not counted in the mean).
 *
 * @param input - Indices: 1-D with `offsets`, or 2-D `[bags, n]`.
 * @param weight - The table `[numEmbeddings, dim]`.
 * @param options - Offsets, mode, per-sample weights, padding index, includeLastOffset, maxNorm.
 * @returns `[bags, dim]`.
 * @example
 * ```ts
 * import { embeddingBag } from 'omai-engine/nn'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(embeddingBag([0, 1, 2], [[1, 0], [3, 0], [5, 2]], { offsets: [0, 2], mode: 'sum' })) // => [[4, 0], [5, 2]]
 * ```
 */
export function embeddingBag(input: TensorLike, weight: TensorLike, options: EmbeddingBagOptions = {}): Value {
  const [w] = promoteFloat(weight) as [Value]
  const idx = indices(input)
  const flat = toFlat(idx)
  const dim = w.shape[1] ?? 0
  let starts: number[]
  if (idx.shape.length === 2) {
    const [bags = 0, m = 0] = idx.shape
    starts = Array.from({ length: bags + 1 }, (_, i) => i * m)
  } else {
    if (options.offsets === undefined) throw new OmaiError('value', 'embeddingBag: a 1-D input needs offsets')
    starts = toFlat(indices(options.offsets))
    if (!(options.includeLastOffset ?? false)) starts.push(flat.length)
  }
  const bags = starts.length - 1
  const mode = options.mode ?? 'mean'
  if (options.perSampleWeights !== undefined && mode !== 'sum')
    throw new OmaiError('value', "embeddingBag: perSampleWeights needs mode 'sum'")
  // Each kept position's bag, skipping the padding index.
  const keep: number[] = []
  const bagOf: number[] = []
  for (let b = 0; b < bags; b++) {
    for (let j = starts[b] ?? 0; j < (starts[b + 1] ?? 0); j++) {
      if (options.paddingIdx !== undefined && flat[j] === options.paddingIdx) continue
      keep.push(j)
      bagOf.push(b)
    }
  }
  if (keep.length === 0) return zeros([bags, dim])
  const keepIdx = fromStorage(Int32Array.from(keep), [keep.length], 'int32')
  let rows = embedding(take(reshape(idx, [-1]), keepIdx, 0), w, {
    ...(options.maxNorm === undefined ? {} : { maxNorm: options.maxNorm }),
    ...(options.normType === undefined ? {} : { normType: options.normType }),
  })
  if (options.perSampleWeights !== undefined) {
    const psw = take(reshape(promoteFloat(options.perSampleWeights)[0] as Value, [-1]), keepIdx, 0)
    rows = mul(rows, expandDims(psw, 1))
  }
  const segments = fromStorage(Int32Array.from(bagOf), [bagOf.length], 'int32')
  if (mode === 'max') {
    let at = 0
    const out: Value[] = []
    for (let b = 0; b < bags; b++) {
      const count = bagOf.filter((x) => x === b).length
      out.push(count === 0 ? zeros([1, dim]) : max(sliceAxis(rows, 0, at, at + count), 0, true))
      at += count
    }
    return concat(out, 0)
  }
  const total = segmentSum(rows, segments, bags)
  if (mode === 'sum') return total
  const counts = new Float64Array(bags)
  for (const b of bagOf) counts[b] = (counts[b] ?? 0) + 1
  return div(
    total,
    fromStorage(
      Float64Array.from(counts, (c) => Math.max(c, 1)),
      [bags, 1],
      'float64',
    ),
  )
}
