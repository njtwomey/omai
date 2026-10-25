/**
 * Position encodings for attention models, which tell a permutation-blind attention layer where each token sits: the
 * Transformer's sinusoidal encodings, learned absolute positions, rotary position embeddings (RoPE, which rotate
 * queries and keys so scores depend on relative position) and ALiBi's linear distance biases.
 */

import { WhiteboxError } from '../../base/index.ts'
import {
  add,
  concat,
  fromStorage,
  mul,
  neg,
  promoteFloat,
  reshape,
  sliceAxis,
  stack,
  type Tensor,
  type TensorLike,
  type Value,
} from '../../tensor/index.ts'

/**
 * Sinusoidal position encodings (Vaswani et al. 2017): PE[p, 2i] = sin(p / base^(2i / d)) and
 * PE[p, 2i + 1] = cos(p / base^(2i / d)), added to token embeddings so attention can use relative offsets.
 *
 * @param length - Number of positions.
 * @param dim - Encoding width (even).
 * @param options - `base` (default 10000) and the first position `offset` (default 0).
 * @returns `[length, dim]`.
 * @concept positional-encoding
 * @example
 * ```ts
 * import { sinusoidalPositions } from 'whitebox-engine/nn'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(sinusoidalPositions(2, 2)) // => [[0, 1], [0.8414709848078965, 0.5403023058681398]]
 * ```
 */
export function sinusoidalPositions(
  length: number,
  dim: number,
  options: { readonly base?: number; readonly offset?: number } = {},
): Tensor {
  if (dim % 2 !== 0) throw new WhiteboxError('value', 'sinusoidalPositions: the width must be even')
  const base = options.base ?? 10000
  const offset = options.offset ?? 0
  const out = new Float64Array(length * dim)
  for (let p = 0; p < length; p++) {
    for (let i = 0; i < dim / 2; i++) {
      const angle = (p + offset) / base ** ((2 * i) / dim)
      out[p * dim + 2 * i] = Math.sin(angle)
      out[p * dim + 2 * i + 1] = Math.cos(angle)
    }
  }
  return fromStorage(out, [length, dim], 'float64')
}

/**
 * Learned absolute positions (BERT, GPT-2): rows `offset .. offset + length` of a trainable table, to add to token
 * embeddings.
 *
 * @param table - The position table `[maxLength, dim]`.
 * @param length - Number of positions.
 * @param offset - The first position (default 0; a decoding step's cache length).
 * @returns `[length, dim]`, differentiable in the table.
 * @example
 * ```ts
 * import { learnedPositions } from 'whitebox-engine/nn'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(learnedPositions([[0, 0], [1, 1], [2, 2]], 2, 1)) // => [[1, 1], [2, 2]]
 * ```
 */
export function learnedPositions(table: TensorLike, length: number, offset = 0): Value {
  const [t] = promoteFloat(table) as [Value]
  if (offset + length > (t.shape[0] ?? 0))
    throw new WhiteboxError('value', `learnedPositions: positions up to ${offset + length} exceed the table`)
  return sliceAxis(t, 0, offset, offset + length)
}

/** The angles of rotary position embeddings. */
export interface RotaryAngles {
  /** cos of each angle `[length, dim / 2]`. */
  readonly cos: Tensor
  /** sin of each angle `[length, dim / 2]`. */
  readonly sin: Tensor
}

/** Options of {@link rotaryAngles}. */
export interface RotaryOptions {
  /** Frequency base (default 10000; LLaMA 3 uses 500000). */
  readonly base?: number
  /** The first position (default 0). */
  readonly offset?: number
  /** Divide positions by this factor (linear position interpolation for longer contexts; default 1). */
  readonly scale?: number
  /** Explicit positions instead of offset .. offset + length. */
  readonly positions?: readonly number[]
}

/**
 * The angles theta_{p, i} = p base^(-2i / d) of rotary position embeddings (Su et al. 2021) for positions p and
 * frequency pairs i.
 *
 * @param length - Number of positions.
 * @param dim - The rotated width (even; often the head size).
 * @param options - Base, offset, interpolation scale or explicit positions.
 * @returns cos and sin tables `[length, dim / 2]`.
 * @reference Su, Lu, Pan, Murtadha, Wen and Liu (2021), RoFormer: enhanced transformer with rotary position embedding.
 * @example
 * ```ts
 * import { rotaryAngles } from 'whitebox-engine/nn'
 * rotaryAngles(4, 8).cos.shape // => [4, 4]
 * ```
 */
export function rotaryAngles(length: number, dim: number, options: RotaryOptions = {}): RotaryAngles {
  if (dim % 2 !== 0) throw new WhiteboxError('value', 'rotaryAngles: the width must be even')
  const base = options.base ?? 10000
  const scale = options.scale ?? 1
  const positions = options.positions ?? Array.from({ length }, (_, p) => p + (options.offset ?? 0))
  const half = dim / 2
  const cos = new Float64Array(positions.length * half)
  const sin = new Float64Array(positions.length * half)
  for (const [r, p] of positions.entries()) {
    for (let i = 0; i < half; i++) {
      const angle = (p / scale) * base ** ((-2 * i) / dim)
      cos[r * half + i] = Math.cos(angle)
      sin[r * half + i] = Math.sin(angle)
    }
  }
  return {
    cos: fromStorage(cos, [positions.length, half], 'float64'),
    sin: fromStorage(sin, [positions.length, half], 'float64'),
  }
}

/**
 * Rotate pairs of features by position-dependent angles (rotary position embeddings): applied to queries and keys,
 * it makes q_m . k_n depend on the offset m - n only. Pairs are the two halves of the last axis (GPT-NeoX and LLaMA,
 * the default) or adjacent features (GPT-J and the RoFormer paper, `interleaved`).
 *
 * @param x - `[..., length, dim]`.
 * @param angles - From {@link rotaryAngles} for the same length and dim (or a prefix of the width).
 * @param interleaved - Pair adjacent features instead of halves (default false).
 * @returns The rotated values, differentiable in x.
 * @concept positional-encoding
 * @example
 * ```ts
 * import { applyRotary, rotaryAngles } from 'whitebox-engine/nn'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(applyRotary([[1, 0]], rotaryAngles(1, 2))) // => [[1, 0]]
 * ```
 */
export function applyRotary(x: TensorLike, angles: RotaryAngles, interleaved = false): Value {
  const [v] = promoteFloat(x) as [Value]
  const d = v.shape[v.shape.length - 1] ?? 0
  const half = angles.cos.shape[1] ?? 0
  const rot = 2 * half
  if (rot > d) throw new WhiteboxError('shape', 'applyRotary: the angles are wider than the input')
  const head = rot === d ? v : sliceAxis(v, -1, 0, rot)
  const tail = rot === d ? undefined : sliceAxis(v, -1, rot, d)
  let out: Value
  if (interleaved) {
    const pairs = reshape(head, [...head.shape.slice(0, -1), half, 2])
    const a = reshape(sliceAxis(pairs, -1, 0, 1), [...head.shape.slice(0, -1), half])
    const b = reshape(sliceAxis(pairs, -1, 1, 2), [...head.shape.slice(0, -1), half])
    const ra = add(mul(a, angles.cos), mul(neg(b), angles.sin))
    const rb = add(mul(b, angles.cos), mul(a, angles.sin))
    out = reshape(stack([ra, rb], ra.shape.length), head.shape)
  } else {
    const a = sliceAxis(head, -1, 0, half)
    const b = sliceAxis(head, -1, half, rot)
    out = concat([add(mul(a, angles.cos), mul(neg(b), angles.sin)), add(mul(b, angles.cos), mul(a, angles.sin))], -1)
  }
  return tail === undefined ? out : concat([out, tail], -1)
}

/**
 * The head slopes of ALiBi (Press et al. 2022): a geometric sequence 2^(-8 / n), 2^(-16 / n), ... for n heads, with
 * the paper's interleaving rule when n is not a power of two.
 *
 * @param numHeads - Number of heads.
 * @returns One slope per head.
 * @reference Press, Smith and Lewis (2022), Train short, test long: attention with linear biases enables input length
 *   extrapolation.
 * @example
 * ```ts
 * import { alibiSlopes } from 'whitebox-engine/nn'
 * alibiSlopes(4) // => [0.25, 0.0625, 0.015625, 0.00390625]
 * ```
 */
export function alibiSlopes(numHeads: number): number[] {
  const powerOfTwo = (n: number): number[] => {
    const start = 2 ** -(2 ** -(Math.log2(n) - 3))
    return Array.from({ length: n }, (_, i) => start * start ** i)
  }
  if (Number.isInteger(Math.log2(numHeads))) return powerOfTwo(numHeads)
  const closest = 2 ** Math.floor(Math.log2(numHeads))
  const extra = powerOfTwo(2 * closest).filter((_, i) => i % 2 === 0)
  return [...powerOfTwo(closest), ...extra.slice(0, numHeads - closest)]
}

/**
 * The ALiBi attention bias `[H, L, S]`: -slope_h |i - j| (or slope_h (j - i) for j <= i when causal, the paper's
 * form), added to the attention scores in place of position embeddings.
 *
 * @param numHeads - Number of heads.
 * @param length - Query length.
 * @param sourceLength - Key length (default the query length).
 * @param causal - Use the one-sided form (default true).
 * @returns The bias.
 * @example
 * ```ts
 * import { alibiBias } from 'whitebox-engine/nn'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(alibiBias(1, 2)) // => [[[0, 0], [-0.00390625, 0]]]
 * ```
 */
export function alibiBias(numHeads: number, length: number, sourceLength: number = length, causal = true): Tensor {
  const slopes = alibiSlopes(numHeads)
  const out = new Float64Array(numHeads * length * sourceLength)
  // Queries are aligned with the last keys (a decoding step attends to the whole cache).
  const shift = sourceLength - length
  for (let h = 0; h < numHeads; h++) {
    for (let i = 0; i < length; i++) {
      for (let j = 0; j < sourceLength; j++) {
        const distance = j - (i + shift)
        out[(h * length + i) * sourceLength + j] =
          (slopes[h] ?? 0) * (causal ? Math.min(distance, 0) : -Math.abs(distance))
      }
    }
  }
  return fromStorage(out, [numHeads, length, sourceLength], 'float64')
}
