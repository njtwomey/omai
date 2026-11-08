/**
 * A language model built by hand with a known circuit, to test and teach interpretability methods against: a
 * one-layer transformer over six tokens whose head 0 copies the "name" tokens it has seen (0, 1, 2) into the stream,
 * so the unembedding raises the logit of a name that appeared earlier in the prompt, whose head 1 does the same for
 * the "filler" tokens (3, 4, 5) and whose feed-forward writes nothing. Patching and attribution should therefore find
 * head 0, at the last position, and the position of the name, and nothing else.
 */
import { blocks, type Params } from '../nn/index.ts'
import { key } from '../random/index.ts'
import { tensor, type Value } from '../tensor/index.ts'

/** The truth of {@link plantedCopyCircuit}. */
export interface PlantedTruth {
  /** The block and head that copy the names. */
  readonly layer: number
  readonly head: number
  /** The tokens that head copies (a name earlier in the prompt raises its own logit at the end). */
  readonly nameTokens: readonly number[]
  /** The tokens the other head copies; they never change a difference between two names. */
  readonly fillerTokens: readonly number[]
}

/** A hand-built model and what it computes. */
export interface PlantedCircuit {
  readonly lm: blocks.TransformerLM
  readonly truth: PlantedTruth
}

const identity = (n: number): number[][] =>
  Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)))

/**
 * The copying circuit: embeddings are one-hot, attention is uniform over the prefix (zero queries and keys, no
 * rotary positions), head 0 passes the one-hot's name coordinates, head 1 its filler coordinates, the output
 * projection is the identity and the feed-forward's output weights are zero. At the last position the stream is the
 * last token plus the mean of the earlier names and fillers, so with tied unembedding a name seen in the prompt gets
 * the larger logit.
 *
 * @returns The model and its truth.
 * @concept mechanistic-interpretability
 * @example
 * ```ts
 * import { plantedCopyCircuit } from 'omai-engine/explain'
 * plantedCopyCircuit().truth.head // => 0
 * ```
 */
export function plantedCopyCircuit(): PlantedCircuit {
  const lm = new blocks.TransformerLM(6, 6, 1, 2, { rope: false, key: key(0) })
  const zero = (r: number, c: number): Value =>
    tensor(Array.from({ length: r }, () => Array.from({ length: c }, () => 0)))
  const p = lm.params as Record<string, Params>
  const stack = p['stack'] as Record<string, Params>
  const blk = (stack['blocks'] as Record<string, Params>)['0'] as Record<string, Params>
  const attn = blk['attn'] as Record<string, Params>
  const next: Params = {
    ...p,
    embed: { weight: tensor(identity(6)) },
    stack: {
      ...stack,
      blocks: {
        '0': {
          ...blk,
          attn: {
            ...attn,
            q_proj: { weight: zero(6, 6) },
            k_proj: { weight: zero(6, 6) },
            v_proj: { weight: tensor(identity(6)) },
            o_proj: { weight: tensor(identity(6)) },
          },
          ffn: { ...(blk['ffn'] as Params), down_proj: { weight: zero(6, 16) } },
        },
      },
    },
  }
  lm.setParams(next)
  return { lm, truth: { layer: 0, head: 0, nameTokens: [0, 1, 2], fillerTokens: [3, 4, 5] } }
}
