/**
 * A transformer language model opened up along its residual stream, the view mechanistic interpretability works in.
 * `residualModel` rebuilds the forward pass of a pre-norm `TransformerLM` from its parts (embedding, attention
 * heads, feed-forward blocks, final norm and unembedding) so that every place where a component writes into the
 * stream is a named site: a hook sees the activation there and may replace it, and every site's value is recorded.
 * Logit and tuned lenses, activation and path patching, attribution patching and direct logit attribution are all
 * functions of this one object. Hooks are ordinary tensor code, so a run can be differentiated with `grad`.
 */
import { MdmlError } from '../base/index.ts'
import {
  applyRotary,
  attention,
  type blocks,
  type GroupedQueryAttention,
  linear,
  type Params,
  rotaryAngles,
  splitHeads,
} from '../nn/index.ts'
import {
  add,
  asTensor,
  div,
  fromStorage,
  mean,
  mul,
  reshape,
  sliceAxis,
  sqrt,
  square,
  sub,
  type TensorLike,
  take,
  toFlat,
  type Value,
  zeros,
} from '../tensor/index.ts'

/** Token ids, one row per sequence. */
export type TokenRows = readonly (readonly number[])[]

/**
 * A place where a component writes into the residual stream (or where the stream itself can be read):
 * `embed` (token embeddings), `layer<i>.head<h>` (head h's contribution to the stream, after the output projection),
 * `layer<i>.attn` (the attention sub-block's output, the sum of its heads and the output bias), `layer<i>.mid` (the
 * stream after attention), `layer<i>.mlp` (the feed-forward output) and `layer<i>.out` (the stream after the block).
 */
export type Site = 'embed' | `layer${number}.${'head' | 'attn' | 'mid' | 'mlp' | 'out'}${number | ''}`

/** A hook sees a site and its activation and returns a replacement (or nothing, to leave it). */
export type SiteHook = (site: string, value: Value) => Value | undefined

/** One forward pass of a {@link ResidualModel}. */
export interface ResidualRun {
  /** The logits `[B, T, V]`. */
  readonly logits: Value
  /** The residual stream before the first block (the embeddings) and after each block: `numLayers + 1` values `[B, T, D]`. */
  readonly residual: readonly Value[]
  /** The activation at every site, as it was after the hook. */
  readonly sites: ReadonlyMap<string, Value>
}

/** A transformer opened along its residual stream. */
export interface ResidualModel {
  readonly numLayers: number
  readonly numHeads: number
  readonly headDim: number
  readonly dim: number
  readonly vocabSize: number
  /** Every site name, in the order a forward pass visits them. */
  readonly sites: readonly string[]
  /** A forward pass over token rows, with an optional hook at every site. */
  run(tokens: TokenRows, hook?: SiteHook): ResidualRun
  /** The final normalisation and the unembedding: a residual stream `[..., D]` to logits `[..., V]`. */
  unembed(residual: Value): Value
  /** The unembedding matrix `[V, D]` (the token embedding when tied). */
  readonly unembedding: Value
  /**
   * The final normalisation made linear: `x` normalised with the statistics of `reference` (the full stream), without
   * the bias, so contributions to the stream map to contributions to the logits and sum to them.
   */
  finalNorm(x: Value, reference: Value): Value
  /** The final normalisation's bias `[D]`, if it has one: its own constant contribution to the logits. */
  readonly finalNormBias: Value | undefined
}

/**
 * The site name of a head's contribution.
 *
 * @param layer - The block.
 * @param head - The head.
 * @returns `layer<layer>.head<head>`.
 * @example
 * ```ts
 * import { headSite } from 'mdml-engine/explain'
 * headSite(1, 0) // => 'layer1.head0'
 * ```
 */
export function headSite(layer: number, head: number): string {
  return `layer${layer}.head${head}`
}

function tokensTensor(tokens: TokenRows): Value {
  const t = tokens[0]?.length ?? 0
  if (tokens.length === 0 || t === 0 || tokens.some((r) => r.length !== t))
    throw new MdmlError('shape', 'residualModel: give equal-length, non-empty token rows')
  return fromStorage(Int32Array.from(tokens.flat()), [tokens.length, t], 'int32')
}

/**
 * Open a pre-norm `TransformerLM` along its residual stream. The forward pass is rebuilt from the model's own
 * sub-modules and parameters, so with no hook it equals `lm.forward` (a test checks it), and the attention is computed
 * head by head so a head's contribution is a site of its own.
 *
 * @param lm - A decoder-only language model with pre-norm blocks and no cross-attention.
 * @param params - Parameters to run with (default the model's own); pass traced parameters to differentiate through the run.
 * @returns The opened model.
 * @throws MdmlError with code `value` for a post-norm or cross-attending block.
 * @concept mechanistic-interpretability
 * @example
 * ```ts
 * import { residualModel } from 'mdml-engine/explain'
 * import { blocks } from 'mdml-engine/nn'
 * import { key } from 'mdml-engine/random'
 * const lm = new blocks.TransformerLM(7, 8, 2, 2, { key: key(0) })
 * const m = residualModel(lm)
 * [m.numLayers, m.sites.length, m.run([[1, 2, 3]]).logits.shape] // => [2, 13, [1, 3, 7]]
 * ```
 */
export function residualModel(lm: blocks.TransformerLM, params: Params = lm.params): ResidualModel {
  const stack = lm.part<blocks.TransformerStack>('stack')
  const layers = stack.blockList
  for (const b of layers) {
    if (!b.preNorm || b.crossAttention)
      throw new MdmlError('value', 'residualModel: only pre-norm blocks without cross-attention are opened')
  }
  const embedWeight = (params['embed'] as Params)['weight'] as Value
  const stackParams = params['stack'] as Params
  const blockParams = (i: number): Params => ((stackParams['blocks'] as Params)[String(i)] ?? {}) as Params
  const first = layers[0]
  const numHeads = first?.numHeads ?? 1
  const attn0 = first === undefined ? undefined : headAttention(first)
  const headDim = attn0?.headDim ?? lm.dim / numHeads
  const unembedding: Value = lm.tieEmbeddings ? embedWeight : ((params['lm_head'] as Params)['weight'] as Value)
  const sites: string[] = ['embed']
  layers.forEach((_, i) => {
    for (let h = 0; h < numHeads; h++) sites.push(headSite(i, h))
    sites.push(`layer${i}.attn`, `layer${i}.mid`, `layer${i}.mlp`, `layer${i}.out`)
  })

  const unembed = (r: Value): Value => {
    const normed = stack.has('norm') ? stack.part('norm').apply((stackParams['norm'] ?? {}) as Params, r) : r
    return linear(normed, unembedding)
  }

  const normModule = stack.has('norm') ? stack.part('norm') : undefined
  const normParams = (stackParams['norm'] ?? {}) as Params
  const normEps = (normModule as { eps?: number } | undefined)?.eps ?? 1e-6
  const finalNorm = (x: Value, reference: Value): Value => {
    if (normModule === undefined) return x
    const centre = normModule.kind === 'LayerNorm'
    const c = (v: Value): Value => (centre ? sub(v, mean(v, -1, true)) : v)
    const scale = sqrt(add(mean(square(c(reference)), -1, true), normEps))
    const out = div(c(x), scale)
    return normParams['weight'] === undefined ? out : mul(out, normParams['weight'] as Value)
  }

  return {
    numLayers: layers.length,
    finalNorm,
    finalNormBias: normParams['bias'] as Value | undefined,
    numHeads,
    headDim,
    dim: lm.dim,
    vocabSize: lm.vocabSize,
    sites,
    unembedding,
    unembed,
    run: (tokens, hook) => {
      const record = new Map<string, Value>()
      const visit = (site: string, v: Value): Value => {
        const replaced = hook?.(site, v) ?? v
        record.set(site, replaced)
        return replaced
      }
      const ids = tokensTensor(tokens)
      const length = ids.shape[1] ?? 0
      let h = visit('embed', gather(embedWeight, ids))
      const residual: Value[] = [h]
      layers.forEach((block, i) => {
        const bp = blockParams(i)
        const a = headAttention(block)
        const normed = block.part('attn_norm').apply((bp['attn_norm'] ?? {}) as Params, h) as Value
        const heads = a.heads((bp['attn'] ?? {}) as Params, normed, length)
        let out: Value = heads.bias ?? (zeros(h.shape) as Value)
        heads.contributions.forEach((c, k) => {
          out = add(out, visit(headSite(i, k), c))
        })
        const attnOut = visit(`layer${i}.attn`, out)
        const mid = visit(`layer${i}.mid`, add(h, attnOut))
        const ffnIn = block.part('ffn_norm').apply((bp['ffn_norm'] ?? {}) as Params, mid) as Value
        const mlp = visit(`layer${i}.mlp`, block.part('ffn').apply((bp['ffn'] ?? {}) as Params, ffnIn) as Value)
        h = visit(`layer${i}.out`, add(mid, mlp))
        residual.push(h)
      })
      return { logits: unembed(h), residual, sites: record }
    },
  }
}

function gather(table: Value, ids: Value): Value {
  const [b = 1, t = 1] = ids.shape
  const rows = Array.from(toFlat(asTensor(ids)))
  const d = table.shape[1] ?? 0
  // Embedding lookup as a differentiable take, so gradients reach the table when it is traced.
  return reshape(take(table, rows, 0), [b, t, d])
}

/** What {@link headAttention} computes for a block. */
interface HeadAttention {
  readonly headDim: number
  heads(
    params: Params,
    x: Value,
    length: number,
  ): { readonly contributions: readonly Value[]; readonly bias: Value | undefined }
}

/** The block's attention, computed head by head: each head's `z_h W_o[:, h]` is its contribution to the stream. */
function headAttention(block: blocks.TransformerBlock): HeadAttention {
  const attn = block.part<GroupedQueryAttention>('attn')
  return {
    headDim: attn.headDim,
    heads: (p, x, length) => {
      const lin = (name: string, v: Value): Value => {
        const q = p[name] as Params
        return linear(v, q['weight'] as Value, q['bias'] as Value | undefined)
      }
      let q = splitHeads(lin('q_proj', x), attn.numHeads)
      let k = splitHeads(lin('k_proj', x), attn.numKvHeads)
      const v = splitHeads(lin('v_proj', x), attn.numKvHeads)
      if (attn.rope !== false) {
        const angles = rotaryAngles(length, attn.headDim, { base: attn.rope })
        q = applyRotary(q, angles)
        k = applyRotary(k, angles)
      }
      let mask: Value | undefined
      if (attn.causal) {
        const data = new Uint8Array(length * length)
        for (let i = 0; i < length; i++)
          for (let j = 0; j < length; j++)
            data[i * length + j] = j <= i && (attn.window === undefined || i - j < attn.window) ? 1 : 0
        mask = fromStorage(data, [length, length], 'bool')
      }
      const z = attention(q, k, v, {
        enableGqa: true,
        ...(mask === undefined ? {} : { mask }),
        ...(attn.softCap === undefined ? {} : { softCap: attn.softCap }),
      }).output
      const o = p['o_proj'] as Params
      const wo = o['weight'] as Value
      const [b = 1, , l = 1] = z.shape
      const contributions = Array.from({ length: attn.numHeads }, (_, h) => {
        const zh = reshape(sliceAxis(z, 1, h, h + 1), [b, l, attn.headDim])
        return linear(zh, sliceAxis(wo, 1, h * attn.headDim, (h + 1) * attn.headDim))
      })
      return { contributions, bias: o['bias'] as Value | undefined }
    },
  }
}

/**
 * A tensor of token rows as arrays (a convenience for callers that hold tokens as tensors).
 *
 * @param tokens - Rows of ids, or a `[B, T]` tensor.
 * @returns The rows.
 * @example
 * ```ts
 * import { tokenRows } from 'mdml-engine/explain'
 * tokenRows([[1, 2], [3, 4]]) // => [[1, 2], [3, 4]]
 * ```
 */
export function tokenRows(tokens: TokenRows | TensorLike): TokenRows {
  if (Array.isArray(tokens) && tokens.every((r) => Array.isArray(r))) return tokens as TokenRows
  const t = asTensor(tokens as TensorLike)
  const [b = 1, l = 1] = t.shape
  const flat = Array.from(toFlat(t))
  return Array.from({ length: b }, (_, i) => flat.slice(i * l, (i + 1) * l))
}
