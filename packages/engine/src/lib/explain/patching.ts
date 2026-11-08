/**
 * Finding the circuit that computes an answer: activation patching copies one component's activation from a clean run
 * into a corrupted one (or the reverse) and measures how much of the behaviour moves with it, causal tracing
 * (Meng et al. 2022) is the same experiment with noise on a subject's embeddings as the corruption, path patching
 * (Wang et al. 2022, Goldowsky-Dill et al. 2023) lets a sender's change reach the output only through chosen
 * receivers, attribution patching (Nanda 2023) replaces the many patched runs by one gradient, and direct logit
 * attribution splits the logit difference exactly into what each component writes to the stream. All of them work
 * on a {@link residualModel} and a scalar metric of the logits, such as the logit difference between two answers.
 */
import { grad } from '../autodiff/index.ts'
import { OmaiError } from '../base/index.ts'
import { type Key, normal } from '../random/index.ts'
import {
  add,
  asTensor,
  concat,
  fromStorage,
  logSoftmax,
  mul,
  reshape,
  sliceAxis,
  softmax,
  sub,
  sum,
  type Tensor,
  toFlat,
  type Value,
  zeros,
} from '../tensor/index.ts'
import { headSite, type ResidualModel, type SiteHook } from './residual.ts'

/** A scalar of a run's logits `[B, T, V]` whose change a patch is measured by. */
export type PatchMetric = (logits: Value) => Value

function at(logits: Value, position: number): Value {
  const t = logits.shape[1] ?? 1
  const p = position < 0 ? t + position : position
  return reshape(sliceAxis(logits, 1, p, p + 1), [logits.shape[0] ?? 1, logits.shape[2] ?? 1])
}

/**
 * The logit difference between a correct and a wrong next token at one position, the usual metric of circuit
 * analysis (batch size one).
 *
 * @param correct - The token the model should favour.
 * @param wrong - The token it should favour less.
 * @param position - The position (default the last; negative counts from the end).
 * @returns The metric.
 * @example
 * ```ts
 * import { logitDifference } from 'omai-engine/explain'
 * import { tensor, toFlat } from 'omai-engine/tensor'
 * toFlat(logitDifference(0, 1)(tensor([[[2, 0.5]]]))) // => [1.5]
 * ```
 */
export function logitDifference(correct: number, wrong: number, position = -1): PatchMetric {
  return (logits) => {
    const row = at(logits, position)
    return sub(sliceAxis(row, 1, correct, correct + 1), sliceAxis(row, 1, wrong, wrong + 1))
  }
}

/**
 * The log probability of a token at one position (batch size one).
 *
 * @param token - The token.
 * @param position - The position (default the last).
 * @returns The metric.
 * @example
 * ```ts
 * import { tokenLogProbability } from 'omai-engine/explain'
 * import { tensor, toFlat } from 'omai-engine/tensor'
 * toFlat(tokenLogProbability(0)(tensor([[[0, 0]]]))) // => [-0.6931471805599453]
 * ```
 */
export function tokenLogProbability(token: number, position = -1): PatchMetric {
  return (logits) => sliceAxis(logSoftmax(at(logits, position), -1), 1, token, token + 1)
}

/**
 * The probability of a token at one position (batch size one), the metric of causal tracing.
 *
 * @param token - The token.
 * @param position - The position (default the last).
 * @returns The metric.
 * @example
 * ```ts
 * import { tokenProbability } from 'omai-engine/explain'
 * import { tensor, toFlat } from 'omai-engine/tensor'
 * toFlat(tokenProbability(1)(tensor([[[0, 0]]]))) // => [0.5]
 * ```
 */
export function tokenProbability(token: number, position = -1): PatchMetric {
  return (logits) => sliceAxis(softmax(at(logits, position), -1), 1, token, token + 1)
}

const scalarOf = (v: Value): number => toFlat(asTensor(v))[0] as number

/** A corrupted input: its tokens, and a hook that perturbs the run further (noise on an embedding). */
export interface Corrupted {
  readonly tokens: readonly number[]
  readonly perturb?: SiteHook
}

const corruptedOf = (c: readonly number[] | Corrupted): Corrupted =>
  Array.isArray(c) ? { tokens: c as number[] } : (c as Corrupted)

/** The sites a patching study visits by default: the embeddings, and each block's attention, feed-forward and output. */
export function defaultSites(model: ResidualModel, heads = false): string[] {
  const out = ['embed']
  for (let l = 0; l < model.numLayers; l++) {
    if (heads) for (let h = 0; h < model.numHeads; h++) out.push(headSite(l, h))
    out.push(`layer${l}.attn`, `layer${l}.mlp`, `layer${l}.out`)
  }
  return out
}

/** `v` with row `p` (the position axis of `[B, T, D]`) taken from `donor`. */
function replaceAt(v: Value, donor: Value, p: number | undefined): Value {
  if (p === undefined) return donor
  const t = v.shape[1] ?? 1
  const parts = [
    ...(p > 0 ? [sliceAxis(v, 1, 0, p)] : []),
    sliceAxis(donor, 1, p, p + 1),
    ...(p < t - 1 ? [sliceAxis(v, 1, p + 1, t)] : []),
  ]
  return concat(parts, 1)
}

/** Options of {@link activationPatching}. */
export interface PatchingOptions {
  /** The scalar to measure. */
  readonly metric: PatchMetric
  /** The sites to patch (default {@link defaultSites}). */
  readonly sites?: readonly string[]
  /** Patch the heads individually when the default sites are used. */
  readonly heads?: boolean
  /** The positions to patch one at a time (default every position); `'all'` patches the whole sequence at once. */
  readonly positions?: readonly number[] | 'all'
  /**
   * `denoise` runs the corrupted prompt and restores the clean activation (is this site sufficient?, the default);
   * `noise` runs the clean prompt and plants the corrupted activation (is it necessary?).
   */
  readonly direction?: 'denoise' | 'noise'
}

/** The map of patching effects over sites and positions. */
export interface PatchingResult {
  readonly sites: readonly string[]
  /** The patched positions; `-1` for the whole sequence at once. */
  readonly positions: readonly number[]
  /** The metric on the clean run. */
  readonly clean: number
  /** The metric on the corrupted run. */
  readonly corrupted: number
  /** The metric after each patch `[sites, positions]`. */
  readonly patched: Tensor
  /**
   * The share of the clean-corrupted gap recovered `[sites, positions]`: for `denoise`, `(patched - corrupted) /
   * (clean - corrupted)`; for `noise`, `(clean - patched) / (clean - corrupted)`, the share destroyed.
   */
  readonly recovered: Tensor
}

/**
 * Activation patching: for every site and position, run one prompt with that single activation taken from the other
 * and see how much of the metric's clean-to-corrupted gap moves. A component that restores the whole gap when
 * patched alone carries the information; one whose removal destroys it is needed. Patching a site with its own clean
 * activation changes nothing, so a clean run is a fixed point.
 *
 * @param model - The opened model.
 * @param clean - The clean prompt.
 * @param corrupted - The corrupted prompt (same length), or tokens with a perturbation of the run (causal tracing).
 * @param options - The metric, sites, positions and direction.
 * @returns The clean and corrupted metric and the map of patched metrics and recovered shares.
 * @throws OmaiError with code `shape` if the prompts differ in length.
 * @concept mechanistic-interpretability
 * @example
 * ```ts
 * import { activationPatching, logitDifference, residualModel } from 'omai-engine/explain'
 * import { blocks } from 'omai-engine/nn'
 * import { key } from 'omai-engine/random'
 * const m = residualModel(new blocks.TransformerLM(6, 8, 1, 2, { key: key(0) }))
 * const r = activationPatching(m, [0, 3, 3], [1, 3, 3], { metric: logitDifference(0, 1), sites: ['embed'] })
 * r.sites // => ['embed']
 * ```
 */
export function activationPatching(
  model: ResidualModel,
  clean: readonly number[],
  corrupted: readonly number[] | Corrupted,
  options: PatchingOptions,
): PatchingResult {
  const c = corruptedOf(corrupted)
  if (c.tokens.length !== clean.length)
    throw new OmaiError('shape', 'activationPatching: the prompts need equal length')
  const direction = options.direction ?? 'denoise'
  const sites = options.sites ?? defaultSites(model, options.heads ?? false)
  const positions =
    options.positions === 'all' ? [-1] : (options.positions ?? Array.from({ length: clean.length }, (_, i) => i))
  const cleanRun = model.run([clean])
  const corruptRun = model.run([c.tokens], c.perturb)
  const cleanMetric = scalarOf(options.metric(cleanRun.logits))
  const corruptMetric = scalarOf(options.metric(corruptRun.logits))
  const donor = direction === 'denoise' ? cleanRun : corruptRun
  const gap = cleanMetric - corruptMetric
  const patched = new Float64Array(sites.length * positions.length)
  const recovered = new Float64Array(sites.length * positions.length)
  sites.forEach((site, si) => {
    positions.forEach((p, pi) => {
      const hook: SiteHook = (name, v) => {
        if (name !== site) return direction === 'denoise' ? c.perturb?.(name, v) : undefined
        const d = donor.sites.get(site)
        if (d === undefined) throw new OmaiError('value', `activationPatching: unknown site '${site}'`)
        return replaceAt(v, d, p < 0 ? undefined : p)
      }
      const run = direction === 'denoise' ? model.run([c.tokens], hook) : model.run([clean], hook)
      const m = scalarOf(options.metric(run.logits))
      patched[si * positions.length + pi] = m
      const share = gap === 0 ? 0 : direction === 'denoise' ? (m - corruptMetric) / gap : (cleanMetric - m) / gap
      recovered[si * positions.length + pi] = share
    })
  })
  const shape = [sites.length, positions.length]
  return {
    sites,
    positions,
    clean: cleanMetric,
    corrupted: corruptMetric,
    patched: fromStorage(patched, shape, 'float64'),
    recovered: fromStorage(recovered, shape, 'float64'),
  }
}

/** Options of {@link causalTrace}. */
export interface CausalTraceOptions {
  /** The positions of the subject tokens, whose embeddings are corrupted. */
  readonly subject: readonly number[]
  /** The metric to restore (usually {@link tokenProbability} of the answer). */
  readonly metric: PatchMetric
  /** The key of the noise. */
  readonly key: Key
  /** The noise's standard deviation, in units of the embeddings' (default 3 times their standard deviation). */
  readonly noise?: number
  /** Which sites to restore, by kind: the stream (`out`), the feed-forward (`mlp`) and attention (`attn`) outputs. */
  readonly kinds?: readonly ('out' | 'mlp' | 'attn')[]
}

/** The result of {@link causalTrace}: one map `[layers, positions]` of recovered shares per kind of site. */
export interface CausalTrace {
  readonly clean: number
  readonly corrupted: number
  readonly effects: Readonly<Record<string, Tensor>>
  /** The recovered share by restoring the subject's embeddings alone, per position of the subject. */
  readonly embedding: Tensor
}

/**
 * Causal tracing (ROME): corrupt the subject's embeddings with Gaussian noise, then restore one clean activation at a
 * time (the stream, the feed-forward or the attention output at one layer and position) and record how much of the
 * metric returns. The result is a map over layers and positions for each kind of site.
 *
 * @param model - The opened model.
 * @param tokens - The prompt.
 * @param options - Subject, metric, noise and kinds.
 * @returns The maps of recovered shares.
 * @concept mechanistic-interpretability
 * @example
 * ```ts
 * import { causalTrace, residualModel, tokenProbability } from 'omai-engine/explain'
 * import { blocks } from 'omai-engine/nn'
 * import { key } from 'omai-engine/random'
 * const m = residualModel(new blocks.TransformerLM(6, 8, 2, 2, { key: key(0) }))
 * const r = causalTrace(m, [0, 1, 2, 3], { subject: [1], metric: tokenProbability(4), key: key(1), kinds: ['out'] })
 * r.effects['out']?.shape // => [2, 4]
 * ```
 */
export function causalTrace(model: ResidualModel, tokens: readonly number[], options: CausalTraceOptions): CausalTrace {
  const run = model.run([tokens])
  const embed = asTensor(run.sites.get('embed') as Value)
  const flat = toFlat(embed)
  const mean = flat.reduce((a, b) => a + b, 0) / flat.length
  const sd = Math.sqrt(flat.reduce((a, b) => a + (b - mean) ** 2, 0) / flat.length)
  const sigma = options.noise ?? 3 * sd
  const eps = normal(options.key, [1, tokens.length, model.dim]) as Tensor
  const mask = fromStorage(
    Float64Array.from({ length: tokens.length }, (_, p) => (options.subject.includes(p) ? sigma : 0)),
    [1, tokens.length, 1],
    'float64',
  )
  const perturb: SiteHook = (site, v) => (site === 'embed' ? add(v, mul(eps, mask)) : undefined)
  const kinds = options.kinds ?? ['out', 'mlp', 'attn']
  const corrupted: Corrupted = { tokens, perturb }
  const effects: Record<string, Tensor> = {}
  let clean = 0
  let corruptedMetric = 0
  for (const kind of kinds) {
    const sites = Array.from({ length: model.numLayers }, (_, l) => `layer${l}.${kind}`)
    const r = activationPatching(model, tokens, corrupted, { metric: options.metric, sites })
    effects[kind] = r.recovered
    clean = r.clean
    corruptedMetric = r.corrupted
  }
  const emb = activationPatching(model, tokens, corrupted, { metric: options.metric, sites: ['embed'] })
  return { clean, corrupted: corruptedMetric, effects, embedding: reshape(emb.recovered, [tokens.length]) as Tensor }
}

/** Options of {@link attributionPatching}. */
export interface AttributionPatchingOptions {
  readonly metric: PatchMetric
  readonly sites?: readonly string[]
  readonly heads?: boolean
  /** `denoise` (default) approximates patching clean into corrupted; `noise` the reverse. */
  readonly direction?: 'denoise' | 'noise'
}

/** The linear approximation of every patch at once. */
export interface AttributionPatching {
  readonly sites: readonly string[]
  /** The approximate change in the metric from patching each site at each position `[sites, positions]`. */
  readonly attribution: Tensor
  readonly clean: number
  readonly corrupted: number
}

/**
 * Attribution patching: the first-order estimate of every single-site patch from one forward and backward pass. On the
 * corrupted run the effect of patching a site at a position is approximately `(a_clean - a_corrupted) . dm/da`, summed
 * over the activation's features; one backward pass gives the gradient at every site. Fast, and wrong exactly where
 * the metric is nonlinear in the activation (attention patterns saturate), so it ranks sites well and sizes them
 * loosely.
 *
 * @param model - The opened model.
 * @param clean - The clean prompt.
 * @param corrupted - The corrupted prompt (same length).
 * @param options - The metric, sites and direction.
 * @returns The estimated changes `[sites, positions]`.
 * @concept mechanistic-interpretability
 * @example
 * ```ts
 * import { attributionPatching, logitDifference, residualModel } from 'omai-engine/explain'
 * import { blocks } from 'omai-engine/nn'
 * import { key } from 'omai-engine/random'
 * const m = residualModel(new blocks.TransformerLM(6, 8, 1, 2, { key: key(0) }))
 * attributionPatching(m, [0, 3, 3], [1, 3, 3], { metric: logitDifference(0, 1), sites: ['embed'] }).attribution.shape // => [1, 3]
 * ```
 */
export function attributionPatching(
  model: ResidualModel,
  clean: readonly number[],
  corrupted: readonly number[],
  options: AttributionPatchingOptions,
): AttributionPatching {
  if (corrupted.length !== clean.length)
    throw new OmaiError('shape', 'attributionPatching: the prompts need equal length')
  const direction = options.direction ?? 'denoise'
  const sites = options.sites ?? defaultSites(model, options.heads ?? false)
  const cleanRun = model.run([clean])
  const corruptRun = model.run([corrupted])
  const base = direction === 'denoise' ? corrupted : clean
  const other = direction === 'denoise' ? cleanRun : corruptRun
  const own = direction === 'denoise' ? corruptRun : cleanRun
  const t = clean.length
  const zeroDeltas: Record<string, Value> = {}
  for (const s of sites) zeroDeltas[s] = zeros([1, t, model.dim]) as Value
  const metricOf = (deltas: Record<string, Value>): Value =>
    sum(
      options.metric(
        model.run([base], (site, v) => (deltas[site] === undefined ? undefined : add(v, deltas[site]))).logits,
      ),
    )
  const g = grad<Record<string, Value>>(metricOf)(zeroDeltas)
  const out = new Float64Array(sites.length * t)
  sites.forEach((s, si) => {
    const diff = toFlat(asTensor(sub(other.sites.get(s) as Value, own.sites.get(s) as Value)))
    const gradient = toFlat(asTensor(g[s] as Value))
    const d = model.dim
    for (let p = 0; p < t; p++) {
      let acc = 0
      for (let k = 0; k < d; k++) acc += (diff[p * d + k] as number) * (gradient[p * d + k] as number)
      out[si * t + p] = acc
    }
  })
  return {
    sites,
    attribution: fromStorage(out, [sites.length, t], 'float64'),
    clean: scalarOf(options.metric(cleanRun.logits)),
    corrupted: scalarOf(options.metric(corruptRun.logits)),
  }
}

/** One sender's path-patching result. */
export interface PathPatchingResult {
  readonly sender: string
  readonly receivers: readonly string[] | 'logits'
  /** The metric on the clean run. */
  readonly clean: number
  /** The metric on the corrupted run. */
  readonly corrupted: number
  /** The metric when the sender's corrupted activation reaches only the receivers. */
  readonly patched: number
  /** `(patched - clean) / (corrupted - clean)`: the share of the corruption's effect carried by those paths. */
  readonly effect: number
}

/** Options of {@link pathPatching}. */
export interface PathPatchingOptions {
  readonly metric: PatchMetric
  /** The components that read the sender (their sites); `'logits'` (default) is the direct path to the output. */
  readonly receivers?: readonly string[] | 'logits'
}

/** The sites that write into the stream and are frozen to their clean value in a path patch. */
const isComponent = (site: string): boolean => site === 'embed' || /\.(head\d+|mlp)$/.test(site)

/**
 * Path patching: run the clean prompt, change one sender component to its corrupted activation, and let that change
 * reach the metric only through the chosen receivers. Every other component's output is held at its clean value, so
 * the effect measured is that of the paths sender -> receivers -> output (receivers `'logits'` is the direct path
 * through the residual stream). Where activation patching says which components matter, path patching says which
 * connections between them do.
 *
 * @param model - The opened model.
 * @param clean - The clean prompt.
 * @param corrupted - The corrupted prompt (same length).
 * @param senders - The sender sites (heads, feed-forwards or `embed`).
 * @param options - The metric and the receivers.
 * @returns One result per sender.
 * @concept mechanistic-interpretability
 * @example
 * ```ts
 * import { logitDifference, pathPatching, residualModel } from 'omai-engine/explain'
 * import { blocks } from 'omai-engine/nn'
 * import { key } from 'omai-engine/random'
 * const m = residualModel(new blocks.TransformerLM(6, 8, 1, 2, { key: key(0) }))
 * pathPatching(m, [0, 3, 3], [1, 3, 3], ['embed'], { metric: logitDifference(0, 1) })[0]?.sender // => 'embed'
 * ```
 */
export function pathPatching(
  model: ResidualModel,
  clean: readonly number[],
  corrupted: readonly number[],
  senders: readonly string[],
  options: PathPatchingOptions,
): PathPatchingResult[] {
  if (corrupted.length !== clean.length) throw new OmaiError('shape', 'pathPatching: the prompts need equal length')
  const receivers = options.receivers ?? 'logits'
  const cleanRun = model.run([clean])
  const corruptRun = model.run([corrupted])
  const cleanMetric = scalarOf(options.metric(cleanRun.logits))
  const corruptMetric = scalarOf(options.metric(corruptRun.logits))
  return senders.map((sender) => {
    const run = model.run([clean], (site, v) => {
      if (site === sender) return corruptRun.sites.get(site)
      if (isComponent(site) && (receivers === 'logits' || !receivers.includes(site)))
        return cleanRun.sites.get(site) ?? v
      return undefined
    })
    const patched = scalarOf(options.metric(run.logits))
    const gap = corruptMetric - cleanMetric
    return {
      sender,
      receivers,
      clean: cleanMetric,
      corrupted: corruptMetric,
      patched,
      effect: gap === 0 ? 0 : (patched - cleanMetric) / gap,
    }
  })
}

/** The exact split of a logit difference over the components that write to the stream. */
export interface LogitAttribution {
  /** The components: `embed`, every head, every feed-forward. */
  readonly components: readonly string[]
  /** What each writes to the logit difference. */
  readonly contribution: readonly number[]
  /** The final normalisation's bias, a constant (zero without one). */
  readonly bias: number
  /** The sum of the contributions and the bias. */
  readonly total: number
  /** The model's own logit difference; equal to `total`. */
  readonly logitDifference: number
}

/**
 * Direct logit attribution: the logit difference between two tokens at a position is, up to the final normalisation's
 * scale (frozen at its actual value), linear in the residual stream, which is a sum of what the embeddings, every
 * head and every feed-forward wrote there. Each component's share is its output, normalised, dotted with the
 * difference of the two tokens' unembedding vectors; the shares sum to the logit difference exactly. It sees only the
 * direct path to the output, so a head that matters by feeding a later one reads as small (path patching sees it).
 *
 * @param model - The opened model.
 * @param tokens - The prompt.
 * @param correct - The token whose logit is raised.
 * @param wrong - The token it is compared with.
 * @param position - The position (default the last).
 * @returns The contribution of every component and the totals.
 * @concept mechanistic-interpretability
 * @example
 * ```ts
 * import { directLogitAttribution, residualModel } from 'omai-engine/explain'
 * import { blocks } from 'omai-engine/nn'
 * import { key } from 'omai-engine/random'
 * const m = residualModel(new blocks.TransformerLM(6, 8, 1, 2, { key: key(0) }))
 * const a = directLogitAttribution(m, [0, 3, 3], 0, 1)
 * Math.abs(a.total - a.logitDifference) < 1e-9 // => true
 * ```
 */
export function directLogitAttribution(
  model: ResidualModel,
  tokens: readonly number[],
  correct: number,
  wrong: number,
  position = -1,
): LogitAttribution {
  const run = model.run([tokens])
  const t = tokens.length
  const p = position < 0 ? t + position : position
  const final = run.residual[run.residual.length - 1] as Value
  const row = (v: Value): Value => reshape(sliceAxis(v, 1, p, p + 1), [1, model.dim])
  const direction = sub(
    sliceAxis(model.unembedding, 0, correct, correct + 1),
    sliceAxis(model.unembedding, 0, wrong, wrong + 1),
  )
  const dir = reshape(direction, [model.dim, 1])
  const components = model.sites.filter((s) => s === 'embed' || /\.(head\d+|mlp)$/.test(s))
  const reference = row(final)
  const contribution = components.map((s) =>
    scalarOf(sum(mul(model.finalNorm(row(run.sites.get(s) as Value), reference), reshape(dir, [1, model.dim])))),
  )
  const bias =
    model.finalNormBias === undefined
      ? 0
      : scalarOf(sum(mul(reshape(model.finalNormBias, [1, model.dim]), reshape(dir, [1, model.dim]))))
  const total = contribution.reduce((a, b) => a + b, 0) + bias
  return {
    components,
    contribution,
    bias,
    total,
    logitDifference: scalarOf(logitDifference(correct, wrong, p)(run.logits)),
  }
}
