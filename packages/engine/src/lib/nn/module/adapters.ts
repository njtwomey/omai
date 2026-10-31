/**
 * Parameter-efficient fine-tuning as module wrappers: LoRA (low-rank updates of chosen weight matrices, with merge and
 * unmerge), bottleneck adapters, prompt tuning (learnt input embeddings prepended to a sequence), prefix tuning for an
 * attention layer, and BitFit (train only the biases), each freezing the base network so that only the small addition
 * is trained, with parameter-count accounting (`parameterCounts`) to show how small it is.
 */

import { MdmlError } from '../../base/index.ts'
import { foldIn, type Key } from '../../random/index.ts'
import {
  add,
  asTensor,
  concat,
  expandDims,
  isValue,
  item,
  matmul,
  mul,
  sizeOf,
  type TensorLike,
  treeFlatten,
  treeMap,
  treePaths,
  type Value,
  zeros,
} from '../../tensor/index.ts'
import { gelu, relu } from '../functional/activations.ts'
import { normalInit, torchDefaultUniform } from '../init.ts'
import { type AttentionInput, type AttentionOutput, MultiheadAttention } from './attention.ts'
import { Linear } from './layers.ts'
import { type ApplyContext, childScope, initKey, Module, type Params } from './module.ts'

type AnyModule = Module<never, unknown>

/** A nested parameter tree entry at a path (undefined when absent). */
function getIn(tree: unknown, path: readonly string[]): unknown {
  let node = tree
  for (const k of path) {
    if (typeof node !== 'object' || node === null) return undefined
    node = (node as Record<string, unknown>)[k]
  }
  return node
}

/** A copy of a parameter tree with the entry at a path replaced. */
function setIn(tree: Params, path: readonly string[], value: unknown): Params {
  const [head, ...rest] = path
  if (head === undefined) return tree
  const child = rest.length === 0 ? value : setIn((tree[head] ?? {}) as Params, rest, value)
  return { ...tree, [head]: child }
}

/**
 * Which weight matrix a LoRA adapter adapts: a name (the key of its factors under `lora`), the path of the weight in
 * the base module's parameters, and, for a weight that stacks several projections (the packed query, key and value of
 * `MultiheadAttention`), the rows `[from, to)` the adapter updates.
 */
export interface LoRATarget {
  readonly name: string
  readonly path: readonly string[]
  readonly rows?: readonly [number, number]
}

/** Options of {@link LoRA}. */
export interface LoRAOptions {
  /** The rank r of the update B A. */
  readonly rank: number
  /** The scaling numerator: the update is multiplied by alpha / r (default r, so a scale of 1). */
  readonly alpha?: number
  /** The weights to adapt (default: a `Linear`'s weight, or the query and value projections of an attention layer). */
  readonly targets?: readonly LoRATarget[]
  /** Scale by alpha / sqrt(r) instead (rank-stabilised LoRA, Kalajdzievski 2023). */
  readonly rankStabilised?: boolean
  readonly key?: Key
}

/**
 * The weights LoRA adapts by default: the weight of a `Linear`; for a `MultiheadAttention` the query and value
 * projections (the choice of the LoRA paper), which `attention` can widen to any of `q`, `k`, `v` and `out`.
 *
 * @param base - The module.
 * @param attention - For an attention layer, which projections.
 * @returns The targets.
 * @throws MdmlError with code `value` for a module this does not know.
 * @example
 * ```ts
 * import { loraTargets, Linear } from 'mdml-engine/nn'
 * loraTargets(new Linear(3, 2)).map((t) => t.name) // => ['weight']
 * ```
 */
export function loraTargets(
  base: AnyModule,
  attention: readonly ('q' | 'k' | 'v' | 'out')[] = ['q', 'v'],
): LoRATarget[] {
  if (base instanceof Linear) return [{ name: 'weight', path: ['weight'] }]
  if (base instanceof MultiheadAttention) {
    const e = base.embedDim
    const parts = { q: 0, k: 1, v: 2 } as const
    return attention.map((which): LoRATarget => {
      if (which === 'out') return { name: 'out', path: ['out_proj', 'weight'] }
      if (base.params['in_proj_weight'] !== undefined) {
        const i = parts[which]
        return { name: which, path: ['in_proj_weight'], rows: [i * e, (i + 1) * e] }
      }
      return { name: which, path: [`${which}_proj_weight`] }
    })
  }
  throw new MdmlError('value', `loraTargets: no default targets for a ${base.kind}; give them`)
}

/**
 * Low-rank adaptation (LoRA, Hu et al. 2021) of a module: each target weight W (out by in) is replaced by
 * W + (alpha / r) B A, with A (r by in) initialised like a Linear layer and B (out by r) at zero, so the adapted
 * module starts as the base. The base is frozen (`requiresGrad(false)`); only the factors under `lora` are trained.
 * The functional form is the base module applied to the merged weights, so `merged` is exactly what the adapted
 * module computes with, and a deployment can drop the adapters by loading it.
 *
 * Dropout on the adapter's input, which the PEFT library offers, is not part of this wrapper.
 *
 * @concept lora
 * @example
 * ```ts
 * import { LoRA, Linear } from 'mdml-engine/nn'
 * import { key } from 'mdml-engine/random'
 * const lora = new LoRA(new Linear(8, 4, { key: key(0) }), { rank: 2, key: key(1) })
 * lora.loraParameterCount() // => 24
 * ```
 */
export class LoRA<I, O> extends Module<I, O> {
  readonly kind = 'LoRA'
  readonly base: Module<I, O>
  readonly rank: number
  readonly alpha: number
  readonly targets: readonly LoRATarget[]
  /** The multiplier alpha / r (alpha / sqrt(r) when rank-stabilised). */
  readonly scaling: number

  constructor(base: Module<I, O>, options: LoRAOptions) {
    super()
    if (!Number.isInteger(options.rank) || options.rank < 1)
      throw new MdmlError('value', `LoRA: rank must be a positive integer, got ${options.rank}`)
    this.base = base
    this.rank = options.rank
    this.alpha = options.alpha ?? options.rank
    this.targets = options.targets ?? loraTargets(base as unknown as AnyModule)
    this.scaling = options.rankStabilised === true ? this.alpha / Math.sqrt(this.rank) : this.alpha / this.rank
    base.requiresGrad(false)
    this.params = { base: base.params, lora: this.initLora(initKey('LoRA', options.key)) }
    this.buffers = { ...(Object.keys(base.buffers).length === 0 ? {} : { base: base.buffers }) }
  }

  private weightShape(base: Params, t: LoRATarget): readonly [number, number] {
    const w = getIn(base, t.path) as Value | undefined
    if (w === undefined || w.shape.length !== 2)
      throw new MdmlError('value', `LoRA: target ${t.name} (${t.path.join('.')}) is not a matrix of the base module`)
    const rows = t.rows === undefined ? w.shape[0] : t.rows[1] - t.rows[0]
    return [rows ?? 0, w.shape[1] ?? 0]
  }

  private initLora(key: Key): Params {
    const out: Record<string, unknown> = {}
    this.targets.forEach((t, i) => {
      const [dOut, dIn] = this.weightShape(this.base.params, t)
      out[t.name] = {
        A: torchDefaultUniform(foldIn(key, i), [this.rank, dIn], dIn),
        B: zeros([dOut, this.rank]),
      }
    })
    return out
  }

  override initParams(key: Key): Params {
    return { base: this.base.initParams(foldIn(key, 0)), lora: this.initLora(foldIn(key, 1)) }
  }

  override initBuffers(): Params {
    const b = this.base.initBuffers()
    return Object.keys(b).length === 0 ? {} : { base: b }
  }

  override children(): readonly (readonly [string, AnyModule])[] {
    return [['base', this.base as unknown as AnyModule]]
  }

  /** The scaled updates (alpha / r) B A of each target, keyed by its name. */
  delta(params: Params = this.params): Record<string, Value> {
    const lora = params['lora'] as Params
    const out: Record<string, Value> = {}
    for (const t of this.targets) {
      const f = lora[t.name] as { A: Value; B: Value }
      out[t.name] = mul(matmul(f.B, f.A), this.scaling)
    }
    return out
  }

  private shifted(base: Params, params: Params, sign: 1 | -1): Params {
    const deltas = this.delta(params)
    let out = base
    for (const t of this.targets) {
      const w = getIn(out, t.path) as Value
      let d = deltas[t.name] as Value
      if (sign === -1) d = mul(d, -1)
      if (t.rows !== undefined) {
        const [from, to] = t.rows
        const total = w.shape[0] ?? 0
        const cols = w.shape[1] ?? 0
        const parts: Value[] = []
        if (from > 0) parts.push(zeros([from, cols]))
        parts.push(d)
        if (to < total) parts.push(zeros([total - to, cols]))
        d = concat(parts, 0)
      }
      out = setIn(out, t.path, add(w, d))
    }
    return out
  }

  /**
   * The base module's parameters with the updates folded in (W + alpha / r B A), the weights of the adapted module as
   * a plain base: `base.apply(lora.merged(), x)` is what the wrapper computes.
   */
  merged(params: Params = this.params): Params {
    return this.shifted(params['base'] as Params, params, 1)
  }

  /** Undo {@link merged}: subtract the updates from merged base parameters, giving the original base weights back. */
  unmerge(mergedBase: Params, params: Params = this.params): Params {
    return this.shifted(mergedBase, params, -1)
  }

  protected override prepare(x: unknown): unknown {
    return (this.base as unknown as { prepare(x: unknown): unknown }).prepare(x)
  }

  apply(params: Params, x: I, ctx: ApplyContext = {}): O {
    const scope = childScope(ctx, this.buffers)
    const y = this.base.apply(this.merged(params), x, scope.child('base', 0))
    scope.done()
    return y
  }

  outputShape(inputShape: readonly number[]): number[] {
    return this.base.outputShape(inputShape)
  }

  /** The number of trainable adapter parameters: the sum of r (d_in + d_out) over the targets. */
  loraParameterCount(): number {
    return this.targets.reduce((n, t) => {
      const [dOut, dIn] = this.weightShape(this.base.params, t)
      return n + loraParameterCount(dIn, dOut, this.rank)
    }, 0)
  }

  override extraRepr(): string {
    return `${this.base.kind}, rank=${this.rank}, alpha=${this.alpha}, targets=[${this.targets.map((t) => t.name).join(', ')}]`
  }
}

/**
 * The parameters of one LoRA adapter on a d_out by d_in weight: r (d_in + d_out).
 *
 * @param dIn - Input width of the weight.
 * @param dOut - Output width.
 * @param rank - The rank r.
 * @returns r (d_in + d_out).
 * @example
 * ```ts
 * import { loraParameterCount } from 'mdml-engine/nn'
 * loraParameterCount(768, 768, 8) // => 12288
 * ```
 */
export function loraParameterCount(dIn: number, dOut: number, rank: number): number {
  return rank * (dIn + dOut)
}

/** A module's parameters counted by whether they are trained. */
export interface ParameterCounts {
  readonly total: number
  readonly trainable: number
  readonly frozen: number
  /** trainable / total. */
  readonly trainableShare: number
}

/**
 * The parameters of a module counted by whether they are trained (the module's own `trainableMask`, so frozen bases,
 * BitFit and any `requiresGrad` are respected): how few a parameter-efficient method actually trains.
 *
 * @param module - The module.
 * @returns Total, trainable and frozen counts and the trainable share.
 * @example
 * ```ts
 * import { parameterCounts, Linear } from 'mdml-engine/nn'
 * parameterCounts(new Linear(3, 2).requiresGrad(false)).trainable // => 0
 * ```
 */
export function parameterCounts(module: AnyModule): ParameterCounts {
  const leaves = treeFlatten(module.params).leaves
  const mask = treeFlatten(module.trainableMask()).leaves
  let total = 0
  let trainable = 0
  leaves.forEach((leaf, i) => {
    const n = sizeOf(leaf.shape)
    total += n
    const m = mask[i]
    if (m !== undefined && item(m) !== 0) trainable += n
  })
  return { total, trainable, frozen: total - trainable, trainableShare: total === 0 ? 0 : trainable / total }
}

/**
 * Train only the parameters a predicate selects and freeze the rest, in place: the module's `trainableMask` becomes
 * the selection, which `fit` and `applyGradients` honour.
 *
 * @param module - The module (changed).
 * @param select - Whether the parameter at a dotted path (`0.weight`) is trained.
 * @returns The module.
 * @example
 * ```ts
 * import { trainOnly, parameterCounts, Linear } from 'mdml-engine/nn'
 * parameterCounts(trainOnly(new Linear(3, 2), (p) => p === 'bias')).trainable // => 2
 * ```
 */
export function trainOnly<M extends AnyModule>(module: M, select: (path: string) => boolean): M {
  const pathMask = (): Params => {
    const paths = Object.keys(treePaths(module.params))
    const flat = treeFlatten(module.params)
    const values = flat.leaves.map((_, i) => asTensor(select(paths[i] ?? '') ? 1 : 0))
    return rebuild(module.params, values)
  }
  module.trainableMask = pathMask
  return module
}

/** A tree of the structure of `tree` whose leaves are the given values in flatten order. */
function rebuild(tree: Params, values: readonly Value[]): Params {
  let i = 0
  return treeMap(() => values[i++] as Value, tree) as Params
}

/**
 * BitFit (Ben Zaken et al. 2022): train only the bias terms, freezing every weight, in place.
 *
 * @param module - The module (changed).
 * @returns The module.
 * @concept bitfit
 * @example
 * ```ts
 * import { bitFit, parameterCounts, mlp } from 'mdml-engine/nn'
 * parameterCounts(bitFit(mlp([4, 8, 2]))).trainable // => 10
 * ```
 */
export function bitFit<M extends AnyModule>(module: M): M {
  return trainOnly(module, (path) => path === 'bias' || path.endsWith('.bias'))
}

/** Options of {@link BottleneckAdapter}. */
export interface BottleneckAdapterOptions {
  readonly activation?: 'gelu' | 'relu'
  /** The standard deviation of the down-projection's initial weights relative to its default; the up-projection starts at zero, so the adapter starts as the identity. */
  readonly key?: Key
}

/**
 * A bottleneck adapter (Houlsby et al. 2019): x + up(act(down(x))), a residual through a narrow hidden layer, with the
 * up-projection at zero so that inserting it changes nothing until it is trained.
 *
 * @concept adapters
 * @example
 * ```ts
 * import { BottleneckAdapter } from 'mdml-engine/nn'
 * new BottleneckAdapter(16, 4).parameterCount() // => 16 * 4 + 4 + 4 * 16 + 16
 * ```
 */
export class BottleneckAdapter extends Module {
  readonly kind = 'BottleneckAdapter'
  readonly width: number
  readonly bottleneck: number
  readonly activation: 'gelu' | 'relu'
  readonly down: Linear
  readonly up: Linear

  constructor(width: number, bottleneck: number, options: BottleneckAdapterOptions = {}) {
    super()
    this.width = width
    this.bottleneck = bottleneck
    this.activation = options.activation ?? 'gelu'
    const key = initKey('BottleneckAdapter', options.key)
    this.down = new Linear(width, bottleneck, { key: foldIn(key, 0) })
    this.up = new Linear(bottleneck, width, { key: foldIn(key, 1) })
    this.up.params = { ...this.up.params, weight: zeros([width, bottleneck]), bias: zeros([width]) }
    this.params = { down: this.down.params, up: this.up.params }
  }

  override initParams(key: Key): Params {
    return {
      down: this.down.initParams(foldIn(key, 0)),
      up: { weight: zeros([this.width, this.bottleneck]), bias: zeros([this.width]) },
    }
  }

  override children(): readonly (readonly [string, Module])[] {
    return [
      ['down', this.down],
      ['up', this.up],
    ]
  }

  apply(params: Params, x: Value): Value {
    const h = this.down.apply(params['down'] as Params, x)
    const a = this.activation === 'gelu' ? gelu(h) : relu(h)
    return add(x, this.up.apply(params['up'] as Params, a))
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape]
  }

  override extraRepr(): string {
    return `width=${this.width}, bottleneck=${this.bottleneck}, activation=${this.activation}`
  }
}

/**
 * A module followed by an adapter on its output (the adapter placed after a sub-layer, as in Houlsby's transformer),
 * with the base frozen so that only the adapter trains. Parameters are `base` and `adapter`.
 *
 * @param base - The module to adapt (frozen).
 * @param adapter - The module applied to its output (default a {@link BottleneckAdapter} of the given width).
 * @returns The adapted module.
 * @concept adapters
 * @example
 * ```ts
 * import { withAdapter, parameterCounts, BottleneckAdapter, Linear } from 'mdml-engine/nn'
 * parameterCounts(withAdapter(new Linear(4, 16), new BottleneckAdapter(16, 2))).trainable // => 82
 * ```
 */
export function withAdapter(base: Module, adapter: Module): Module {
  return new Adapted(base, adapter)
}

class Adapted extends Module {
  readonly kind = 'Adapted'
  readonly base: Module
  readonly adapter: Module

  constructor(base: Module, adapter: Module) {
    super()
    this.base = base.requiresGrad(false)
    this.adapter = adapter
    this.params = { base: base.params, adapter: adapter.params }
    this.buffers = { ...(Object.keys(base.buffers).length === 0 ? {} : { base: base.buffers }) }
  }

  override initParams(key: Key): Params {
    return { base: this.base.initParams(foldIn(key, 0)), adapter: this.adapter.initParams(foldIn(key, 1)) }
  }

  override children(): readonly (readonly [string, Module])[] {
    return [
      ['base', this.base],
      ['adapter', this.adapter],
    ]
  }

  apply(params: Params, x: Value, ctx: ApplyContext = {}): Value {
    const scope = childScope(ctx, this.buffers)
    const h = this.base.apply(params['base'] as Params, x, scope.child('base', 0))
    const y = this.adapter.apply(params['adapter'] as Params, h, scope.child('adapter', 1))
    scope.done()
    return y
  }

  outputShape(inputShape: readonly number[]): number[] {
    return this.adapter.outputShape(this.base.outputShape(inputShape))
  }
}

/**
 * Prompt tuning (Lester et al. 2021): a frozen module on sequences whose input has a learnt matrix of `nTokens`
 * embeddings prepended along the sequence axis (axis -2 of `[..., L, E]`); only the prompt trains. The output is the
 * base module's on the longer sequence.
 *
 * @concept prompt-tuning
 * @example
 * ```ts
 * import { PromptTuning, Sequential } from 'mdml-engine/nn'
 * import { zeros } from 'mdml-engine/tensor'
 * import { key } from 'mdml-engine/random'
 * new PromptTuning(new Sequential(), 3, 4, { key: key(0) }).forward(zeros([2, 5, 4])).shape // => [2, 8, 4]
 * ```
 */
export class PromptTuning extends Module {
  readonly kind = 'PromptTuning'
  readonly base: Module
  readonly nTokens: number
  readonly dim: number

  constructor(base: Module, nTokens: number, dim: number, options: { readonly key?: Key; readonly std?: number } = {}) {
    super()
    this.base = base.requiresGrad(false)
    this.nTokens = nTokens
    this.dim = dim
    this.params = {
      base: base.params,
      prompt: normalInit(foldIn(initKey('PromptTuning', options.key), 0), [nTokens, dim], 0, options.std ?? 0.5),
    }
    this.buffers = { ...(Object.keys(base.buffers).length === 0 ? {} : { base: base.buffers }) }
  }

  override initParams(key: Key): Params {
    return { base: this.base.initParams(foldIn(key, 0)), prompt: normalInit(foldIn(key, 1), [this.nTokens, this.dim]) }
  }

  override children(): readonly (readonly [string, Module])[] {
    return [['base', this.base]]
  }

  apply(params: Params, x: Value, ctx: ApplyContext = {}): Value {
    const prompt = params['prompt'] as Value
    const rank = x.shape.length
    const lead = x.shape.slice(0, rank - 2)
    // Broadcast the prompt over the leading (batch) axes by adding it to zeros of the right shape.
    const tiled = add(zeros([...lead, this.nTokens, this.dim]), prompt)
    const scope = childScope(ctx, this.buffers)
    const y = this.base.apply(params['base'] as Params, concat([tiled, x], rank - 2), scope.child('base', 0))
    scope.done()
    return y
  }

  outputShape(inputShape: readonly number[]): number[] {
    const s = [...inputShape]
    s[s.length - 2] = (s[s.length - 2] ?? 0) + this.nTokens
    return this.base.outputShape(s)
  }
}

/**
 * Prefix tuning of one attention layer (Li and Liang 2021): `length` learnt key rows and value rows are prepended to
 * the layer's keys and values so that every query can attend to them, and only they train (the attention layer is
 * frozen). The prefix lives in the layer's input space and passes through its key and value projections, the
 * reparametrisation the paper used for stability; `attnMask` is not supported (padding masks and `isCausal` are, with
 * the prefix always visible).
 *
 * @concept prefix-tuning
 * @example
 * ```ts
 * import { PrefixAttention, MultiheadAttention } from 'mdml-engine/nn'
 * import { zeros } from 'mdml-engine/tensor'
 * import { key } from 'mdml-engine/random'
 * const attn = new MultiheadAttention(8, 2, { batchFirst: true, key: key(0) })
 * new PrefixAttention(attn, 3, { key: key(1) }).forward(zeros([2, 5, 8])).weights?.shape // => [2, 5, 8]
 * ```
 */
export class PrefixAttention extends Module<Value | AttentionInput, AttentionOutput> {
  readonly kind = 'PrefixAttention'
  readonly base: MultiheadAttention
  readonly length: number

  constructor(base: MultiheadAttention, length: number, options: { readonly key?: Key; readonly std?: number } = {}) {
    super()
    this.base = base.requiresGrad(false)
    this.length = length
    const key = initKey('PrefixAttention', options.key)
    const std = options.std ?? 0.5
    this.params = {
      base: base.params,
      prefix_key: normalInit(foldIn(key, 0), [length, base.embedDim], 0, std),
      prefix_value: normalInit(foldIn(key, 1), [length, base.embedDim], 0, std),
    }
  }

  override initParams(key: Key): Params {
    const e = this.base.embedDim
    return {
      base: this.base.initParams(foldIn(key, 0)),
      prefix_key: normalInit(foldIn(key, 1), [this.length, e]),
      prefix_value: normalInit(foldIn(key, 2), [this.length, e]),
    }
  }

  override children(): readonly (readonly [string, AnyModule])[] {
    return [['base', this.base as unknown as AnyModule]]
  }

  protected override prepare(x: unknown): unknown {
    return (this.base as unknown as { prepare(x: unknown): unknown }).prepare(x)
  }

  apply(params: Params, x: Value | AttentionInput, ctx: ApplyContext = {}): AttentionOutput {
    const input: AttentionInput = isValue(x) ? { query: x } : x
    if (input.attnMask !== undefined) throw new MdmlError('value', 'PrefixAttention: attnMask is not supported')
    const q = asTensor(input.query as TensorLike)
    const k = input.key === undefined ? q : asTensor(input.key)
    const v = input.value === undefined ? k : asTensor(input.value)
    const unbatched = q.shape.length === 2
    const batchFirst = this.base.batchFirst
    const prefix = (p: Value, like: Value): Value => {
      if (unbatched) return p
      const [a = 0, b = 0, e = 0] = like.shape
      // [P, E] to [B, P, E] (batch first) or [P, B, E].
      return batchFirst ? add(zeros([a, this.length, e]), p) : add(zeros([this.length, b, e]), expandDims(p, 1))
    }
    const axis = unbatched || !batchFirst ? 0 : 1
    const extend = (t: Value, p: Value): Value => concat([prefix(p, t), t], axis)
    const l = q.shape[unbatched ? 0 : batchFirst ? 1 : 0] ?? 0
    const s = k.shape[unbatched ? 0 : batchFirst ? 1 : 0] ?? 0
    // True = not allowed: the prefix is always visible, the sequence is causal.
    const attnMask =
      input.isCausal === true
        ? asTensor(
            Array.from({ length: l }, (_, i) =>
              Array.from({ length: this.length + s }, (_, j) => (j >= this.length && j - this.length > i ? 1 : 0)),
            ),
            'bool',
          )
        : undefined
    let padding: TensorLike | undefined
    if (input.keyPaddingMask !== undefined) {
      const m = asTensor(input.keyPaddingMask)
      padding = concat([zeros([...m.shape.slice(0, -1), this.length], { dtype: 'bool' }), m], m.shape.length - 1)
    }
    const { isCausal: _c, ...rest } = input
    return this.base.apply(
      params['base'] as Params,
      {
        ...rest,
        query: q,
        key: extend(k, params['prefix_key'] as Value),
        value: extend(v, params['prefix_value'] as Value),
        ...(attnMask === undefined ? {} : { attnMask }),
        ...(padding === undefined ? {} : { keyPaddingMask: padding }),
      },
      ctx,
    )
  }

  outputShape(inputShape: readonly number[]): number[] {
    return this.base.outputShape(inputShape)
  }
}
