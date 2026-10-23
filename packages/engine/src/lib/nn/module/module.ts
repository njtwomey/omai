import { WhiteboxError } from '../../base/index.ts'
import { type Context, defaultContext } from '../../context/index.ts'
import { foldIn, type Key } from '../../random/index.ts'
import {
  asTensor,
  isTensor,
  isTraced,
  primalOf,
  type Tensor,
  type TensorLike,
  treeCount,
  treeFlatten,
  treeMap,
  treePaths,
  treeUnflatten,
  type Value,
} from '../../tensor/index.ts'
import type { TapContext } from '../../trace/index.ts'

/** A module's parameters: a tree of tensors (nested records keyed like PyTorch's state-dict paths). */
export type Params = Readonly<Record<string, unknown>>

/**
 * A parameter: one trainable leaf of a {@link Params} tree. PyTorch wraps parameters in a class; here a parameter is
 * a plain value in the tree, and freezing (PyTorch's `requires_grad = False`) is a property of the module
 * ({@link Module.requiresGrad}).
 */
export type Parameter = Value

/**
 * What a forward pass may need besides its input, all passed and returned as data: the training flag, a key for
 * stochastic layers, the buffers to read (running statistics), a sink for updated buffers, a tap context for
 * activations and the module's path (for tap names).
 */
export interface ApplyContext {
  /** Training mode (default true, as a freshly built PyTorch module). */
  readonly train?: boolean
  /** The key for dropout and other stochastic layers. */
  readonly key?: Key
  /** The module's buffers (default: the buffers it holds). */
  readonly buffers?: Params
  /** Receives the module's updated buffers when a forward pass changes them (batch statistics in training). */
  readonly onBuffers?: (buffers: Params) => void
  /** Collects activations that layers tap (attention weights, hidden states). */
  readonly taps?: TapContext
  /** The module's path inside its parents (`encoder.layers.0`), the prefix of its tap names. */
  readonly path?: string
}

// Default initialisation keys are numbered per context, so a program that builds its modules in the same order from
// the same seed gets the same weights (as with torch.manual_seed), and replacing the default context starts afresh.
const counters = new WeakMap<Context, number>()

/**
 * The key a module uses for initialisation: the given one, or the next key of the default context.
 *
 * @param kind - The module type (part of the derived key's name).
 * @param key - An explicit key, returned as is.
 * @returns The key.
 * @example
 * ```ts
 * import { initKey } from 'whitebox-engine/nn'
 * import { key } from 'whitebox-engine/random'
 * initKey('Linear', key(3)).words[0] === key(3).words[0] // => true
 * ```
 */
export function initKey(kind: string, key?: Key): Key {
  if (key !== undefined) return key
  const ctx = defaultContext()
  const n = counters.get(ctx) ?? 0
  counters.set(ctx, n + 1)
  return ctx.key('nn', kind, n)
}

/**
 * Whether a forward pass runs in training mode (the `train` flag, true when absent).
 *
 * @param ctx - The apply context.
 * @returns The flag.
 * @example
 * ```ts
 * import { isTraining } from 'whitebox-engine/nn'
 * isTraining({}) // => true
 * ```
 */
export function isTraining(ctx: ApplyContext = {}): boolean {
  return ctx.train ?? true
}

/**
 * Record an activation under the module's path (`<path>.<name>`) when the context carries taps; returns the value.
 *
 * @param ctx - The apply context.
 * @param name - The activation's name within the module.
 * @param value - The value.
 * @returns `value`, unchanged.
 * @example
 * ```ts
 * import { tap } from 'whitebox-engine/nn'
 * import { tapContext } from 'whitebox-engine/trace'
 * import { tensor } from 'whitebox-engine/tensor'
 * const taps = tapContext()
 * tap({ taps, path: 'attn' }, 'weights', tensor([1]))
 * [...taps.entries.keys()] // => ['attn.weights']
 * ```
 */
export function tap(ctx: ApplyContext, name: string, value: Value): Value {
  if (ctx.taps === undefined) return value
  // Inside a differentiated forward pass the value is traced: record its concrete primal when it has one.
  let concrete: Value | undefined = value
  while (concrete !== undefined && isTraced(concrete)) concrete = (concrete as { primal?: Value }).primal
  if (concrete !== undefined)
    ctx.taps.tap(ctx.path === undefined || ctx.path === '' ? name : `${ctx.path}.${name}`, concrete)
  return value
}

/**
 * A tree with every traced leaf replaced by its concrete primal (buffers computed inside a differentiated function).
 *
 * @param tree - A tree of values.
 * @returns The same tree with concrete leaves.
 * @example
 * ```ts
 * import { concreteTree } from 'whitebox-engine/nn'
 * import { tensor } from 'whitebox-engine/tensor'
 * Object.keys(concreteTree({ a: tensor([1]) })) // => ['a']
 * ```
 */
export function concreteTree<T>(tree: T): T {
  return treeMap((v) => (isTraced(v) ? primalOf(v) : v), tree)
}

/** Routes an apply context to named children and gathers the buffers they update. */
export interface ChildScope {
  /** The context for a child: its key (folded by index), buffers, buffer sink, taps and path. */
  child(name: string, index: number): ApplyContext
  /** Report the gathered buffer updates to the parent's sink (call once, after the children have run). */
  done(): void
}

/**
 * The context routing of a composite module: each child gets its share of the buffers, its own key and path, and the
 * buffers it updates are merged into the parent's buffers when the composite finishes.
 *
 * @param ctx - The composite's context.
 * @param own - The composite's own buffers (used when the context carries none).
 * @returns The scope.
 * @example
 * ```ts
 * import { childScope } from 'whitebox-engine/nn'
 * childScope({ path: 'net' }, {}).child('0', 0).path // => 'net.0'
 * ```
 */
export function childScope(ctx: ApplyContext, own: Params): ChildScope {
  const base = ctx.buffers ?? own
  const updates: Record<string, unknown> = {}
  let changed = false
  return {
    child: (name, index) => {
      const b = base[name] as Params | undefined
      return {
        ...(ctx.train === undefined ? {} : { train: ctx.train }),
        ...(ctx.key === undefined ? {} : { key: foldIn(ctx.key, index) }),
        ...(b === undefined ? {} : { buffers: b }),
        ...(ctx.onBuffers === undefined
          ? {}
          : {
              onBuffers: (next: Params) => {
                updates[name] = next
                changed = true
              },
            }),
        ...(ctx.taps === undefined ? {} : { taps: ctx.taps }),
        path: ctx.path === undefined || ctx.path === '' ? name : `${ctx.path}.${name}`,
      }
    },
    done: () => {
      if (changed && ctx.onBuffers !== undefined) ctx.onBuffers({ ...base, ...updates })
    },
  }
}

const nonEmpty = (tree: unknown): boolean => treeFlatten(tree).leaves.length > 0

/**
 * The parameter (or buffer) tree of a composite: each child's tree under its name (empty trees left out), merged with
 * the composite's own leaves.
 *
 * @param children - The named children's trees.
 * @param own - The composite's own entries.
 * @returns The tree.
 * @example
 * ```ts
 * import { assemble } from 'whitebox-engine/nn'
 * import { tensor } from 'whitebox-engine/tensor'
 * Object.keys(assemble([['0', { w: tensor([1]) }], ['1', {}]])) // => ['0']
 * ```
 */
export function assemble(children: readonly (readonly [string, Params])[], own: Params = {}): Params {
  const out: Record<string, unknown> = { ...own }
  for (const [name, tree] of children) if (nonEmpty(tree)) out[name] = tree
  return out
}

/**
 * A neural network module in PyTorch's shape (forward, parameters, buffers, state dicts, train and eval modes) over a
 * pure functional core: `apply(params, x, ctx)` computes the output from explicit parameters, so `grad` of a loss with
 * respect to the parameters is just `grad` of a function of `params`. Buffers (running statistics) are read from the
 * context and updated through it, so the functional form stays pure; `forward` applies the updates to the module, as
 * PyTorch does.
 *
 * `I` and `O` are the input and output types: a value for most layers, records or tuples for attention and
 * recurrent layers.
 *
 * @concept modules
 */
export abstract class Module<I = Value, O = Value> {
  /** The module type, as PyTorch names it (`Linear`, `Sequential`, `ReLU`). */
  abstract readonly kind: string
  params: Params
  /** Non-trainable state (running statistics), keyed like the parameters. */
  buffers: Params
  training = true
  /** Whether the parameters are trained (false after `requiresGrad(false)`: frozen). */
  trainable = true

  protected constructor(params: Params = {}, buffers: Params = {}) {
    this.params = params
    this.buffers = buffers
  }

  /** The output for explicit parameters (the functional form). */
  abstract apply(params: Params, x: I, ctx?: ApplyContext): O

  /** The output shape for an input shape (the main input's, for modules with several; used by `describe`). */
  abstract outputShape(inputShape: readonly number[]): number[]

  /** A short description of the configuration, as PyTorch prints it (`in_features=2, out_features=3`). */
  extraRepr(): string {
    return ''
  }

  /** Named submodules (composites override this). */
  children(): readonly (readonly [string, Module<never, unknown>])[] {
    return []
  }

  /**
   * Fresh parameters from a key, as constructing the module with that key would give (PyTorch's reset_parameters).
   * Composites assemble their children's, with the key folded by each child's index.
   */
  initParams(key: Key): Params {
    return assemble(this.children().map(([name, child], i) => [name, child.initParams(foldIn(key, i))] as const))
  }

  /** Fresh buffers (running statistics at their initial values). */
  initBuffers(): Params {
    return assemble(this.children().map(([name, child]) => [name, child.initBuffers()] as const))
  }

  /** Re-initialise the parameters and buffers in place. */
  resetParameters(key: Key): this {
    this.setParams(this.initParams(key))
    return this.setBuffers(this.initBuffers())
  }

  /** Convert a forward input (tensor-likes become tensors); modules with structured inputs override it. */
  protected prepare(x: unknown): unknown {
    return asTensor(x as TensorLike)
  }

  /** The output for the module's own parameters and buffers; in training mode, updated buffers are kept. */
  forward(x: I extends Value ? TensorLike : I, ctx: ApplyContext = {}): O {
    let next: Params | undefined
    const y = this.apply(this.params, this.prepare(x) as I, {
      train: this.training,
      buffers: this.buffers,
      onBuffers: (b) => {
        next = b
      },
      ...ctx,
    })
    if (next !== undefined && ctx.onBuffers === undefined) this.setBuffers(concreteTree(next))
    return y
  }

  /** Every parameter tensor. */
  parameters(): Value[] {
    return treeFlatten(this.params).leaves
  }

  /** Parameters keyed by PyTorch-style paths (`0.weight`). */
  namedParameters(): Record<string, Value> {
    return treePaths(this.params)
  }

  /** Buffers keyed by path (`0.running_mean`). */
  namedBuffers(): Record<string, Value> {
    return treePaths(this.buffers)
  }

  /** The number of parameters. */
  parameterCount(): number {
    return treeCount(this.params)
  }

  /** Concrete parameters and buffers keyed by path (PyTorch's state_dict). */
  stateDict(): Record<string, Tensor> {
    const out: Record<string, Tensor> = {}
    for (const [k, v] of [...Object.entries(this.namedParameters()), ...Object.entries(this.namedBuffers())]) {
      if (!isTensor(v)) throw new WhiteboxError('value', `stateDict: ${k} is traced`)
      out[k] = v
    }
    return out
  }

  /**
   * Replace the parameters and buffers with those of a state dict (same paths and shapes). With `strict` (the
   * default) every parameter and buffer must be present; otherwise missing buffers keep their values.
   */
  loadStateDict(dict: Readonly<Record<string, TensorLike>>, options: { readonly strict?: boolean } = {}): this {
    const strict = options.strict ?? true
    const load = (tree: Params, what: 'parameter' | 'buffer'): Params => {
      const { leaves, def } = treeFlatten(tree)
      // treePaths and treeFlatten visit leaves in the same (sorted) order.
      const next = Object.keys(treePaths(tree)).map((p, i) => {
        const old = leaves[i] as Value
        const v = dict[p]
        if (v === undefined) {
          if (strict || what === 'parameter') throw new WhiteboxError('value', `loadStateDict: missing ${p}`)
          return old
        }
        const t = asTensor(v, old.dtype)
        if (t.shape.length !== old.shape.length || t.shape.some((d, j) => d !== old.shape[j])) {
          throw new WhiteboxError(
            'shape',
            `loadStateDict: ${p} has shape [${t.shape.join(', ')}], expected [${old.shape.join(', ')}]`,
          )
        }
        return t
      })
      return treeUnflatten(def, next) as Params
    }
    this.setParams(load(this.params, 'parameter'))
    return this.setBuffers(load(this.buffers, 'buffer'))
  }

  /** Replace the parameter tree and hand each child its part, so children stay in step with their parent. */
  setParams(params: Params): this {
    this.params = params
    for (const [name, child] of this.children()) {
      const p = params[name]
      if (p !== undefined) child.setParams(p as Params)
    }
    return this
  }

  /** Replace the buffer tree and hand each child its part. */
  setBuffers(buffers: Params): this {
    this.buffers = buffers
    for (const [name, child] of this.children()) {
      const b = buffers[name]
      if (b !== undefined) child.setBuffers(b as Params)
    }
    return this
  }

  /** Switch training mode (dropout and batch statistics) on or off, recursively. */
  train(mode = true): this {
    this.training = mode
    for (const [, child] of this.children()) child.train(mode)
    return this
  }

  /** Switch to evaluation mode. */
  eval(): this {
    return this.train(false)
  }

  /** Freeze (false) or unfreeze (true) the parameters, recursively (PyTorch's requires_grad_). */
  requiresGrad(flag = true): this {
    this.trainable = flag
    for (const [, child] of this.children()) child.requiresGrad(flag)
    return this
  }

  /** A tree of the parameters' shape whose leaves are 1 for trainable and 0 for frozen parameters. */
  trainableMask(): Params {
    const own = this.trainable ? 1 : 0
    const childNames = new Set(this.children().map(([n]) => n))
    const out: Record<string, unknown> = {}
    for (const [k, v] of Object.entries(this.params)) {
      if (!childNames.has(k)) out[k] = treeMap(() => asTensor(own), v)
    }
    for (const [name, child] of this.children()) if (this.params[name] !== undefined) out[name] = child.trainableMask()
    return out
  }
}

/** The functional form of a module: `init` and `apply` over explicit parameters and buffers. */
export interface Functional<I = Value, O = Value> {
  /** Fresh parameters from a key. */
  init(key: Key): Params
  /** Fresh buffers. */
  initBuffers(): Params
  /** The output for parameters and a context (pass `buffers` in the context to read them). */
  apply(params: Params, x: I, ctx?: ApplyContext): O
  /** The output and the updated buffers. */
  applyWithState(
    params: Params,
    buffers: Params,
    x: I,
    ctx?: ApplyContext,
  ): { readonly output: O; readonly buffers: Params }
}

/**
 * The functional form of a module (Flax's `init`/`apply`, torch.func.functional_call): pure functions of explicit
 * parameters and buffers, for `grad`, `vmap` and custom training loops.
 *
 * @param module - The module.
 * @returns `init`, `initBuffers`, `apply` and `applyWithState`.
 * @concept modules
 * @example
 * ```ts
 * import { functional, Linear } from 'whitebox-engine/nn'
 * import { key } from 'whitebox-engine/random'
 * const f = functional(new Linear(2, 1))
 * f.apply(f.init(key(0)), [[1, 2]]).shape // => [1, 1]
 * ```
 */
export function functional<I, O>(module: Module<I, O>): Functional<I, O> {
  return {
    init: (key) => module.initParams(key),
    initBuffers: () => module.initBuffers(),
    apply: (params, x, ctx = {}) => module.apply(params, x, { buffers: module.buffers, ...ctx }),
    applyWithState: (params, buffers, x, ctx = {}) => {
      let next = buffers
      const output = module.apply(params, x, {
        ...ctx,
        buffers,
        onBuffers: (b) => {
          next = b
        },
      })
      return { output, buffers: next }
    },
  }
}
