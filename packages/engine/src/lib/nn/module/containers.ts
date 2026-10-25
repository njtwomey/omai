/**
 * Containers that compose modules: `Sequential`, `ModuleList` and `ModuleDict` as in torch.nn, a `Residual` connection,
 * `Parallel` branches combined by sum, concatenation or a function, `Composite` (the base of the named-part blocks such
 * as transformer and U-Net blocks), `Lambda` for small glue functions, and the `mlp` builder.
 */

import { MdmlError } from '../../base/index.ts'
import { foldIn, type Key } from '../../random/index.ts'
import { add, concat, stack, type Value } from '../../tensor/index.ts'
import { ReLU } from './activations.ts'
import { Linear } from './layers.ts'
import { type ApplyContext, assemble, childScope, Module, type Params } from './module.ts'

type AnyModule = Module<never, unknown>

const paramsOf = (children: readonly (readonly [string, AnyModule])[]): Params =>
  assemble(children.map(([n, c]) => [n, c.params] as const))
const buffersOf = (children: readonly (readonly [string, AnyModule])[]): Params =>
  assemble(children.map(([n, c]) => [n, c.buffers] as const))
const sub = (tree: Params, name: string): Params => (tree[name] ?? {}) as Params

/**
 * Modules applied in order (torch.nn.Sequential). Parameters are keyed by position, as PyTorch's state dict
 * (`0.weight`, `2.bias`); each child gets its own key for stochastic layers and its share of the buffers.
 *
 * @concept modules
 * @example
 * ```ts
 * import { Sequential, Linear, ReLU } from 'mdml-engine/nn'
 * import { key } from 'mdml-engine/random'
 * const net = new Sequential(new Linear(2, 8, { key: key(0) }), new ReLU(), new Linear(8, 1, { key: key(1) }))
 * Object.keys(net.stateDict()) // => ['0.bias', '0.weight', '2.bias', '2.weight']
 * ```
 */
export class Sequential extends Module {
  readonly kind: string = 'Sequential'
  readonly layers: readonly Module[]

  constructor(...layers: readonly Module[]) {
    const named = layers.map((l, i) => [String(i), l] as const)
    super(paramsOf(named), buffersOf(named))
    this.layers = layers
  }

  apply(params: Params, x: Value, ctx: ApplyContext = {}): Value {
    const scope = childScope(ctx, this.buffers)
    const y = this.layers.reduce((h, layer, i) => layer.apply(sub(params, String(i)), h, scope.child(String(i), i)), x)
    scope.done()
    return y
  }

  outputShape(inputShape: readonly number[]): number[] {
    return this.layers.reduce<number[]>((s, layer) => layer.outputShape(s), [...inputShape])
  }

  override children(): readonly (readonly [string, Module])[] {
    return this.layers.map((l, i) => [String(i), l] as const)
  }
}

/**
 * A list of modules (torch.nn.ModuleList): it holds and names them (`0`, `1`, ...) for parameters, state dicts and
 * modes; the parent decides how to call them.
 *
 * @concept modules
 * @example
 * ```ts
 * import { ModuleList, Linear } from 'mdml-engine/nn'
 * import { key } from 'mdml-engine/random'
 * new ModuleList([new Linear(2, 2, { key: key(0) }), new Linear(2, 2, { key: key(1) })]).parameterCount() // => 12
 * ```
 */
export class ModuleList<M extends AnyModule = Module> extends Module<never, never> {
  readonly kind = 'ModuleList'
  readonly modules: readonly M[]

  constructor(modules: readonly M[]) {
    const named = modules.map((m, i) => [String(i), m] as const)
    super(paramsOf(named), buffersOf(named))
    this.modules = modules
  }

  /** The module at an index. */
  at(i: number): M {
    const m = this.modules[i]
    if (m === undefined) throw new MdmlError('value', `ModuleList: no module at ${i}`)
    return m
  }

  /** The number of modules. */
  get length(): number {
    return this.modules.length
  }

  apply(): never {
    throw new MdmlError('value', 'ModuleList has no forward; call its modules')
  }

  outputShape(): number[] {
    throw new MdmlError('value', 'ModuleList has no forward')
  }

  override children(): readonly (readonly [string, AnyModule])[] {
    return this.modules.map((m, i) => [String(i), m] as const)
  }
}

/**
 * Modules by name (torch.nn.ModuleDict), in insertion order; parameters are keyed `<name>.<parameter>`.
 *
 * @concept modules
 * @example
 * ```ts
 * import { ModuleDict, Linear, ReLU } from 'mdml-engine/nn'
 * import { key } from 'mdml-engine/random'
 * Object.keys(new ModuleDict({ proj: new Linear(2, 1, { key: key(0) }), act: new ReLU() }).stateDict()) // => ['proj.bias', 'proj.weight']
 * ```
 */
export class ModuleDict<M extends AnyModule = Module> extends Module<never, never> {
  readonly kind = 'ModuleDict'
  readonly modules: Readonly<Record<string, M>>

  constructor(modules: Readonly<Record<string, M>>) {
    super(paramsOf(Object.entries(modules)), buffersOf(Object.entries(modules)))
    this.modules = modules
  }

  /** The module of a name. */
  get(name: string): M {
    const m = this.modules[name]
    if (m === undefined) throw new MdmlError('value', `ModuleDict: no module ${name}`)
    return m
  }

  /** The names, in insertion order. */
  keys(): string[] {
    return Object.keys(this.modules)
  }

  apply(): never {
    throw new MdmlError('value', 'ModuleDict has no forward; call its modules')
  }

  outputShape(): number[] {
    throw new MdmlError('value', 'ModuleDict has no forward')
  }

  override children(): readonly (readonly [string, AnyModule])[] {
    return Object.entries(this.modules)
  }
}

/**
 * A residual connection: y = body(x) + shortcut(x), the shortcut being the identity unless given (a projection when
 * shapes change). Parameters are keyed `body.*` and `shortcut.*`.
 *
 * @concept residual-connections
 * @example
 * ```ts
 * import { Residual, Linear } from 'mdml-engine/nn'
 * import { key } from 'mdml-engine/random'
 * new Residual(new Linear(3, 3, { key: key(0) })).outputShape([2, 3]) // => [2, 3]
 * ```
 */
export class Residual extends Module {
  readonly kind = 'Residual'
  readonly body: Module
  readonly shortcut: Module | undefined

  constructor(body: Module, shortcut?: Module) {
    const named: (readonly [string, Module])[] = [['body', body]]
    if (shortcut !== undefined) named.push(['shortcut', shortcut])
    super(paramsOf(named), buffersOf(named))
    this.body = body
    this.shortcut = shortcut
  }

  apply(params: Params, x: Value, ctx: ApplyContext = {}): Value {
    const scope = childScope(ctx, this.buffers)
    const y = this.body.apply(sub(params, 'body'), x, scope.child('body', 0))
    const s =
      this.shortcut === undefined ? x : this.shortcut.apply(sub(params, 'shortcut'), x, scope.child('shortcut', 1))
    scope.done()
    return add(y, s)
  }

  outputShape(inputShape: readonly number[]): number[] {
    return this.body.outputShape(inputShape)
  }

  override children(): readonly (readonly [string, Module])[] {
    return this.shortcut === undefined
      ? [['body', this.body]]
      : [
          ['body', this.body],
          ['shortcut', this.shortcut],
        ]
  }
}

/** How {@link Parallel} combines its branches. */
export type Combine = 'sum' | 'concat' | 'stack' | ((outputs: readonly Value[]) => Value)

/**
 * Branches applied to the same input, their outputs combined: summed, concatenated or stacked along an axis (an
 * Inception block, a multi-scale head), or by a function.
 *
 * @concept modules
 * @example
 * ```ts
 * import { Parallel, Linear } from 'mdml-engine/nn'
 * import { key } from 'mdml-engine/random'
 * new Parallel([new Linear(3, 2, { key: key(0) }), new Linear(3, 4, { key: key(1) })], 'concat').outputShape([5, 3]) // => [5, 6]
 * ```
 */
export class Parallel extends Module {
  readonly kind = 'Parallel'
  readonly branches: readonly Module[]
  readonly combine: Combine
  readonly axis: number

  constructor(branches: readonly Module[], combine: Combine = 'sum', axis = -1) {
    const named = branches.map((b, i) => [String(i), b] as const)
    super(paramsOf(named), buffersOf(named))
    this.branches = branches
    this.combine = combine
    this.axis = axis
  }

  apply(params: Params, x: Value, ctx: ApplyContext = {}): Value {
    const scope = childScope(ctx, this.buffers)
    const outs = this.branches.map((b, i) => b.apply(sub(params, String(i)), x, scope.child(String(i), i)))
    scope.done()
    if (typeof this.combine === 'function') return this.combine(outs)
    if (this.combine === 'concat') return concat(outs, this.axis)
    if (this.combine === 'stack')
      return stack(outs, this.axis < 0 ? (outs[0]?.shape.length ?? 0) + 1 + this.axis : this.axis)
    return outs.reduce((a, b) => add(a, b))
  }

  outputShape(inputShape: readonly number[]): number[] {
    const shapes = this.branches.map((b) => b.outputShape(inputShape))
    const first = shapes[0] ?? [...inputShape]
    if (this.combine === 'concat') {
      const a = this.axis < 0 ? first.length + this.axis : this.axis
      return first.map((n, i) => (i === a ? shapes.reduce((s, sh) => s + (sh[i] ?? 0), 0) : n))
    }
    if (this.combine === 'stack') {
      const a = this.axis < 0 ? first.length + 1 + this.axis : this.axis
      return [...first.slice(0, a), shapes.length, ...first.slice(a)]
    }
    return first
  }

  override children(): readonly (readonly [string, Module])[] {
    return this.branches.map((b, i) => [String(i), b] as const)
  }
}

/** Calls the parts of a {@link Composite} with their parameters and routed contexts. */
export interface PartRunner {
  /** Apply a part to an input. */
  run<I, O>(name: string, x: I): O
  /** The routed context of a part (for parts called in a custom way). */
  context(name: string): ApplyContext
  /** Report updated buffers to the parent (call once, after the parts have run). */
  done(): void
}

/**
 * A module made of named parts (the base of blocks such as residual, transformer and U-Net blocks): parameters and
 * buffers are assembled from the parts under their names, `initParams` folds the key by each part's position, and
 * `parts(params, ctx)` runs each part with its share of the parameters, buffers, key and path.
 *
 * @concept modules
 */
export abstract class Composite<I = Value, O = Value> extends Module<I, O> {
  readonly partList: readonly (readonly [string, AnyModule])[]

  protected constructor(parts: Readonly<Record<string, AnyModule | undefined>>, own: Params = {}) {
    const list = Object.entries(parts).filter((e): e is [string, AnyModule] => e[1] !== undefined)
    super(
      assemble(
        list.map(([n, c]) => [n, c.params] as const),
        own,
      ),
      buffersOf(list),
    )
    this.partList = list
  }

  override children(): readonly (readonly [string, AnyModule])[] {
    return this.partList
  }

  /** A part by name. */
  part<M extends AnyModule = Module>(name: string): M {
    const found = this.partList.find(([n]) => n === name)
    if (found === undefined) throw new MdmlError('value', `${this.kind}: no part ${name}`)
    return found[1] as M
  }

  /** Whether a part exists. */
  has(name: string): boolean {
    return this.partList.some(([n]) => n === name)
  }

  /** A runner for this forward pass. */
  protected parts(params: Params, ctx: ApplyContext): PartRunner {
    const scope = childScope(ctx, this.buffers)
    const index = (name: string): number => this.partList.findIndex(([n]) => n === name)
    return {
      run: <PI, PO>(name: string, x: PI): PO =>
        (this.part(name) as unknown as Module<PI, PO>).apply(sub(params, name), x, scope.child(name, index(name))),
      context: (name) => scope.child(name, index(name)),
      done: () => scope.done(),
    }
  }
}

/**
 * A parameter-free module from a function of the input (Keras's Lambda), for reshapes and small glue in a
 * {@link Sequential}.
 *
 * @example
 * ```ts
 * import { Lambda } from 'mdml-engine/nn'
 * import { mul } from 'mdml-engine/tensor'
 * new Lambda((x) => mul(x, 2)).outputShape([3]) // => [3]
 * ```
 */
export class Lambda extends Module {
  readonly kind = 'Lambda'
  private readonly fn: (x: Value, ctx: ApplyContext) => Value
  private readonly shapeFn: (s: readonly number[]) => number[]

  constructor(fn: (x: Value, ctx: ApplyContext) => Value, outputShape?: (inputShape: readonly number[]) => number[]) {
    super()
    this.fn = fn
    this.shapeFn = outputShape ?? ((s) => [...s])
  }

  apply(_params: Params, x: Value, ctx: ApplyContext = {}): Value {
    return this.fn(x, ctx)
  }

  outputShape(inputShape: readonly number[]): number[] {
    return this.shapeFn(inputShape)
  }
}

/**
 * A multilayer perceptron: Linear layers of the given widths with an activation between them.
 *
 * @param widths - Layer widths from input to output (at least two).
 * @param options - The hidden activation (default ReLU) and an initialisation key.
 * @returns The network.
 * @example
 * ```ts
 * import { mlp } from 'mdml-engine/nn'
 * mlp([4, 16, 16, 1]).parameterCount() // => 369
 * ```
 */
export function mlp(
  widths: readonly number[],
  options: { readonly activation?: () => Module; readonly key?: Key; readonly bias?: boolean } = {},
): Sequential {
  const layers: Module[] = []
  for (let i = 0; i + 1 < widths.length; i++) {
    layers.push(
      new Linear(widths[i] ?? 0, widths[i + 1] ?? 0, {
        ...(options.key === undefined ? {} : { key: foldIn(options.key, i) }),
        ...(options.bias === undefined ? {} : { bias: options.bias }),
      }),
    )
    if (i + 2 < widths.length) layers.push(options.activation === undefined ? new ReLU() : options.activation())
  }
  return new Sequential(...layers)
}
