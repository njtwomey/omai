import { WhiteboxError } from '../../base/index.ts'
import { type Context, defaultContext } from '../../context/index.ts'
import type { Key } from '../../random/index.ts'
import {
  asTensor,
  isTensor,
  type Tensor,
  type TensorLike,
  treeCount,
  treeFlatten,
  treePaths,
  treeUnflatten,
  type Value,
} from '../../tensor/index.ts'

/** A module's parameters: a tree of tensors (nested records keyed like PyTorch's state-dict paths). */
export type Params = Readonly<Record<string, unknown>>

/** What a forward pass may need besides its input: the training flag and a key for stochastic layers. */
export interface ApplyContext {
  readonly train?: boolean
  readonly key?: Key
}

// Default initialisation keys are numbered per context, so a program that builds its modules in the same order from
// the same seed gets the same weights (as with torch.manual_seed), and replacing the default context starts afresh.
const counters = new WeakMap<Context, number>()

/** The key a module uses for initialisation: the given one, or the next key of the default context. */
export function initKey(kind: string, key?: Key): Key {
  if (key !== undefined) return key
  const ctx = defaultContext()
  const n = counters.get(ctx) ?? 0
  counters.set(ctx, n + 1)
  return ctx.key('nn', kind, n)
}

/**
 * A neural network module in PyTorch's shape (forward, parameters, state dicts, train and eval modes) over a pure
 * functional core: `apply(params, x, ctx)` computes the output from explicit parameters, so `grad` of a loss with
 * respect to the parameters is just `grad` of a function of `params`.
 *
 * @concept modules
 */
export abstract class Module {
  /** The module type, as PyTorch names it (`Linear`, `Sequential`, `ReLU`). */
  abstract readonly kind: string
  params: Params
  training = true

  protected constructor(params: Params) {
    this.params = params
  }

  /** The output for explicit parameters (the functional form). */
  abstract apply(params: Params, x: Value, ctx?: ApplyContext): Value

  /** The output shape for an input shape (used by `describe`). */
  abstract outputShape(inputShape: readonly number[]): number[]

  /** A short description of the configuration, as PyTorch prints it (`in_features=2, out_features=3`). */
  extraRepr(): string {
    return ''
  }

  /** Named submodules (containers override this). */
  children(): readonly (readonly [string, Module])[] {
    return []
  }

  /** The output for the module's own parameters. */
  forward(x: TensorLike, ctx: ApplyContext = {}): Value {
    return this.apply(this.params, asTensor(x), { train: this.training, ...ctx })
  }

  /** Every parameter tensor. */
  parameters(): Value[] {
    return treeFlatten(this.params).leaves
  }

  /** Parameters keyed by PyTorch-style paths (`0.weight`). */
  namedParameters(): Record<string, Value> {
    return treePaths(this.params)
  }

  /** The number of parameters. */
  parameterCount(): number {
    return treeCount(this.params)
  }

  /** Concrete parameters keyed by path (PyTorch's state_dict). */
  stateDict(): Record<string, Tensor> {
    const out: Record<string, Tensor> = {}
    for (const [k, v] of Object.entries(this.namedParameters())) {
      if (!isTensor(v)) throw new WhiteboxError('value', `stateDict: parameter ${k} is traced`)
      out[k] = v
    }
    return out
  }

  /** Replace the parameters with those of a state dict (same paths and shapes). */
  loadStateDict(dict: Readonly<Record<string, TensorLike>>): this {
    const { leaves, def } = treeFlatten(this.params)
    const paths = Object.keys(treePaths(this.params))
    // treePaths and treeFlatten visit leaves in the same (sorted) order.
    const next = paths.map((p, i) => {
      const v = dict[p]
      if (v === undefined) throw new WhiteboxError('value', `loadStateDict: missing ${p}`)
      const t = asTensor(v)
      const old = leaves[i] as Value
      if (t.shape.length !== old.shape.length || t.shape.some((d, j) => d !== old.shape[j])) {
        throw new WhiteboxError(
          'shape',
          `loadStateDict: ${p} has shape [${t.shape.join(', ')}], expected [${old.shape.join(', ')}]`,
        )
      }
      return t
    })
    return this.setParams(treeUnflatten(def, next) as Params)
  }

  /** Replace the parameter tree (containers also hand each child its part). */
  setParams(params: Params): this {
    this.params = params
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
}
