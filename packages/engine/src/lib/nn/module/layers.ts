import { WhiteboxError } from '../../base/index.ts'
import { foldIn, type Key, uniform } from '../../random/index.ts'
import {
  cast,
  div,
  flatten as flattenOp,
  greater,
  mul,
  reshape,
  sigmoid,
  tanh,
  type Value,
} from '../../tensor/index.ts'
import { type ActivationName, activation, gelu, leakyRelu, relu, silu } from '../functional/activations.ts'
import { linear } from '../functional/linear.ts'
import { torchDefaultUniform } from '../init.ts'
import { type ApplyContext, initKey, Module, type Params } from './module.ts'

/** Options of {@link Linear}. */
export interface LinearOptions {
  readonly bias?: boolean
  readonly key?: Key
}

/**
 * A fully connected layer y = x W^T + b (torch.nn.Linear), initialised as PyTorch does: weight and bias uniform on
 * +-1/sqrt(inFeatures).
 *
 * @concept modules
 * @example
 * ```ts
 * import { Linear } from 'whitebox-engine/nn'
 * new Linear(3, 2).parameterCount() // => 8
 * ```
 */
export class Linear extends Module {
  readonly kind = 'Linear'
  readonly inFeatures: number
  readonly outFeatures: number
  readonly bias: boolean

  constructor(inFeatures: number, outFeatures: number, options: LinearOptions = {}) {
    const k = initKey('Linear', options.key)
    const bias = options.bias ?? true
    super({
      weight: torchDefaultUniform(foldIn(k, 0), [outFeatures, inFeatures], inFeatures),
      ...(bias ? { bias: torchDefaultUniform(foldIn(k, 1), [outFeatures], inFeatures) } : {}),
    })
    this.inFeatures = inFeatures
    this.outFeatures = outFeatures
    this.bias = bias
  }

  apply(params: Params, x: Value): Value {
    return linear(x, params['weight'] as Value, params['bias'] as Value | undefined)
  }

  outputShape(inputShape: readonly number[]): number[] {
    if (inputShape[inputShape.length - 1] !== this.inFeatures) {
      throw new WhiteboxError(
        'shape',
        `Linear: expected ${this.inFeatures} input features, got [${inputShape.join(', ')}]`,
      )
    }
    return [...inputShape.slice(0, -1), this.outFeatures]
  }

  override extraRepr(): string {
    return `in_features=${this.inFeatures}, out_features=${this.outFeatures}, bias=${this.bias}`
  }
}

/** A parameter-free module applying a function elementwise. */
class Elementwise extends Module {
  readonly kind: string
  private readonly fn: (x: Value) => Value

  constructor(kind: string, fn: (x: Value) => Value) {
    super({})
    this.kind = kind
    this.fn = fn
  }

  apply(_params: Params, x: Value): Value {
    return this.fn(x)
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape]
  }
}

/** ReLU as a module (torch.nn.ReLU). @example new ReLU().kind // => 'ReLU' */
export class ReLU extends Elementwise {
  constructor() {
    super('ReLU', relu)
  }
}

/** Tanh as a module (torch.nn.Tanh). @example new Tanh().kind // => 'Tanh' */
export class Tanh extends Elementwise {
  constructor() {
    super('Tanh', tanh)
  }
}

/** Sigmoid as a module (torch.nn.Sigmoid). @example new Sigmoid().kind // => 'Sigmoid' */
export class Sigmoid extends Elementwise {
  constructor() {
    super('Sigmoid', sigmoid)
  }
}

/** GELU as a module (torch.nn.GELU). @example new GELU().kind // => 'GELU' */
export class GELU extends Elementwise {
  constructor(approximate: 'none' | 'tanh' = 'none') {
    super('GELU', (x) => gelu(x, approximate))
  }
}

/** SiLU as a module (torch.nn.SiLU). @example new SiLU().kind // => 'SiLU' */
export class SiLU extends Elementwise {
  constructor() {
    super('SiLU', silu)
  }
}

/** Leaky ReLU as a module (torch.nn.LeakyReLU). @example new LeakyReLU(0.2).kind // => 'LeakyReLU' */
export class LeakyReLU extends Elementwise {
  constructor(slope = 0.01) {
    super('LeakyReLU', (x) => leakyRelu(x, slope))
  }
}

/** The identity (torch.nn.Identity). @example new Identity().kind // => 'Identity' */
export class Identity extends Elementwise {
  constructor() {
    super('Identity', (x) => x)
  }
}

/** An activation module by name. @example activationModule('tanh').kind // => 'tanh' */
export function activationModule(name: ActivationName): Module {
  const f = activation(name)
  return new Elementwise(name, (x) => f(x))
}

/**
 * Flatten all but the first axis (torch.nn.Flatten with its defaults).
 *
 * @example
 * ```ts
 * import { Flatten } from 'whitebox-engine/nn'
 * new Flatten().outputShape([2, 3, 4]) // => [2, 12]
 * ```
 */
export class Flatten extends Module {
  readonly kind = 'Flatten'

  constructor() {
    super({})
  }

  apply(_params: Params, x: Value): Value {
    if (x.shape.length < 2) return flattenOp(x)
    return reshape(x, [x.shape[0] ?? 0, x.shape.slice(1).reduce((a, b) => a * b, 1)])
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [inputShape[0] ?? 1, inputShape.slice(1).reduce((a, b) => a * b, 1)]
  }
}

/**
 * Dropout (torch.nn.Dropout): in training, zero each element with probability p and scale the rest by 1 / (1 - p),
 * using the key in the apply context; the identity in evaluation mode.
 *
 * @example
 * ```ts
 * import { Dropout } from 'whitebox-engine/nn'
 * new Dropout(0.5).eval().forward([1, 2]).shape // => [2]
 * ```
 */
export class Dropout extends Module {
  readonly kind = 'Dropout'
  readonly p: number

  constructor(p = 0.5) {
    super({})
    if (!(p >= 0 && p < 1)) throw new WhiteboxError('value', `Dropout: p must be in [0, 1), got ${p}`)
    this.p = p
  }

  apply(_params: Params, x: Value, ctx: ApplyContext = {}): Value {
    if (ctx.train === false || this.p === 0) return x
    if (ctx.key === undefined)
      throw new WhiteboxError('value', 'Dropout: training mode needs a key in the apply context')
    const keep = cast(greater(uniform(ctx.key, x.shape), this.p), x.dtype)
    return div(mul(x, keep), 1 - this.p)
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape]
  }

  override extraRepr(): string {
    return `p=${this.p}`
  }
}
