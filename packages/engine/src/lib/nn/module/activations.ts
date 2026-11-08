/**
 * The activation functions as parameter-free modules in torch.nn's shape (`ReLU`, `GELU`, `SiLU`, `Softmax` and the
 * rest), plus `PReLU` with learnt slopes and `activationModule` to build one by name, for use inside `Sequential` and
 * other containers.
 */

import { OmaiError } from '../../base/index.ts'
import { full, logSoftmax, mul, sigmoid, softmax, tanh, type Value } from '../../tensor/index.ts'
import {
  type ActivationName,
  activation,
  celu,
  elu,
  gelu,
  glu,
  hardshrink,
  hardsigmoid,
  hardswish,
  hardtanh,
  leakyRelu,
  logsigmoid,
  mish,
  prelu,
  relu,
  relu6,
  selu,
  silu,
  softplus,
  softshrink,
  softsign,
  tanhshrink,
  threshold,
} from '../functional/activations.ts'
import { Module, type Params } from './module.ts'

/** A parameter-free module applying a shape-preserving function. */
class Elementwise extends Module {
  readonly kind: string
  private readonly fn: (x: Value) => Value
  private readonly repr: string

  constructor(kind: string, fn: (x: Value) => Value, repr = '') {
    super()
    this.kind = kind
    this.fn = fn
    this.repr = repr
  }

  apply(_params: Params, x: Value): Value {
    return this.fn(x)
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape]
  }

  override extraRepr(): string {
    return this.repr
  }
}

/** ReLU as a module (torch.nn.ReLU). @example new ReLU().kind // => 'ReLU' */
export class ReLU extends Elementwise {
  constructor() {
    super('ReLU', relu)
  }
}

/** ReLU6 as a module (torch.nn.ReLU6). @example new ReLU6().kind // => 'ReLU6' */
export class ReLU6 extends Elementwise {
  constructor() {
    super('ReLU6', relu6)
  }
}

/** Leaky ReLU as a module (torch.nn.LeakyReLU). @example new LeakyReLU(0.2).kind // => 'LeakyReLU' */
export class LeakyReLU extends Elementwise {
  constructor(slope = 0.01) {
    super('LeakyReLU', (x) => leakyRelu(x, slope), `negative_slope=${slope}`)
  }
}

/** ELU as a module (torch.nn.ELU). @example new ELU().kind // => 'ELU' */
export class ELU extends Elementwise {
  constructor(alpha = 1) {
    super('ELU', (x) => elu(x, alpha), `alpha=${alpha}`)
  }
}

/** CELU as a module (torch.nn.CELU). @example new CELU().kind // => 'CELU' */
export class CELU extends Elementwise {
  constructor(alpha = 1) {
    super('CELU', (x) => celu(x, alpha), `alpha=${alpha}`)
  }
}

/** SELU as a module (torch.nn.SELU). @example new SELU().kind // => 'SELU' */
export class SELU extends Elementwise {
  constructor() {
    super('SELU', selu)
  }
}

/** GELU as a module (torch.nn.GELU). @example new GELU().kind // => 'GELU' */
export class GELU extends Elementwise {
  constructor(approximate: 'none' | 'tanh' = 'none') {
    super('GELU', (x) => gelu(x, approximate), `approximate='${approximate}'`)
  }
}

/** SiLU as a module (torch.nn.SiLU). @example new SiLU().kind // => 'SiLU' */
export class SiLU extends Elementwise {
  constructor() {
    super('SiLU', silu)
  }
}

/** Mish as a module (torch.nn.Mish). @example new Mish().kind // => 'Mish' */
export class Mish extends Elementwise {
  constructor() {
    super('Mish', mish)
  }
}

/** Softplus as a module (torch.nn.Softplus). @example new Softplus().kind // => 'Softplus' */
export class Softplus extends Elementwise {
  constructor(beta = 1, threshold = 20) {
    super('Softplus', (x) => softplus(x, beta, threshold), `beta=${beta}, threshold=${threshold}`)
  }
}

/** Softsign as a module (torch.nn.Softsign). @example new Softsign().kind // => 'Softsign' */
export class Softsign extends Elementwise {
  constructor() {
    super('Softsign', softsign)
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

/** Log-sigmoid as a module (torch.nn.LogSigmoid). @example new LogSigmoid().kind // => 'LogSigmoid' */
export class LogSigmoid extends Elementwise {
  constructor() {
    super('LogSigmoid', logsigmoid)
  }
}

/** Hard tanh as a module (torch.nn.Hardtanh). @example new Hardtanh().kind // => 'Hardtanh' */
export class Hardtanh extends Elementwise {
  constructor(min = -1, max = 1) {
    super('Hardtanh', (x) => hardtanh(x, min, max), `min_val=${min}, max_val=${max}`)
  }
}

/** Hard sigmoid as a module (torch.nn.Hardsigmoid). @example new Hardsigmoid().kind // => 'Hardsigmoid' */
export class Hardsigmoid extends Elementwise {
  constructor() {
    super('Hardsigmoid', hardsigmoid)
  }
}

/** Hard swish as a module (torch.nn.Hardswish). @example new Hardswish().kind // => 'Hardswish' */
export class Hardswish extends Elementwise {
  constructor() {
    super('Hardswish', hardswish)
  }
}

/** Tanh shrink as a module (torch.nn.Tanhshrink). @example new Tanhshrink().kind // => 'Tanhshrink' */
export class Tanhshrink extends Elementwise {
  constructor() {
    super('Tanhshrink', tanhshrink)
  }
}

/** Soft shrinkage as a module (torch.nn.Softshrink). @example new Softshrink().kind // => 'Softshrink' */
export class Softshrink extends Elementwise {
  constructor(lambda = 0.5) {
    super('Softshrink', (x) => softshrink(x, lambda), `${lambda}`)
  }
}

/** Hard shrinkage as a module (torch.nn.Hardshrink). @example new Hardshrink().kind // => 'Hardshrink' */
export class Hardshrink extends Elementwise {
  constructor(lambda = 0.5) {
    super('Hardshrink', (x) => hardshrink(x, lambda), `${lambda}`)
  }
}

/** Threshold as a module (torch.nn.Threshold). @example new Threshold(0.1, 20).kind // => 'Threshold' */
export class Threshold extends Elementwise {
  constructor(thresholdValue: number, value: number) {
    super('Threshold', (x) => threshold(x, thresholdValue, value), `threshold=${thresholdValue}, value=${value}`)
  }
}

/**
 * Softmax along an axis as a module (torch.nn.Softmax).
 *
 * @example
 * ```ts
 * import { Softmax } from 'omai-engine/nn'
 * new Softmax(-1).kind // => 'Softmax'
 * ```
 */
export class Softmax extends Elementwise {
  constructor(axis = -1) {
    super('Softmax', (x) => softmax(x, axis), `dim=${axis}`)
  }
}

/** Log-softmax along an axis as a module (torch.nn.LogSoftmax). @example new LogSoftmax().kind // => 'LogSoftmax' */
export class LogSoftmax extends Elementwise {
  constructor(axis = -1) {
    super('LogSoftmax', (x) => logSoftmax(x, axis), `dim=${axis}`)
  }
}

/** Softmin, softmax(-x), along an axis (torch.nn.Softmin). @example new Softmin().kind // => 'Softmin' */
export class Softmin extends Elementwise {
  constructor(axis = -1) {
    super('Softmin', (x) => softmax(mul(x, -1), axis), `dim=${axis}`)
  }
}

/**
 * The gated linear unit as a module (torch.nn.GLU): halves the split axis.
 *
 * @example
 * ```ts
 * import { GLU } from 'omai-engine/nn'
 * new GLU().outputShape([2, 6]) // => [2, 3]
 * ```
 */
export class GLU extends Module {
  readonly kind = 'GLU'
  readonly axis: number

  constructor(axis = -1) {
    super()
    this.axis = axis
  }

  apply(_params: Params, x: Value): Value {
    return glu(x, this.axis)
  }

  outputShape(inputShape: readonly number[]): number[] {
    const a = this.axis < 0 ? inputShape.length + this.axis : this.axis
    return inputShape.map((n, i) => (i === a ? n / 2 : n))
  }

  override extraRepr(): string {
    return `dim=${this.axis}`
  }
}

/**
 * PReLU with learnt slopes (torch.nn.PReLU): one shared slope, or one per channel, initialised to 0.25.
 *
 * @example
 * ```ts
 * import { PReLU } from 'omai-engine/nn'
 * new PReLU(3).parameterCount() // => 3
 * ```
 */
export class PReLU extends Module {
  readonly kind = 'PReLU'
  readonly numParameters: number
  readonly init: number

  constructor(numParameters = 1, init = 0.25) {
    super()
    if (numParameters < 1) throw new OmaiError('value', 'PReLU: need at least one parameter')
    this.numParameters = numParameters
    this.init = init
    this.params = this.initParams()
  }

  override initParams(): Params {
    return { weight: full([this.numParameters], this.init) }
  }

  apply(params: Params, x: Value): Value {
    return prelu(x, params['weight'] as Value)
  }

  outputShape(inputShape: readonly number[]): number[] {
    return [...inputShape]
  }

  override extraRepr(): string {
    return `num_parameters=${this.numParameters}`
  }
}

/**
 * An activation module by name.
 *
 * @param name - The activation.
 * @returns The module.
 * @example
 * ```ts
 * import { activationModule } from 'omai-engine/nn'
 * activationModule('tanh').kind // => 'tanh'
 * ```
 */
export function activationModule(name: ActivationName): Module {
  const f = activation(name)
  return new Elementwise(name, (x) => f(x))
}
