/**
 * The continuous normalising flow (FFJORD, Grathwohl et al. 2019): a neural ODE `dz/dt = f(z, t)` as an invertible
 * map whose log-density change is the integral of the trace of the Jacobian (the instantaneous change of variables,
 * `d log p / dt = -tr(df/dz)`), with the trace taken exactly (one backward pass per dimension) or estimated by
 * Hutchinson's estimator (one backward pass with a fixed noise vector, unbiased). The block integrates the state and
 * the accumulated trace together with any solver of the neural ODE blocks.
 */

import { vjp } from '../../autodiff/index.ts'
import { OmaiError } from '../../base/index.ts'
import { foldIn, type Key, key as makeKey, normal } from '../../random/index.ts'
import {
  add,
  asTensor,
  concat,
  mul,
  reshape,
  sign,
  sliceAxis,
  sum,
  type TensorLike,
  tensor,
  type Value,
  zeros,
} from '../../tensor/index.ts'
import { Tanh } from '../module/activations.ts'
import { Composite, mlp } from '../module/containers.ts'
import { type ApplyContext, initKey, type Module, type Params } from '../module/module.ts'
import type { OdeSolver } from './continuous.ts'

/** How the trace of the Jacobian is computed. */
export type TraceEstimator = 'exact' | 'hutchinson'

/** The noise of Hutchinson's estimator: Rademacher (+-1, lower variance) or Gaussian. */
export type HutchinsonNoise = 'rademacher' | 'gaussian'

/** Options of {@link CNF}. */
export interface CnfOptions {
  /** The ODE solver (any of `dynamics/ode`'s, for example `odeSolver('RK4', { dt: 0.1 })`). */
  readonly solver: OdeSolver
  /** The vector field `[D (+1 for time)] -> [D]` (default: a tanh perceptron with `hidden` widths). */
  readonly field?: Module
  /** Hidden widths of the default field (default `[32, 32]`). */
  readonly hidden?: readonly number[]
  /** The trace: `exact` (default) or `hutchinson`. */
  readonly trace?: TraceEstimator
  /** The Hutchinson noise (default Rademacher). */
  readonly noise?: HutchinsonNoise
  /** Give the field the time as an extra last feature (default true with the default field, false with a given one). */
  readonly timeDependent?: boolean
  /** Integration interval of the map from data to the base distribution (default `[0, 1]`). */
  readonly interval?: readonly [number, number]
  readonly key?: Key
}

/** The input of {@link CNF.apply}: data rows and the key of the Hutchinson noise. */
export interface CnfInput {
  readonly x: TensorLike
  readonly key?: Key
}

/** The output of {@link CNF.apply}. */
export interface CnfOutput {
  /** The image of the data in the base space `[B, D]`. */
  readonly z: Value
  /** The integral of the trace along the path `[B]`: `log p(x) = log p_base(z) + logDet`. */
  readonly logDet: Value
}

/**
 * `-1/2 (D log 2 pi + |z|^2)`, the log density of a standard normal base per row.
 *
 * @param z - Base rows `[B, D]`.
 * @returns The log densities `[B]`.
 * @example
 * ```ts
 * import { standardNormalLogProb } from 'omai-engine/nn/blocks'
 * import { item } from 'omai-engine/tensor'
 * item(standardNormalLogProb([[0, 0]])) // => -1.8378770664093453
 * ```
 */
export function standardNormalLogProb(z: TensorLike): Value {
  const v = asTensor(z)
  const d = v.shape[v.shape.length - 1] ?? 1
  return mul(add(sum(mul(v, v), -1), d * Math.log(2 * Math.PI)), -0.5)
}

/**
 * A continuous normalising flow: the data `x` is carried to the base space by the ODE `dz/dt = f(z, t)` over `interval`,
 * and the log-density follows from the instantaneous change of variables. With the default standard normal base,
 * `logProb(x) = log N(z(1)) + integral of tr(df/dz) dt`. Sampling integrates the same field backwards from base
 * noise. Trace estimation is `exact` (a vector-Jacobian product per dimension) or `hutchinson` (one product with a
 * fixed noise vector per forward pass, unbiased: its variance is on the tape of a stochastic run).
 *
 * @reference Grathwohl, Chen, Bettencourt, Sutskever and Duvenaud (2019), FFJORD: free-form continuous dynamics for scalable reversible generative models.
 * @example
 * ```ts
 * import { CNF } from 'omai-engine/nn/blocks'
 * import { odeSolver } from 'omai-engine/dynamics/ode'
 * import { key } from 'omai-engine/random'
 * const flow = new CNF(2, { solver: odeSolver('RK4', { dt: 0.25 }), hidden: [8], key: key(0) })
 * flow.logProb(flow.params, [[0.5, -0.5]]).shape // => [1]
 * ```
 */
export class CNF extends Composite<CnfInput, CnfOutput> {
  readonly kind = 'CNF'
  readonly dim: number
  readonly solver: OdeSolver
  readonly trace: TraceEstimator
  readonly noise: HutchinsonNoise
  readonly timeDependent: boolean
  readonly interval: readonly [number, number]

  /**
   * @param dim - The data dimension `D`.
   * @param options - The solver, field, trace estimator and interval.
   */
  constructor(dim: number, options: CnfOptions) {
    const k = initKey('CNF', options.key)
    const timeDependent = options.timeDependent ?? options.field === undefined
    super({
      field:
        options.field ??
        mlp([dim + (timeDependent ? 1 : 0), ...(options.hidden ?? [32, 32]), dim], {
          activation: () => new Tanh(),
          key: foldIn(k, 0),
        }),
    })
    this.dim = dim
    this.solver = options.solver
    this.trace = options.trace ?? 'exact'
    this.noise = options.noise ?? 'rademacher'
    this.timeDependent = timeDependent
    this.interval = options.interval ?? [0, 1]
  }

  protected override prepare(x: unknown): unknown {
    const input = x as CnfInput
    return input.x === undefined ? { x: x as TensorLike } : input
  }

  /** The vector field `f(z, t)` for explicit parameters. */
  vectorField(params: Params, t: number, z: Value, ctx: ApplyContext = {}): Value {
    const input = this.timeDependent ? concat([z, add(zeros([...z.shape.slice(0, -1), 1]), t)], -1) : z
    return this.part('field').apply((params['field'] ?? {}) as Params, input, ctx)
  }

  /** The trace of the Jacobian of the field at `(t, z)` per row `[B, 1]`, exactly or with the noise vector `eps`. */
  jacobianTrace(params: Params, t: number, z: Value, eps?: Value, ctx: ApplyContext = {}): Value {
    const { out, pullback } = vjp((s: Value) => this.vectorField(params, t, s, ctx), z)
    if (this.trace === 'hutchinson') {
      if (eps === undefined) throw new OmaiError('value', 'CNF: the Hutchinson estimator needs a noise vector')
      return sum(mul(pullback(eps), eps), -1, true)
    }
    // Exact: column i of the Jacobian's transpose dotted with e_i, one pullback per dimension.
    const rows = out.shape.slice(0, -1)
    let total: Value = zeros([...rows, 1])
    for (let i = 0; i < this.dim; i++) {
      const e = tensor(Array.from({ length: this.dim }, (_, j) => (j === i ? 1 : 0)))
      const seed = add(zeros(out.shape), e)
      total = add(total, sliceAxis(pullback(seed), -1, i, i + 1))
    }
    return total
  }

  private noiseFor(rows: number, key: Key | undefined): Value | undefined {
    if (this.trace !== 'hutchinson') return undefined
    const k = key ?? makeKey(0)
    if (this.noise === 'gaussian') return normal(k, [rows, this.dim])
    // Rademacher: the sign of a Gaussian draw.
    return sign(normal(k, [rows, this.dim]))
  }

  private integrate(
    params: Params,
    state: Value,
    times: readonly number[],
    flip: boolean,
    eps: Value | undefined,
    ctx: ApplyContext,
    withTrace = true,
  ): Value {
    const d = this.dim
    const field = (t: number, y: Value): Value => {
      const time = flip ? this.interval[0] + this.interval[1] - t : t
      const z = sliceAxis(y, -1, 0, d)
      const dz = this.vectorField(params, time, z, ctx)
      const dTrace = withTrace ? this.jacobianTrace(params, time, z, eps, ctx) : zeros([...z.shape.slice(0, -1), 1])
      // Backwards in time the field and the accumulated trace both change sign.
      return flip ? concat([mul(dz, -1), mul(dTrace, -1)], -1) : concat([dz, dTrace], -1)
    }
    const ys = this.solver(field, state, times)
    return ys[ys.length - 1] as Value
  }

  apply(params: Params, input: CnfInput, ctx: ApplyContext = {}): CnfOutput {
    const x = asTensor(input.x)
    const rows = x.shape[0] ?? 1
    const eps = this.noiseFor(rows, input.key ?? ctx.key)
    const y0 = concat([x, zeros([rows, 1])], -1)
    const y1 = this.integrate(params, y0, [this.interval[0], this.interval[1]], false, eps, ctx)
    return { z: sliceAxis(y1, -1, 0, this.dim), logDet: reshape(sliceAxis(y1, -1, this.dim, this.dim + 1), [-1]) }
  }

  /**
   * The log density of data under the flow: `log p_base(z(1)) + integral of the trace`.
   *
   * @param params - The parameters.
   * @param x - Data rows `[B, D]`.
   * @param options - The base log density (default standard normal), the noise key and the apply context.
   * @returns `[B]`.
   */
  logProb(
    params: Params,
    x: TensorLike,
    options: { readonly baseLogProb?: (z: Value) => Value; readonly key?: Key; readonly ctx?: ApplyContext } = {},
  ): Value {
    const out = this.apply(params, { x, ...(options.key === undefined ? {} : { key: options.key }) }, options.ctx ?? {})
    return add((options.baseLogProb ?? standardNormalLogProb)(out.z), out.logDet)
  }

  /**
   * Carry base samples back to data space by integrating the field backwards.
   *
   * @param params - The parameters.
   * @param z - Base rows `[B, D]`.
   * @param ctx - The apply context.
   * @returns The data rows `[B, D]`.
   */
  sample(params: Params, z: TensorLike, ctx: ApplyContext = {}): Value {
    const v = asTensor(z)
    const rows = v.shape[0] ?? 1
    // dx/ds = -f(x, t0 + t1 - s) from s = t0 to t1 runs the flow in reverse; the trace is not needed.
    const eps = this.noiseFor(rows, ctx.key)
    const y1 = this.integrate(
      params,
      concat([v, zeros([rows, 1])], -1),
      [this.interval[0], this.interval[1]],
      true,
      eps,
      ctx,
    )
    return sliceAxis(y1, -1, 0, this.dim)
  }

  outputShape(s: readonly number[]): number[] {
    return [...s]
  }

  override extraRepr(): string {
    return `dim=${this.dim}, trace=${this.trace}, interval=[${this.interval.join(', ')}]`
  }
}
