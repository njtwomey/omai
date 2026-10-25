/**
 * Continuous-depth blocks: neural ODEs, where a network defines the derivative of a hidden state and the output is the
 * solution of that differential equation at given times. `ODEBlock` (Chen et al. 2018, with gradients through the
 * solver or by the adjoint method), its augmented and second-order variants, the neural controlled differential
 * equation `CDEBlock` driven by an input path, and the neural SDE `SDEBlock` with learnt drift and diffusion.
 */

import { customVjp, vjp } from '../../autodiff/index.ts'
import { WhiteboxError } from '../../base/index.ts'
import type { Key } from '../../random/index.ts'
import {
  add,
  asTensor,
  concat,
  matmul,
  mul,
  neg,
  reshape,
  sizeOf,
  sliceAxis,
  stack,
  sub,
  type TensorLike,
  treeFlatten,
  treeUnflatten,
  type Value,
  zeros,
} from '../../tensor/index.ts'
import { Composite } from '../module/containers.ts'
import type { ApplyContext, Module, Params } from '../module/module.ts'

/**
 * An ODE solver: the solution of dy/dt = f(t, y) from y(times[0]) = y0 at each of the increasing `times`. Any solver
 * of `dynamics/ode` (fixed-step or adaptive) fits this shape; solvers built from tensor operations are differentiable,
 * so gradients can flow through their steps.
 */
export type OdeSolver = (f: (t: number, y: Value) => Value, y0: Value, times: readonly number[]) => Value[]

/**
 * An SDE solver: the solution of dy = f(t, y) dt + g(t, y) dW (diagonal noise) at each of the increasing `times`,
 * driven by a key (any solver of `stochastic`, such as Euler-Maruyama or Milstein).
 */
export type SdeSolver = (
  f: (t: number, y: Value) => Value,
  g: (t: number, y: Value) => Value,
  y0: Value,
  times: readonly number[],
  key: Key,
) => Value[]

/** How an ODE block's gradients are computed. */
export type OdeGradient = 'backprop' | 'adjoint'

/** Options of {@link ODEBlock}. */
export interface OdeBlockOptions {
  readonly solver: OdeSolver
  /** Integration times (default [0, 1]); the output is the state at the last one. */
  readonly times?: readonly number[]
  /** Backpropagate through the solver's steps (default) or solve the adjoint equation backwards (constant memory). */
  readonly gradient?: OdeGradient
  /** Give the vector field the time as an extra last feature (default false). */
  readonly timeDependent?: boolean
  /** Return the states at every time `[len(times), ...]` instead of the last. */
  readonly trajectory?: boolean
}

const withTime = (y: Value, t: number): Value => {
  const shape = [...y.shape.slice(0, -1), 1]
  return concat([y, add(zeros(shape), t)], -1)
}

/**
 * Integrate one segment [t0, t1] of dy/dt = field(params, t, y), differentiable in y0 and the parameters by the
 * adjoint method (Chen et al. 2018): the backward pass solves the augmented system for (y, a = dL/dy, dL/dtheta)
 * from t1 back to t0 with the same solver, in reversed time.
 */
function adjointSegment(
  solver: OdeSolver,
  field: (params: Params, t: number, y: Value) => Value,
  params: Params,
  t0: number,
  t1: number,
): (y0: Value) => Value {
  const { leaves, def } = treeFlatten(params)
  const n = leaves.length
  const paramsOf = (vs: readonly Value[]): Params => treeUnflatten(def, vs) as Params
  const forward = (y0: Value, ...theta: Value[]): Value => {
    const ys = solver((t, y) => field(paramsOf(theta), t, y), y0, [t0, t1])
    return ys[ys.length - 1] as Value
  }
  const segment = customVjp({
    name: 'odeAdjoint',
    forward: (...inputs: Value[]) => forward(inputs[0] as Value, ...inputs.slice(1)),
    vjp: (g, inputs, output) => {
      const theta = inputs.slice(1)
      const ySize = sizeOf(output.shape)
      const sizes = theta.map((p) => sizeOf(p.shape))
      const total = 2 * ySize + sizes.reduce((a, b) => a + b, 0)
      const flat = (v: Value): Value => reshape(v, [-1])
      // Augmented state z = [y, a, a_theta] as one vector; reversed time s = -t runs from -t1 to -t0.
      const dynamics = (s: number, z: Value): Value => {
        const t = -s
        const y = reshape(sliceAxis(z, 0, 0, ySize), output.shape)
        const a = reshape(sliceAxis(z, 0, ySize, 2 * ySize), output.shape)
        const { out, pullback } = vjp((args: readonly Value[]) => field(paramsOf(args.slice(1)), t, args[0] as Value), [
          y,
          ...theta,
        ] as readonly Value[])
        const [dy, ...dtheta] = pullback(a) as readonly Value[]
        // dz/ds = -dz/dt = [-f, a^T df/dy, a^T df/dtheta].
        return concat([flat(neg(out)), flat(dy as Value), ...dtheta.map(flat)], 0)
      }
      const z1 = concat([flat(output), flat(g), zeros([total - 2 * ySize])], 0)
      const zs = solver(dynamics, z1, [-t1, -t0])
      const z0 = zs[zs.length - 1] as Value
      const gy = reshape(sliceAxis(z0, 0, ySize, 2 * ySize), output.shape)
      const gtheta: Value[] = []
      let at = 2 * ySize
      for (const [i, size] of sizes.entries()) {
        gtheta.push(reshape(sliceAxis(z0, 0, at, at + size), (theta[i] as Value).shape))
        at += size
      }
      return [gy, ...gtheta]
    },
  })
  return (y0) => segment(y0, ...(leaves as Value[]).slice(0, n))
}

/**
 * A neural ODE block (Chen et al. 2018): the output is the solution at the last time of dy/dt = f(y) (or f([y, t])),
 * with the vector field f a module and the integration done by any solver passed in. Gradients come from
 * backpropagation through the solver or from the adjoint method; the block's parameters are the field's (`field.*`).
 *
 * @concept neural-ode
 * @reference Chen, Rubanova, Bettencourt and Duvenaud (2018), Neural ordinary differential equations.
 * @example
 * ```ts
 * import { blocks, Linear } from 'whitebox-engine/nn'
 * import { add, mul, type Value } from 'whitebox-engine/tensor'
 * import { key } from 'whitebox-engine/random'
 * const euler = (f: (t: number, y: Value) => Value, y0: Value, ts: readonly number[]): Value[] =>
 *   ts.map((_, i) => i).reduce<Value[]>((ys, i) => (i === 0 ? [y0] : [...ys, add(ys[i - 1] as Value, mul(f(ts[i - 1] ?? 0, ys[i - 1] as Value), (ts[i] ?? 0) - (ts[i - 1] ?? 0)))]), [])
 * new blocks.ODEBlock(new Linear(2, 2, { key: key(0) }), { solver: euler }).forward([[1, 2]]).shape // => [1, 2]
 * ```
 */
export class ODEBlock extends Composite {
  readonly kind: string = 'ODEBlock'
  readonly solver: OdeSolver
  readonly times: readonly number[]
  readonly gradient: OdeGradient
  readonly timeDependent: boolean
  readonly trajectory: boolean

  constructor(field: Module, options: OdeBlockOptions) {
    super({ field })
    this.solver = options.solver
    this.times = options.times ?? [0, 1]
    if (this.times.length < 2 || this.times.some((t, i) => i > 0 && t <= (this.times[i - 1] ?? t)))
      throw new WhiteboxError('value', 'ODEBlock: times must increase')
    this.gradient = options.gradient ?? 'backprop'
    this.timeDependent = options.timeDependent ?? false
    this.trajectory = options.trajectory ?? false
  }

  /** The vector field at (t, y) for explicit parameters (`field.*`). */
  vectorField(params: Params, t: number, y: Value, ctx: ApplyContext = {}): Value {
    const field = this.part('field')
    return field.apply((params['field'] ?? {}) as Params, this.timeDependent ? withTime(y, t) : y, ctx)
  }

  /** The states at every integration time. */
  solve(params: Params, y0: Value, ctx: ApplyContext = {}): Value[] {
    if (this.gradient === 'backprop') return this.solver((t, y) => this.vectorField(params, t, y, ctx), y0, this.times)
    const ys: Value[] = [y0]
    for (let i = 1; i < this.times.length; i++) {
      const seg = adjointSegment(
        this.solver,
        (p, t, y) => this.vectorField({ field: p }, t, y, ctx),
        (params['field'] ?? {}) as Params,
        this.times[i - 1] ?? 0,
        this.times[i] ?? 1,
      )
      ys.push(seg(ys[i - 1] as Value))
    }
    return ys
  }

  apply(params: Params, x: Value, ctx: ApplyContext = {}): Value {
    const ys = this.solve(params, x, ctx)
    return this.trajectory ? stack(ys, 0) : (ys[ys.length - 1] as Value)
  }

  outputShape(s: readonly number[]): number[] {
    return this.trajectory ? [this.times.length, ...s] : [...s]
  }

  override extraRepr(): string {
    return `times=[${this.times.join(', ')}], gradient=${this.gradient}`
  }
}

/**
 * An augmented neural ODE (Dupont, Doucet and Teh 2019): the state is padded with `augmentDim` zeros before
 * integration, so trajectories can pass around each other in the extra dimensions; the field maps the augmented width
 * to itself. The output keeps the augmented features unless `strip`.
 *
 * @reference Dupont, Doucet and Teh (2019), Augmented neural ODEs.
 * @example
 * ```ts
 * import { blocks, Linear } from 'whitebox-engine/nn'
 * import { key } from 'whitebox-engine/random'
 * import type { Value } from 'whitebox-engine/tensor'
 * const still = (_f: unknown, y0: Value, ts: readonly number[]): Value[] => ts.map(() => y0)
 * new blocks.AugmentedODEBlock(new Linear(3, 3, { key: key(0) }), 1, { solver: still }).forward([[1, 2]]).shape // => [1, 3]
 * ```
 */
export class AugmentedODEBlock extends ODEBlock {
  override readonly kind = 'AugmentedODEBlock'
  readonly augmentDim: number
  readonly strip: boolean

  constructor(field: Module, augmentDim: number, options: OdeBlockOptions & { readonly strip?: boolean }) {
    super(field, options)
    this.augmentDim = augmentDim
    this.strip = options.strip ?? false
  }

  override apply(params: Params, x: Value, ctx: ApplyContext = {}): Value {
    const width = x.shape[x.shape.length - 1] ?? 0
    const padded = concat([x, zeros([...x.shape.slice(0, -1), this.augmentDim])], -1)
    const y = super.apply(params, padded, ctx)
    return this.strip ? sliceAxis(y, -1, 0, width) : y
  }

  override outputShape(s: readonly number[]): number[] {
    const out = [...s.slice(0, -1), (s[s.length - 1] ?? 0) + (this.strip ? 0 : this.augmentDim)]
    return this.trajectory ? [this.times.length, ...out] : out
  }
}

/**
 * A second-order neural ODE (Norcliffe et al. 2020): x'' = f([x, x']), integrated as the first-order system of
 * position and velocity (the field maps `[..., 2 D]` to the acceleration `[..., D]`); the initial velocity is zero
 * unless given. The output is the position at the last time.
 *
 * @reference Norcliffe, Bodnar, Day, Simidjievski and Lio (2020), On second order behaviour in augmented neural ODEs.
 * @example
 * ```ts
 * import { blocks, Linear } from 'whitebox-engine/nn'
 * import { key } from 'whitebox-engine/random'
 * import type { Value } from 'whitebox-engine/tensor'
 * const still = (_f: unknown, y0: Value, ts: readonly number[]): Value[] => ts.map(() => y0)
 * new blocks.SecondOrderODEBlock(new Linear(4, 2, { key: key(0) }), { solver: still }).forward([[1, 2]]).shape // => [1, 2]
 * ```
 */
export class SecondOrderODEBlock extends ODEBlock {
  override readonly kind = 'SecondOrderODEBlock'

  override vectorField(params: Params, t: number, state: Value, ctx: ApplyContext = {}): Value {
    const d = (state.shape[state.shape.length - 1] ?? 0) / 2
    const v = sliceAxis(state, -1, d, 2 * d)
    const acc = super.vectorField(params, t, state, ctx)
    return concat([v, acc], -1)
  }

  override apply(
    params: Params,
    x: Value | { readonly x: TensorLike; readonly v: TensorLike },
    ctx: ApplyContext = {},
  ): Value {
    const pos = 'x' in (x as object) ? asTensor((x as { x: TensorLike }).x) : (x as Value)
    const vel = 'v' in (x as object) ? asTensor((x as { v: TensorLike }).v) : mul(pos, 0)
    const d = pos.shape[pos.shape.length - 1] ?? 0
    const ys = this.solve(params, concat([pos, vel], -1), ctx)
    const last = (y: Value): Value => sliceAxis(y, -1, 0, d)
    return this.trajectory ? stack(ys.map(last), 0) : last(ys[ys.length - 1] as Value)
  }
}

/** Options of {@link CDEBlock}. */
export interface CdeBlockOptions {
  readonly solver: OdeSolver
  /** Observation times (default 0, 1, 2, ...). */
  readonly times?: readonly number[]
  /** Return the hidden state at every observation time `[B, T, H]` instead of the last `[B, H]`. */
  readonly trajectory?: boolean
}

/**
 * A neural controlled differential equation (Kidger et al. 2020): dz = f(z) dX, the hidden state driven by the
 * piecewise-linear interpolation of an observed path X `[B, T, C]`; z0 = initial(X_0), and the field maps z `[B, H]` to
 * a matrix `[B, H C]`. Integrated segment by segment (dX/dt is constant on each), so irregular sampling is natural.
 *
 * @reference Kidger, Morrill, Foster and Lyons (2020), Neural controlled differential equations for irregular time
 *   series.
 * @example
 * ```ts
 * import { blocks, Linear } from 'whitebox-engine/nn'
 * import { key } from 'whitebox-engine/random'
 * import type { Value } from 'whitebox-engine/tensor'
 * const still = (_f: unknown, y0: Value, ts: readonly number[]): Value[] => ts.map(() => y0)
 * new blocks.CDEBlock(new Linear(2, 3, { key: key(0) }), new Linear(3, 6, { key: key(1) }), { solver: still }).forward([[[0, 1], [1, 2]]]).shape // => [1, 3]
 * ```
 */
export class CDEBlock extends Composite {
  readonly kind = 'CDEBlock'
  readonly solver: OdeSolver
  readonly times: readonly number[] | undefined
  readonly trajectory: boolean

  constructor(initial: Module, field: Module, options: CdeBlockOptions) {
    super({ initial, field })
    this.solver = options.solver
    this.times = options.times
    this.trajectory = options.trajectory ?? false
  }

  apply(params: Params, path: Value, ctx: ApplyContext = {}): Value {
    const [b = 1, t = 1, c = 1] = path.shape
    const times = this.times ?? Array.from({ length: t }, (_, i) => i)
    if (times.length !== t) throw new WhiteboxError('shape', 'CDEBlock: one time per observation')
    const p = this.parts(params, ctx)
    const at = (k: number): Value => reshape(sliceAxis(path, 1, k, k + 1), [b, c])
    let z = p.run<Value, Value>('initial', at(0))
    const h = z.shape[1] ?? 1
    const field = this.part('field')
    const fp = (params['field'] ?? {}) as Params
    const states: Value[] = [z]
    for (let k = 0; k + 1 < t; k++) {
      const t0 = times[k] ?? k
      const t1 = times[k + 1] ?? k + 1
      const slope = mul(sub(at(k + 1), at(k)), 1 / (t1 - t0))
      const rhs = (_t: number, y: Value): Value =>
        reshape(matmul(reshape(field.apply(fp, y, ctx), [b, h, c]), reshape(slope, [b, c, 1])), [b, h])
      const ys = this.solver(rhs, z, [t0, t1])
      z = ys[ys.length - 1] as Value
      states.push(z)
    }
    p.done()
    return this.trajectory ? stack(states, 1) : z
  }

  outputShape(s: readonly number[]): number[] {
    const h = (this.part('initial') as Module).outputShape([s[0] ?? 1, s[2] ?? 1])[1] ?? 1
    return this.trajectory ? [s[0] ?? 1, s[1] ?? 1, h] : [s[0] ?? 1, h]
  }
}

/** Options of {@link SDEBlock}. */
export interface SdeBlockOptions {
  readonly solver: SdeSolver
  readonly times?: readonly number[]
  readonly trajectory?: boolean
}

/**
 * A neural SDE block (Tzen and Raginsky 2019; Li et al. 2020): dy = f(y) dt + g(y) dW with drift and diagonal
 * diffusion modules, integrated by the SDE solver passed in with the key in the apply context.
 *
 * @reference Li, Wong, Chen and Duvenaud (2020), Scalable gradients for stochastic differential equations.
 * @example
 * ```ts
 * import { blocks, Linear } from 'whitebox-engine/nn'
 * import { key } from 'whitebox-engine/random'
 * import type { Value } from 'whitebox-engine/tensor'
 * const still = (_f: unknown, _g: unknown, y0: Value, ts: readonly number[]): Value[] => ts.map(() => y0)
 * new blocks.SDEBlock(new Linear(2, 2, { key: key(0) }), new Linear(2, 2, { key: key(1) }), { solver: still }).forward([[1, 2]], { key: key(2) }).shape // => [1, 2]
 * ```
 */
export class SDEBlock extends Composite {
  readonly kind = 'SDEBlock'
  readonly solver: SdeSolver
  readonly times: readonly number[]
  readonly trajectory: boolean

  constructor(drift: Module, diffusion: Module, options: SdeBlockOptions) {
    super({ drift, diffusion })
    this.solver = options.solver
    this.times = options.times ?? [0, 1]
    this.trajectory = options.trajectory ?? false
  }

  apply(params: Params, x: Value, ctx: ApplyContext = {}): Value {
    if (ctx.key === undefined) throw new WhiteboxError('value', 'SDEBlock: needs a key in the apply context')
    const p = this.parts(params, ctx)
    const f = (_t: number, y: Value): Value => p.run<Value, Value>('drift', y)
    const g = (_t: number, y: Value): Value => p.run<Value, Value>('diffusion', y)
    const ys = this.solver(f, g, x, this.times, ctx.key)
    p.done()
    return this.trajectory ? stack(ys, 0) : (ys[ys.length - 1] as Value)
  }

  outputShape(s: readonly number[]): number[] {
    return this.trajectory ? [this.times.length, ...s] : [...s]
  }
}
