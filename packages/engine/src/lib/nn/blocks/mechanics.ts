/**
 * Blocks that conserve what they learn: the Hamiltonian neural network (Greydanus et al. 2019), a network for the
 * energy `H(q, p)` whose symplectic gradient `(dq/dt, dp/dt) = (dH/dp, -dH/dq)` is the vector field, so the learned
 * energy is conserved along the flow by construction; and the Lagrangian neural network (Cranmer et al. 2020), a
 * network for `L(q, qdot)` whose Euler-Lagrange equation gives the acceleration, `qddot = (d2L/dqdot2)^-1
 * (dL/dq - d2L/dqdot dq qdot)`. Both need derivatives of the network with respect to its inputs inside the loss, so
 * training them is second- and third-order differentiation of the primitives.
 */

import { grad, jacobian } from '../../autodiff/index.ts'
import { OmaiError } from '../../base/index.ts'
import { solve } from '../../numerics/linalg/index.ts'
import { foldIn, type Key } from '../../random/index.ts'
import {
  asTensor,
  concat,
  matmul,
  neg,
  reshape,
  sliceAxis,
  stack,
  sub,
  sum,
  type TensorLike,
  type Value,
} from '../../tensor/index.ts'
import { Softplus } from '../module/activations.ts'
import { Composite, mlp } from '../module/containers.ts'
import { type ApplyContext, initKey, type Module, type Params } from '../module/module.ts'
import type { OdeSolver } from './continuous.ts'

/** Options of {@link HamiltonianNet}. */
export interface HamiltonianOptions {
  /** The energy network `[2 n] -> [1]` (default: a perceptron with softplus activations over `hidden` widths). */
  readonly energy?: Module
  /** Hidden widths of the default energy network (default `[64, 64]`). */
  readonly hidden?: readonly number[]
  readonly key?: Key
}

/**
 * A Hamiltonian neural network on the state `(q, p)` of `n` degrees of freedom (`q` the first `n` entries). The module
 * computes the time derivative `(dH/dp, -dH/dq)` of a state `[B, 2 n]`; {@link HamiltonianNet.energy} is the learned
 * energy and {@link HamiltonianNet.flow} integrates it with any ODE solver. Because the field is the symplectic
 * gradient, `dH/dt = 0` along the exact flow, so energy drifts only by the integrator's error.
 *
 * @reference Greydanus, Dzamba and Yosinski (2019), Hamiltonian neural networks.
 * @example
 * ```ts
 * import { HamiltonianNet } from 'omai-engine/nn/blocks'
 * import { key } from 'omai-engine/random'
 * new HamiltonianNet(1, { hidden: [16], key: key(0) }).forward([[0.5, 0]]).shape // => [1, 2]
 * ```
 */
export class HamiltonianNet extends Composite {
  readonly kind = 'HamiltonianNet'
  readonly freedom: number

  /**
   * @param freedom - The number `n` of position coordinates (the state has `2 n` entries).
   * @param options - The energy network or its widths, and the key.
   */
  constructor(freedom: number, options: HamiltonianOptions = {}) {
    const key = initKey('HamiltonianNet', options.key)
    super({
      energy:
        options.energy ??
        mlp([2 * freedom, ...(options.hidden ?? [64, 64]), 1], {
          activation: () => new Softplus(),
          key: foldIn(key, 0),
        }),
    })
    this.freedom = freedom
  }

  /** The learned energy of each state row `[B]`. */
  energy(params: Params, state: TensorLike, ctx: ApplyContext = {}): Value {
    return reshape(this.part('energy').apply((params['energy'] ?? {}) as Params, asTensor(state), ctx), [-1])
  }

  /** The gradient of the energy with respect to the state `[B, 2 n]`: `(dH/dq, dH/dp)`. */
  energyGradient(params: Params, state: TensorLike, ctx: ApplyContext = {}): Value {
    return grad((s: Value) => sum(this.energy(params, s, ctx)))(asTensor(state)) as Value
  }

  apply(params: Params, state: Value, ctx: ApplyContext = {}): Value {
    const n = this.freedom
    const g = this.energyGradient(params, state, ctx)
    return concat([sliceAxis(g, -1, n, 2 * n), neg(sliceAxis(g, -1, 0, n))], -1)
  }

  /**
   * Integrate the learned dynamics from `state` and return the states at `times`.
   *
   * @param params - The parameters.
   * @param state - The initial state `[B, 2 n]`.
   * @param times - Increasing times starting at the initial one.
   * @param solver - An ODE solver.
   * @returns One state per time.
   */
  flow(params: Params, state: TensorLike, times: readonly number[], solver: OdeSolver): Value[] {
    return solver((_t, y) => this.apply(params, y), asTensor(state), times)
  }

  outputShape(s: readonly number[]): number[] {
    return [...s]
  }
}

/** Options of {@link LagrangianNet}. */
export interface LagrangianOptions {
  /** The Lagrangian network `[2 n] -> [1]` on `(q, qdot)` (default: a perceptron with softplus activations). */
  readonly lagrangian?: Module
  /** Hidden widths of the default network (default `[64, 64]`). */
  readonly hidden?: readonly number[]
  readonly key?: Key
}

/**
 * A Lagrangian neural network on `(q, qdot)` of `n` degrees of freedom: the network gives `L(q, qdot)` and the module
 * computes the time derivative `(qdot, qddot)` of a state `[B, 2 n]` from the Euler-Lagrange equation, solving with
 * the Hessian of `L` in the velocities. A network that makes this Hessian singular has no acceleration; with a
 * kinetic term positive definite in `qdot` (as for mechanical systems) it is well defined.
 *
 * @reference Cranmer, Greydanus, Hoyer, Battaglia, Spergel and Ho (2020), Lagrangian neural networks.
 * @example
 * ```ts
 * import { LagrangianNet } from 'omai-engine/nn/blocks'
 * import { Lambda } from 'omai-engine/nn'
 * import { mul, sliceAxis, square, sub, toArray } from 'omai-engine/tensor'
 * // L = (qdot^2 - q^2) / 2: a unit harmonic oscillator, qddot = -q.
 * const L = new Lambda((x) => mul(sub(square(sliceAxis(x, -1, 1, 2)), square(sliceAxis(x, -1, 0, 1))), 0.5))
 * toArray(new LagrangianNet(1, { lagrangian: L }).accelerations({}, [[1, 0]])) // => [[-1]]
 * ```
 */
export class LagrangianNet extends Composite {
  readonly kind = 'LagrangianNet'
  readonly freedom: number

  /**
   * @param freedom - The number `n` of position coordinates (the state has `2 n` entries).
   * @param options - The Lagrangian network or its widths, and the key.
   */
  constructor(freedom: number, options: LagrangianOptions = {}) {
    const key = initKey('LagrangianNet', options.key)
    super({
      lagrangian:
        options.lagrangian ??
        mlp([2 * freedom, ...(options.hidden ?? [64, 64]), 1], {
          activation: () => new Softplus(),
          key: foldIn(key, 0),
        }),
    })
    this.freedom = freedom
  }

  /** The Lagrangian of each state row `[B]`. */
  lagrangian(params: Params, state: TensorLike, ctx: ApplyContext = {}): Value {
    return reshape(this.part('lagrangian').apply((params['lagrangian'] ?? {}) as Params, asTensor(state), ctx), [-1])
  }

  /** The accelerations `qddot` of each state row `[B, n]`, by the Euler-Lagrange equation. */
  accelerations(params: Params, state: TensorLike, ctx: ApplyContext = {}): Value {
    const n = this.freedom
    const s = asTensor(state)
    if (s.shape[s.shape.length - 1] !== 2 * n)
      throw new OmaiError('shape', `LagrangianNet: the state needs ${2 * n} entries`)
    const rows = s.shape[0] ?? 1
    const single = (x: Value): Value => sum(this.lagrangian(params, reshape(x, [1, 2 * n]), ctx))
    const out: Value[] = []
    for (let r = 0; r < rows; r++) {
      const x = reshape(sliceAxis(s, 0, r, r + 1), [2 * n])
      const dL = (z: Value): Value => grad(single)(z) as Value
      const hessianOf = jacobian(dL)(x)
      const H = hessianOf
      const gradient = dL(x)
      const dLdq = sliceAxis(gradient, 0, 0, n)
      const mixed = sliceAxis(sliceAxis(H, 0, n, 2 * n), 1, 0, n)
      const velocity = sliceAxis(x, 0, n, 2 * n)
      const mass = sliceAxis(sliceAxis(H, 0, n, 2 * n), 1, n, 2 * n)
      const rhs = reshape(sub(dLdq, reshape(matmul(mixed, reshape(velocity, [n, 1])), [n])), [n, 1])
      out.push(reshape(solve(mass, rhs), [n]))
    }
    return stack(out, 0)
  }

  apply(params: Params, state: Value, ctx: ApplyContext = {}): Value {
    const n = this.freedom
    return concat([sliceAxis(state, -1, n, 2 * n), this.accelerations(params, state, ctx)], -1)
  }

  /**
   * Integrate the learned dynamics from `state` and return the states at `times`.
   *
   * @param params - The parameters.
   * @param state - The initial state `[B, 2 n]`.
   * @param times - Increasing times starting at the initial one.
   * @param solver - An ODE solver.
   * @returns One state per time.
   */
  flow(params: Params, state: TensorLike, times: readonly number[], solver: OdeSolver): Value[] {
    return solver((_t, y) => this.apply(params, y), asTensor(state), times)
  }

  outputShape(s: readonly number[]): number[] {
    return [...s]
  }
}
