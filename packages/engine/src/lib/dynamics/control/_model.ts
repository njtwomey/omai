// Discrete-time control models and their derivatives along trajectories (the machinery behind iLQR and MPC).
import { grad, jacfwd, jvp } from '../../autodiff/index.ts'
import {
  fromStorage,
  mul,
  reshape,
  sliceAxis,
  sum,
  type Tensor,
  toFlat,
  type Value,
  zeros,
} from '../../tensor/index.ts'
import type { DiscreteModel } from './trajectory.ts'

/** Per-step derivatives of the dynamics and the cost along a trajectory. */
export interface StepDerivatives {
  readonly fx: number[][][]
  readonly fu: number[][][]
  readonly lx: number[][]
  readonly lu: number[][]
  readonly lxx: number[][][]
  readonly luu: number[][][]
  readonly lux: number[][][]
}

export const f64 = (xs: readonly number[], shape: number[]): Tensor =>
  fromStorage(Float64Array.from(xs), shape, 'float64')

const basisRows = (T: number, d: number, i: number): Tensor => {
  const data = new Float64Array(T * d)
  for (let t = 0; t < T; t++) data[t * d + i] = 1
  return fromStorage(data, [T, d], 'float64')
}

function scatterColumn(out: number[][][], flat: readonly number[], d: number, j: number): void {
  for (const [t, rows] of out.entries()) for (let r = 0; r < d; r++) (rows[r] as number[])[j] = flat[t * d + r] ?? 0
}

const blank = (T: number, r: number, c: number): number[][][] =>
  Array.from({ length: T }, () => Array.from({ length: r }, () => new Array<number>(c).fill(0)))

/** The derivatives along a trajectory, one batched forward-mode pass per coordinate for all steps at once. */
export function stepDerivatives(
  model: DiscreteModel,
  states: readonly (readonly number[])[],
  actions: readonly (readonly number[])[],
): StepDerivatives {
  const T = states.length
  const n = model.stateSize
  const m = model.actionSize
  const X = f64(states.flat(), [T, n])
  const U = f64(actions.flat(), [T, m])
  const fx = blank(T, n, n)
  const fu = blank(T, n, m)
  for (let i = 0; i < n; i++) {
    scatterColumn(fx, toFlat(jvp((x: Value) => model.dynamics(x, U), X, basisRows(T, n, i)).tangent), n, i)
  }
  for (let j = 0; j < m; j++) {
    scatterColumn(fu, toFlat(jvp((u: Value) => model.dynamics(X, u), U, basisRows(T, m, j)).tangent), n, j)
  }
  type Z = { readonly x: Value; readonly u: Value }
  const g = grad((z: Z) => sum(model.cost(z.x, z.u))) as (z: Z) => Z
  const gz = g({ x: X, u: U })
  const gx = toFlat(gz.x)
  const gu = toFlat(gz.u)
  const lx = Array.from({ length: T }, (_, t) => gx.slice(t * n, (t + 1) * n))
  const lu = Array.from({ length: T }, (_, t) => gu.slice(t * m, (t + 1) * m))
  const lxx = blank(T, n, n)
  const luu = blank(T, m, m)
  const lux = blank(T, m, n)
  const zx = zeros([T, n])
  const zu = zeros([T, m])
  for (let i = 0; i < n; i++) {
    const t = jvp(g, { x: X, u: U }, { x: basisRows(T, n, i), u: zu }).tangent
    scatterColumn(lxx, toFlat(t.x), n, i)
    scatterColumn(lux, toFlat(t.u), m, i)
  }
  for (let j = 0; j < m; j++) {
    const t = jvp(g, { x: X, u: U }, { x: zx, u: basisRows(T, m, j) }).tangent
    scatterColumn(luu, toFlat(t.u), m, j)
  }
  return { fx, fu, lx, lu, lxx, luu, lux }
}

/** The Hessian of v . f(x, u) over (x, u) at one step (the second-order dynamics term of DDP), `[(n+m)^2]`. */
export function weightedDynamicsHessian(
  model: DiscreteModel,
  x: readonly number[],
  u: readonly number[],
  v: readonly number[],
): Float64Array {
  const n = model.stateSize
  const m = model.actionSize
  const w = f64(v, [n])
  const scalar = (z: Value): Value => {
    const xs = reshape(sliceAxis(z, 0, 0, n), [1, n])
    const us = reshape(sliceAxis(z, 0, n, n + m), [1, m])
    return sum(mul(reshape(model.dynamics(xs, us), [n]), w))
  }
  const h = jacfwd(grad(scalar))(f64([...x, ...u], [n + m]))
  return Float64Array.from(toFlat(h))
}

/** Roll out from a state under actions: the states visited (start first) and the total cost. */
export function rollout(
  model: DiscreteModel,
  state: readonly number[],
  actions: readonly (readonly number[])[],
): { readonly states: number[][]; readonly cost: number } {
  let x: Value = f64(state, [model.stateSize])
  const states = [[...state]]
  let cost = 0
  for (const a of actions) {
    const u = f64(a, [model.actionSize])
    cost += toFlat(model.cost(x, u))[0] ?? 0
    x = model.dynamics(x, u)
    states.push(toFlat(x))
  }
  if (model.terminalCost !== undefined) cost += toFlat(model.terminalCost(x))[0] ?? 0
  return { states, cost }
}
