import { describe, expect, it } from 'vitest'
import { foldIn, key, uniform } from '../random/index.ts'
import { tensor, toArray, toFlat } from '../tensor/index.ts'
import {
  acrobotDynamics,
  cartPoleDynamics,
  continuousMountainCarDynamics,
  pendulumDynamics,
} from './classic-control.ts'
import {
  cartDoublePendulumDynamics,
  cartDoublePendulumParameters,
  doubleIntegratorDynamics,
  doubleIntegratorParameters,
  planarQuadrotorDynamics,
  planarQuadrotorParameters,
  pointMassDynamics,
  pointMassParameters,
  reacher2dDynamics,
  reacher2dParameters,
} from './continuous-control.ts'
import {
  acrobotModel,
  type ControlModel,
  cartDoublePendulumModel,
  cartPoleModel,
  controlModelEnv,
  doubleIntegratorModel,
  linearize,
  modelStepError,
  mountainCarContinuousModel,
  pendulumModel,
  planarQuadrotorModel,
  pointMassModel,
  reacher2dModel,
  rolloutModel,
  sequenceCosts,
  trajectoryDerivatives,
} from './models.ts'

const draw = (seed: number, lo: readonly number[], hi: readonly number[]): number[] => {
  const u = uniform(key(seed), [lo.length]).data
  return lo.map((l, i) => l + ((hi[i] ?? 0) - l) * (u[i] ?? 0))
}

interface Pair {
  readonly name: string
  readonly model: ControlModel
  readonly env: (s: readonly number[], a: readonly number[]) => readonly number[]
  readonly state: readonly [readonly number[], readonly number[]]
  readonly action: readonly [readonly number[], readonly number[]] | 'discrete'
  readonly tol?: number
}

const pm = pointMassParameters()
const re = reacher2dParameters()
const di = doubleIntegratorParameters()
const qu = planarQuadrotorParameters()
const cdp = cartDoublePendulumParameters()

const pairs: readonly Pair[] = [
  {
    name: 'CartPole',
    model: cartPoleModel(),
    env: (s, a) => cartPoleDynamics(s as never, (a[0] ?? 0) > 0 ? 1 : 0),
    state: [
      [-1, -1, -0.2, -1],
      [1, 1, 0.2, 1],
    ],
    action: 'discrete',
  },
  {
    name: 'Pendulum',
    model: pendulumModel(),
    env: (s, a) => pendulumDynamics(s as never, a[0] ?? 0).state,
    state: [
      [-3, -3],
      [3, 3],
    ],
    action: [[-2], [2]],
  },
  {
    // gymnasium stores this state in float32: the model agrees to float32 precision.
    name: 'MountainCarContinuous',
    model: mountainCarContinuousModel(),
    env: (s, a) => continuousMountainCarDynamics(s as never, a[0] ?? 0),
    state: [
      [-1, -0.03],
      [0.4, 0.03],
    ],
    action: [[-1], [1]],
    tol: 1e-6,
  },
  {
    name: 'Acrobot',
    model: acrobotModel(),
    env: (s, a) => acrobotDynamics(s as never, Math.round(a[0] ?? 0) + 1),
    state: [
      [-0.5, -0.5, -1, -1],
      [0.5, 0.5, 1, 1],
    ],
    action: 'discrete',
  },
  {
    name: 'PointMass',
    model: pointMassModel(),
    env: (s, a) => pointMassDynamics([...s, 0, 0], a, pm).slice(0, 4),
    state: [
      [-0.2, -0.2, -0.1, -0.1],
      [0.2, 0.2, 0.1, 0.1],
    ],
    action: [
      [-1, -1],
      [1, 1],
    ],
  },
  {
    name: 'Reacher2D',
    model: reacher2dModel(),
    env: (s, a) => reacher2dDynamics([...s, 0, 0], a, re).slice(0, 4),
    state: [
      [-3, -3, -5, -5],
      [3, 3, 5, 5],
    ],
    action: [
      [-1, -1],
      [1, 1],
    ],
  },
  {
    name: 'DoubleIntegrator',
    model: doubleIntegratorModel(),
    env: (s, a) => doubleIntegratorDynamics(s as never, a, di),
    state: [
      [-2, -2],
      [2, 2],
    ],
    action: [[-10], [10]],
  },
  {
    name: 'PlanarQuadrotor',
    model: planarQuadrotorModel(),
    env: (s, a) => planarQuadrotorDynamics([...s, 0, 0], a, qu).slice(0, 6),
    state: [
      [-1, -1, -1, -2, -2, -3],
      [1, 1, 1, 2, 2, 3],
    ],
    action: [
      [0, 0],
      [qu.maxThrust, qu.maxThrust],
    ],
  },
  {
    name: 'CartDoublePendulum',
    model: cartDoublePendulumModel(),
    env: (s, a) => cartDoublePendulumDynamics(s, a, cdp),
    state: [
      [-1, -3, -3, -2, -5, -5],
      [1, 3, 3, 2, 5, 5],
    ],
    action: [[-20], [20]],
  },
]

describe('differentiable models of the control environments', () => {
  for (const pair of pairs) {
    it(`steps ${pair.name} as its environment does`, () => {
      const discrete = pair.model.discreteActions ?? []
      for (let i = 0; i < 20; i++) {
        const s = draw(2 * i, pair.state[0], pair.state[1])
        const a = pair.action === 'discrete' ? (discrete[i % discrete.length] ?? [0]) : draw(2 * i + 1, ...pair.action)
        expect(modelStepError(pair.model, pair.env, s, a)).toBeLessThan(pair.tol ?? 1e-12)
      }
    })

    it(`linearises ${pair.name} like central differences`, () => {
      const s = draw(99, pair.state[0], pair.state[1])
      const a = pair.action === 'discrete' ? (pair.model.equilibrium?.action ?? [0]) : draw(98, ...pair.action)
      const { A, B } = linearize(pair.model, s, a)
      const f = (x: readonly number[], u: readonly number[]): number[] =>
        toFlat(pair.model.dynamics(tensor([...x]), tensor([...u])))
      const h = 1e-6
      const a2 = toArray(A) as number[][]
      const b2 = toArray(B) as number[][]
      for (let j = 0; j < s.length; j++) {
        const up = f(
          s.map((v, k) => (k === j ? v + h : v)),
          a,
        )
        const dn = f(
          s.map((v, k) => (k === j ? v - h : v)),
          a,
        )
        for (let i = 0; i < s.length; i++) expect(a2[i]?.[j]).toBeCloseTo(((up[i] ?? 0) - (dn[i] ?? 0)) / (2 * h), 5)
      }
      for (let j = 0; j < a.length; j++) {
        const up = f(
          s,
          a.map((v, k) => (k === j ? v + h : v)),
        )
        const dn = f(
          s,
          a.map((v, k) => (k === j ? v - h : v)),
        )
        for (let i = 0; i < s.length; i++) expect(b2[i]?.[j]).toBeCloseTo(((up[i] ?? 0) - (dn[i] ?? 0)) / (2 * h), 5)
      }
    })
  }

  it('keeps every equilibrium fixed', () => {
    for (const pair of pairs) {
      const eq = pair.model.equilibrium
      if (eq === undefined) continue
      const next = toFlat(pair.model.dynamics(tensor([...eq.state]), tensor([...eq.action])))
      for (const [i, v] of next.entries()) expect(Math.abs(v - (eq.state[i] ?? 0))).toBeLessThan(1e-12)
    }
  })

  it('differentiates whole trajectories at once, matching per-step linearisations and the cost derivatives', () => {
    const m = planarQuadrotorModel({ goal: [0.5, -0.5] })
    const states = [0, 1, 2].map((i) => draw(10 + i, [-1, -1, -1, -1, -1, -1], [1, 1, 1, 1, 1, 1]))
    const actions = [0, 1, 2].map((i) => draw(20 + i, [0, 0], [9, 9]))
    const d = trajectoryDerivatives(m, states, actions)
    for (const [t, s] of states.entries()) {
      const { A, B } = linearize(m, s, actions[t] ?? [])
      const a = toArray(A) as number[][]
      const b = toArray(B) as number[][]
      for (let i = 0; i < 6; i++) {
        for (let j = 0; j < 6; j++) expect(d.fx[t]?.[i]?.[j]).toBeCloseTo(a[i]?.[j] ?? 0, 12)
        for (let j = 0; j < 2; j++) expect(d.fu[t]?.[i]?.[j]).toBeCloseTo(b[i]?.[j] ?? 0, 12)
      }
    }
    // The quadrotor cost is quadratic: its Hessian is constant and diagonal.
    expect(d.luu[0]).toEqual([
      [0.02, 0],
      [0, 0.02],
    ])
    expect(d.lxx[2]?.[2]?.[2]).toBeCloseTo(0.2, 14)
    expect(d.lux[1]).toEqual([
      [0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0],
    ])
    const s0 = states[0] ?? []
    expect(d.lx[0]?.[0]).toBeCloseTo(2 * ((s0[0] ?? 0) - 0.5), 12)
    // Pendulum: the cost Hessian in the angle is 2 away from the wrap.
    const p = trajectoryDerivatives(pendulumModel(), [[0.5, 0.2]], [[1]])
    expect(p.lxx[0]?.[0]?.[0]).toBeCloseTo(2, 12)
    expect(p.luu[0]?.[0]?.[0]).toBeCloseTo(0.002, 12)
  })

  it('scores batches of plans as sequential rollouts do, and runs as an environment', () => {
    const m = cartDoublePendulumModel()
    const s = [0.1, 0.05, -0.05, 0, 0, 0]
    const plans = [0, 1, 2].map((i) =>
      Array.from({ length: 5 }, (_, t) => [toFlat(uniform(foldIn(key(i), t), [1], { low: -5, high: 5 }))[0] ?? 0]),
    )
    const batch = sequenceCosts(m, s, plans)
    for (const [i, plan] of plans.entries()) expect(batch[i]).toBeCloseTo(rolloutModel(m, s, plan).cost, 10)
    const env = controlModelEnv(m)
    const r = env.step(s, [1], key(0))
    expect(r.reward).toBeCloseTo(-rolloutModel(m, s, [[1]]).cost, 12)
    expect(env.reset(key(0)).state).toEqual([0, 0, 0, 0, 0, 0])
  })
})
