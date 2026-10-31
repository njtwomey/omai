import { describe, expect, it } from 'vitest'
import { grad } from '../autodiff/index.ts'
import { add, cos, item, mul, sub, sum, tensor, toFlat, type Value } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { fieldDivergence } from './fields.ts'
import {
  doublePendulumEnergy,
  doublePendulumField,
  doublePendulumFlipTime,
  doublePendulumHamiltonian,
  doublePendulumLagrangian,
  hamiltonianField,
  hamiltonianOfLagrangian,
  lagrangianEnergy,
  lagrangianField,
} from './mechanics.ts'
import { rosslerField, rosslerFixedPoints, thomasField } from './named-systems.ts'
import { flowAlgorithm } from './ode/flow.ts'

const PARAMS = { m1: 1.3, m2: 0.7, l1: 0.9, l2: 1.4, g: 9.81 }
const STATES = [
  [0.4, -0.3, 0.2, 0.5],
  [2.5, 1.9, -1.1, 0.7],
  [-1.2, 2.8, 3.0, -2.0],
]

describe('the Euler-Lagrange equations by autodiff', () => {
  it('derives the oscillator and the pendulum from their Lagrangians', () => {
    const osc = lagrangianField((q, qd) => mul(sum(sub(mul(qd, qd), mul(q, q))), 0.5), 1)
    expect(toFlat(osc(0, tensor([2, 3])))).toEqual([3, -2])
    // A pendulum of length l: L = l^2 w^2 / 2 + g l cos(theta), so theta'' = -(g / l) sin(theta).
    const l = 1.7
    const g = 9.81
    const pend = lagrangianField((q, qd) => sum(add(mul(mul(qd, qd), 0.5 * l * l), mul(cos(q), g * l))), 1)
    const out = toFlat(pend(0, tensor([0.8, 0.3])))
    expect(out[0]).toBeCloseTo(0.3, 14)
    expect(out[1]).toBeCloseTo(-(g / l) * Math.sin(0.8), 12)
  })

  it('derives the double pendulum’s accelerations: the closed form is what the derivation gives', () => {
    const derived = lagrangianField(doublePendulumLagrangian(PARAMS), 2)
    const closed = doublePendulumField(PARAMS)
    for (const s of STATES) {
      const a = toFlat(derived(0, tensor(s)))
      const b = toFlat(closed(0, tensor(s)))
      for (let i = 0; i < 4; i++) expect(a[i]).toBeCloseTo(b[i] as number, 9)
    }
  })

  it('has the energy function equal to the kinetic plus potential energy', () => {
    const E = lagrangianEnergy(doublePendulumLagrangian(PARAMS), 2)
    for (const s of STATES) expect(E(s)).toBeCloseTo(doublePendulumEnergy(s, PARAMS), 10)
    expect(doublePendulumEnergy([0, 0, 0, 0])).toBeCloseTo(-3 * 9.81, 12)
  })

  it('conserves the energy along the flow, to the order of the integrator', () => {
    const drift = (dt: number): number => {
      const steps = Math.round(5 / dt)
      const alg = flowAlgorithm(doublePendulumField(PARAMS), [1.2, 0.5, 0, 0], { dt, steps })
      const end = runAlgorithm(alg, { maxSteps: steps }).state
      return Math.abs(doublePendulumEnergy(end.y, PARAMS) - doublePendulumEnergy([1.2, 0.5, 0, 0], PARAMS))
    }
    expect(drift(0.01)).toBeLessThan(1e-5)
    expect(drift(0.02) / drift(0.01)).toBeGreaterThan(8)
  })
})

describe('the Legendre transform', () => {
  it('gives a Hamiltonian that equals the energy at the conjugate momenta, and Hamilton’s equations that agree with the Lagrangian flow', () => {
    const L = doublePendulumLagrangian(PARAMS)
    const { H, field } = doublePendulumHamiltonian(PARAMS)
    for (const s of STATES) {
      const q = tensor(s.slice(0, 2))
      const qd = tensor(s.slice(2))
      const p = grad((v: Value) => L(q, v))(qd)
      expect(item(H(q, p))).toBeCloseTo(doublePendulumEnergy(s, PARAMS), 9)
      // dq/dt of Hamilton's equations is the velocity the state started with.
      const f = toFlat(field(0, tensor([...s.slice(0, 2), ...toFlat(p)])))
      expect(f[0]).toBeCloseTo(s[2] as number, 9)
      expect(f[1]).toBeCloseTo(s[3] as number, 9)
    }
  })

  it('makes Hamilton’s equations the right field for a simple example', () => {
    const f = hamiltonianField((q, p) => mul(sum(add(mul(q, q), mul(p, p))), 0.5), 1)
    expect(toFlat(f(0, tensor([2, 3])))).toEqual([3, -2])
    const H = hamiltonianOfLagrangian((q, qd) => mul(sum(sub(mul(qd, qd), mul(q, q))), 0.5))
    expect(item(H(tensor([2]), tensor([3])))).toBeCloseTo(6.5, 14)
  })
})

describe('the double pendulum’s motion', () => {
  const separation = (a: number[], b: number[], t: number): number => {
    const steps = Math.round(t / 0.005)
    const run = (s: number[]) =>
      runAlgorithm(flowAlgorithm(doublePendulumField(), s, { dt: 0.005, steps }), { maxSteps: steps }).state.y
    const ya = run(a)
    const yb = run(b)
    return Math.hypot(...Array.from(ya).map((v, i) => v - (yb[i] as number)))
  }

  it('amplifies a tiny difference when the swings are large and not when they are small', () => {
    const eps = 1e-9
    const chaotic = separation([2.5, 2.0, 0, 0], [2.5 + eps, 2.0, 0, 0], 15) / eps
    const regular = separation([0.1, 0.05, 0, 0], [0.1 + eps, 0.05, 0, 0], 15) / eps
    expect(chaotic).toBeGreaterThan(1e3)
    expect(regular).toBeLessThan(50)
  })

  it('times the first flip: never from small angles, quickly from near the top, and as the integration says', () => {
    expect(doublePendulumFlipTime(0.1, 0.1, {}, { tMax: 5 })).toBe(Number.POSITIVE_INFINITY)
    expect(doublePendulumFlipTime(Math.PI, 0.5)).toBe(0)
    const t = doublePendulumFlipTime(2.9, 2.9)
    expect(t).toBeGreaterThan(0)
    expect(t).toBeLessThan(3)
    // The same crossing found from the field by the stepped integration.
    const steps = 1000
    const dt = 0.005
    const alg = flowAlgorithm(doublePendulumField(), [2.9, 2.9, 0, 0], {
      dt,
      steps,
      stop: (_t, y) =>
        Math.abs(y[0] as number) >= Math.PI || Math.abs(y[1] as number) >= Math.PI ? 'stopped' : undefined,
    })
    const crossed = runAlgorithm(alg, { maxSteps: steps }).state.t
    expect(Math.abs(t - crossed)).toBeLessThan(2 * dt)
  })
})

describe('Rossler and Thomas', () => {
  it('has the Rossler fixed points make the field vanish, in the order near then far', () => {
    for (const params of [{}, { a: 0.1, b: 0.3, c: 4 }]) {
      const pts = rosslerFixedPoints(params)
      expect(pts.length).toBe(2)
      for (const p of pts) expect(Math.hypot(...toFlat(rosslerField(params)(0, tensor(p))))).toBeLessThan(1e-12)
      expect(Math.abs(pts[0]?.[0] ?? 0)).toBeLessThan(Math.abs(pts[1]?.[0] ?? 0))
    }
    expect(rosslerFixedPoints({ a: 1, b: 1, c: 1 })).toEqual([])
  })

  it('has the Thomas field of constant divergence -3b and the origin as an equilibrium', () => {
    for (const b of [0.1, 0.208186, 0.5]) {
      const f = thomasField({ b })
      expect(fieldDivergence(f, [0.3, -1.2, 2.2])).toBeCloseTo(-3 * b, 12)
      expect(toFlat(f(0, tensor([0, 0, 0])))).toEqual([0, 0, 0])
    }
  })

  it('stays bounded on the Rossler attractor and amplifies a difference of 1e-8 a hundredfold over 80 time units', () => {
    const run = (x0: number): Float64Array => {
      const alg = flowAlgorithm(rosslerField(), [x0, 1, 1], { dt: 0.01, steps: 8000 })
      return runAlgorithm(alg, { maxSteps: 8000 }).state.y
    }
    const a = run(1)
    expect(Math.max(...a.map(Math.abs))).toBeLessThan(40)
    const b = run(1 + 1e-8)
    expect(Math.hypot(...Array.from(a).map((v, i) => v - (b[i] as number)))).toBeGreaterThan(1e-6)
  })
})
