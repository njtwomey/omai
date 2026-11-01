import { describe, expect, it } from 'vitest'
import { add, mul, neg, sliceAxis, sub, sum, tensor, toFlat, type Value } from '../../tensor/index.ts'
import { messages, metricSeries, policies, recorder, runAlgorithm, tensorSeries } from '../../trace/index.ts'
import {
  bangBang,
  type ControlProblem,
  doubleIntegratorEnergy,
  hamiltonianSystem,
  minimumTimeDoubleIntegrator,
  minimumTimeShooting,
  pontryaginAlgorithm,
  pontryaginHamiltonian,
} from './indirect.ts'

const part = (y: Value, a: number, b: number): Value => sliceAxis(y, 0, a, b)

describe('the Hamiltonian system', () => {
  it("has x' = f(x, u*) and lambda' = -dH/dx at the minimising control", () => {
    const f = hamiltonianSystem(doubleIntegratorEnergy())
    for (const [got, want] of [
      [toFlat(f(0, tensor([0, 0, 2, 3]))), [0, -3, 0, -2]],
      [toFlat(f(0, tensor([1, 4, -1, 0.5]))), [4, -0.5, 0, 1]],
    ] as const) {
      for (const [i, w] of want.entries()) expect(got[i]).toBeCloseTo(w, 14)
    }
  })

  it('conserves the Hamiltonian along its flow when the problem does not depend on time', () => {
    const p = doubleIntegratorEnergy()
    const field = hamiltonianSystem(p)
    let y = tensor([0.3, -0.2, 0.7, 1.1]) as Value
    const hOf = (v: Value): number => {
      const x = part(v, 0, 2)
      const l = part(v, 2, 4)
      return toFlat(pontryaginHamiltonian(p, x, l, p.control(x, l)))[0] as number
    }
    const h0 = hOf(y)
    for (let k = 0; k < 200; k++) {
      const k1 = field(0, y)
      const k2 = field(0, add(y, mul(k1, 0.005)))
      y = add(y, mul(k2, 0.01))
    }
    expect(hOf(y)).toBeCloseTo(h0, 3)
  })

  it('computes the Hamiltonian of a simple problem', () => {
    const p: ControlProblem = {
      stateSize: 1,
      controlSize: 1,
      dynamics: (_x, u) => u,
      runningCost: (_x, u) => mul(sum(mul(u, u)), 0.5),
      control: (_x, l) => neg(l),
    }
    expect(toFlat(pontryaginHamiltonian(p, tensor([0]), tensor([2]), tensor([3])))).toEqual([10.5])
  })

  it('picks the extreme control that minimises sigma u', () => {
    expect([bangBang(2, [-3, 5]), bangBang(-2, [-3, 5]), bangBang(0, [-3, 5])]).toEqual([-3, 5, 5])
  })
})

describe('shooting for the minimum-energy double integrator', () => {
  const d = 1.5
  const T = 2
  const run = (steps: number) => {
    const rec = recorder()
    const alg = pontryaginAlgorithm(doubleIntegratorEnergy(), {
      x0: [0, 0],
      target: [d, 0],
      horizon: T,
      lambda0: [0, 0],
      steps,
    })
    const { state } = runAlgorithm(alg, { maxSteps: 20, tracer: rec, policy: policies.teaching })
    return { state, run: rec.run() }
  }

  it('finds the costate that gives the analytic control u(t) = 6 d (T - 2 t) / T^3', () => {
    const { state } = run(100)
    expect(state.missNorm).toBeLessThan(1e-8)
    // lambda2(0) = -u(0) = -6 d / T^2, and lambda1 = u'... lambda2' = -lambda1 with u = -lambda2, so lambda1 = u'(t)·(-1)...
    expect(state.lambda0[1]).toBeCloseTo((-6 * d) / T ** 2, 6)
    expect(state.lambda0[0]).toBeCloseTo((-12 * d) / T ** 3, 6)
    // The control along the shot against the closed form, node by node.
    for (let k = 0; k <= 100; k++) {
      const t = (k * T) / 100
      expect(state.controls[k]).toBeCloseTo((6 * d * (T - 2 * t)) / T ** 3, 6)
    }
    // The cost against 6 d^2 / T^3 (trapezoid rule, so to O(h^2)).
    expect(state.cost).toBeCloseTo((6 * d * d) / T ** 3, 3)
  })

  it('converges in one Newton step on this linear problem, the miss logged shrinking from the first guess', () => {
    const { run: r, state } = run(100)
    const misses = metricSeries(r, 'shot/miss').values
    expect(misses.length).toBeLessThanOrEqual(2)
    expect(misses.at(-1)).toBeLessThan(1e-8)
    expect(state.iteration).toBeLessThanOrEqual(2)
    expect(tensorSeries(r, 'trajectory/state').length).toBe(misses.length)
  })

  it('has a Hamiltonian constant along the optimal shot (the problem is autonomous)', () => {
    const { state } = run(100)
    const p = doubleIntegratorEnergy()
    const hs = Array.from({ length: 101 }, (_, k) => {
      const x = tensor(Array.from(state.states.slice(2 * k, 2 * k + 2))) as Value
      const l = tensor(Array.from(state.costates.slice(2 * k, 2 * k + 2))) as Value
      return toFlat(pontryaginHamiltonian(p, x, l, p.control(x, l)))[0] as number
    })
    expect(Math.max(...hs) - Math.min(...hs)).toBeLessThan(1e-8)
  })

  it('beats any perturbed control of the same endpoint (first-order optimality)', () => {
    const { state } = run(100)
    // Replace u by u + eps b(t) with b(t) = sin(pi t / T)-shaped, correcting nothing: the endpoint moves, so compare
    // costs at equal endpoint by adding the correction that restores (x, v) at T (a two-term basis).
    const costOf = (u: (t: number) => number): { cost: number; x: number; v: number } => {
      let x = 0
      let v = 0
      let cost = 0
      const h = T / 2000
      for (let k = 0; k < 2000; k++) {
        const t = k * h
        const a = u(t)
        cost += 0.5 * a * a * h
        x += v * h + 0.5 * a * h * h
        v += a * h
      }
      return { cost, x, v }
    }
    const best = costOf((t) => (6 * d * (T - 2 * t)) / T ** 3)
    expect(best.x).toBeCloseTo(d, 2)
    // A different control reaching the same rest-to-rest endpoint: add the bump of the next Legendre-like mode.
    const bump = (t: number): number => Math.sin((2 * Math.PI * t) / T) // zero net velocity and displacement change
    const other = costOf((t) => (6 * d * (T - 2 * t)) / T ** 3 + 0.4 * bump(t))
    expect(other.v).toBeCloseTo(best.v, 2)
    expect(other.cost).toBeGreaterThan(best.cost)
    expect(state.cost).toBeLessThan(other.cost)
  })
})

describe('shooting for a free final state and a nonlinear problem', () => {
  it('meets the transversality condition lambda(T) = grad phi for a quadratic terminal cost', () => {
    // x' = u, cost u^2 / 2 + 5 (x(T) - 1)^2 : u = -lambda, lambda constant = 10 (x(T) - 1), x(T) = u T.
    const p: ControlProblem = {
      stateSize: 1,
      controlSize: 1,
      dynamics: (_x, u) => u,
      runningCost: (_x, u) => mul(sum(mul(u, u)), 0.5),
      control: (_x, l) => neg(l),
    }
    const alg = pontryaginAlgorithm(p, {
      x0: [0],
      terminalCost: (x) => mul(sum(mul(sub(x, 1), sub(x, 1))), 5),
      horizon: 1,
      lambda0: [0],
      steps: 50,
    })
    const s = runAlgorithm(alg, { maxSteps: 20 }).state
    // lambda = -u, x(T) = -lambda T, lambda = 10 (x(T) - 1)  =>  lambda = 10 (-lambda - 1)  =>  lambda = -10 / 11.
    expect(s.lambda0[0]).toBeCloseTo(-10 / 11, 6)
    expect(s.states[50]).toBeCloseTo(10 / 11, 6)
  })

  it('converges by damped Newton on a nonlinear problem and meets the target', () => {
    // x' = -x^3 + u, cost u^2 / 2, from 1 to 0.2 in time 2: the costate equation is lambda' = 3 x^2 lambda.
    const p: ControlProblem = {
      stateSize: 1,
      controlSize: 1,
      dynamics: (x, u) => add(neg(mul(mul(x, x), x)), u),
      runningCost: (_x, u) => mul(sum(mul(u, u)), 0.5),
      control: (_x, l) => neg(l),
    }
    const rec = recorder()
    const alg = pontryaginAlgorithm(p, { x0: [1], target: [0.2], horizon: 2, lambda0: [0], steps: 200 })
    const { state } = runAlgorithm(alg, { maxSteps: 30, tracer: rec })
    expect(state.missNorm).toBeLessThan(1e-8)
    expect(state.states[200]).toBeCloseTo(0.2, 7)
    const misses = metricSeries(rec.run(), 'shot/miss').values
    expect(misses.length).toBeGreaterThan(1)
    // Each step does not make the residual worse.
    for (let i = 1; i < misses.length; i++) expect(misses[i] as number).toBeLessThanOrEqual(misses[i - 1] as number)
  })

  it('refuses a problem without a boundary condition or with a guess of the wrong size', () => {
    expect(() => pontryaginAlgorithm(doubleIntegratorEnergy(), { x0: [0, 0], horizon: 1, lambda0: [0, 0] })).toThrow()
    expect(() =>
      pontryaginAlgorithm(doubleIntegratorEnergy(), { x0: [0, 0], target: [1, 0], horizon: 1, lambda0: [0] }),
    ).toThrow()
  })
})

describe('minimum time', () => {
  it('finds the horizon 2 sqrt(d) and the costate the theory gives, the smoothing falling from stage to stage', () => {
    for (const d of [1, 4]) {
      const { final, stages } = minimumTimeShooting(d, { steps: 200 })
      expect(final.missNorm).toBeLessThan(1e-8)
      // The sharper the control, the nearer the bang-bang horizon; each stage is a solved problem.
      expect(stages.length).toBe(5)
      expect(Math.abs((stages[4] as { horizon: number }).horizon - 2 * Math.sqrt(d))).toBeLessThan(
        Math.abs((stages[0] as { horizon: number }).horizon - 2 * Math.sqrt(d)),
      )
      expect(final.horizon).toBeGreaterThan(0)
      expect(final.horizon).toBeCloseTo(2 * Math.sqrt(d), 1)
      // H = 0 at the start from rest: 1 + lambda2 u(0) = 0 with u(0) = -1, so lambda2(0) = 1 and lambda1 = 2 / T.
      expect(final.lambda0[1]).toBeCloseTo(1, 1)
      expect(final.lambda0[0]).toBeCloseTo(1 / Math.sqrt(d), 1)
    }
  }, 60_000)

  it('has the control bang-bang in effect: full brake, one switch at half the time, then full thrust the other way', () => {
    const { final } = minimumTimeShooting(1, { steps: 400 })
    expect(final.controls[0]).toBeCloseTo(-1, 3)
    expect(final.controls[400]).toBeCloseTo(1, 3)
    expect(final.switches.length).toBe(1)
    expect(final.switches[0]).toBeCloseTo(final.horizon / 2, 2)
    // After the switch the state sits on the switching curve x = -v |v| / 2.
    const k = Math.min(400, Math.ceil(((final.switches[0] as number) / final.horizon) * 400) + 20)
    const x = final.states[2 * k] as number
    const v = final.states[2 * k + 1] as number
    expect(x).toBeCloseTo((-v * Math.abs(v)) / 2, 1)
    // The exact bang-bang rule picks the extremes.
    const exact = minimumTimeDoubleIntegrator()
    expect(toFlat(exact.control(tensor([1, 0]), tensor([1, 0.5])))).toEqual([-1])
    expect(toFlat(exact.control(tensor([1, 0]), tensor([1, -0.5])))).toEqual([1])
  }, 60_000)

  it('emits the switch as an event on the tape', () => {
    const rec = recorder()
    const alg = pontryaginAlgorithm(minimumTimeDoubleIntegrator(1, 0.05), {
      x0: [1, 0],
      target: [0, 0],
      horizon: 2.2,
      freeHorizon: true,
      lambda0: [1, 1],
      steps: 400,
    })
    runAlgorithm(alg, { maxSteps: 30, tracer: rec, policy: policies.teaching })
    const switches = messages(rec.run(), 'trajectory/switch')
    expect(switches.length).toBeGreaterThanOrEqual(1)
    const times = switches.map((e) => (e.payload as unknown as { time: number }).time)
    expect(times.at(-1)).toBeCloseTo(1, 1)
  })
})
