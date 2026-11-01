import { describe, expect, it } from 'vitest'
import { tf } from '../../systems/index.ts'
import { add, mul, sin, sum } from '../../tensor/index.ts'
import { policies, recorder, runAlgorithm, stateToJson } from '../../trace/index.ts'
import { antiWindups, disturbances, qpMethods, setpoints } from './control-options.ts'
import {
  pid,
  pidLoopAlgorithm,
  pidLoopContract,
  relayAnalysis,
  relayAutotune,
  relayAutotuneAlgorithm,
  simulatePid,
} from './pid.ts'
import { linearMpc, mpcContract, nonlinearMpc, recedingHorizon, recedingHorizonAlgorithm } from './predictive.ts'
import type { DiscreteModel } from './trajectory.ts'

const setpoint = (t: number): number => (t < 2 ? 1 : 0.5)
const disturbance = (t: number): number => (t > 3 ? 0.1 : 0)

// What the loops returned before they became runs of algorithms (13 significant digits).
const OLD_PID = {
  y: [0, 0.2509062967536, 0.9083066085744, 0.9560322287258, 0.656913095203, 0.4786316334865],
  at: [0, 50, 150, 250, 350, 499],
  u: [3.015, 2.181725921883, -0.1405744259282, 0.430092266987],
  uAt: [0, 50, 250, 499],
  iae: 1.486914419799,
  overshoot: 1.096550777715,
}
const OLD_RELAY = { ku: 7.052264164522108, tu: 3.849999999986595, amplitude: 0.18054337089930084 }
const OLD_MPC = {
  x: [1.030060879558e-8, -1.132992025264e-8],
  u: [-0.5, -0.5, -3.359606555511e-15, 1.167658557506e-15, 0.08001931483764, 0.5, 0.3053132072491],
  cost0: -192.8347487545,
}
const OLD_NMPC = { x: 2.317506455762e-8, u: [-2.091655376718, -0.216446088215, -0.02241088537531] }

const A = [
  [1, 1],
  [0, 1],
]
const B = [[0.5], [1]]
const Q = [
  [1, 0],
  [0, 1],
]
const plant = (x: readonly number[], u: readonly number[]): number[] => [
  (x[0] ?? 0) + (x[1] ?? 0) + 0.5 * (u[0] ?? 0),
  (x[1] ?? 0) + (u[0] ?? 0),
]
const mpcCfg = { A, B, Q, R: [[0.1]], horizon: 10, uMin: -0.5, uMax: 0.5, xMin: [-10, -1], xMax: [10, 1] }

const near = (a: number, b: number): void => {
  expect(Math.abs(a - b)).toBeLessThan(1e-11 * Math.max(1, Math.abs(b)))
}

describe('pidLoopAlgorithm', () => {
  const p = tf([1], [1, 2, 1])
  const make = () =>
    pidLoopAlgorithm(p, pid({ kp: 3, ki: 1.5, kd: 1, dt: 0.01 }), { duration: 5, setpoint, disturbance })

  it('is simulatePid: the loop run to the end gives the numbers it gave before', () => {
    const s = runAlgorithm(make(), { maxSteps: 500 }).state
    OLD_PID.at.forEach((i, k) => {
      near(s.samples[i]?.y ?? 0, OLD_PID.y[k] ?? 0)
    })
    OLD_PID.uAt.forEach((i, k) => {
      near(s.samples[i]?.u ?? 0, OLD_PID.u[k] ?? 0)
    })
    const r = simulatePid(p, pid({ kp: 3, ki: 1.5, kd: 1, dt: 0.01 }), { duration: 5, setpoint, disturbance })
    near(r.iae, OLD_PID.iae)
    near(r.overshoot, OLD_PID.overshoot)
    expect(r.y).toEqual(s.samples.map((q) => q.y))
  })

  it('has a state that is JSON, and a run resumes from a checkpoint to the same end', () => {
    const alg = make()
    const whole = runAlgorithm(alg, { maxSteps: 500 }).state
    const half = runAlgorithm(alg, { maxSteps: 200 }).state
    const stored = JSON.parse(JSON.stringify(alg.serialise?.(half)))
    const rest = runAlgorithm(alg, { maxSteps: 300, start: { state: stored, step: 200, serialised: true } })
    expect(rest.steps).toBe(500)
    expect(stateToJson(rest.state)).toEqual(stateToJson(whole))
  })

  it('logs its contract by path with the right shapes', () => {
    const rec = recorder()
    runAlgorithm(make(), {
      maxSteps: 6,
      tracer: rec,
      policy: { ...policies.minimal, metrics: true, tensors: true, roles: ['output'], detail: 'full' },
    })
    const metric = (name: string): number[] =>
      rec.events.flatMap((e) => (e.type === 'metric' && e.name === name ? [e.value] : []))
    expect(metric('pid/u')).toHaveLength(6)
    expect(metric('pid/error')[0]).toBeCloseTo(1, 12)
    expect(metric('pid/setpoint')).toEqual(Array(6).fill(1))
    expect(metric('pid/integral')).toHaveLength(6)
    expect(rec.events.flatMap((e) => (e.type === 'tensor' && e.name === 'plant/x' ? [e.shape[0]] : []))).toEqual(
      Array(6).fill(2),
    )
    expect(Object.keys(pidLoopContract)).toHaveLength(7)
  })

  it('anti-windup keeps the actuator inside its limits and the integral from running away', () => {
    const slow = tf([1], [10, 1])
    const run = (kind: string): number => {
      const c = pid({ kp: 5, ki: 2, dt: 0.01, outputMin: -1, outputMax: 1, ...antiWindups.make({ kind }) })
      return simulatePid(slow, c, { duration: 60, setpoint: 0.8 }).overshoot
    }
    expect(run('clamping')).toBeLessThan(run('none'))
    expect(run('back-calculation')).toBeLessThan(run('none'))
    expect(antiWindups.where((q) => q.hasGain)).toEqual(['back-calculation'])
  })
})

describe('relayAutotuneAlgorithm', () => {
  it('is relayAutotune: the limit cycle is read off the algorithm run to the end', () => {
    const plantR = tf([1], [1, 3, 3, 1])
    const options = { dt: 0.005, duration: 60, hysteresis: 0.01 }
    const alg = relayAutotuneAlgorithm(plantR, options)
    const s = runAlgorithm(alg, { maxSteps: 12000 }).state
    const read = relayAnalysis(s.samples, { dt: 0.005 })
    near(read.ku, OLD_RELAY.ku)
    near(read.tu, OLD_RELAY.tu)
    near(read.amplitude, OLD_RELAY.amplitude)
    expect(relayAutotune(plantR, options).gains).toEqual(read.gains)
    // The relay's output alternates between plus and minus the amplitude (it is held in the integral memory).
    expect(new Set(s.samples.map((q) => Math.abs(q.u)))).toEqual(new Set([1]))
    // Resume mid-cycle.
    const half = runAlgorithm(alg, { maxSteps: 5000 }).state
    const stored = JSON.parse(JSON.stringify(alg.serialise?.(half)))
    const rest = runAlgorithm(alg, { maxSteps: 7000, start: { state: stored, step: 5000, serialised: true } })
    expect(stateToJson(rest.state)).toEqual(stateToJson(s))
  })
})

describe('recedingHorizonAlgorithm', () => {
  it('is recedingHorizon with a linear planner: the same inputs, costs and states as before', () => {
    const run = recedingHorizon(linearMpc(mpcCfg), plant, [5, 0], { steps: 20 })
    OLD_MPC.x.forEach((v, i) => {
      near(run.x[i] ?? 0, v)
    })
    OLD_MPC.u.forEach((v, i) => {
      expect(Math.abs((run.inputs[i]?.[0] ?? 0) - v)).toBeLessThan(1e-9)
    })
    near(run.costs[0] ?? 0, OLD_MPC.cost0)
    expect(run.status).toBe('optimal')
  })

  const model: DiscreteModel = {
    stateSize: 1,
    actionSize: 1,
    dynamics: (x, u) => add(add(x, mul(sin(x), 0.3)), u),
    cost: (x, u) => add(sum(mul(x, x), -1), mul(sum(mul(u, u), -1), 0.1)),
  }
  const nlPlant = (x: readonly number[], u: readonly number[]): number[] => [
    (x[0] ?? 0) + 0.3 * Math.sin(x[0] ?? 0) + (u[0] ?? 0),
  ]

  it('is recedingHorizon with a nonlinear planner, and a resumed run keeps the warm start', () => {
    const alg = recedingHorizonAlgorithm(nonlinearMpc(model, { horizon: 6 }), nlPlant, [2], { steps: 8 })
    const whole = runAlgorithm(alg, { maxSteps: 8 }).state
    near(whole.x[0] ?? 0, OLD_NMPC.x)
    OLD_NMPC.u.forEach((v, i) => {
      near(whole.inputs[i]?.[0] ?? 0, v)
    })
    // A fresh planner (no memory of its own) resumes from the stored state to the same end.
    const half = runAlgorithm(alg, { maxSteps: 4 }).state
    expect(half.warm.length).toBe(6)
    const stored = JSON.parse(JSON.stringify(alg.serialise?.(half)))
    const again = recedingHorizonAlgorithm(nonlinearMpc(model, { horizon: 6 }), nlPlant, [2], { steps: 8 })
    const rest = runAlgorithm(again, { maxSteps: 4, start: { state: stored, step: 4, serialised: true } })
    expect(stateToJson(rest.state)).toEqual(stateToJson(whole))
  })

  it('has a state that is JSON, resumes (linear), and logs the plan, the prediction and the cost by path', () => {
    const alg = recedingHorizonAlgorithm(linearMpc(mpcCfg), plant, [5, 0], { steps: 12 })
    const whole = runAlgorithm(alg, { maxSteps: 12 }).state
    const half = runAlgorithm(alg, { maxSteps: 5 }).state
    const stored = JSON.parse(JSON.stringify(alg.serialise?.(half)))
    const rest = runAlgorithm(alg, { maxSteps: 7, start: { state: stored, step: 5, serialised: true } })
    expect(stateToJson(rest.state)).toEqual(stateToJson(whole))
    const rec = recorder()
    runAlgorithm(alg, {
      maxSteps: 3,
      tracer: rec,
      policy: { ...policies.minimal, metrics: true, tensors: true, roles: ['output'], detail: 'full' },
    })
    const shapes = (name: string): number[][] =>
      rec.events.flatMap((e) => (e.type === 'tensor' && e.name === name ? [[...e.shape]] : []))
    expect(shapes('mpc/plan')).toEqual(Array(3).fill([10, 1]))
    expect(shapes('mpc/predicted')).toEqual(Array(3).fill([10, 2]))
    expect(shapes('mpc/u')).toEqual(Array(3).fill([1]))
    expect(Object.keys(mpcContract)).toHaveLength(5)
  })

  it('respects its constraints at every step (an invariant of the loop)', () => {
    const s = runAlgorithm(recedingHorizonAlgorithm(linearMpc(mpcCfg), plant, [5, 0], { steps: 25 }), {
      maxSteps: 25,
    }).state
    expect(s.inputs.every((u) => Math.abs(u[0] ?? 0) <= 0.5 + 1e-9)).toBe(true)
    expect(s.states.every((x) => Math.abs(x[1] ?? 0) <= 1 + 1e-6)).toBe(true)
  })
})

describe('control option tables', () => {
  it('run every option of the QP, setpoint and disturbance tables', () => {
    for (const name of qpMethods.names) {
      const mpc = linearMpc({
        A,
        B,
        Q,
        R: [[0.1]],
        horizon: 6,
        uMin: -1,
        uMax: 1,
        method: qpMethods.make({ kind: name }),
      })
      expect(Math.abs(mpc.solve([2, 0]).u[0] ?? 9)).toBeLessThanOrEqual(1 + 1e-6)
    }
    const p = tf([1], [1, 1])
    for (const sp of setpoints.names) {
      for (const dist of disturbances.names) {
        const r = simulatePid(p, pid({ kp: 2, ki: 2, dt: 0.05 }), {
          duration: 5,
          setpoint: setpoints.make({ kind: sp }),
          disturbance: disturbances.make({ kind: dist }),
        })
        expect(r.y.every(Number.isFinite)).toBe(true)
      }
    }
    expect(setpoints.where((q) => q.periodic)).toEqual(['square', 'sine'])
    expect(disturbances.make({ kind: 'step', size: 0.3, at: 1 })(2)).toBe(0.3)
  })
})
