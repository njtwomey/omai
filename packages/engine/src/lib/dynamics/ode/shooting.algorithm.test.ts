import { describe, expect, it } from 'vitest'
import { concat, neg, sliceAxis, sub, toFlat, type Value } from '../../tensor/index.ts'
import { policies, recorder, runAlgorithm } from '../../trace/index.ts'
import {
  multipleShooting,
  multipleShootingAlgorithm,
  shootContract,
  shooting,
  shootingOf,
  singleShootingAlgorithm,
} from './boundary.ts'
import { shootingAccuracies } from './boundary-options.ts'

const viaJson = <T>(x: T): T => JSON.parse(JSON.stringify(x)) as T
// y'' = -y on [0, pi/2] with y(0) = 0, y(pi/2) = 1: the initial slope is 1.
const f = (_t: number, y: Value): Value => concat([sliceAxis(y, 0, 1, 2), neg(sliceAxis(y, 0, 0, 1))], 0)
const bc = (ya: Value, yb: Value): Value => concat([sliceAxis(ya, 0, 0, 1), sub(sliceAxis(yb, 0, 0, 1), 1)], 0)
const span: [number, number] = [0, Math.PI / 2]

describe('singleShootingAlgorithm', () => {
  it('takes one Newton iteration per step on the miss, and reaches the slope of shooting', () => {
    const run = runAlgorithm(singleShootingAlgorithm(f, bc, span, [0, 0.5]), { maxSteps: 50 })
    expect(run.status).toBe('converged')
    const r = shootingOf(run.state, 2)
    expect(r.y0.data[1]).toBeCloseTo(1, 8)
    expect(toFlat(shooting(f, bc, span, [0, 0.5]).y0)).toEqual(toFlat(r.y0))
    expect(r.iterations).toBe(run.steps)
    // A linear problem is solved by one Newton step; the miss is then zero.
    expect(run.steps).toBeLessThanOrEqual(3)
  })

  it('logs the miss falling and the slope, serialises and resumes', () => {
    const rec = recorder()
    const alg = singleShootingAlgorithm(f, bc, span, [0, 0.2])
    const whole = runAlgorithm(alg, {
      maxSteps: 50,
      tracer: rec,
      policy: { ...policies.minimal, metrics: true, detail: 'full' },
    })
    const miss = rec.events.flatMap((e) => (e.type === 'metric' && e.name === 'shoot/miss' ? [e.value] : []))
    expect(miss).toHaveLength(whole.steps)
    expect(miss.at(-1) ?? 1).toBeLessThan(1e-8)
    expect(Object.keys(shootContract)).toHaveLength(7)
    const part = runAlgorithm(alg, { maxSteps: 1 }).state
    const rest = runAlgorithm(alg, {
      maxSteps: 50,
      start: { state: viaJson(alg.serialise?.(part) ?? null), step: 1, serialised: true },
    })
    expect(rest.state.x).toEqual(whole.state.x)
    expect(shootingAccuracies.make({ kind: 'fast' }).method).toBe('RK45')
  })
})

describe('multipleShootingAlgorithm', () => {
  it('matches the segments and the boundary conditions, as multipleShooting does', () => {
    const nodes = [0, 0.8, Math.PI / 2]
    const guesses = [
      [0, 0.5],
      [0.5, 0.5],
    ]
    const run = runAlgorithm(multipleShootingAlgorithm(f, bc, nodes, guesses), { maxSteps: 50 })
    expect(run.status).toBe('converged')
    const direct = multipleShooting(f, bc, nodes, guesses)
    expect(direct.converged).toBe(true)
    const r = shootingOf(run.state, 2)
    expect(r.y0.data[1]).toBeCloseTo(1, 8)
    expect(Array.from(toFlat(direct.nodes))).toEqual(Array.from(run.state.x))
    // The state at the second node is sin(0.8), cos(0.8).
    expect(run.state.x[2]).toBeCloseTo(Math.sin(0.8), 8)
    expect(() => multipleShootingAlgorithm(f, bc, [0], [[0, 0]])).toThrow(/two nodes/)
    expect(() => multipleShootingAlgorithm(f, bc, [0, 1, 2], [[0, 0]])).toThrow(/guesses/)
  })
})
