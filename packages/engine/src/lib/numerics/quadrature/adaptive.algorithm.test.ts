import { describe, expect, it } from 'vitest'
import { grad } from '../../autodiff/index.ts'
import { abs, exp, item, mul, neg, sin, sqrt, square, tensor } from '../../tensor/index.ts'
import { policies, recorder, runAlgorithm, stateToJson } from '../../trace/index.ts'
import { quad, quadAlgorithm, quadContract } from './adaptive.ts'

describe('quad is a run of quadAlgorithm', () => {
  // What quad returned before it became a run of the algorithm (taken from the previous implementation).
  it('gives the numbers it gave before, exactly', () => {
    const g = quad((x) => exp(neg(square(x))), Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY)
    expect([item(g.value), g.error, g.intervals.length, g.evaluations, g.converged]).toEqual([
      1.7724538509055154,
      1.7190269193375743e-8,
      5,
      135,
      true,
    ])
    const r = quad((x) => sqrt(x), 0, 1)
    expect([item(r.value), r.error, r.intervals.length, r.evaluations]).toEqual([
      0.6666666670773991, 7.109463204287009e-9, 11, 315,
    ])
    const capped = quad((x) => sqrt(x), 0, 1, { limit: 4 })
    expect([item(capped.value), capped.error, capped.converged]).toEqual([
      0.6666672614708262,
      0.000010295454516688618,
      false,
    ])
    const osc = quad((x) => sin(x), 0, 20, { epsabs: 1e-12, epsrel: 1e-12 })
    expect([item(osc.value), osc.intervals.length, osc.evaluations]).toEqual([0.591917938186608, 8, 225])
    const pts = quad((x) => abs(x), -1, 2, { points: [0] })
    expect([item(pts.value), pts.intervals.length]).toEqual([2.5, 2])
    const rev = quad((x) => sin(x), 3, 0)
    expect(item(rev.value)).toBe(-1.9899924966004452)
  })

  it('stays differentiable in what the integrand closes over', () => {
    const d = grad((t: ReturnType<typeof tensor>) => quad((x) => mul(t, square(x)), 0, 1).value)(tensor(2))
    expect(item(d)).toBeCloseTo(1 / 3, 10)
  })
})

describe('quadAlgorithm', () => {
  const make = () => quadAlgorithm((x) => sqrt(x), 0, 1)

  it('has a state that is JSON, and a run resumes to the same end', () => {
    const alg = make()
    const whole = runAlgorithm(alg, { maxSteps: 50 })
    const half = runAlgorithm(alg, { maxSteps: 4 }).state
    const stored = JSON.parse(JSON.stringify(alg.serialise?.(half)))
    const rest = runAlgorithm(make(), { maxSteps: 46, start: { state: stored, step: 4, serialised: true } })
    expect(stateToJson(rest.state)).toEqual(stateToJson(whole.state))
    expect(whole.status).toBe('converged')
    expect(item(alg.value(whole.state))).toBe(item(make().value(rest.state)))
  })

  it('logs the estimate, the error, the intervals and the split each step', () => {
    const rec = recorder()
    runAlgorithm(make(), {
      maxSteps: 3,
      tracer: rec,
      policy: { ...policies.minimal, metrics: true, tensors: true, roles: ['state', 'output'], detail: 'full' },
    })
    const metric = (n: string): number[] =>
      rec.events.flatMap((e) => (e.type === 'metric' && e.name === n ? [e.value] : []))
    expect(metric('quad/count')).toEqual([2, 3, 4])
    expect(metric('quad/evaluations')).toEqual([45, 75, 105])
    const rows = rec.events.flatMap((e) => (e.type === 'tensor' && e.name === 'quad/intervals' ? [e.shape] : []))
    expect(rows.map((s) => s[0] ?? 0)).toEqual([8, 12, 16])
    expect(Object.keys(quadContract)).toContain('quad/split')
  })

  it('splits the interval with the largest error, so the total error never grows past a split', () => {
    const alg = make()
    let s = alg.init(undefined as never)
    const ctx = { t: 0, wants: () => false, log: () => undefined, emit: () => undefined } as never
    for (let i = 0; i < 6; i++) {
      const worst = Math.max(...s.intervals.map((v) => v.error))
      const before = s.error
      s = alg.step(s, ctx)
      const [a, b] = s.split as readonly [number, number]
      expect(s.intervals.some((v) => v.a === a && v.b === b)).toBe(false)
      expect(before - s.error).toBeGreaterThan(-worst)
      // The intervals tile [0, 1] without gaps.
      expect(s.intervals[0]?.a).toBe(0)
      expect(s.intervals.at(-1)?.b).toBe(1)
      for (const [k, v] of s.intervals.slice(1).entries()) expect(v.a).toBe(s.intervals[k]?.b)
    }
  })

  it('stops at the subinterval limit, and rejects a reversed range', () => {
    const r = runAlgorithm(
      quadAlgorithm((x) => sqrt(x), 0, 1, { limit: 5 }),
      { maxSteps: 100 },
    )
    expect(r.state.intervals).toHaveLength(5)
    expect(r.status).toBe('completed')
    expect(() => quadAlgorithm((x) => x, 1, 0)).toThrow()
  })
})
