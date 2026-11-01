import { describe, expect, it } from 'vitest'
import { policies, recorder, runAlgorithm } from '../trace/index.ts'
import { backtracking } from './line-search.ts'
import { backtrackingAlgorithm, hagerZhangAlgorithm, lineSearchContract } from './line-search-algorithms.ts'
import { hagerZhang } from './line-search-interval.ts'
import { lineSearchRules } from './line-search-options.ts'
import { bracket, bracketAlgorithm } from './scalar.ts'

const viaJson = <T>(x: T): T => JSON.parse(JSON.stringify(x)) as T
const quad = { at: (a: number) => ({ value: (a - 0.3) ** 2, slope: 2 * (a - 0.3) }) }
const quartic = {
  at: (a: number) => ({ value: (a - 0.7) ** 4 + 0.1 * a, slope: 4 * (a - 0.7) ** 3 + 0.1 }),
  valueAt: (a: number) => (a - 0.7) ** 4 + 0.1 * a,
}
const hardStart = { value: 0.7 ** 4, slope: -4 * 0.7 ** 3 + 0.1, alpha: 6 }

describe('backtrackingAlgorithm', () => {
  it('evaluates one trial step per step and ends on the Armijo condition', () => {
    const run = runAlgorithm(backtrackingAlgorithm(quartic, hardStart), { maxSteps: 200 })
    expect(run.status).toBe('converged')
    expect(run.steps).toBe(run.state.trials.length)
    const a = run.state.outcome?.alpha ?? Number.NaN
    expect(quartic.valueAt(a)).toBeLessThanOrEqual(hardStart.value + 1e-4 * a * hardStart.slope)
    expect(a).toBeLessThan(6)
    const wrapped = backtracking()(quartic, hardStart)
    expect(wrapped.alpha).toBe(a)
    expect(wrapped.trials).toEqual(run.state.trials)
    expect(wrapped.evaluations).toBe(run.steps)
  })

  it('halves the step by rho in the halving variant', () => {
    const run = runAlgorithm(backtrackingAlgorithm(quad, { value: 0.09, slope: -0.6 }, { interpolation: 'halving' }), {
      maxSteps: 60,
    })
    expect(run.state.trials.map((t) => t.alpha)).toEqual([1, 0.5])
    expect(run.state.outcome?.message).toBe('Armijo condition satisfied')
  })

  it('fails on an ascent direction and when the step becomes too small', () => {
    const up = runAlgorithm(backtrackingAlgorithm(quad, { value: 0.09, slope: 0.6 }), { maxSteps: 5 })
    expect([up.status, up.steps]).toEqual(['failed', 0])
    const never = { at: (a: number) => ({ value: 1 + a, slope: 1 }) }
    const small = runAlgorithm(
      backtrackingAlgorithm(never, { value: 1, slope: -1 }, { interpolation: 'halving', maxEvaluations: 6 }),
      { maxSteps: 60 },
    )
    expect(small.status).toBe('failed')
    expect(small.state.trials).toHaveLength(6)
  })

  it('serialises and resumes to the same end', () => {
    const alg = backtrackingAlgorithm(quartic, hardStart)
    const whole = runAlgorithm(alg, { maxSteps: 200 })
    const half = runAlgorithm(alg, { maxSteps: 2 }).state
    const stored = viaJson(alg.serialise?.(half) ?? null)
    const rest = runAlgorithm(alg, { maxSteps: 200, start: { state: stored, step: 2, serialised: true } })
    expect(rest.state.trials).toEqual(whole.state.trials)
    expect(rest.state.outcome).toEqual(whole.state.outcome)
    expect(rest.steps).toBe(whole.steps)
  })

  it('logs the step, phi, the flags and the interval', () => {
    const rec = recorder()
    runAlgorithm(backtrackingAlgorithm(quartic, hardStart), {
      maxSteps: 200,
      tracer: rec,
      policy: { ...policies.minimal, metrics: true, tensors: true, roles: ['output'], detail: 'full' },
    })
    const names = new Set(rec.events.flatMap((e) => (e.type === 'metric' ? [e.name] : [])))
    expect([...names].sort()).toEqual(['ls/alpha', 'ls/armijo', 'ls/bracket', 'ls/phi'])
    const armijo = rec.events.flatMap((e) => (e.type === 'metric' && e.name === 'ls/armijo' ? [e.value] : []))
    expect(armijo.at(-1)).toBe(1)
    expect(Object.keys(lineSearchContract)).toHaveLength(7)
  })
})

describe('hagerZhangAlgorithm', () => {
  it('reaches a point meeting the Wolfe conditions, as the function does', () => {
    const run = runAlgorithm(hagerZhangAlgorithm(quartic, hardStart), { maxSteps: 100 })
    expect(run.status).toBe('converged')
    const a = run.state.outcome?.alpha ?? Number.NaN
    expect(a).toBeGreaterThan(0)
    expect(Math.abs(quartic.at(a).slope)).toBeLessThanOrEqual(0.9 * Math.abs(hardStart.slope))
    const wrapped = hagerZhang()(quartic, hardStart)
    expect(wrapped.alpha).toBe(a)
    expect(wrapped.trials).toEqual(run.state.trials)
    expect(wrapped.evaluations).toBe(run.state.trials.length)
    expect(wrapped.gradEvaluations).toBe(run.state.trials.length)
  })

  it('resumes from a JSON state at every step to the same outcome', () => {
    const alg = hagerZhangAlgorithm(quartic, hardStart)
    const whole = runAlgorithm(alg, { maxSteps: 100 })
    for (let cut = 1; cut < whole.steps; cut++) {
      const part = runAlgorithm(alg, { maxSteps: cut }).state
      const stored = viaJson(alg.serialise?.(part) ?? null)
      const rest = runAlgorithm(alg, { maxSteps: 100, start: { state: stored, step: cut, serialised: true } })
      expect(rest.state.trials).toEqual(whole.state.trials)
      expect(rest.state.outcome?.alpha).toBe(whole.state.outcome?.alpha)
    }
  })

  it('logs the slope and curvature flag and a shrinking interval', () => {
    const rec = recorder()
    runAlgorithm(hagerZhangAlgorithm(quartic, hardStart), {
      maxSteps: 100,
      tracer: rec,
      policy: { ...policies.minimal, metrics: true, tensors: true, roles: ['output'], detail: 'full' },
    })
    const metric = (name: string): number[] =>
      rec.events.flatMap((e) => (e.type === 'metric' && e.name === name ? [e.value] : []))
    expect(metric('ls/curvature').length).toBe(metric('ls/alpha').length)
    expect(metric('ls/bracket').every((w) => w > 0)).toBe(true)
    expect(rec.events.filter((e) => e.type === 'tensor' && e.name === 'ls/interval').length).toBe(
      metric('ls/alpha').length,
    )
  })

  it('rejects constants outside 0 < delta < sigma < 1', () => {
    expect(() => hagerZhang({ delta: 0.9, sigma: 0.5 })).toThrow(/c1 < c2/)
  })
})

describe('bracketAlgorithm', () => {
  it('walks downhill one extrapolation per step and agrees with bracket', () => {
    const f = (x: number): number => (x - 3) ** 2
    const run = runAlgorithm(bracketAlgorithm(f), { maxSteps: 50 })
    const s = run.state
    expect(run.status).toBe('converged')
    expect(s.fb).toBeLessThan(s.fa)
    expect(s.fb).toBeLessThan(s.fc)
    expect(bracket(f)).toEqual({ xa: s.xa, xb: s.xb, xc: s.xc, fa: s.fa, fb: s.fb, fc: s.fc, nfev: s.nfev })
  })

  it('resumes from JSON and fails on a function with no minimum', () => {
    const f = (x: number): number => Math.cosh(x - 40) // far from the start, still unimodal
    const alg = bracketAlgorithm(f)
    const whole = runAlgorithm(alg, { maxSteps: 200 })
    const part = runAlgorithm(alg, { maxSteps: 2 }).state
    const rest = runAlgorithm(alg, {
      maxSteps: 200,
      start: { state: viaJson(alg.serialise?.(part) ?? null), step: 2, serialised: true },
    })
    expect(rest.state).toEqual(whole.state)
    expect(() => bracket((x) => -x, { maxIterations: 5 })).toThrow(/bracket/)
    expect(
      runAlgorithm(
        bracketAlgorithm((x) => -x, { maxIterations: 5 }),
        { maxSteps: 50 },
      ).status,
    ).toBe('failed')
  })
})

describe('lineSearchRules', () => {
  it('names the rules and builds a stepped search from each', () => {
    expect(lineSearchRules.names).toEqual(['backtracking', 'halving', 'hager-zhang'])
    for (const kind of lineSearchRules.names) {
      const run = runAlgorithm(lineSearchRules.make({ kind })(quartic, hardStart), { maxSteps: 200 })
      expect(run.status).toBe('converged')
    }
    expect(lineSearchRules.properties({ kind: 'hager-zhang' }).usesSlope).toBe(true)
  })
})
