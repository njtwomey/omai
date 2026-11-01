import { describe, expect, it } from 'vitest'
import { gamma } from '../dists/index.ts'
import { foldIn, key, uniform } from '../random/index.ts'
import { policies, recorder, runAlgorithm, stateToJson } from '../trace/index.ts'
import { historyArray } from './paths.ts'
import { pointProcess, pointProcesses } from './point-process-options.ts'
import {
  coxProcess,
  coxProcessAlgorithm,
  hawkesProcess,
  hawkesProcessAlgorithm,
  inhomogeneousPoisson,
  markedPoissonProcess,
  pointProcessContract,
  poissonProcess,
  poissonProcessAlgorithm,
  renewalProcess,
  renewalProcessAlgorithm,
  thinningAlgorithm,
} from './point-processes.ts'

const stats = (a: Float64Array): { n: number; first: number[]; sum: number } => ({
  n: a.length,
  first: Array.from(a.slice(0, 3)),
  sum: Array.from(a).reduce((x, y) => x + y, 0),
})

describe('the simulators give the events they gave before, exactly', () => {
  // Taken from the previous implementation, where the same Algorithm was private.
  it('poisson, thinning, renewal, hawkes, cox and marked', () => {
    expect(stats(poissonProcess(key(4), 3, 5))).toEqual({
      n: 15,
      first: [0.01087214186896665, 0.04962709212543795, 0.16908903690383914],
      sum: 36.497487628826924,
    })
    expect(stats(inhomogeneousPoisson(key(4), (t) => 2 + Math.sin(t), 6, 3))).toEqual({
      n: 12,
      first: [0.016195736982243324, 0.4782662988062229, 1.5160960596532],
      sum: 29.987514549680668,
    })
    expect(stats(renewalProcess(key(4), gamma(2, 2), 8))).toEqual({
      n: 9,
      first: [1.7986312609527055, 2.579838580270602, 2.855009390583341],
      sum: 45.06408602043779,
    })
    expect(stats(hawkesProcess(key(4), { mu: 1, alpha: 0.5, beta: 1 }, 8))).toEqual({
      n: 14,
      first: [0.04858721094672997, 0.9727283345946892, 3.617795802349863],
      sum: 67.34220013984422,
    })
    const c = coxProcess(
      key(4),
      (k) => {
        const r = 1 + 4 * (uniform(k, [1]).data[0] ?? 0)
        return { intensity: () => r, max: r }
      },
      4,
    )
    expect(stats(c.events)).toEqual({
      n: 10,
      first: [0.023967887199790604, 0.14010514525524748, 2.127342309932162],
      sum: 22.555953740092143,
    })
    expect(markedPoissonProcess(key(4), 3, 5, gamma(1, 1)).times.length).toBe(12)
  })
})

const make = {
  poisson: () => poissonProcessAlgorithm(3, 5),
  thinning: () => thinningAlgorithm((t) => 2 + Math.sin(t), 6, 3),
  renewal: () => renewalProcessAlgorithm(gamma(2, 2), 8),
  hawkes: () => hawkesProcessAlgorithm({ mu: 1, alpha: 0.5, beta: 1 }, 8),
} as const

describe('the point-process algorithms: state, paths, invariants', () => {
  for (const [name, build] of Object.entries(make)) {
    it(`${name}: a state that is JSON resumes to the same events`, () => {
      const alg = build()
      const whole = runAlgorithm(alg, { key: key(4), maxSteps: 10_000 })
      const half = runAlgorithm(alg, { key: key(4), maxSteps: 5 }).state
      const stored = JSON.parse(JSON.stringify(alg.serialise?.(half)))
      const rest = runAlgorithm(alg, {
        key: key(4),
        maxSteps: 9_995,
        start: { state: stored, step: 5, serialised: true },
      })
      expect(stateToJson(rest.state)).toEqual(stateToJson(whole.state))
      expect(historyArray(rest.state.events)).toEqual(historyArray(whole.state.events))
      expect(whole.status).toBe('completed')
    })

    it(`${name}: logs the candidate, whether it was kept and the count each step`, () => {
      const rec = recorder()
      const r = runAlgorithm(build(), {
        key: key(4),
        maxSteps: 10_000,
        tracer: rec,
        policy: { ...policies.minimal, metrics: true, detail: 'full' },
      })
      const metric = (n: string): number[] =>
        rec.events.flatMap((e) => (e.type === 'metric' && e.name === n ? [e.value] : []))
      expect(metric('pp/time')).toHaveLength(r.steps)
      expect(metric('pp/accepted')).toHaveLength(r.steps)
      const kept = metric('pp/accepted').reduce((a, b) => a + b, 0)
      expect(kept).toBe(r.state.count)
      expect(metric('pp/count').at(-1)).toBe(r.state.count)
      const times = historyArray(r.state.events)
      expect(times.every((t, i) => i === 0 || t > (times[i - 1] ?? 0))).toBe(true)
    })
  }

  it('thinning: only candidates under the intensity are kept, and the others are thinned away', () => {
    const rec = recorder()
    const r = runAlgorithm(
      thinningAlgorithm((t) => (t < 3 ? 0 : 3), 6, 3),
      {
        key: key(1),
        maxSteps: 10_000,
        tracer: rec,
        policy: { ...policies.minimal, metrics: true, detail: 'full' },
      },
    )
    const metric = (n: string): number[] =>
      rec.events.flatMap((e) => (e.type === 'metric' && e.name === n ? [e.value] : []))
    const time = metric('pp/time')
    const intensity = metric('pp/intensity')
    const accepted = metric('pp/accepted')
    time.forEach((t, i) => {
      if (t < 3) expect(accepted[i]).toBe(0)
      if (intensity[i] === 0) expect(accepted[i]).toBe(0)
    })
    expect(r.steps).toBeGreaterThan(r.state.count)
    expect(historyArray(r.state.events).every((t) => t >= 3)).toBe(true)
  })

  it('hawkes: the excitation is non-negative, and the intensity logged is mu plus it', () => {
    const r = runAlgorithm(make.hawkes(), { key: key(7), maxSteps: 10_000 })
    expect(r.state.memory).toBeGreaterThanOrEqual(0)
  })

  it('cox: the algorithm run with foldIn(key, 1) gives coxProcess’s events', () => {
    const random = (k: ReturnType<typeof key>) => {
      const r = 1 + 4 * (uniform(k, [1]).data[0] ?? 0)
      return { intensity: () => r, max: r }
    }
    const { algorithm } = coxProcessAlgorithm(key(4), random, 4)
    const s = runAlgorithm(algorithm, { key: foldIn(key(4), 1), maxSteps: 10_000 }).state
    expect(historyArray(s.events)).toEqual(Array.from(coxProcess(key(4), random, 4).events))
  })

  it('has a contract with the paths a page reads, and rejects a bad horizon or rate', () => {
    expect(Object.keys(pointProcessContract)).toEqual(['pp/time', 'pp/intensity', 'pp/accepted', 'pp/count'])
    expect(() => poissonProcessAlgorithm(2, 0)).toThrow()
    expect(() => poissonProcessAlgorithm(0, 1)).toThrow()
    expect(() => thinningAlgorithm(() => 1, 1, 0)).toThrow()
  })
})

describe('pointProcesses: the slot', () => {
  it('each family runs and produces increasing events inside the horizon', () => {
    for (const kind of pointProcesses.names) {
      const r = runAlgorithm(pointProcess({ kind }, 6), { key: key(2), maxSteps: 100_000 })
      const times = historyArray(r.state.events)
      expect(r.state.finished).toBe(true)
      expect(times.every((t, i) => t <= 6 && (i === 0 || t > (times[i - 1] ?? 0)))).toBe(true)
    }
  })

  it('knows which families are self-exciting or thinned', () => {
    expect(pointProcesses.where((p) => p.selfExciting)).toEqual(['hawkes'])
    expect(pointProcesses.where((p) => p.thinned)).toEqual(['sinusoidal', 'hawkes'])
    expect(pointProcesses.where((p) => p.memoryless)).toEqual(['poisson', 'renewal-gamma', 'renewal-weibull'])
  })
})
