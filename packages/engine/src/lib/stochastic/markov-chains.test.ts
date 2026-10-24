import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { recorder } from '../trace/index.ts'
import {
  ctmcAlgorithm,
  ctmcDistribution,
  ctmcHittingTimes,
  ctmcPaths,
  ctmcStationary,
  embeddedChain,
  simulateCtmc,
  trajectoryAt,
  uniformise,
} from './ctmc.ts'
import {
  absorption,
  classifyStates,
  expectedHittingTimes,
  hittingProbabilities,
  isReversible,
  meanReturnTimes,
  mixingTime,
  occupation,
  simulateChain,
  spectralGap,
  stationaryDistribution,
  stationaryDistributions,
} from './markov-chains.ts'
import { valuesAt } from './paths.ts'

const weather = [
  [0.7, 0.2, 0.1],
  [0.3, 0.4, 0.3],
  [0.2, 0.3, 0.5],
]

describe('classification', () => {
  it('finds classes, periods, recurrence and absorbing states', () => {
    const c = classifyStates([
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [1, 0, 0, 0],
      [0.5, 0, 0, 0.5],
    ])
    expect(c.classes).toEqual([[0, 1, 2], [3]])
    expect(c.periods).toEqual([3, 1])
    expect(c.closed).toEqual([true, false])
    expect(c.irreducible).toBe(false)
    const a = classifyStates(weather)
    expect(a.irreducible && a.aperiodic).toBe(true)
    expect(
      classifyStates([
        [1, 0],
        [0.5, 0.5],
      ]).absorbing,
    ).toEqual([0])
  })

  it('rejects matrices that are not stochastic', () => {
    expect(() =>
      classifyStates([
        [0.5, 0.4],
        [0, 1],
      ]),
    ).toThrow()
    expect(() =>
      classifyStates([
        [1.5, -0.5],
        [0, 1],
      ]),
    ).toThrow()
    expect(() => classifyStates([[1, 0]])).toThrow()
    expect(() =>
      stationaryDistribution([
        [1, 0],
        [0, 1],
      ]),
    ).toThrow()
  })
})

describe('stationary laws, hitting and absorption', () => {
  it('gives one stationary law per closed class', () => {
    const s = stationaryDistributions([
      [0.5, 0.5, 0, 0],
      [0.5, 0.5, 0, 0],
      [0.1, 0.1, 0.4, 0.4],
      [0, 0, 0, 1],
    ])
    expect(s.shape).toEqual([2, 4])
    expect(toFlat(s)).toEqual([0.5, 0.5, 0, 0, 0, 0, 0, 1])
  })

  it("solves gambler's ruin by hitting probabilities and absorption, with Kac's return times", () => {
    const p = 0.4
    const n = 5
    const rows = Array.from({ length: n + 1 }, (_, i) =>
      Array.from({ length: n + 1 }, (_, j) => {
        if (i === 0 || i === n) return i === j ? 1 : 0
        return j === i + 1 ? p : j === i - 1 ? 1 - p : 0
      }),
    )
    const r = (1 - p) / p
    const h = toFlat(hittingProbabilities(rows, [n]))
    for (let i = 0; i <= n; i++) expect(h[i]).toBeCloseTo((1 - r ** i) / (1 - r ** n), 12)
    const a = absorption(rows)
    expect(a.transient).toEqual([1, 2, 3, 4])
    const win = toFlat(a.probabilities).filter((_, k) => k % 2 === 1)
    for (let i = 1; i < n; i++) expect(win[i - 1]).toBeCloseTo(h[i] ?? 0, 12)
    // expected duration (i - n h_i) / (q - p)
    const steps = toFlat(a.expectedSteps)
    for (let i = 1; i < n; i++) expect(steps[i - 1]).toBeCloseTo((i - n * (h[i] ?? 0)) / (1 - 2 * p), 10)
    expect(toFlat(a.varianceSteps).every((v) => v > 0)).toBe(true)
    const pi = toFlat(stationaryDistribution(weather))
    toFlat(meanReturnTimes(weather)).forEach((m, i) => {
      expect(m).toBeCloseTo(1 / (pi[i] ?? 1), 12)
    })
  })

  it('gives infinite hitting times when the target can be missed', () => {
    const k = toFlat(
      expectedHittingTimes(
        [
          [1, 0, 0],
          [0.5, 0, 0.5],
          [0, 0, 1],
        ],
        [2],
      ),
    )
    expect(k).toEqual([Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, 0])
    expect(() => absorption(weather)).toThrow()
    expect(() => hittingProbabilities(weather, [])).toThrow()
    expect(() => hittingProbabilities(weather, [7])).toThrow()
  })
})

describe('mixing, spectra and reversibility', () => {
  it('mixes at the rate of the spectral gap', () => {
    const m = mixingTime(weather, { epsilon: 1e-3 })
    expect(m.distances[0]).toBeGreaterThan(0.5)
    for (let t = 1; t < m.distances.length; t++)
      expect(m.distances[t]).toBeLessThanOrEqual((m.distances[t - 1] ?? 1) + 1e-15)
    const g = spectralGap(weather)
    // d(t) decays like (1 - gap)^t: the mixing time is of order relaxation time times log(1 / eps)
    expect(m.time).toBeLessThan(10 * g.relaxationTime * Math.log(1e3))
    expect(g.moduli[0]).toBeCloseTo(1, 12)
    expect(() => mixingTime(weather, { epsilon: 2 })).toThrow()
  })

  it('detects reversibility of birth-death chains and non-reversibility of cycles', () => {
    expect(
      isReversible([
        [0.5, 0.5, 0],
        [0.3, 0.2, 0.5],
        [0, 0.6, 0.4],
      ]),
    ).toBe(true)
    expect(isReversible(weather)).toBe(false)
    expect(spectralGap([[1]]).gap).toBe(1)
  })
})

describe('simulation', () => {
  it('visits states with the stationary frequencies', () => {
    const p = simulateChain(key(1), weather, 0, 60, { paths: 500 })
    expect(p.values.shape).toEqual([500, 61])
    const last = toFlat(valuesAt(p, -1))
    const pi = toFlat(stationaryDistribution(weather))
    for (let s = 0; s < 3; s++) {
      const f = last.filter((x) => x === s).length / 500
      expect(Math.abs(f - (pi[s] ?? 0))).toBeLessThan(4 * Math.sqrt(((pi[s] ?? 0) * (1 - (pi[s] ?? 0))) / 500))
    }
    expect(occupation(p, 3).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
    expect(() => simulateChain(key(0), weather, 5, 3)).toThrow()
    const many = simulateChain(key(2), weather, [0, 1, 2], 2)
    expect(many.values.shape).toEqual([3, 3])
  })
})

describe('continuous-time chains', () => {
  const q = [
    [-2, 1.5, 0.5],
    [1, -1.5, 0.5],
    [0.5, 0.5, -1],
  ]

  it('Gillespie trajectories spend the stationary fraction of time in each state', () => {
    const rec = recorder()
    const traj = simulateCtmc(key(3), q, 0, 400, { tracer: rec })
    expect(traj.times[0]).toBe(0)
    const time = new Float64Array(3)
    for (let i = 0; i < traj.times.length; i++) {
      const end = traj.times[i + 1] ?? 400
      time[traj.states[i] ?? 0] = (time[traj.states[i] ?? 0] ?? 0) + end - (traj.times[i] ?? 0)
    }
    const pi = toFlat(ctmcStationary(q))
    for (let s = 0; s < 3; s++) expect(Math.abs((time[s] ?? 0) / 400 - (pi[s] ?? 0))).toBeLessThan(0.06)
    expect(rec.run().events.some((e) => e.type === 'metric' && e.name === 'state')).toBe(true)
    expect(Array.from(trajectoryAt(traj, [0]))).toEqual([0])
  })

  it('stops at absorbing states and caps the number of jumps', () => {
    const t = simulateCtmc(
      key(4),
      [
        [-1, 1],
        [0, 0],
      ],
      0,
      1e6,
    )
    expect(Array.from(t.states)).toEqual([0, 1])
    expect(() => simulateCtmc(key(5), q, 0, 1e6, { maxJumps: 10 })).toThrow()
    expect(() => ctmcAlgorithm(q, 0, 0)).toThrow()
    expect(() =>
      ctmcAlgorithm(
        [
          [1, -1],
          [0, 0],
        ],
        0,
        1,
      ),
    ).toThrow()
  })

  it('grid paths have the law p0 exp(t Q)', () => {
    const p = ctmcPaths(key(6), q, 0, [0, 0.4, 1.1], { paths: 2000 })
    const end = toFlat(valuesAt(p, -1))
    const law = toFlat(ctmcDistribution(q, [1, 0, 0], 1.1))
    for (let s = 0; s < 3; s++) {
      const f = end.filter((x) => x === s).length / 2000
      expect(Math.abs(f - (law[s] ?? 0))).toBeLessThan(4 * Math.sqrt(((law[s] ?? 0) * (1 - (law[s] ?? 0))) / 2000))
    }
  })

  it('has hitting times, jump chains and uniformisations', () => {
    const k = toFlat(
      ctmcHittingTimes(
        [
          [-1, 1, 0],
          [0, -1, 1],
          [0, 0, 0],
        ],
        [2],
      ),
    )
    expect(k).toEqual([2, 1, 0])
    expect(
      toFlat(
        ctmcHittingTimes(
          [
            [0, 0],
            [1, -1],
          ],
          [1],
        ),
      ),
    ).toEqual([Number.POSITIVE_INFINITY, 0])
    expect(Array.from(embeddedChain(q).rates)).toEqual([2, 1.5, 1])
    expect(() => uniformise(q, 1)).toThrow()
    expect(uniformise(q, 4).rate).toBe(4)
  })
})
