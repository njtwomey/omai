import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../testing/check.ts'
import { meanOf, varianceOf } from '../base/index.ts'
import { cycleGraph, starGraph } from '../graph/index.ts'
import { key } from '../random/index.ts'
import { tensor, toFlat, type Value } from '../tensor/index.ts'
import { recorder } from '../trace/index.ts'
import { occupation } from './markov-chains.ts'
import { valuesAt } from './paths.ts'
import {
  countSelfAvoidingWalks,
  gamblersRuin,
  graphRandomWalk,
  graphWalkStationary,
  latticeWalk,
  levyFlight,
  meanSquaredDisplacement,
  selfAvoidingWalks,
  simpleRandomWalk,
  simpleWalkPmf,
  simulateGamblersRuin,
} from './random-walks.ts'

describe('simple and biased walks', () => {
  it('have mean n (2p - 1) and variance 4 n p (1 - p)', () => {
    const w = simpleRandomWalk(key(1), 40, { p: 0.6, paths: 3000 })
    const end = toFlat(valuesAt(w, -1))
    expect(Math.abs(meanOf(end) - 40 * 0.2)).toBeLessThan(4 * Math.sqrt((40 * 4 * 0.24) / 3000))
    expect(Math.abs(varianceOf(end) - 40 * 4 * 0.24)).toBeLessThan(5 * 38.4 * Math.sqrt(2 / 3000))
    expect(end.every((x) => Math.abs(x) % 2 === 0)).toBe(true)
    expect(() => simpleRandomWalk(key(0), 3, { p: 2 })).toThrow()
  })

  it('has a pmf summing to one that differentiates in p', () => {
    const k = Array.from({ length: 21 }, (_, i) => i - 10)
    expect(toFlat(simpleWalkPmf(k, 10, 0.3)).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
    expectConformant((p: Value) => simpleWalkPmf([-4, 0, 2, 3], 6, p), [tensor(0.35)])
  })

  it('lattice walks spread diffusively, E|X_n|^2 = n', () => {
    const w = latticeWalk(key(2), 30, { dim: 3, paths: 2000 })
    const msd = meanSquaredDisplacement(w)
    expect(Math.abs((msd[30] ?? 0) - 30)).toBeLessThan(3)
    expect(msd[1]).toBeCloseTo(1, 12)
  })

  it('Levy flights are superdiffusive with heavy-tailed steps', () => {
    const f = levyFlight(key(3), 50, { alpha: 1.5, paths: 500 })
    const first = toFlat(valuesAt(f, 1))
    const steps: number[] = []
    for (let i = 0; i < first.length; i += 2) steps.push(Math.hypot(first[i] ?? 0, first[i + 1] ?? 0))
    // P(L > l) = l^{-alpha}: the median step length is 2^{1/alpha}
    const sorted = [...steps].sort((a, b) => a - b)
    expect(Math.abs((sorted[250] ?? 0) - 2 ** (1 / 1.5))).toBeLessThan(0.15)
    expect(steps.every((s) => s >= 1 - 1e-12)).toBe(true)
    expect(() => levyFlight(key(0), 3, { alpha: 0 })).toThrow()
  })
})

describe('walks on graphs', () => {
  it('visit nodes in proportion to their degrees', () => {
    const g = starGraph(4)
    const w = graphRandomWalk(key(4), g, 0, 41, { paths: 400, laziness: 0.5 })
    const freq = occupation({ times: w.times, values: valuesAt(w, -1) }, 5)
    const pi = toFlat(graphWalkStationary(g))
    for (let s = 0; s < 5; s++) expect(Math.abs((freq[s] ?? 0) - (pi[s] ?? 0))).toBeLessThan(4 * Math.sqrt(0.25 / 400))
    expect(() => graphWalkStationary(cycleGraph(3, { directed: true }))).toThrow()
  })
})

describe('self-avoiding walks', () => {
  it('Rosenbluth weights estimate the number of walks', () => {
    const s = selfAvoidingWalks(key(5), 10, { walks: 3000 })
    const w = Array.from(s.weights)
    const exact = countSelfAvoidingWalks(10)
    expect(exact).toBe(44100)
    expect(Math.abs(meanOf(w) - exact)).toBeLessThan(4 * Math.sqrt(varianceOf(w) / w.length))
    expect(s.positions.shape).toEqual([3000, 11, 2])
    // a walk never revisits a site
    const p = toFlat(s.positions)
    for (let j = 0; j < 20; j++) {
      const sites = new Set<string>()
      for (let i = 0; i <= (s.lengths[j] ?? 0); i++) sites.add(`${p[(j * 11 + i) * 2]},${p[(j * 11 + i) * 2 + 1]}`)
      expect(sites.size).toBe((s.lengths[j] ?? 0) + 1)
    }
  })

  it('records trapped walks with zero weight and refuses huge enumerations', () => {
    const s = selfAvoidingWalks(key(6), 60, { walks: 200 })
    const trapped = Array.from(s.lengths).filter((l) => l < 60).length
    expect(Array.from(s.weights).filter((w) => w === 0).length).toBe(trapped)
    expect(() => countSelfAvoidingWalks(30)).toThrow()
    expect(countSelfAvoidingWalks(3, 1)).toBe(2)
  })
})

describe("gambler's ruin", () => {
  it('simulations match the closed forms', () => {
    const exact = gamblersRuin(3, 8, 0.45)
    const rec = recorder()
    const sim = simulateGamblersRuin(key(7), 3, 8, 0.45, { gamblers: 4000, tracer: rec })
    const won = Array.from(sim.won)
    const p = meanOf(won)
    expect(Math.abs(p - exact.winProbability)).toBeLessThan(4 * Math.sqrt((p * (1 - p)) / 4000))
    const rounds = Array.from(sim.rounds)
    expect(Math.abs(meanOf(rounds) - exact.expectedDuration)).toBeLessThan(4 * Math.sqrt(varianceOf(rounds) / 4000))
    expect(rec.run().events.some((e) => e.type === 'metric' && e.name === 'ruined')).toBe(true)
  })

  it('handles the edges and the fair game', () => {
    expect(gamblersRuin(0, 5, 0.3).winProbability).toBe(0)
    expect(gamblersRuin(5, 5, 0.3).winProbability).toBeCloseTo(1, 14)
    expect(gamblersRuin(2, 4).expectedDuration).toBe(4)
    // continuity at p = 1/2
    expect(gamblersRuin(3, 10, 0.5 + 1e-9).winProbability).toBeCloseTo(0.3, 6)
    expect(() => gamblersRuin(6, 5)).toThrow()
    expect(() => simulateGamblersRuin(key(0), 50, 100, 0.5, { gamblers: 5, maxRounds: 10 })).toThrow()
  })
})
