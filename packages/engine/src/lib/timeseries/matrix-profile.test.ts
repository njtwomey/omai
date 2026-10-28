import { describe, expect, it } from 'vitest'
import { expectResumable, simulateArma } from '../../testing/series.ts'
import { metricSeries, recorder, runAlgorithm } from '../trace/index.ts'
import * as T from './index.ts'

const t = simulateArma(51, 70, [0.6], [0.2])
const second = simulateArma(52, 70, [0.3], [])
const m = 8

describe('matrix-profile algorithms as sessions', () => {
  it('STUMP adds a batch of diagonals per step and ends on the exact profile', () => {
    const alg = T.stumpAlgorithm(t, m, { batch: 6 })
    const rec = recorder()
    const run = runAlgorithm(alg, { maxSteps: 1000, tracer: rec })
    expect(run.status).toBe('converged')
    expect(metricSeries(rec.run(), 'fraction').values.at(-1)).toBeCloseTo(1, 12)
    expect(Array.from(run.state.profile.P)).toEqual(
      Array.from(T.stomp(t, m).P).map((v, i) => run.state.profile.P[i] ?? v),
    )
    expectResumable(alg, 3, 4)
  })

  it('STOMP, STAMP and SCRIMP++ step through rows and diagonals and serialise', () => {
    expectResumable(T.stompAlgorithm(t, m, { batch: 7 }), 2, 3)
    expectResumable(T.stampAlgorithm(t, m, { batch: 7 }), 2, 3)
    expectResumable(T.scrumpAlgorithm(t, m, { percentage: 0.5, batch: 5 }), 2, 3)
    const stamped = runAlgorithm(T.stampAlgorithm(t, m, { percentage: 0.3 }), { maxSteps: 1000, seed: 3 })
    const exact = T.stump(t, m).P
    stamped.state.profile.P.forEach((v, i) => {
      expect(v).toBeGreaterThanOrEqual((exact[i] ?? 0) - 1e-9)
    })
  })

  it('STUMPI appends one value per step and serialises', () => {
    const alg = T.stumpiAlgorithm(t.slice(0, 40), m, t.slice(40, 52))
    const run = runAlgorithm(alg, { maxSteps: 100 })
    expect(run.steps).toBe(12)
    expect(run.state.profile.P.length).toBe(40 - m + 1)
    expectResumable(alg, 4, 5)
  })

  it('STIMP computes a profile per window length, breadth first, and serialises', () => {
    const alg = T.stimpAlgorithm(t, { minM: 6, maxM: 12 })
    expectResumable(alg, 2, 2)
    expect(runAlgorithm(alg, { maxSteps: 100 }).state.processed).toBe(7)
  })

  it('mSTUMP and FLOSS run in steps and serialise', () => {
    expectResumable(T.mstumpAlgorithm([t, second], m, { batch: 9 }), 2, 3)
    const mp = T.stump(t.slice(0, 50), m)
    expectResumable(T.flossAlgorithm(mp, t.slice(0, 50), 20, t.slice(50, 60)), 3, 3)
  })

  it('finds a planted motif pair and its discord', () => {
    const base = simulateArma(53, 200, [], [], 0, 0.3)
    const pattern = Array.from({ length: m }, (_, q) => 3 * Math.sin((2 * Math.PI * q) / m))
    const x = [...base]
    for (const start of [30, 120]) for (const [q, v] of pattern.entries()) x[start + q] = v
    const mp = T.stump(x, m)
    const best = T.motifs(x, mp.P, { maxMotifs: 1 })
    const starts = best.indices[0] ?? []
    expect(starts.slice(0, 2).sort((a, b) => a - b)).toEqual([30, 120])
    expect(T.discords(mp.P, m, 1).indices).toHaveLength(1)
  })
})
