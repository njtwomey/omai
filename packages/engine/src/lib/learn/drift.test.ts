import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { recorder, runAlgorithm } from '../trace/index.ts'
import {
  adwin,
  adwinAlgorithm,
  ddm,
  ddmAlgorithm,
  eddm,
  eddmAlgorithm,
  kswin,
  kswinAlgorithm,
  populationStabilityIndex,
} from './drift.ts'

describe('drift', () => {
  it('ddm detects an abrupt rise in error rate and stays calm on stationary error bits', () => {
    const stationary = Array.from({ length: 80 }, (_, i) => (i % 25 === 0 ? 1 : 0))
    expect(ddm(stationary).drifts.length).toBe(0)

    const drifting = [...stationary, ...Array.from({ length: 40 }, () => 1)]
    const rec = recorder()
    const res = ddm(drifting, { tracer: rec })
    expect(res.drifts.length).toBeGreaterThan(0)
    expect(res.drifts[0]).toBeGreaterThanOrEqual(80)

    const alg = ddmAlgorithm(drifting)
    const run = runAlgorithm(alg, { maxSteps: 50 })
    const json = alg.serialise?.(run.state)
    expect(json).toBeDefined()
    expect(alg.deserialise?.(json as NonNullable<typeof json>)?.t).toBe(50)
  })

  it('eddm detects shrinking inter-error distances and serialises state', () => {
    const slow = Array.from({ length: 120 }, (_, i) => (i % 10 === 0 ? 1 : 0))
    const fast = Array.from({ length: 40 }, () => 1)
    const stream = [...slow, ...fast]
    const res = eddm(stream, { minErrors: 10 })
    expect(res.drifts.length).toBeGreaterThan(0)
    expect(res.drifts[0]).toBeGreaterThanOrEqual(120)

    const alg = eddmAlgorithm(stream, { minErrors: 10 })
    const run = runAlgorithm(alg, { maxSteps: 60 })
    const json = alg.serialise?.(run.state)
    expect(json).toBeDefined()
    expect(alg.deserialise?.(json as NonNullable<typeof json>)?.t).toBe(60)
  })

  it('adwin detects a shift in mean, shrinks its window, and serialises state', () => {
    const stream = [...Array.from({ length: 45 }, () => 0), ...Array.from({ length: 45 }, () => 1)]
    const res = adwin(stream, { delta: 0.01 })
    expect(res.drifts.length).toBeGreaterThan(0)
    expect(res.drifts[0]).toBeGreaterThanOrEqual(45)
    // Window should have shrunk after dropping the older prefix.
    expect(res.windowSizes[res.windowSizes.length - 1]).toBeLessThan(stream.length)

    const alg = adwinAlgorithm(stream, { delta: 0.01 })
    const run = runAlgorithm(alg, { maxSteps: 55 })
    const json = alg.serialise?.(run.state)
    expect(json).toBeDefined()
    expect(alg.deserialise?.(json as NonNullable<typeof json>)?.t).toBe(55)
  })

  it('kswin detects distribution shift via two-sample KS and serialises state', () => {
    const s0 = Array.from({ length: 45 }, (_, i) => Math.sin(i) * 0.2)
    const s1 = Array.from({ length: 35 }, (_, i) => 5 + Math.sin(i) * 0.2)
    const stream = [...s0, ...s1]
    const res = kswin(stream, { windowSize: 40, statSize: 15, alpha: 0.01, key: key(4) })
    expect(res.drifts.length).toBeGreaterThan(0)
    expect(res.drifts[0]).toBeGreaterThanOrEqual(45)

    const alg = kswinAlgorithm(stream, { windowSize: 40, statSize: 15, alpha: 0.01, key: key(4) })
    const run = runAlgorithm(alg, { maxSteps: 50 })
    const json = alg.serialise?.(run.state)
    expect(json).toBeDefined()
    expect(alg.deserialise?.(json as NonNullable<typeof json>)?.t).toBe(50)
  })

  it('populationStabilityIndex is the published sum (q - p) ln(q / p) over reference-quantile bins', () => {
    // Reference 1..10 in five quantile bins of two values each; the current sample has counts 4, 1, 1, 2, 2 of ten.
    const ref = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    const cur = [1, 1.5, 2, 2.5, 3, 5, 7, 7.5, 9, 10]
    const r = populationStabilityIndex(ref, cur, { nBins: 5 })
    expect(r.edges.map((e) => Math.round(e * 10) / 10)).toEqual([1, 2.8, 4.6, 6.4, 8.2, 10])
    expect(r.referenceProportions).toEqual([0.2, 0.2, 0.2, 0.2, 0.2])
    expect(r.currentProportions).toEqual([0.4, 0.1, 0.1, 0.2, 0.2])
    const expected = [0.4, 0.1, 0.1, 0.2, 0.2].reduce((a, q) => a + (q - 0.2) * Math.log(q / 0.2), 0)
    expect(r.psi).toBeCloseTo(expected, 12)
    expect(expected).toBeCloseTo(0.4 * Math.log(2), 12)
    expect(r.bins.reduce((a, b) => a + b, 0)).toBeCloseTo(r.psi, 14)
    expect(r.status).toBe('drift')

    expect(populationStabilityIndex(ref, ref, { nBins: 5 }).psi).toBe(0)
    const far = populationStabilityIndex(
      ref,
      ref.map((v) => v + 20),
      { nBins: 5, strategy: 'uniform' },
    )
    expect(far.status).toBe('drift')
  })
})
