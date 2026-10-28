import { describe, expect, it } from 'vitest'
import { expectResumable, simulateArma } from '../../testing/series.ts'
import { metricSeries, recorder, runAlgorithm } from '../trace/index.ts'
import * as T from './index.ts'

const noise = simulateArma(61, 300, [], [], 0, 1)
const shifted = noise.map((v, t) => v + (t >= 150 ? 2 : 0))

describe('CUSUM', () => {
  it('stays quiet on in-control noise and alarms soon after a shift', () => {
    const quiet = T.cusum(noise, { target: 0, sigma: 1, threshold: 8 })
    expect(quiet.alarms).toHaveLength(0)
    const run = T.cusum(shifted, { target: 0, sigma: 1 })
    const first = run.alarms[0]
    expect(first?.direction).toBe('up')
    expect(first?.index).toBeGreaterThanOrEqual(150)
    expect(first?.index).toBeLessThan(165)
  })

  it('matches the defining recursion', () => {
    const x = [0.3, -0.2, 1.4, 2.2, 0.9, -1.1, -2.5, -1.7]
    const s = T.cusumAlgorithm(x, { target: 0, sigma: 1, threshold: 100 })
    const rec = recorder()
    runAlgorithm(s, { maxSteps: 100, tracer: rec })
    const high = metricSeries(rec.run(), 'high').values.slice(1)
    let h = 0
    let l = 0
    x.forEach((v, t) => {
      h = Math.max(0, h + v - 0.5)
      l = Math.max(0, l - v - 0.5)
      expect(high[t]).toBeCloseTo(h, 12)
    })
    expect(l).toBeGreaterThan(0)
  })

  it('detects a downward shift and estimates the target from the warm-up', () => {
    const down = noise.map((v, t) => v - (t >= 150 ? 2.5 : 0) + 10)
    const a = T.cusum(down, { warmup: 50 }).alarms[0]
    expect(a?.direction).toBe('down')
    expect(a?.index).toBeGreaterThanOrEqual(150)
  })

  it('serialises, resumes and refuses bad settings', () => {
    expectResumable(T.cusumAlgorithm(shifted, { warmup: 30 }), 60, 80)
    expectResumable(T.cusumAlgorithm(shifted, { target: 0, sigma: 1 }), 155, 20)
    expect(() => T.cusum(noise, { threshold: -1 })).toThrow()
    expect(() => T.cusum(noise, { sigma: 0, target: 0 })).toThrow()
  })
})

describe('Page-Hinkley', () => {
  it('stays quiet on a stationary stream and alarms after a rise', () => {
    expect(T.pageHinkley(noise, { threshold: 60, delta: 0.1 }).alarms).toHaveLength(0)
    const first = T.pageHinkley(shifted, { threshold: 30, delta: 0.1, direction: 'up' }).alarms[0]
    expect(first?.direction).toBe('up')
    expect(first?.index).toBeGreaterThanOrEqual(150)
    expect(first?.index).toBeLessThan(200)
  })

  it('detects a fall with the direction both', () => {
    const fall = noise.map((v, t) => v - (t >= 150 ? 3 : 0))
    expect(T.pageHinkley(fall, { threshold: 30, delta: 0.1 }).alarms[0]?.direction).toBe('down')
  })

  it('records both statistics, serialises and resumes', () => {
    const alg = T.pageHinkleyAlgorithm(shifted, { threshold: 30, delta: 0.1 })
    const rec = recorder()
    runAlgorithm(alg, { maxSteps: 400, tracer: rec })
    expect(metricSeries(rec.run(), 'statisticUp').values.length).toBeGreaterThan(100)
    expectResumable(alg, 100, 100)
    expect(() => T.pageHinkley(noise, { alpha: 0 })).toThrow()
  })
})
