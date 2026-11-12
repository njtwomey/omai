import { bayesianNeuralNetwork } from 'omai-engine/probprog/models'
import { describe, expect, it } from 'vitest'
import { fit, type Size } from './compute.ts'
import { POINTS } from './data.ts'
import { readResult } from './reading.ts'

const SMALL: Size = { chains: 2, warmup: 40, draws: 40, lines: 8, grid: 71 }

describe('the bayesian-neural-network computation', () => {
  it('draws the library benchmark data, to three decimals', () => {
    const library = bayesianNeuralNetwork.data.data as { x: number[]; y: number[] }
    expect(POINTS.x).toHaveLength(16)
    POINTS.x.forEach((x, i) => {
      expect(Math.abs(x - (library.x[i] as number))).toBeLessThan(6e-4)
      expect(Math.abs((POINTS.y[i] as number) - (library.y[i] as number))).toBeLessThan(6e-4)
    })
  })

  it('returns plain data of the right shapes on a small run', () => {
    const said: string[] = []
    const r = fit(1, (s) => said.push(s), SMALL)
    expect(r.grid).toHaveLength(71)
    expect(r.mean).toHaveLength(71)
    expect(r.lower).toHaveLength(71)
    expect(r.upper).toHaveLength(71)
    expect(r.draws).toBe(80)
    expect(r.fan).toHaveLength(8)
    expect(r.fan.every((curve) => curve.length === 71)).toBe(true)
    expect(said.length).toBeGreaterThanOrEqual(2)
    // The result crosses the worker boundary, so it must survive a structured clone.
    expect(structuredClone(r)).toEqual(r)
  })

  it('has a band that contains the mean, and a reading the prose can quote', () => {
    const r = fit(1, undefined, SMALL)
    r.grid.forEach((_, i) => {
      expect(r.lower[i]).toBeLessThanOrEqual(r.mean[i] as number)
      expect(r.upper[i]).toBeGreaterThanOrEqual(r.mean[i] as number)
    })
    const reading = readResult(r)
    expect(reading.dataWidth).toBeGreaterThan(0)
    expect(reading.gapWidth).toBeGreaterThan(0)
    expect(Number.isFinite(reading.rightRatio)).toBe(true)
  })

  it('draws different networks for a different seed', () => {
    const a = fit(1, undefined, SMALL)
    const b = fit(2, undefined, SMALL)
    expect(a.fan).not.toEqual(b.fan)
  })
})
