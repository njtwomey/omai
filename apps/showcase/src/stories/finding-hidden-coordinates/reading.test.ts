import { describe, expect, it } from 'vitest'
import type { Result } from './compute.ts'
import { readResult } from './reading.ts'

const made = (recovered: readonly number[], spread: number, correlation: number, residual: number): Result => ({
  seed: 1,
  truth: [-1, 0, 1],
  recovered,
  spread: [spread, spread, spread],
  y: [],
  clean: [],
  grid: [],
  curves: [],
  residual,
  noise: 0.1,
  correlation,
  error: 0,
  sweeps: 70,
  settled: false,
})

describe('what the prose reads from a result', () => {
  it('says a run found the coordinate, and compares the stated spread with the miss', () => {
    const r = readResult(made([-0.9, 0.1, 1.1], 0.3, 0.99, 0.08))
    expect(r.found).toBe(true)
    expect(r.miss).toBeCloseTo(0.1, 12)
    expect(r.spreadIs).toBe('wider than')
    expect(r.inside).toBe(3)
    expect(r.curvesFit).toBe(true)
    expect([r.low, r.high]).toEqual([-1, 1])
  })

  it('says a run missed, with a narrow stated spread, when the recovered coordinate does not follow the truth', () => {
    const r = readResult(made([0.5, -0.8, 0.9], 0.05, 0.2, 0.4))
    expect(r.found).toBe(false)
    expect(r.spreadIs).toBe('narrower than')
    expect(r.inside).toBe(0)
    expect(r.curvesFit).toBe(false)
  })
})
