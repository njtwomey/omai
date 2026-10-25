import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { type BoidsOptions, boidsAlgorithm, polarisation } from './flocking.ts'

const run = (options: BoidsOptions, steps = 300) =>
  runAlgorithm(boidsAlgorithm({ count: 120, ...options }), { key: key(4), maxSteps: steps }).state

const meanNearest = (s: ReturnType<typeof run>): number => {
  let total = 0
  for (let i = 0; i < s.x.length; i++) {
    let m = Number.POSITIVE_INFINITY
    for (let j = 0; j < s.x.length; j++)
      if (j !== i) m = Math.min(m, Math.hypot((s.x[i] ?? 0) - (s.x[j] ?? 0), (s.y[i] ?? 0) - (s.y[j] ?? 0)))
    total += m
  }
  return total / s.x.length
}

describe('boids', () => {
  it('aligns through alignment, and not without it', () => {
    expect(polarisation(run({}))).toBeGreaterThan(0.8)
    expect(polarisation(run({ alignment: 0 }))).toBeLessThan(0.3)
  })

  it('keeps its distance through separation, and stays in the world at speeds within the limits', () => {
    const s = run({})
    expect(meanNearest(s)).toBeGreaterThan(3 * meanNearest(run({ separation: 0 })))
    for (let i = 0; i < s.x.length; i++) {
      expect(s.x[i]).toBeGreaterThanOrEqual(0)
      expect(s.x[i]).toBeLessThan(100)
      const v = Math.hypot(s.vx[i] ?? 0, s.vy[i] ?? 0)
      expect(v).toBeGreaterThanOrEqual(0.5 - 1e-12)
      expect(v).toBeLessThanOrEqual(2 + 1e-12)
    }
  })

  it('is a pure function of its key, and rejects a bad flock', () => {
    expect(Array.from(run({}, 20).x)).toEqual(Array.from(run({}, 20).x))
    expect(() => boidsAlgorithm({ count: 0 })).toThrow()
    expect(() => boidsAlgorithm({ minSpeed: 3, maxSpeed: 2 })).toThrow()
  })
})
