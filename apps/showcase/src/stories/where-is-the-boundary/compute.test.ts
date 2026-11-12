import { describe, expect, it } from 'vitest'
import { fit } from './compute.ts'
import { drawPoints, POINTS, shuffled } from './data.ts'
import { readData, readResult } from './reading.ts'

describe('the where-is-the-boundary computation', () => {
  it('has forty points of two classes that are the same every time', () => {
    const a = drawPoints()
    expect(a.y).toHaveLength(POINTS)
    expect(drawPoints().x).toEqual(a.x)
    const read = readData(a)
    expect(read.first + read.second).toBe(POINTS)
    expect(read.first).toBeGreaterThan(5)
    expect(read.second).toBeGreaterThan(5)
    // another order holds the same points
    const s = shuffled(a, 3)
    expect(s.y.filter((c) => c === 1)).toHaveLength(read.second)
    expect([...s.x].sort()).toEqual([...a.x].sort())
  })

  it('returns plain data of the right shapes on a small run', () => {
    const said: string[] = []
    const r = fit(1, (s) => said.push(s), { points: 20, grid: 9 })
    expect(r.points.y).toHaveLength(20)
    expect(r.xs).toHaveLength(9)
    expect(r.ys).toHaveLength(9)
    expect(r.probability).toHaveLength(9)
    expect(r.probability.every((row) => row.length === 9)).toBe(true)
    expect(r.doubt.every((row) => row.every((v) => v > 0))).toBe(true)
    expect(r.probability.every((row) => row.every((v) => v >= 0 && v <= 1))).toBe(true)
    expect(r.settled).toBe(true)
    expect(r.kernel.lengthScale).toBeGreaterThan(0)
    expect(said.length).toBeGreaterThanOrEqual(3)
    // The result crosses the worker boundary, so it must survive a structured clone.
    expect(structuredClone(r)).toEqual(r)
  }, 30000)

  it('on the real run puts most points on their own side, and is unsure far from the data', () => {
    const r = fit(1)
    const read = readResult(r)
    expect(r.probability).toHaveLength(41)
    expect(read.right).toBeGreaterThanOrEqual(34)
    // far from every point the chance is close to a half and the doubt about the score is greater than near the points
    expect(read.corners.every((c) => Math.abs(c - 0.5) < 0.25)).toBe(true)
    expect(read.doubtFar).toBeGreaterThan(read.doubtNear)
  }, 60000)

  it('gives the same map whatever order the points are handed over in', () => {
    const a = fit(1, undefined, { points: 40, grid: 11 })
    const b = fit(2, undefined, { points: 40, grid: 11 })
    const gaps = a.probability.flatMap((row, i) =>
      row.map((v, j) => Math.abs(v - (b.probability[i]?.[j] ?? Number.NaN))),
    )
    expect(Math.max(...gaps)).toBeLessThan(0.01)
  }, 60000)
})
