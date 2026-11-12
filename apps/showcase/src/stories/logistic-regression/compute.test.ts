import { multivariateNormal } from 'omai-engine/dists'
import { key } from 'omai-engine/random'
import { matmul, mean, sigmoid, tensor, toFlat, transpose } from 'omai-engine/tensor'
import { describe, expect, it } from 'vitest'
import { expectedChance, fit, type Size } from './compute.ts'
import { drawPoints, ROWS } from './data.ts'
import { readData, readResult } from './reading.ts'

const SMALL: Size = { grid: 9, density: 12 }

describe('the logistic-regression computation', () => {
  it('draws the same forty points every time, about half of them yes', () => {
    const a = drawPoints()
    expect(a.y).toHaveLength(ROWS)
    expect(drawPoints()).toEqual(a)
    const read = readData(a)
    expect(read.yes + read.no).toBe(ROWS)
    expect(read.yes).toBeGreaterThan(10)
    expect(read.no).toBeGreaterThan(10)
  })

  it('returns plain data of the right shapes on a small run', () => {
    const said: string[] = []
    const r = fit(1, (s) => said.push(s), SMALL)
    expect(r.xs).toHaveLength(9)
    expect(r.chance).toHaveLength(9)
    expect(r.chance.every((row) => row.length === 9 && row.every((v) => v >= 0 && v <= 1))).toBe(true)
    expect(r.doubt.every((row) => row.every((v) => v > 0))).toBe(true)
    expect(r.joint.density).toHaveLength(12)
    expect(r.joint.density.every((row) => row.length === 12)).toBe(true)
    expect(r.curves.axis).toHaveLength(200)
    expect(r.curves.first).toHaveLength(200)
    expect(r.settled).toBe(true)
    expect(r.count).toBe(ROWS)
    expect(said.length).toBeGreaterThanOrEqual(3)
    // The result crosses the worker boundary, so it must survive a structured clone.
    expect(structuredClone(r)).toEqual(r)
  })

  it('finds weights with the signs the points were drawn with, and most points on their own side', () => {
    const r = fit(1)
    expect(r.weights.first.mean).toBeGreaterThan(0)
    expect(r.weights.second.mean).toBeLessThan(0)
    expect(r.weights.first.lo).toBeLessThan(r.weights.first.hi)
    // the true weights are carried in the result, for the figures to mark
    expect([r.truth.offset, r.truth.first, r.truth.second]).toEqual([0.4, 1.3, -0.9])
    expect(r.right).toBeGreaterThanOrEqual(30)
    const read = readResult(r)
    expect(read.first.factor).toBeGreaterThan(1)
    expect(read.second.factor).toBeLessThan(1)
    // the score is surer in the middle of the points than at the corners
    expect(read.doubtCorners).toBeGreaterThan(read.doubtMiddle)
    // the densities integrate to about one
    const step = (r.curves.axis[1] as number) - (r.curves.axis[0] as number)
    expect(r.curves.first.reduce((a, b) => a + b, 0) * step).toBeCloseTo(1, 2)
  })

  it('gives the same belief whatever order the points are handed over in', () => {
    const a = fit(1, undefined, SMALL)
    const b = fit(2, undefined, SMALL)
    expect(Math.abs(a.weights.first.mean - b.weights.first.mean)).toBeLessThan(1e-6)
    expect(Math.abs(a.weights.second.hi - b.weights.second.hi)).toBeLessThan(1e-6)
  })

  it('shows the true posterior predictive: the field matches an average over 200,000 draws of the belief, near and far', () => {
    const r = fit(1, undefined, SMALL)
    const centre = tensor([...r.belief.mean])
    const covariance = tensor(r.belief.covariance.map((row) => [...row]))
    // the middle of the data, its edge, the corners of the figure, and places far outside it where the score's variance is large
    const places = [
      [0, 0],
      [1, -1],
      [3.5, 3.5],
      [-3.5, 3.5],
      [3.5, -3.5],
      [10, -10],
      [-12, -12],
      [15, 5],
    ]
    const rows = tensor(places.map((p) => [1, ...p]))
    const quadrature = expectedChance(rows, centre, covariance)
    // the score's spread is at least 5 (variance 25) at the far places, which the rule must still handle
    expect(Math.max(...toFlat(quadrature.doubt))).toBeGreaterThan(5)
    const draws = multivariateNormal(centre, { covariance }).sample(key(7), [200000])
    const brute = toFlat(mean(sigmoid(matmul(draws, transpose(rows))), 0))
    toFlat(quadrature.chance).forEach((c, i) => {
      expect(Math.abs(c - (brute[i] as number))).toBeLessThan(0.006)
    })
  }, 120000)

  it('has a straight one-half line and lines for 0.2 and 0.8 that are farther apart near the edge than in the middle', () => {
    const r = fit(1)
    expect(r.gaps.edge).toBeGreaterThan(r.gaps.centre)
    // along each row of the map the chance crosses one half where the mean score is zero: the same place as the straight line
    const [offset, first, second] = [r.belief.mean[0] as number, r.belief.mean[1] as number, r.belief.mean[2] as number]
    r.ys.forEach((y, row) => {
      const line = -(offset + second * y) / first
      const cells = r.chance[row] as number[]
      const i = cells.findIndex((c, k) => k > 0 && ((cells[k - 1] as number) - 0.5) * (c - 0.5) <= 0)
      // the straight line falls in the cell where the chance crosses one half
      if (i > 0 && Math.abs(line) < 3.4) {
        expect(line).toBeGreaterThanOrEqual((r.xs[i - 1] as number) - 1e-6)
        expect(line).toBeLessThanOrEqual((r.xs[i] as number) + 1e-6)
      }
    })
  })
})
