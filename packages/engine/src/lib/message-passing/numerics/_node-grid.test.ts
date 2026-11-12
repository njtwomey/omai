// The zoomed grid of nodes over one to three coordinates against integrals known in closed form: a Gaussian times a Gaussian (a narrow
// likelihood against a wide cavity, the case the zoom exists for), a product of two Gaussians in two coordinates, and a density with
// two separated peaks, whose nodes it must keep both of.
import { describe, expect, it } from 'vitest'
import { ImproperMessageError } from '../core/errors.ts'
import { zoomNodeGrid } from './_node-grid.ts'

const LOG_2PI = Math.log(2 * Math.PI)
const logNormal = (x: number, mean: number, variance: number): number =>
  -0.5 * (LOG_2PI + Math.log(variance) + (x - mean) ** 2 / variance)

/** The mass, mean and variance of coordinate `i` of a grid. */
function moments(grid: ReturnType<typeof zoomNodeGrid>, i: number): { mass: number; mean: number; variance: number } {
  let sum = 0
  let first = 0
  for (let k = 0; k < grid.count; k++) {
    sum += grid.weights[k] as number
    first += (grid.weights[k] as number) * (grid.nodes[k * grid.q + i] as number)
  }
  const mean = first / sum
  let second = 0
  for (let k = 0; k < grid.count; k++)
    second += (grid.weights[k] as number) * ((grid.nodes[k * grid.q + i] as number) - mean) ** 2
  return { mass: Math.exp(grid.top) * grid.total * grid.volume, mean, variance: second / sum }
}

describe('zoomNodeGrid', () => {
  it('finds the mass and moments of a narrow Gaussian likelihood against a wide cavity in one coordinate', () => {
    // cavity N(0, 1), likelihood N(y = 1.3; x, 0.05^2): the product has mass N(1.3; 0, 1 + 0.0025) and a posterior
    // sd of about 0.05, forty times narrower than the window of 16 cavity sds.
    const noise = 0.05 ** 2
    const grid = zoomNodeGrid(
      [{ lo: -8, hi: 8 }],
      161,
      (x) => logNormal(x[0] as number, 0, 1) + logNormal(1.3, x[0] as number, noise),
    )
    const m = moments(grid, 0)
    const variance = 1 / (1 + 1 / noise)
    expect(m.mass).toBeCloseTo(Math.exp(logNormal(1.3, 0, 1 + noise)), 8)
    expect(m.mean).toBeCloseTo((1.3 / noise) * variance, 8)
    expect(m.variance).toBeCloseTo(variance, 8)
  })

  it('does the same in two coordinates, with the first varying fastest and the cell volume of the last grid', () => {
    const grid = zoomNodeGrid(
      [
        { lo: -6, hi: 6 },
        { lo: -6, hi: 6 },
      ],
      61,
      (x) =>
        logNormal(x[0] as number, 0, 1) +
        logNormal(x[1] as number, 0, 1) +
        logNormal(0.8, x[0] as number, 0.1) +
        logNormal(-0.4, x[1] as number, 0.1),
    )
    expect(grid.count).toBe(61 * 61)
    expect(grid.nodes[1 * 2 + 0]).not.toBe(grid.nodes[0])
    expect(grid.nodes[1 * 2 + 1]).toBe(grid.nodes[1])
    const a = moments(grid, 0)
    const b = moments(grid, 1)
    const v = 1 / (1 + 1 / 0.1)
    expect(a.mean).toBeCloseTo((0.8 / 0.1) * v, 6)
    expect(b.mean).toBeCloseTo((-0.4 / 0.1) * v, 6)
    expect(a.variance).toBeCloseTo(v, 6)
    const mass = Math.exp(logNormal(0.8, 0, 1.1) + logNormal(-0.4, 0, 1.1))
    expect(a.mass).toBeCloseTo(mass, 6)
  })

  it('keeps both peaks of a density with two separated ones', () => {
    const grid = zoomNodeGrid([{ lo: -8, hi: 8 }], 241, (x) =>
      Math.log(
        0.5 * Math.exp(logNormal(x[0] as number, -3, 0.01)) + 0.5 * Math.exp(logNormal(x[0] as number, 3, 0.01)),
      ),
    )
    const m = moments(grid, 0)
    expect(m.mass).toBeCloseTo(1, 3)
    expect(m.mean).toBeCloseTo(0, 3)
    expect(m.variance).toBeCloseTo(9.01, 1)
  })

  it('refuses a grid with no finite weight, as an improper message', () => {
    expect(() => zoomNodeGrid([{ lo: -1, hi: 1 }], 11, () => Number.NEGATIVE_INFINITY)).toThrow(ImproperMessageError)
  })
})
