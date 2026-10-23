import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import { grad } from '../../autodiff/index.ts'
import { maxOf, minOf } from '../../base/index.ts'
import { item, sum, type Tensor, tensor, toFlat, type Value } from '../../tensor/index.ts'
import * as I from './index.ts'

describe('grid and scattered interpolation against numpy and scipy', () => {
  for (const c of loadFixture('numerics/interpolate/multivariate').cases) {
    it(c.name, () => {
      const t = c.inputs.map((v) => decode(v) as Tensor)
      const p = c.params
      const actual: Value =
        c.fn === 'interp'
          ? I.interp(t[0] as Tensor, t[1] as Tensor, t[2] as Tensor, {
              ...(p['left'] === undefined ? {} : { left: p['left'] as number, right: p['right'] as number }),
              ...(p['period'] === undefined ? {} : { period: p['period'] as number }),
            })
          : c.fn === 'interp1d'
            ? I.interp1d(t[0] as Tensor, t[1] as Tensor, { kind: p['kind'] as I.GridMethod })(t[2] as Tensor)
            : c.fn === 'regularGrid'
              ? I.regularGridInterpolator([t[0] as Tensor, t[1] as Tensor], t[2] as Tensor, {
                  method: p['method'] as NonNullable<I.RegularGridOptions['method']>,
                })(t[3] as Tensor)
              : c.fn === 'regularGrid3'
                ? I.regularGridInterpolator([t[0] as Tensor, t[1] as Tensor, t[2] as Tensor], t[3] as Tensor, {
                    method: p['method'] as NonNullable<I.RegularGridOptions['method']>,
                  })(t[4] as Tensor)
                : c.fn === 'griddata'
                  ? I.griddata(t[0] as Tensor, t[1] as Tensor, t[2] as Tensor, p['method'] as 'nearest' | 'linear')
                  : I.rbfInterpolator(t[0] as Tensor, t[1] as Tensor, {
                      kernel: p['kernel'] as I.RbfKernel,
                      epsilon: p['epsilon'] as number,
                      ...(p['smoothing'] === undefined
                        ? {}
                        : { smoothing: p['smoothing'] as number, degree: p['degree'] as number }),
                    })(t[2] as Tensor)
      expectMatches(actual, c.output, c.rtol, c.atol, c.name)
    })
  }
})

describe('scattered interpolation properties', () => {
  const sites: [number, number][] = [
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
    [0.4, 0.6],
    [0.7, 0.2],
    [0.2, 0.3],
    [0.8, 0.8],
  ]
  const linear = sites.map(([x, y]) => 2 * x - 3 * y + 1)
  const queries: [number, number][] = [
    [0.5, 0.5],
    [0.1, 0.9],
    [0.66, 0.31],
  ]

  it('natural-neighbour weights are a partition of unity that reproduces linear functions', () => {
    const w = toFlat(I.naturalNeighbourWeights(sites, queries))
    const n = sites.length
    queries.forEach(([x, y], r) => {
      const row = w.slice(r * n, (r + 1) * n)
      expect(row.every((v) => v >= -1e-14)).toBe(true)
      expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
      expect(row.reduce((a, b, i) => a + b * (linear[i] ?? 0), 0)).toBeCloseTo(2 * x - 3 * y + 1, 11)
      // The weights also reproduce the coordinates (the local-coordinates property).
      expect(row.reduce((a, b, i) => a + b * (sites[i]?.[0] ?? 0), 0)).toBeCloseTo(x, 11)
    })
    expect(toFlat(I.naturalNeighbour(sites, linear)([[1, 1]]))[0]).toBe(linear[3])
    expect(Number.isNaN(toFlat(I.naturalNeighbourWeights(sites, [[2, 2]]))[0])).toBe(true)
  })

  it('inverse-distance weighting interpolates the data and stays within its range', () => {
    const f = I.inverseDistanceWeighting(sites, linear)
    expect(toFlat(f(sites))).toEqual(linear)
    const inner = toFlat(f(queries))
    for (const v of inner) {
      expect(v).toBeGreaterThanOrEqual(minOf(linear))
      expect(v).toBeLessThanOrEqual(maxOf(linear))
    }
  })

  it('RBF and grid interpolants are differentiable in the query points', () => {
    const f = I.rbfInterpolator(sites, linear)
    const g = toFlat(grad((q: Value) => sum(f(q)))(tensor([[0.3, 0.4]])) as Value)
    expect(g[0]).toBeCloseTo(2, 8)
    expect(g[1]).toBeCloseTo(-3, 8)
    const axis = [0, 1, 2, 3]
    const grid = I.regularGridInterpolator(
      [axis, axis],
      axis.map((x) => axis.map((y) => 3 * x + y)),
      { method: 'cubic' },
    )
    const gg = toFlat(grad((q: Value) => sum(grid(q)))(tensor([[0.7, 1.2]])) as Value)
    expect(gg[0]).toBeCloseTo(3, 10)
    expect(gg[1]).toBeCloseTo(1, 10)
    expect(item(grad((x: Value) => sum(I.interp(x, [0, 1, 2], [0, 2, 3])))(tensor(1.5)) as Value)).toBeCloseTo(1, 12)
  })
})
