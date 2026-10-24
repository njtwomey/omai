import { describe, expect, it } from 'vitest'
import { spmv, toDense } from '../../numerics/sparse/index.ts'
import { toArray, toFlat } from '../../tensor/index.ts'
import {
  applyDirichlet,
  boundaryValue,
  dirichlet,
  grid1d,
  grid2d,
  gridPoints,
  laplacian1d,
  laplacian2d,
  neumann,
  periodic,
  robin,
} from './grids.ts'

describe('grids and boundary conditions', () => {
  it('builds grids and validates them', () => {
    const g = grid1d(-1, 1, 5)
    expect(Array.from(g.x)).toEqual([-1, -0.5, 0, 0.5, 1])
    expect(() => grid1d(0, 1, 1)).toThrow()
    expect(() => grid1d(1, 0, 3)).toThrow()
    expect(gridPoints(grid2d([0, 1], 2, [0, 2], 3)).shape).toEqual([2, 3, 2])
    expect(() => robin(1, 0)).toThrow()
    expect(boundaryValue(periodic(), 1)).toBe(0)
  })

  it('the 1-D Laplacian is exact on quadratics, with every kind of condition', () => {
    const g = grid1d(0, 1, 6)
    const quad = Array.from(g.x, (x) => x * x)
    for (const [left, right] of [
      [dirichlet(), dirichlet(1)],
      [neumann(0), dirichlet(1)],
      [dirichlet(), neumann(2)],
      [robin(1, 1, 0), robin(1, 1, 3)],
    ] as const) {
      const op = laplacian1d(g, left, right)
      const u = applyDirichlet(quad, op, 0)
      const lu = toFlat(spmv(op.matrix, u))
      const f = op.forcing(0)
      const fixed = new Set(op.fixed.map((q) => q.index))
      lu.forEach((v, i) => {
        if (!fixed.has(i)) expect(v + (f[i] ?? 0)).toBeCloseTo(2, 9)
      })
    }
    expect(() => laplacian1d(g, periodic(), dirichlet())).toThrow()
  })

  it('periodic Laplacians wrap around and the 2-D operator is the Kronecker sum', () => {
    const op = laplacian1d(grid1d(0, 1, 5), periodic(), periodic())
    const dense = toArray(toDense(op.matrix)) as number[][]
    expect(dense[0]?.[3]).toBeCloseTo(16, 12)
    const d = dirichlet((_t, x, y) => (x ?? 0) + (y ?? 0))
    const g2 = grid2d([0, 1], 4, [0, 1], 4)
    const op2 = laplacian2d(g2, { left: d, right: d, bottom: d, top: d })
    const u = applyDirichlet(new Float64Array(16), op2, 0)
    expect(u[15]).toBe(2)
    // Linear data has zero Laplacian: fill the interior with x + y too.
    for (let i = 1; i < 3; i++) for (let j = 1; j < 3; j++) u[i * 4 + j] = i / 3 + j / 3
    const lu = toFlat(spmv(op2.matrix, u))
    expect(Math.abs(lu[5] ?? 1)).toBeLessThan(1e-12)
    expect(() => laplacian2d(g2, { left: periodic(), right: d, bottom: d, top: d })).toThrow()
  })
})
