import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { toArray, toFlat } from '../../tensor/index.ts'
import { solve } from '../linalg/index.ts'
import * as S from './index.ts'

const ones = (n: number): number[] => new Array<number>(n).fill(1)

describe('multigrid', () => {
  it('builds the model problems and the interpolation', () => {
    expect(toArray(S.toDense(S.poisson1d(3)))).toEqual([
      [2, -1, 0],
      [-1, 2, -1],
      [0, -1, 2],
    ])
    expect(S.poisson2d(3).shape).toEqual([9, 9])
    const p = S.prolongation([3])
    expect(toArray(S.toDense(p))).toEqual([[0.5], [1], [0.5]])
    expect(S.prolongation([7, 7]).shape).toEqual([49, 9])
    expect(() => S.prolongation([4])).toThrow(/odd/)
  })

  it('reproduces smooth fine-grid functions exactly: P applied to a linear function on the coarse grid is linear', () => {
    const p = S.prolongation([7])
    const coarse = [1, 2, 3]
    const fine = toFlat(S.spmv(p, coarse))
    // Coarse point j sits at fine index 2j+1: the linear function 1,2,3 continues linearly to the fine grid.
    expect(fine).toEqual([0.5, 1, 1.5, 2, 2.5, 3, 1.5])
  })

  it('converges at a rate that does not depend on the grid size, unlike the smoother alone', () => {
    const rate = (n: number, levels: number, smootherOnly = false): number => {
      const b = ones(n)
      const result = smootherOnly
        ? S.stationarySolve(S.poisson1d(n), b, { method: 'jacobi', omega: 2 / 3, rtol: 0, maxIterations: 10 })
        : S.multigridSolve(S.poisson1d(n), b, { shape: [n], levels, rtol: 0, maxIterations: 10 })
      const h = result.history
      return ((h[10] ?? 1) / (h[4] ?? 1)) ** (1 / 6)
    }
    const small = rate(15, 2)
    const large = rate(127, 6)
    expect(small).toBeLessThan(0.35)
    expect(large).toBeLessThan(0.35)
    expect(Math.abs(large - small)).toBeLessThan(0.1)
    expect(rate(127, 0, true)).toBeGreaterThan(0.95)
  })

  it('a W-cycle contracts at least as fast as a V-cycle on the same grids', () => {
    const run = (cycle: 'v' | 'w'): number =>
      S.multigridSolve(S.poisson2d(15), ones(225), { shape: [15, 15], levels: 3, cycle, rtol: 1e-10 }).iterations
    expect(run('w')).toBeLessThanOrEqual(run('v'))
  })

  it('solves a 2-D Poisson system to the direct solution', () => {
    const a = S.poisson2d(7)
    const r = S.multigridSolve(a, ones(49), { shape: [7, 7], levels: 2, rtol: 1e-12 })
    expect(r.converged).toBe(true)
    const direct = toFlat(solve(S.toDense(a), ones(49)))
    toFlat(r.x).forEach((v, i) => {
      expect(v).toBeCloseTo(direct[i] ?? 0, 9)
    })
  })

  it('the coarse-grid correction removes the smooth error that the smoother leaves', () => {
    const n = 31
    const a = S.poisson1d(n)
    const exact = toFlat(solve(S.toDense(a), ones(n)))
    const smoothError = Array.from({ length: n }, (_, i) => Math.sin(((i + 1) * Math.PI) / (n + 1)))
    const x0 = exact.map((v, i) => v + (smoothError[i] ?? 0))
    const cycle = S.multigridCycle(a, ones(n), x0, { shape: [n], levels: 2 })
    const err = (x: readonly number[]): number => Math.hypot(...x.map((v, i) => v - (exact[i] ?? 0)))
    expect(err(toFlat(cycle.pre))).toBeGreaterThan(0.9 * err(x0))
    expect(err(toFlat(cycle.corrected))).toBeLessThan(0.1 * err(x0))
  })

  it('is the identity on a fixed point for any smoother', () => {
    fc.assert(
      fc.property(
        fc.constantFrom('jacobi', 'gauss-seidel', 'sor'),
        fc.integer({ min: 0, max: 3 }),
        (smoother, sweeps) => {
          const a = S.poisson1d(7)
          const exact = toFlat(solve(S.toDense(a), ones(7)))
          const c = S.multigridCycle(a, ones(7), exact, { shape: [7], smoother, preSweeps: sweeps, postSweeps: sweeps })
          toFlat(c.post).forEach((v, i) => {
            expect(v).toBeCloseTo(exact[i] ?? 0, 10)
          })
        },
      ),
      { numRuns: 12 },
    )
  })

  it('rejects mismatched grids', () => {
    expect(() => S.multigridAlgorithm(S.poisson1d(7), ones(7), { shape: [15] })).toThrow(/prolongation/)
    expect(() => S.multigridAlgorithm(S.poisson1d(7), ones(7), {})).toThrow(/shape/)
  })
})
