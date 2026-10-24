import { describe, expect, it } from 'vitest'
import { grad } from '../../autodiff/index.ts'
import { spmv } from '../../numerics/sparse/index.ts'
import { item, mul, sliceAxis, sub, sum, tensor, toFlat, type Value } from '../../tensor/index.ts'
import { solveIvp } from '../ode/solve-ivp.ts'
import { dirichlet, grid1d, grid2d, neumann, periodic } from './grids.ts'
import { firstDerivative1d, semiDiscretise, semiDiscretise2d } from './method-of-lines.ts'

describe('method of lines', () => {
  it('first-derivative matrices are second-order accurate', () => {
    const err = (n: number): number => {
      const g = grid1d(0, 1, n)
      const d = toFlat(
        spmv(
          firstDerivative1d(g),
          Array.from(g.x, (x) => Math.sin(3 * x)),
        ),
      )
      let e = 0
      g.x.forEach((x, i) => {
        e = Math.max(e, Math.abs((d[i] ?? 0) - 3 * Math.cos(3 * x)))
      })
      return e
    }
    expect(Math.log2(err(21) / err(41))).toBeGreaterThan(1.8)
  })

  it('advection-diffusion with a time-dependent Dirichlet value', () => {
    const g = grid1d(0, 1, 41)
    const mol = semiDiscretise(g, {
      diffusion: 0.01,
      velocity: 1,
      advectionScheme: 'upwind',
      left: dirichlet((t) => Math.sin(t)),
      right: neumann(),
    })
    const sol = solveIvp(mol.field, [0, 1], new Float64Array(41), { method: 'BDF', rtol: 1e-6, atol: 1e-9 })
    expect(sol.success).toBe(true)
    expect(Math.abs((toFlat(sol.ys.at(-1) as Value)[0] ?? 0) - Math.sin(1))).toBeLessThan(1e-5)
  })

  it('periodic reaction-diffusion is differentiable in its parameters', () => {
    const g = grid1d(0, 1, 33)
    const u0 = tensor(Array.from(g.x, (x) => 0.5 + 0.1 * Math.cos(2 * Math.PI * x)))
    const loss = (r: Value): Value => {
      const mol = semiDiscretise(g, {
        diffusion: 0.01,
        left: periodic(),
        right: periodic(),
        reaction: (u) => mul(mul(u, sub(1, u)), r),
      })
      const y = solveIvp(mol.field, [0, 1], u0, { method: 'RK4', dt: 0.01, tEval: [1] }).y
      return sum(sliceAxis(y, 1, 16, 17))
    }
    const g0 = item(grad(loss)(tensor(1)))
    const h = 1e-5
    const fd = (item(loss(tensor(1 + h))) - item(loss(tensor(1 - h)))) / (2 * h)
    expect(Math.abs(g0 - fd)).toBeLessThan(1e-6)
  })

  it('2-D diffusion keeps the boundary and decays the mode', () => {
    const d = dirichlet()
    const grid = grid2d([0, 1], 11, [0, 1], 11)
    const mol = semiDiscretise2d(grid, { diffusion: 1, bcs: { left: d, right: d, bottom: d, top: d } })
    const u0 = Array.from(
      { length: 121 },
      (_, k) => Math.sin(Math.PI * (mol.x[2 * k] ?? 0)) * Math.sin(Math.PI * (mol.x[2 * k + 1] ?? 0)),
    )
    const u = toFlat(solveIvp(mol.field, [0, 0.05], u0, { method: 'Radau', rtol: 1e-6, atol: 1e-9, tEval: [0.05] }).y)
    expect(Math.abs((u[60] ?? 0) - Math.exp(-2 * Math.PI ** 2 * 0.05))).toBeLessThan(0.01)
    expect(u[0]).toBe(0)
  })
})
