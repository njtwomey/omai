import { describe, expect, it } from 'vitest'
import { maxOf } from '../../base/index.ts'
import { recorder } from '../../trace/index.ts'
import { advection1d, heat, heat1d, poisson, wave1d } from './finite-difference.ts'
import { dirichlet, grid1d, grid2d, laplacian1d, laplacian2d, neumann, periodic, robin } from './grids.ts'

const last = (u: { data: ArrayLike<number> }, n: number): number[] => Array.from(u.data).slice(-n)

describe('finite-difference schemes', () => {
  it('heat schemes reproduce the discrete decay of a sine mode', () => {
    const g = grid1d(0, 1, 21)
    const r = 0.4
    const dt = (r * g.dx * g.dx) / 1
    const steps = 50
    const u0 = Array.from(g.x, (x) => Math.sin(Math.PI * x))
    const s2 = Math.sin((Math.PI * g.dx) / 2) ** 2
    const factor = {
      explicit: 1 - 4 * r * s2,
      implicit: 1 / (1 + 4 * r * s2),
      'crank-nicolson': (1 - 2 * r * s2) / (1 + 2 * r * s2),
    } as const
    for (const scheme of ['explicit', 'implicit', 'crank-nicolson'] as const) {
      const s = heat(laplacian1d(g, dirichlet(), dirichlet()), u0, { dt, tEnd: steps * dt, scheme })
      const u = last(s.u, 21)
      expect(Math.abs((u[10] ?? 0) - factor[scheme] ** steps)).toBeLessThan(1e-10)
    }
  })

  it('an unstable explicit step fails with a status', () => {
    const g = grid1d(0, 1, 41)
    const s = heat1d((x) => Math.sin(40 * x), g, {
      dt: g.dx * g.dx,
      tEnd: 5,
      scheme: 'explicit',
      left: dirichlet(),
      right: dirichlet(),
    })
    expect(s.status).toBe('failed')
    expect(s.message).toMatch(/non-finite/)
  })

  it('Neumann, Robin and periodic conditions conserve or relax as they should', () => {
    const g = grid1d(0, 1, 41)
    // Insulated rod: the mean (trapezoid) is conserved.
    const s = heat1d((x) => Math.cos(Math.PI * x) + 1, g, { dt: 1e-3, tEnd: 0.5, left: neumann(), right: neumann() })
    const u = last(s.u, 41)
    const mean = u.reduce((a, v, i) => a + v * (i === 0 || i === 40 ? 0.5 : 1), 0) / 40
    expect(Math.abs(mean - 1)).toBeLessThan(1e-10)
    // Robin cooling u + u_n = 0 drives the solution to zero.
    const cool = heat1d(() => 1, g, { dt: 1e-2, tEnd: 5, left: robin(1, 1), right: robin(1, 1), scheme: 'implicit' })
    expect(maxOf(last(cool.u, 41).map(Math.abs))).toBeLessThan(0.05)
    const per = heat1d((x) => Math.sin(2 * Math.PI * x), g, {
      dt: 1e-3,
      tEnd: 0.05,
      left: periodic(),
      right: periodic(),
    })
    const up = last(per.u, 41)
    expect(Math.abs((up[0] ?? 0) - (up[40] ?? 0))).toBeLessThan(1e-12)
  })

  it('the leapfrog wave scheme is exact at Courant number 1', () => {
    const g = grid1d(0, 1, 41)
    const s = wave1d(
      (x) => Math.sin(2 * Math.PI * x),
      () => 0,
      g,
      { dt: g.dx, tEnd: 0.5, left: dirichlet(), right: dirichlet() },
    )
    const u = last(s.u, 41)
    // u = sin(2 pi x) cos(2 pi t): at t = 0.5 the profile is inverted.
    expect(Math.abs((u[10] ?? 0) + 1)).toBeLessThan(1e-12)
  })

  it('advection schemes: upwind is exact at c = 1, Lax-Wendroff is second order', () => {
    const g = grid1d(0, 1, 51)
    const up = advection1d((x) => Math.exp(-50 * (x - 0.5) ** 2), g, { velocity: 1, dt: g.dx, tEnd: 1 })
    expect(Math.abs((last(up.u, 51)[25] ?? 0) - 1)).toBeLessThan(1e-12)
    const err = (n: number): number => {
      const gg = grid1d(0, 1, n)
      const s = advection1d((x) => Math.sin(2 * Math.PI * x), gg, {
        velocity: 1,
        dt: 0.5 * gg.dx,
        tEnd: 1,
        scheme: 'lax-wendroff',
      })
      const u = last(s.u, n)
      return maxOf(u.map((v, i) => Math.abs(v - Math.sin(2 * Math.PI * (gg.x[i] ?? 0)))))
    }
    expect(Math.log2(err(41) / err(81))).toBeGreaterThan(1.8)
    const inflow = advection1d(() => 0, g, {
      velocity: 1,
      dt: 0.5 * g.dx,
      tEnd: 0.5,
      boundary: { inflow: 1 },
      scheme: 'lax-friedrichs',
    })
    expect(last(inflow.u, 51)[0]).toBe(1)
  })

  it('Poisson iterations agree and multigrid converges in few cycles', () => {
    const d = dirichlet()
    const grid = grid2d([0, 1], 17, [0, 1], 17)
    const f = (x: number, y: number): number => -2 * Math.PI ** 2 * Math.sin(Math.PI * x) * Math.sin(Math.PI * y)
    const bcs = { left: d, right: d, bottom: d, top: d }
    const mg = poisson(f, grid, bcs, { method: 'multigrid', tol: 1e-10 })
    expect(mg.converged).toBe(true)
    expect(mg.iterations).toBeLessThan(15)
    const jac = poisson(f, grid, bcs, { method: 'jacobi', tol: 1e-6 })
    const sor = poisson(f, grid, bcs, { method: 'sor', tol: 1e-6 })
    expect(sor.iterations).toBeLessThan(jac.iterations / 5)
    // The discrete solution approximates sin(pi x) sin(pi y) to O(h^2).
    expect(Math.abs((mg.u.data[8 * 17 + 8] ?? 0) - 1)).toBeLessThan(5e-3)
    expect(() => poisson(f, grid2d([0, 1], 10, [0, 1], 10), bcs, { method: 'multigrid' })).toThrow()
    const rec = recorder()
    poisson(f, grid, bcs, { method: 'gauss-seidel', tol: 1e-4, tracer: rec })
    expect(rec.run().events.length).toBeGreaterThan(0)
  })

  it('Neumann Poisson problems solve through GMRES', () => {
    // u'' = -pi^2 cos(pi x) on [0, 1] with u'(0) = 0, u(1) = -1: u = cos(pi x).
    const g = grid1d(0, 1, 81)
    const r = poisson((x) => -(Math.PI ** 2) * Math.cos(Math.PI * x), g, [neumann(), dirichlet(-1)], { method: 'cg' })
    expect(Math.abs((r.u.data[0] ?? 0) - 1)).toBeLessThan(1e-3)
    const lap = laplacian2d(grid2d([0, 1], 5, [0, 1], 5), {
      left: neumann(),
      right: dirichlet(),
      bottom: dirichlet(),
      top: dirichlet(),
    })
    expect(lap.fixed.length).toBe(13)
  })
})
