import { describe, expect, it } from 'vitest'
import { policies, recorder, runAlgorithm, stateToJson } from '../../trace/index.ts'
import { poisson, poissonAlgorithm, poissonContract } from './finite-difference.ts'
import { poissonMethods } from './finite-difference-options.ts'
import { dirichlet, grid1d, grid2d } from './grids.ts'

const d = dirichlet()
const g2 = grid2d([0, 1], 9, [0, 1], 9)
const f2 = (x: number, y: number): number => -2 * Math.PI ** 2 * Math.sin(Math.PI * x) * Math.sin(Math.PI * y)
const bcs = { left: d, right: d, bottom: d, top: d }
const g1 = grid1d(0, 1, 17)

describe('poissonAlgorithm', () => {
  it('is poisson: run to its tolerance it gives the numbers the solver gave before', () => {
    // Before the refactor: multigrid on the 9 x 9 grid took 8 cycles; Gauss-Seidel on the 17-node 1-D problem u'' = 2 took 474 sweeps.
    const mg = runAlgorithm(poissonAlgorithm(f2, g2, bcs, { method: 'multigrid', tol: 1e-10 }), { maxSteps: 50 })
    expect(mg.steps).toBe(8)
    expect(mg.status).toBe('converged')
    expect(mg.state.u[40]).toBeCloseTo(1.01295074669, 10)
    expect(mg.state.u[4 * 9 + 3]).toBeCloseTo(0.9358444623114, 10)
    expect(mg.state.history.slice(0, 3).map((v) => Math.round(v * 1e12) / 1e12)).toEqual([
      1, 0.070892498013, 0.003413065431,
    ])
    const gs = runAlgorithm(
      poissonAlgorithm(() => 2, g1, [d, d], { method: 'gauss-seidel', tol: 1e-8 }),
      {
        maxSteps: 1000,
      },
    )
    expect(gs.steps).toBe(474)
    expect(gs.state.u[4]).toBeCloseTo(-0.1874999979484, 10)
    expect(gs.state.u[8]).toBeCloseTo(-0.2499999973153, 10)
    const fn = poisson(() => 2, g1, [d, d], { method: 'gauss-seidel', tol: 1e-8 })
    expect(fn.iterations).toBe(474)
    expect(fn.residualHistory).toEqual([...gs.state.history])
  })

  it('has a state that is JSON, and a run resumes from a checkpoint to the same end', () => {
    for (const method of ['jacobi', 'gauss-seidel', 'sor', 'multigrid'] as const) {
      const alg = poissonAlgorithm(f2, g2, bcs, { method, tol: 1e-9 })
      const whole = runAlgorithm(alg, { maxSteps: 2000 })
      const half = runAlgorithm(alg, { maxSteps: 5 })
      expect(half.steps).toBe(5)
      const stored = JSON.parse(JSON.stringify(alg.serialise?.(half.state)))
      const rest = runAlgorithm(alg, { maxSteps: 2000, start: { state: stored, step: 5, serialised: true } })
      expect(stateToJson(rest.state)).toEqual(stateToJson(whole.state))
      expect(rest.steps).toBe(whole.steps)
    }
  })

  it('logs its contract by path', () => {
    const rec = recorder()
    runAlgorithm(poissonAlgorithm(f2, g2, bcs, { method: 'sor' }), {
      maxSteps: 4,
      tracer: rec,
      policy: { ...policies.minimal, metrics: true, tensors: true, roles: ['output'], detail: 'full' },
    })
    const metric = (name: string): number[] =>
      rec.events.flatMap((e) => (e.type === 'metric' && e.name === name ? [e.value] : []))
    expect(metric('poisson/residual')).toHaveLength(4)
    expect(metric('poisson/change').every((v) => v > 0)).toBe(true)
    expect(
      rec.events.flatMap((e) => (e.type === 'tensor' && e.name === 'poisson/iterate' ? [e.shape[0]] : [])),
    ).toEqual(Array(4).fill(81))
    expect(Object.keys(poissonContract)).toHaveLength(3)
  })

  it('the multigrid residual falls by a steady factor per cycle and SOR beats Gauss-Seidel beats Jacobi', () => {
    const mg = runAlgorithm(poissonAlgorithm(f2, g2, bcs, { method: 'multigrid', tol: 1e-12 }), { maxSteps: 50 }).state
    const ratios = mg.history.slice(1, 6).map((v, i) => v / (mg.history[i] ?? 1))
    expect(ratios.every((r) => r < 0.3)).toBe(true)
    const steps = (method: 'jacobi' | 'gauss-seidel' | 'sor'): number =>
      runAlgorithm(poissonAlgorithm(f2, g2, bcs, { method, tol: 1e-6 }), { maxSteps: 5000 }).steps
    expect(steps('sor')).toBeLessThan(steps('gauss-seidel'))
    expect(steps('gauss-seidel')).toBeLessThan(steps('jacobi'))
  })

  it('runs every option of the iteration table, with the knobs of its own', () => {
    for (const name of poissonMethods.names) {
      const options = { ...poissonMethods.make({ kind: name }), tol: 1e-6 }
      const run = runAlgorithm(poissonAlgorithm(f2, g2, bcs, options), { maxSteps: 5000 })
      expect(run.status).toBe('converged')
    }
    const more = poissonMethods.make({ kind: 'multigrid', sweeps: 4, coarseSweeps: 20 })
    expect(more).toEqual({ method: 'multigrid', sweeps: 4, coarseSweeps: 20 })
    expect(poissonMethods.where((p) => p.needsNestedGrids)).toEqual(['multigrid'])
    expect(() => poissonAlgorithm(f2, g2, bcs, { method: 'cg' })).toThrow()
    expect(() => poissonAlgorithm(f2, grid2d([0, 1], 10, [0, 1], 10), bcs, { method: 'multigrid' })).toThrow()
  })
})
