import { describe, expect, it } from 'vitest'
import { cos, exp, sin, sqrt, square, sub, tensor, type Value } from '../tensor/index.ts'
import type { RunStatus } from '../trace/index.ts'
import { remez } from './approximation/index.ts'
import { jacobiSvd, powerIteration, subspaceIteration } from './linalg/index.ts'
import { adaptiveSimpson, quad, romberg, tanhSinh } from './quadrature/index.ts'
import { bisect, brentq, broyden, fixedPoint, fsolve, homotopy, newton, ridder, toms748 } from './roots/index.ts'
import { bicgstab, cg, gmres, minres, stationarySolve } from './sparse/index.ts'

// The failure convention of the numeric routines (see `trace/events.ts`): a routine that returns a result reports `status`
// (a `RunStatus`) and `converged` (true exactly when the status is `converged`) and does not throw for non-convergence.
// One row per solver family, each run twice: with a budget that lets it finish, and with one that does not.
interface Outcome {
  readonly converged: boolean
  readonly status: RunStatus
}

const spd = [
  [4, 1, 0],
  [1, 3, 1],
  [0, 1, 2],
]
const rhs = [1, 2, 3]
const general = [
  [1, 2, 3],
  [4, 5, 6],
  [7, 8, 10],
]
const x2 = (x: number): number => x * x - 2
const vector = (v: Value): Value => sub(v, tensor([3, 4]))
const oneInOne = (v: Value): Value => sub(cos(v), v)

const rows: readonly (readonly [string, () => Outcome, () => Outcome])[] = [
  ['bisect', () => bisect(x2, 0, 2), () => bisect(x2, 0, 2, { maxIterations: 3 })],
  ['brentq', () => brentq(x2, 0, 2), () => brentq(x2, 0, 2, { maxIterations: 1 })],
  ['ridder', () => ridder(x2, 0, 2), () => ridder(x2, 0, 2, { maxIterations: 1 })],
  ['toms748', () => toms748(x2, 0, 2), () => toms748(x2, 0, 2, { maxIterations: 1 })],
  [
    'newton',
    () => newton(x2, 1, { fprime: (x) => 2 * x }),
    () => newton(x2, 1, { fprime: (x) => 2 * x, maxIterations: 2 }),
  ],
  [
    'fsolve',
    () => fsolve(vector, [0, 0]),
    () => fsolve((v) => sub(exp(v), tensor([1e6, 1e6])), [0, 0], { maxIterations: 2 }),
  ],
  [
    'broyden',
    () => broyden(vector, [0, 0]),
    () => broyden((v) => sub(exp(v), tensor([1e6, 1e6])), [0, 0], { maxIterations: 2 }),
  ],
  [
    'fixedPoint',
    () => fixedPoint((v) => cos(v), [1]),
    () => fixedPoint((v) => cos(v), [1], { method: 'iteration', maxIterations: 3 }),
  ],
  ['quad', () => quad((x) => sin(x), 0, Math.PI), () => quad((x) => sqrt(x), 0, 1, { limit: 2 })],
  [
    'adaptiveSimpson',
    () => adaptiveSimpson((x) => sin(x), 0, Math.PI),
    () => adaptiveSimpson((x) => sqrt(x), 0, 1, { maxDepth: 1 }),
  ],
  [
    'tanhSinh',
    () => tanhSinh((x) => sin(x), 0, Math.PI),
    () => tanhSinh((x) => sqrt(x), 0, 1, { maxLevel: 1, tol: 1e-14 }),
  ],
  ['romberg', () => romberg((x) => exp(x), 0, 1), () => romberg((x) => sqrt(x), 0, 1, { divmax: 2 })],
  ['cg', () => cg(spd, rhs, { rtol: 1e-10 }), () => cg(spd, rhs, { rtol: 1e-14, maxIterations: 1 })],
  ['gmres', () => gmres(spd, rhs, { rtol: 1e-10 }), () => gmres(spd, rhs, { rtol: 1e-14, maxIterations: 1 })],
  ['minres', () => minres(spd, rhs, { rtol: 1e-10 }), () => minres(spd, rhs, { rtol: 1e-14, maxIterations: 1 })],
  ['bicgstab', () => bicgstab(spd, rhs, { rtol: 1e-10 }), () => bicgstab(spd, rhs, { rtol: 1e-14, maxIterations: 1 })],
  [
    'stationarySolve',
    () => stationarySolve(spd, rhs, { rtol: 1e-8 }),
    () => stationarySolve(spd, rhs, { rtol: 1e-14, maxIterations: 1 }),
  ],
  [
    'powerIteration',
    () => powerIteration(spd, { x0: [1, 1, 1] }),
    () => powerIteration(spd, { x0: [1, 0, 0], maxIterations: 1, tol: 1e-15 }),
  ],
  [
    'subspaceIteration',
    () => subspaceIteration(spd, 1, { tol: 1e-10 }),
    () => subspaceIteration(spd, 1, { tol: 1e-15, maxIterations: 1 }),
  ],
  ['jacobiSvd', () => jacobiSvd(general, { tol: 1e-12 }), () => jacobiSvd(general, { tol: 1e-16, maxSteps: 1 })],
  ['remez', () => remez((x) => x * x, 1), () => remez((x) => Math.exp(x), 3, { maxIterations: 1 })],
]

describe('the numeric solvers share one failure convention', () => {
  it('converged is true exactly when the status is converged, and no family throws when it runs out of budget', () => {
    for (const [name, finishes, runsOut] of rows) {
      expect(finishes(), name).toMatchObject({ converged: true, status: 'converged' })
      expect(runsOut(), name).toMatchObject({ converged: false, status: 'completed' })
    }
  })

  it('fixedPoint and fsolve agree: both return converged false at their iteration cap', () => {
    const f = fsolve(oneInOne, [1], { maxIterations: 1, xtol: 1e-15, ftol: 1e-18 })
    const g = fixedPoint((v) => cos(v), [1], { method: 'iteration', maxIterations: 3 })
    expect([f.converged, f.status, f.message]).toEqual([false, 'completed', 'maximum iterations'])
    expect([g.converged, g.status, g.message]).toEqual([false, 'completed', 'maximum iterations'])
    expect(g.iterations).toBe(3)
  })

  it('newton ends failed when the derivative is zero, with the flag kept beside the status', () => {
    const r = newton((x) => x * x + 1, 0, { fprime: (x) => 2 * x })
    expect([r.converged, r.status, r.flag]).toEqual([false, 'failed', 'derivative was zero'])
  })

  it('homotopy returns the path followed so far with status failed when the corrector fails', () => {
    const r = homotopy((v) => sub(square(v), tensor([-1])), [3])
    expect(r.converged).toBe(false)
    expect(r.status).toBe('failed')
    expect(r.message).toMatch(/^homotopy: the corrector failed at t = /)
    expect(r.path.length).toBeGreaterThanOrEqual(1)
    expect(r.path[0]?.t).toBe(0)
  })
})
