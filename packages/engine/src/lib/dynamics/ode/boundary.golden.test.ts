import { describe, it } from 'vitest'
import { type GoldenFn, runFixture } from '../../../testing/systems.ts'
import { concat, exp, mul, neg, sliceAxis, stack, sub, type Tensor, tensor, type Value } from '../../tensor/index.ts'
import { type BvpSolution, solveBvp } from './boundary.ts'

const row = (y: Value, i: number): Value => sliceAxis(y, 0, i, i + 1)
const probeOf = (sol: BvpSolution, xs: readonly number[]): Value =>
  stack(
    xs.map((x) => sol.sol(x)),
    1,
  )
const linspace = (a: number, b: number, n: number): number[] =>
  Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1))

function bratu(guess: number): BvpSolution {
  const x = linspace(0, 1, 5)
  const y = tensor([x.map(() => guess), x.map(() => 0)])
  return solveBvp(
    (_x, yy) => concat([row(yy, 1), neg(exp(row(yy, 0)))], 0),
    (ya, yb) => concat([row(ya, 0), row(yb, 0)], 0),
    x,
    y,
    { tol: 1e-6 },
  )
}

function sturm(): BvpSolution {
  const x = linspace(0, 1, 5)
  const y = tensor([
    [0, 1, 0, -1, 0],
    [0, 0, 0, 0, 0],
  ])
  return solveBvp(
    (_x, yy, p) => concat([row(yy, 1), neg(mul(mul(p, p), row(yy, 0)))], 0),
    (ya, yb, p) => concat([row(ya, 0), row(yb, 0), sub(row(ya, 1), p)], 0),
    x,
    y,
    { p: [6], tol: 1e-6 },
  )
}

const FNS: Readonly<Record<string, GoldenFn>> = {
  bratu: (_x, p) => probeOf(bratu(p['guess'] as number), linspace(0, 1, 9)),
  sturm_p: () => sturm().p as Tensor,
  sturm_y: () => probeOf(sturm(), linspace(0, 1, 9)),
  singular: () => {
    const xs = linspace(0, Math.PI, 6)
    const sol = solveBvp(
      (_x, yy) => concat([row(yy, 1), neg(row(yy, 0))], 0),
      (ya, yb) => concat([sub(row(ya, 0), 1), row(yb, 0)], 0),
      xs,
      tensor([xs.map((x) => Math.cos(x / 2)), xs.map((x) => -0.5 * Math.sin(x / 2))]),
      {
        S: [
          [0, 0],
          [0, -2],
        ],
        tol: 1e-6,
      },
    )
    return probeOf(sol, linspace(0, Math.PI, 7))
  },
}

describe('dynamics/ode/boundary against scipy.integrate.solve_bvp', () => {
  runFixture('dynamics/ode/boundary', FNS, it)
})
