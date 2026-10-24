import { describe, it } from 'vitest'
import { type GoldenFn, runFixture } from '../../../testing/systems.ts'
import { grad } from '../../autodiff/index.ts'
import { mul, sliceAxis, stack, sub, sum, tensor, type Value } from '../../tensor/index.ts'
import { type AdjointMethod, odeintAdjoint } from './adjoint.ts'

const at = (y: Value, i: number): Value => sum(sliceAxis(y, 0, i, i + 1))
const lotka = (_t: number, y: Value, p: Value): Value => {
  const [a, b] = [at(y, 0), at(y, 1)]
  const [alpha, beta, delta, gamma] = [at(p, 0), at(p, 1), at(p, 2), at(p, 3)]
  return stack([sub(mul(alpha, a), mul(mul(beta, a), b)), sub(mul(mul(delta, a), b), mul(gamma, b))])
}

const solveWith =
  (p: Readonly<Record<string, unknown>>) =>
  (y: Value, q: Value): Value =>
    odeintAdjoint(lotka, y, p['times'] as number[], q, {
      rtol: 1e-11,
      atol: 1e-13,
      adjoint: (p['adjoint'] as AdjointMethod | undefined) ?? 'interpolated',
    })

const FNS: Readonly<Record<string, GoldenFn>> = {
  states: (_x, p) => solveWith(p)(tensor(p['y0'] as number[]), tensor(p['params'] as number[])),
  grad_y0: (_x, p) =>
    grad((y: Value) => {
      const ys = solveWith(p)(y, tensor(p['params'] as number[]))
      return sum(mul(ys, ys))
    })(tensor(p['y0'] as number[])),
  grad_p: (_x, p) =>
    grad((q: Value) => {
      const ys = solveWith(p)(tensor(p['y0'] as number[]), q)
      return sum(mul(ys, ys))
    })(tensor(p['params'] as number[])),
}

describe('dynamics/ode/adjoint against torchdiffeq', () => {
  runFixture('dynamics/ode/adjoint', FNS, it)
})
