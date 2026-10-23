import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import { problem } from '../../../testing/optim-problems.ts'
import {
  concat,
  matmul,
  mul,
  neg,
  softplus,
  square,
  sub,
  sum,
  type Tensor,
  tensor,
  toFlat,
  type Value,
} from '../../tensor/index.ts'
import * as O from '../index.ts'

const solution = (r: O.MinimizeResult): Value => concat([r.x as Value, tensor([r.fun])])

describe('quasi-Newton methods against scipy.optimize.minimize', () => {
  for (const c of loadFixture('optim/minimize/quasi_newton').cases) {
    it(c.name, () => {
      const f = problem(c.params['problem'] as string)
      const x0 = decode(c.inputs[0]) as Tensor
      const r =
        c.fn === 'bfgs'
          ? O.bfgs(f, x0, { gradTol: 1e-8 })
          : c.fn === 'lbfgs'
            ? O.lbfgs(f, x0, { gradTol: 1e-10, ftol: 1e-15 })
            : O.sr1(f, x0, { gradTol: 1e-10, maxIterations: 5000 })
      expect(r.converged).toBe(true)
      expectMatches(solution(r), c.output, c.rtol, c.atol, c.name)
      const nit = c.params['nit'] as number | undefined
      if (nit !== undefined) expect(Math.abs(r.steps - nit)).toBeLessThanOrEqual(2)
    })
  }
})

describe('L-BFGS-B against scipy.optimize.minimize', () => {
  for (const c of loadFixture('optim/minimize/lbfgsb').cases) {
    it(c.name, () => {
      const [x0, lower, upper] = c.inputs.map((v) => decode(v) as Tensor) as [Tensor, Tensor, Tensor]
      const r = O.lbfgsb(problem(c.params['problem'] as string), x0, {
        bounds: { lower: toFlat(lower), upper: toFlat(upper) },
        gradTol: 1e-10,
        ftol: 1e-15,
      })
      expect(r.converged).toBe(true)
      expectMatches(solution(r), c.output, c.rtol, c.atol, c.name)
    })
  }
})

describe('OWL-QN against scikit-learn', () => {
  for (const c of loadFixture('optim/minimize/owlqn').cases) {
    it(c.name, () => {
      const [X, y] = c.inputs.map((v) => decode(v) as Tensor) as [Tensor, Tensor]
      const n = X.shape[0] ?? 1
      const p = X.shape[1] ?? 1
      const r =
        c.fn === 'lasso'
          ? O.owlqn((w: Value) => mul(sum(square(sub(y, matmul(X, w)))), 1 / (2 * n)), new Array(p).fill(0), {
              l1: c.params['alpha'] as number,
              gradTol: 1e-12,
              ftol: 1e-16,
            })
          : O.owlqn(
              (w: Value) => mul(sum(softplus(neg(mul(y, matmul(X, w))))), c.params['C'] as number),
              new Array(p).fill(0),
              { l1: 1, gradTol: 1e-10, ftol: 1e-16 },
            )
      expect(r.converged).toBe(true)
      expectMatches(r.x as Value, c.output, c.rtol, c.atol, c.name)
      // Exact zeros where the reference has them.
      const ref = toFlat(decode(c.output) as Tensor)
      toFlat(r.x as Value).forEach((v, i) => {
        if (ref[i] === 0) expect(v).toBe(0)
      })
    })
  }
})
