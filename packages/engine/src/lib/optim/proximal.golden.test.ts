import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { solve } from '../numerics/linalg/index.ts'
import {
  abs,
  add,
  div,
  eye,
  matmul,
  mul,
  reshape,
  square,
  sub,
  sum,
  swapaxes,
  type Tensor,
  type Value,
} from '../tensor/index.ts'
import * as O from './index.ts'

type Params = Readonly<Record<string, unknown>>
const num = (p: Params, k: string): number => p[k] as number

const OPERATORS: Readonly<Record<string, (x: Tensor, p: Params, rest: readonly Tensor[]) => Value>> = {
  proxL1: (x, p) => O.proxL1(x, num(p, 't')),
  proxL2: (x, p) => O.proxL2(x, num(p, 't')),
  proxElasticNet: (x, p) => O.proxElasticNet(x, num(p, 't'), { l1: num(p, 'l1'), l2: num(p, 'l2') }),
  proxGroupLasso: (x, p) => O.proxGroupLasso(x, num(p, 't'), p['groups'] as number[], p['weights'] as number[]),
  proxLinf: (x, p) => O.proxLinf(x, num(p, 't')),
  proxHuber: (x, p) => O.proxHuber(x, num(p, 't'), num(p, 'delta')),
  proxNuclear: (x, p) => O.proxNuclear(x, num(p, 't')),
  projectSimplex: (x, p) => O.projectSimplex(x, (p['z'] as number | undefined) ?? 1),
  projectL1Ball: (x, p) => O.projectL1Ball(x, num(p, 'r')),
  projectL2Ball: (x, p) => O.projectL2Ball(x, num(p, 'r'), num(p, 'center')),
  projectBox: (x, p) => O.projectBox(x, num(p, 'lo'), num(p, 'hi')),
  projectAffine: (x, _p, [a, b]) => O.projectAffine(x, a as Tensor, b as Tensor),
  projectHalfspace: (x, p, [a]) => O.projectHalfspace(x, a as Tensor, num(p, 'b')),
  projectNuclearBall: (x, p) => O.projectNuclearBall(x, num(p, 'r')),
  projectPsd: (x) => O.projectPsd(x),
  proxTv1d: (x, p) => O.proxTv1d(x, num(p, 'lam')),
  proxTv2d: (x, p) => O.proxTv2d(x, num(p, 'lam'), { isotropic: p['isotropic'] as boolean, maxIterations: 20000 }),
}

const leastSquares =
  (a: Tensor, b: Tensor) =>
  (x: unknown): Value =>
    mul(sum(square(sub(reshape(matmul(a, reshape(x as Value, [-1, 1])), [-1]), b))), 0.5)

const fixture = {
  cases: ['optim/proximal', 'optim/total_variation', 'optim/splitting'].flatMap((f) => loadFixture(f).cases),
}

describe('proximal operators and projections against cvxpy', () => {
  for (const c of fixture.cases.filter((c) => c.fn in OPERATORS)) {
    it(c.name, () => {
      const [x, ...rest] = c.inputs.map((v) => decode(v) as Tensor) as [Tensor, ...Tensor[]]
      const op = OPERATORS[c.fn] as (x: Tensor, p: Params, rest: readonly Tensor[]) => Value
      expectMatches(op(x, c.params, rest), c.output, c.rtol, c.atol, c.name)
    })
  }
})

describe('lasso and elastic net against sklearn', () => {
  for (const c of fixture.cases.filter((c) => c.fn === 'lasso')) {
    const [a, b] = c.inputs.map((v) => decode(v) as Tensor) as [Tensor, Tensor]
    const lam = num(c.params, 'lam')
    const f = leastSquares(a, b)
    const n = a.shape[1] ?? 0
    const x0 = new Float64Array(n)
    const g = (x: Value): Value => mul(sum(abs(x)), lam)
    it(`${c.name} by ISTA`, () => {
      const r = O.ista(f, (x, t) => O.proxL1(x, t * lam), x0, { tol: 1e-15, maxIterations: 50000, g })
      expectMatches(r.x as Value, c.output, c.rtol, c.atol, c.name)
    })
    it(`${c.name} by FISTA with restart`, () => {
      const r = O.fista(f, (x, t) => O.proxL1(x, t * lam), x0, { tol: 1e-15, restart: 'gradient' })
      expectMatches(r.x as Value, c.output, c.rtol, c.atol, c.name)
      expect(r.converged).toBe(true)
    })
    it(`${c.name} by Chambolle-Pock`, () => {
      // F(K x) = |A x - b|^2 / 2 with K = A, G = lam |x|_1.
      const r = O.chambollePock(
        {
          K: a,
          proxG: (x, t) => O.proxL1(x, t * lam),
          proxF: (v, t) => div(add(v, mul(b, t)), 1 + t),
        },
        x0,
        { tol: 1e-11, maxIterations: 200000 },
      )
      expectMatches(r.x as Value, c.output, 1e-5, 1e-6, c.name)
    })
  }
  for (const c of fixture.cases.filter((c) => c.fn === 'elasticNet')) {
    it(`${c.name} by FISTA`, () => {
      const [a, b] = c.inputs.map((v) => decode(v) as Tensor) as [Tensor, Tensor]
      const weights = { l1: num(c.params, 'l1'), l2: num(c.params, 'l2') }
      const r = O.fista(
        leastSquares(a, b),
        (x, t) => O.proxElasticNet(x, t, weights),
        new Float64Array(a.shape[1] ?? 0),
        {
          tol: 1e-15,
        },
      )
      expectMatches(r.x as Value, c.output, c.rtol, c.atol, c.name)
    })
  }
})

describe('ADMM on the lasso against sklearn', () => {
  for (const c of loadFixture('optim/admm').cases) {
    const [a, b] = c.inputs.map((v) => decode(v) as Tensor) as [Tensor, Tensor]
    const lam = num(c.params, 'lam')
    it(`${c.name} with a cached Cholesky factor`, () => {
      const r = O.admmLasso(a, b, lam, { absTol: 1e-12, relTol: 1e-12, adaptiveRho: true })
      expect(r.converged).toBe(true)
      expectMatches(r.z, c.output, c.rtol, c.atol, c.name)
    })
    it(`${c.name} by Douglas-Rachford from proximal operators`, () => {
      // prox of t |A x - b|^2 / 2: (I + t A^T A)^{-1} (v + t A^T b).
      const at = swapaxes(a, 0, 1)
      const n = a.shape[1] ?? 0
      const atb = reshape(matmul(at, reshape(b, [-1, 1])), [-1])
      const proxF = (v: Value, t: number): Value => solve(add(eye(n), mul(matmul(at, a), t)), add(v, mul(atb, t)))
      const r = O.admmProx(proxF, (v, t) => O.proxL1(v, t * lam), new Float64Array(a.shape[1] ?? 0), {
        rho: 1,
        absTol: 1e-10,
        relTol: 1e-10,
        maxIterations: 3000,
      })
      expectMatches(r.z, c.output, 1e-5, 1e-6, c.name)
    })
  }
})
