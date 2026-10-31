import { describe, expect, it } from 'vitest'
import { diabetes } from '../datasets/index.ts'
import { standardScaler } from '../learn/preprocessing/index.ts'
import { toFlat } from '../tensor/index.ts'
import { linearModel } from './model.ts'

describe('linearModel', () => {
  it('fits least squares, poisson and logistic with every solver agreeing', () => {
    const raw = diabetes()
    const data = { ...raw, x: standardScaler().fit(raw).transform(raw.x) }
    const ls = linearModel().fit(data)
    expect(ls.score?.(data)).toBeGreaterThan(0.5)
    expect(ls.exposition[0]?.kind).toBe('coefficientTable')
    const ridge = ['cholesky', 'qr', 'svd', 'irls', 'lbfgs'] as const
    const coefs = ridge.map((solver) => linearModel({ alpha: 0.01, solver, tol: 1e-12 }).fit(data).params.coef)
    for (const c of coefs) for (const [j, v] of c.entries()) expect(v).toBeCloseTo(coefs[0]?.[j] ?? 0, 3)
    const lasso = ['coordinate-descent', 'irls', 'owlqn'] as const
    const l = lasso.map((solver) => linearModel({ alpha: 0.5, l1Ratio: 1, solver, tol: 1e-12 }).fit(data).params.coef)
    for (const c of l) for (const [j, v] of c.entries()) expect(v).toBeCloseTo(l[0]?.[j] ?? 0, 2)
    const y = toFlat(data.y).map((v) => Math.round(v / 50))
    const pois = ['irls', 'lbfgs'] as const
    const p = pois.map(
      (solver) => linearModel({ family: 'poisson', alpha: 0.01, solver }).fit({ x: data.x, y }).params.coef,
    )
    for (const [j, v] of (p[0] ?? []).entries()) expect(v).toBeCloseTo(p[1]?.[j] ?? 0, 4)
  })

  it('takes the solver as a hyperparameter: IRLS, Newton, L-BFGS and Adam fit one logistic model', () => {
    const raw = diabetes()
    const data = { x: standardScaler().fit(raw).transform(raw.x), y: toFlat(raw.y).map((v) => (v > 140 ? 1 : 0)) }
    const base = { family: 'binomial', alpha: 0.02 } as const
    const reference = linearModel({ ...base, solver: 'irls' }).fit(data).params
    expect(reference.solver).toBe('irls')
    const solvers = [
      { kind: 'irls' },
      { kind: 'newton', gradTol: 1e-10 },
      { kind: 'lbfgs', gradTol: 1e-10 },
      { kind: 'gradient', optimiser: { kind: 'adam' }, learningRate: 0.05, gradTol: 1e-7 },
    ]
    for (const solver of solvers) {
      const fit = linearModel({ ...base, solver, maxIter: 6000 }).fit(data).params
      expect(fit.solver).toBe(solver.kind)
      for (const [j, v] of fit.coef.entries()) expect(v, `${solver.kind} ${j}`).toBeCloseTo(reference.coef[j] ?? 0, 3)
      expect(fit.intercept, solver.kind).toBeCloseTo(reference.intercept, 3)
    }
  })

  it('refuses a glmSolvers configuration for an L1 penalty, which is not smooth', () => {
    const data = { x: [[0], [1], [2], [3]], y: [0, 1, 0, 1] }
    expect(() =>
      linearModel({ family: 'binomial', alpha: 0.1, l1Ratio: 1, solver: { kind: 'newton' } }).fit(data),
    ).toThrow(/non-smooth/)
  })
})
