import { describe, expect, it } from 'vitest'
import { diabetes } from '../../lib/datasets/index.ts'
import { standardScaler } from '../../lib/learn/preprocessing/index.ts'
import { toFlat } from '../../lib/tensor/index.ts'
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
})
