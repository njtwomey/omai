import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { metricSeries } from '../trace/index.ts'
import { coefficientsAt, lars, larsPath } from './lars.ts'

const x = [
  [1, 0.9, 0.1],
  [2, 2.1, -0.2],
  [3, 2.8, 0.3],
  [4, 4.2, 0],
  [5, 4.9, -0.1],
  [6, 6.1, 0.2],
]
const y = [1.1, 2.0, 3.2, 3.9, 5.1, 6.0]

describe('least angle regression', () => {
  it('grows one knot per step in a session, with the model at each knot', () => {
    const est = lars({ method: 'lar' })
    const s = est.session({ x, y }, { key: key(0), logs: 'all' })
    s.advance()
    const active = metricSeries(s.run(), 'solver/active').values
    expect(active.slice(0, 3)).toEqual([0, 1, 2])
    const alphas = metricSeries(s.run(), 'solver/alpha').values
    expect(alphas.every((a, i) => i === 0 || a <= (alphas[i - 1] ?? 0) + 1e-12)).toBe(true)
    expect(s.fittedAt(0).params.coef).toEqual([0, 0, 0])
    expect(s.fittedAt(1).params.coef.filter((v) => v !== 0)).toHaveLength(1)
    // The last knot of the plain path is the least squares fit.
    const end = s.fittedAt(s.t)
    expect(end.score({ x, y })).toBeGreaterThan(0.99)
    expect(end.params.path.order).toHaveLength(3)
  })

  it('interpolates the path at a penalty and stops at a number of variables', () => {
    const path = larsPath({ x, y }, { method: 'lasso' })
    const mid = coefficientsAt(path, (path.alphas[1] ?? 0) * 0.5)
    expect(mid.filter((v) => v !== 0).length).toBeGreaterThan(0)
    const f = lars({ method: 'lar', nNonzeroCoefs: 1 }).fit({ x, y })
    expect(f.params.coef.filter((v) => v !== 0)).toHaveLength(1)
    const atAlpha = lars({ alpha: path.alphas[1] ?? 0 }).fit({ x, y })
    expect(toFlat(atAlpha.predict([[3, 3, 0]]))).toHaveLength(1)
    expect(atAlpha.modelAt(0).coef).toEqual(atAlpha.params.path.coefs[atAlpha.params.path.coefs.length - 1])
  })

  it('leaves a collinear column out rather than failing', () => {
    const dup = x.map(([a = 0, b = 0]) => [a, b, a])
    const p = larsPath({ x: dup, y }, { method: 'lar' })
    expect(p.order).toHaveLength(2)
  })
})
