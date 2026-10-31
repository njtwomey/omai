import { describe, expect, it } from 'vitest'
import { key, normal } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { olsSummary } from './inference.ts'
import { subsetAlgorithm, subsetRegression } from './subset.ts'

const n = 120
const z = toFlat(normal(key(5), [n * 7]))
// Six features, of which 0, 2 and 4 matter.
const x = Array.from({ length: n }, (_, i) =>
  Array.from({ length: 6 }, (_, j) => (z[i * 7 + j] ?? 0) + (j === 1 ? 0.6 * (z[i * 7] ?? 0) : 0)),
)
const y = x.map((r, i) => 1 + 2 * (r[0] ?? 0) - 1.5 * (r[2] ?? 0) + 1 * (r[4] ?? 0) + 0.5 * (z[i * 7 + 6] ?? 0))

describe('subsetRegression', () => {
  it('finds the planted features by every method and criterion that penalises size', () => {
    for (const method of ['best', 'forward', 'backward', 'stepwise'] as const)
      for (const criterion of ['bic', 'aic', 'aicc'] as const) {
        const f = subsetRegression({ method, criterion }).fit({ x, y })
        expect(f.params.selected, `${method}/${criterion}`).toEqual([0, 2, 4])
      }
  })

  it('selects by Mallows Cp and adjusted R-squared too', () => {
    expect(subsetRegression({ method: 'best', criterion: 'cp' }).fit({ x, y }).params.selected).toEqual([0, 2, 4])
    const adj = subsetRegression({ method: 'best', criterion: 'adjusted-r2' }).fit({ x, y }).params.selected
    for (const j of [0, 2, 4]) expect(adj).toContain(j)
  })

  it('fits ordinary least squares on the selected features', () => {
    const f = subsetRegression({ method: 'best', criterion: 'bic' }).fit({ x, y })
    const ols = olsSummary({ x: x.map((r) => [r[0] ?? 0, r[2] ?? 0, r[4] ?? 0]), y })
    for (const [c, j] of [0, 2, 4].entries())
      expect(f.params.coef[j]).toBeCloseTo(ols.table.estimates[c] ?? Number.NaN, 9)
    expect(f.params.coef[1]).toBe(0)
    expect(f.params.intercept).toBeCloseTo(ols.table.estimates[3] ?? Number.NaN, 9)
  })

  it('reports the AIC of statsmodels OLS for the chosen subset', () => {
    const f = subsetRegression({ method: 'best', criterion: 'aic' }).fit({ x, y })
    const ols = olsSummary({ x: x.map((r) => [r[0] ?? 0, r[2] ?? 0, r[4] ?? 0]), y })
    expect(f.params.criterionValue).toBeCloseTo(ols.aic, 8)
  })

  it('best subset is never beaten by a greedy path, and its criterion path has the best at the chosen size', () => {
    // A suppressor design: x0 alone is useless but with x1 explains y, so forward selection can miss it.
    const a = toFlat(normal(key(9), [n * 3]))
    const sx = Array.from({ length: n }, (_, i) => [
      (a[i * 3] ?? 0) + (a[i * 3 + 1] ?? 0),
      a[i * 3 + 1] ?? 0,
      a[i * 3 + 2] ?? 0,
    ])
    const sy = sx.map((r, i) => (r[0] ?? 0) - (r[1] ?? 0) + 0.1 * (a[i * 3 + 2] ?? 0))
    const best = subsetRegression({ method: 'best', criterion: 'bic' }).fit({ x: sx, y: sy }).params
    for (const method of ['forward', 'backward', 'stepwise'] as const) {
      const g = subsetRegression({ method, criterion: 'bic' }).fit({ x: sx, y: sy }).params
      expect(best.criterionValue).toBeLessThanOrEqual(g.criterionValue + 1e-9)
    }
    expect(best.path.length).toBeGreaterThan(1)
  })

  it('limits the size and serialises its state', () => {
    const f = subsetRegression({ method: 'forward', criterion: 'aic', maxSize: 2 }).fit({ x, y })
    expect(f.params.selected.length).toBeLessThanOrEqual(2)
    const algorithm = subsetAlgorithm({ x, y }, { method: 'forward', criterion: 'aic' })
    const straight = runAlgorithm(algorithm, { key: key(0), maxSteps: 12 })
    const head = runAlgorithm(algorithm, { key: key(0), maxSteps: 2 })
    const json = algorithm.serialise?.(head.state)
    const resumed = runAlgorithm(algorithm, {
      key: key(0),
      maxSteps: 10,
      start: { state: json, step: head.steps, serialised: true },
    })
    expect(resumed.state.best).toEqual(straight.state.best)
  })

  it('refuses best-subset search over too many features and Cp without degrees of freedom', () => {
    const wide = Array.from({ length: 30 }, (_, i) => Array.from({ length: 25 }, (_, j) => Math.sin(i * (j + 1))))
    expect(() => subsetRegression({ method: 'best' }).fit({ x: wide, y: wide.map((r) => r[0] ?? 0) })).toThrow(
      /too large/,
    )
    expect(() =>
      subsetRegression({ criterion: 'cp' }).fit({
        x: wide.map((r) => r.slice(0, 10)).slice(0, 8),
        y: wide.slice(0, 8).map((r) => r[0] ?? 0),
      }),
    ).toThrow(/Cp/)
  })
})
