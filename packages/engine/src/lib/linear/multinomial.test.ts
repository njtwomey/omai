import { describe, expect, it } from 'vitest'
import { matrixView } from '../learn/index.ts'
import { toFlat } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { linearModel } from './model.ts'
import { multinomial, multinomialAlgorithm, softmaxRows } from './multinomial.ts'

const n = 90
const x = Array.from({ length: n }, (_, i) => [Math.sin(i * 1.3) * 1.5, Math.cos(i * 2.1)])
const y = x.map((r, i) => {
  const s = [0, 0.8 * (r[0] ?? 0) - 0.5, -0.6 * (r[0] ?? 0) + 0.7 * (r[1] ?? 0)].map(
    (v, k) => v + 0.9 * Math.sin(i * 17.3 + k * 5.1),
  )
  return s.indexOf(Math.max(...s))
})

describe('multinomial logit', () => {
  it('has probabilities that sum to one and a stationary point of the likelihood', () => {
    const f = multinomial().fit({ x, y })
    expect(f.params.converged).toBe(true)
    expect(f.params.reference).toBe(0)
    expect(f.params.coef[0]).toEqual([0, 0])
    const p = toFlat(f.predictProba(x))
    for (let i = 0; i < n; i++) expect((p[i * 3] ?? 0) + (p[i * 3 + 1] ?? 0) + (p[i * 3 + 2] ?? 0)).toBeCloseTo(1, 12)
    // Score equations: for each class and feature, sum_i x_ij (1[y_i = c] - p_ic) = 0 (with the intercept column of ones).
    for (let c = 1; c < 3; c++)
      for (const col of [0, 1, -1]) {
        let s = 0
        for (let i = 0; i < n; i++)
          s += (col < 0 ? 1 : (x[i]?.[col] ?? 0)) * ((y[i] === c ? 1 : 0) - (p[i * 3 + c] ?? 0))
        expect(Math.abs(s)).toBeLessThan(1e-7)
      }
    expect(f.score({ x, y })).toBeGreaterThan(0.5)
    expect(f.params.classes).toEqual([0, 1, 2])
  })

  it('is logistic regression for two classes', () => {
    const y2 = y.map((v) => (v === 0 ? 0 : 1))
    const f = multinomial().fit({ x, y: y2 })
    const glm = linearModel({ family: 'binomial', alpha: 0 }).fit({ x, y: y2 })
    f.params.coef[1]?.forEach((v, j) => {
      expect(v).toBeCloseTo(glm.params.coef[j] ?? 0, 6)
    })
    expect(f.params.intercept[1]).toBeCloseTo(glm.params.intercept, 6)
  })

  it('shrinks the weights with a ridge penalty, giving every class its own', () => {
    const plain = multinomial().fit({ x, y }).params
    const ridge = multinomial({ alpha: 0.5 }).fit({ x, y }).params
    expect(ridge.reference).toBeNull()
    expect(ridge.coef[0]?.some((v) => v !== 0)).toBe(true)
    const norm = (c: number[][]): number => c.flat().reduce((a, v) => a + v * v, 0)
    expect(norm(ridge.coef)).toBeLessThan(norm(plain.coef))
  })

  it('treats integer weights as repeated rows and reports a Wald table', () => {
    const w = x.map((_, i) => (i % 3 === 0 ? 2 : 1))
    const a = multinomial().fit({ x, y, weights: w })
    const b = multinomial().fit({
      x: x.flatMap((r, i) => (w[i] === 2 ? [r, r] : [r])),
      y: y.flatMap((v, i) => (w[i] === 2 ? [v, v] : [v])),
    })
    expect(a.params.logLikelihood).toBeCloseTo(b.params.logLikelihood, 8)
    const table = a.exposition[0]
    expect(table?.kind).toBe('coefficientTable')
    if (table?.kind === 'coefficientTable')
      expect(table.names).toEqual(['1:x0', '1:x1', '1:intercept', '2:x0', '2:x1', '2:intercept'])
  })

  it('shifts logits for stability and converges in a few Newton steps', () => {
    expect(Array.from(softmaxRows([1000, 1000, 999], 3)).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
    const problem = { x: matrixView(x, 'test'), codes: y, w: y.map(() => 1), nClasses: 3 }
    const run = runAlgorithm(multinomialAlgorithm(problem), { maxSteps: 50 })
    expect(run.status).toBe('converged')
    expect(run.steps).toBeLessThan(15)
  })
})
