import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { metricSeries } from '../trace/index.ts'
import { gpClassifier } from './classification.ts'

const x = [[-2.1], [-1.6], [-1.1], [-0.6], [-0.2], [0.3], [0.7], [1.2], [1.7], [2.2]]
const y = [0, 0, 0, 0, 1, 0, 1, 1, 1, 1]
const data = { x, y }

describe('gpClassifier', () => {
  it('runs as a session in two phases, the hyperparameter search then the Newton iterations, with probes at an early step', () => {
    const s = gpClassifier.session(data, { key: key(1), logs: 'all' })
    s.advance({ steps: 2 })
    const early = s.fittedAt(2)
    expect(toFlat(early.predictProba([[0]])).length).toBe(2)
    const grid = s.probe(2, 'model/posterior', { resolution: 8 }) as {
      kind: string
      mean: number[]
      variance: number[]
    }
    expect(grid.kind).toBe('posteriorGrid')
    expect(grid.mean).toHaveLength(8)
    s.advance()
    expect(s.status).toBe('converged')
    const lml = metricSeries(s.run(), 'gp/logMarginalLikelihood').values
    expect(lml.length).toBeGreaterThan(4)
    expect(s.state.phase).toBe('done')
    expect(s.fittedAt(s.t).params.logMarginalLikelihood).toBeCloseTo(lml[lml.length - 1] as number, 6)
  })

  it('classifies, with probabilities that order along the line and a latent mean positive for the second class', () => {
    const fitted = gpClassifier.fit(data, { explain: false })
    const p = toFlat(fitted.predictProba([[-2], [0], [2]]))
    expect(p[1] as number).toBeLessThan(p[3] as number)
    expect(p[3] as number).toBeLessThan(p[5] as number)
    expect(toFlat(fitted.predict([[-2], [2]]))).toEqual([0, 1])
    expect(toFlat(fitted.decisionFunction([[2]]))[0]).toBeGreaterThan(0)
    expect(fitted.score(data)).toBeGreaterThanOrEqual(0.8)
    expect(toFlat(fitted.latent([[1]]).variance)[0]).toBeGreaterThan(0)
  })

  it('logs the EP sweeps in its second phase: sites appear one by one and the likelihood is recorded per sweep', () => {
    const est = gpClassifier.with({ method: 'ep', likelihood: 'probit', optimise: false })
    const s = est.session(data, { key: key(1), logs: 'all' })
    s.advance({ steps: 3 })
    const early = s.fittedAt(3).params.posterior
    expect('tau' in early && early.tau.filter((v) => v > 0).length).toBe(3)
    s.advance()
    expect(s.status).toBe('converged')
    expect(metricSeries(s.run(), 'gp/logMarginalLikelihood').values.length).toBeGreaterThan(0)
  })

  it('has quadrature as the exact predictive of the logistic likelihood, close to the five-erf approximation', () => {
    const approximate = gpClassifier.with({ optimise: false }).fit(data, { explain: false })
    const quadrature = gpClassifier.with({ optimise: false, predictive: 'quadrature' }).fit(data, { explain: false })
    const a = toFlat(approximate.predictProba([[-1], [0.5], [3]]))
    const q = toFlat(quadrature.predictProba([[-1], [0.5], [3]]))
    for (const [i, v] of a.entries()) expect(v).toBeCloseTo(q[i] as number, 1)
  })

  it('agrees between the approximations on easy data', () => {
    const laplace = gpClassifier.with({ likelihood: 'probit', optimise: false }).fit(data, { explain: false })
    const ep = gpClassifier.with({ method: 'ep', likelihood: 'probit', optimise: false }).fit(data, { explain: false })
    const a = toFlat(laplace.predictProba([[-1.5], [1.5]]))
    const b = toFlat(ep.predictProba([[-1.5], [1.5]]))
    for (const [i, v] of a.entries()) expect(Math.abs(v - (b[i] as number))).toBeLessThan(0.1)
  })

  it('revives from JSON and rejects what it cannot do', () => {
    const est = gpClassifier.with({ optimise: false })
    const fitted = est.fit(data, { explain: false })
    const again = est.revive(JSON.parse(JSON.stringify(fitted.params)))
    expect(toFlat(again.predictProba([[0.1]]))).toEqual(toFlat(fitted.predictProba([[0.1]])))
    expect(() => est.fit({ x: [[0], [1], [2]], y: [0, 1, 2] })).toThrow(/two classes/)
    expect(() => gpClassifier.with({ method: 'ep' }).fit(data)).toThrow(/probit/)
  })
})
