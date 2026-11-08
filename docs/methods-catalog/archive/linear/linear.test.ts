import { generate } from 'omai-engine/data'
import { makeRegression, moons, sineCurve } from 'omai-engine/datasets'
import { describe, expect, it } from 'vitest'
import { key } from '../../lib/random/index.ts'
import { toFlat } from '../../lib/tensor/index.ts'
import { metricSeries } from '../../lib/trace/index.ts'
import { bayesianRegression } from './bayesianRegression.ts'
import { lassoPath } from './lassoPath.ts'
import { logisticRegression } from './logisticRegression.ts'
import { ridge } from './ridge.ts'

const classes = generate(moons, { knobs: { n: 80 }, key: key(0) }).data
const line = generate(makeRegression, { knobs: { n: 40 }, key: key(0) }).data

describe('logisticRegression', () => {
  it('fits, predicts and is probed in a session', () => {
    const fitted = logisticRegression.fit(classes, { key: key(1) })
    expect(fitted.params.coef.length).toBe(2)
    expect(toFlat(fitted.predict(classes.x)).length).toBe(80)
    expect(toFlat(fitted.predictProba(classes.x)).length).toBe(160)
    const s = logisticRegression.session(classes, { key: key(1), logs: 'all' })
    s.advance({ steps: 3 })
    expect(s.probe(s.t, 'model/params')).toHaveProperty('coef')
  })

  it('predicts with the model at any step of its session, and at the head as fit would', () => {
    const s = logisticRegression.session(classes, { key: key(1) })
    s.advance({ steps: 2 })
    const first = s.fittedAt(1)
    const second = s.fittedAt(2)
    expect(first.classes).toEqual([0, 1])
    expect(toFlat(first.predictProba(classes.x)).length).toBe(160)
    expect(toFlat(first.decisionFunction(classes.x))).not.toEqual(toFlat(second.decisionFunction(classes.x)))
    s.advance()
    expect(s.fittedAt(s.t).params).toEqual(logisticRegression.fit(classes, { key: key(1) }).params)
  })
})

describe('ridge', () => {
  it('fits in one step and shrinks with the penalty', () => {
    const weak = ridge.with({ alpha: 1e-4 }).fit(line, { key: key(1) })
    const strong = ridge.with({ alpha: 100 }).fit(line, { key: key(1) })
    expect(Math.abs(strong.params.coef[0] ?? 0)).toBeLessThan(Math.abs(weak.params.coef[0] ?? 0))
    expect(toFlat(weak.predict(line.x)).length).toBe(40)
  })
})

describe('lassoPath', () => {
  it('walks the penalty grid and projects a regularisationPath', () => {
    const fitted = lassoPath.with({ nAlphas: 8 }).fit(line, { key: key(1) })
    const path = fitted.exposition.find((r) => r.kind === 'regularisationPath')
    expect(path?.kind === 'regularisationPath' && path.penalties.length).toBe(8)
    const s = lassoPath.with({ nAlphas: 8 }).session(line, { key: key(1), logs: 'all' })
    s.advance({ steps: 3 })
    expect(s.t).toBe(3)
  })
})

describe('bayesianRegression', () => {
  it('absorbs one row per step, and its predictive band is a probe of any step', () => {
    const curve = generate(sineCurve, { knobs: { n: 30, noise: 0.2 }, key: key(0) }).data
    const fitted = bayesianRegression.fit(curve, { key: key(1) })
    expect(fitted.params.nObserved).toBe(30)
    expect(fitted.params.mean.length).toBe(4)
    expect(toFlat(fitted.predict(curve.x)).length).toBe(30)
    expect(toFlat(fitted.predictDistribution(curve.x).mean()).length).toBe(30)
    const s = bayesianRegression.session(curve, { key: key(1), logs: 'all' })
    s.advance({ steps: 40 })
    expect(s.t).toBe(30)
    const band = (t: number) => s.probe(t, 'model/predictive') as { mean: number[]; lower: number[]; upper: number[] }
    expect(band(30).mean.length).toBe(30)
    const width = (t: number) => (band(t).upper[0] ?? 0) - (band(t).lower[0] ?? 0)
    expect(width(30)).toBeLessThan(width(0))
    const evidence = metricSeries(s.run(), 'solver/logEvidence').values
    expect(evidence.length).toBeGreaterThan(25)
    expect(evidence.at(-1)).toBeCloseTo(fitted.params.logEvidence, 8)
  })
})
