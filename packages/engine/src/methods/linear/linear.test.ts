import { generate } from 'mdml-engine/data'
import { makeRegression, moons } from 'mdml-engine/datasets'
import { describe, expect, it } from 'vitest'
import { key } from '../../lib/random/index.ts'
import { toFlat } from '../../lib/tensor/index.ts'
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
