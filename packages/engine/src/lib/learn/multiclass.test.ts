import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { toArray, toFlat } from '../tensor/index.ts'
import { metricSeries } from '../trace/index.ts'
import { linearDiscriminant } from './discriminant.ts'
import { revive, serialiseFitted } from './estimator.ts'
import { nestedDichotomies, oneVsOne, oneVsRest, outputCode } from './multiclass.ts'
import { naiveBayes } from './naive-bayes.ts'

const x = [
  [0, 0],
  [1, 0.5],
  [0.4, 1],
  [6, 0],
  [7, 0.4],
  [6.5, 1],
  [0, 6],
  [1, 7],
  [0.5, 6.5],
  [6, 6],
  [7, 6.5],
  [6.4, 7],
]
const y = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
const queries = [
  [0.5, 0.5],
  [6.5, 0.5],
  [0.5, 6.5],
  [6.5, 6.5],
]

describe('multiclass reductions', () => {
  it.each([
    ['one-vs-rest', () => oneVsRest(linearDiscriminant())],
    ['one-vs-one', () => oneVsOne(linearDiscriminant())],
    ['output code', () => outputCode(linearDiscriminant(), { codeSize: 2 })],
    ['nested dichotomies', () => nestedDichotomies(naiveBayes(), { trees: 3 })],
  ])('%s fits, predicts, scores and survives serialisation', (_name, make) => {
    const est = make()
    const f = est.fit({ x, y })
    expect(toFlat(f.predict(queries))).toEqual([0, 1, 2, 3])
    expect(f.score({ x, y })).toBeGreaterThan(0.9)
    const back = revive(est as never, JSON.parse(JSON.stringify(serialiseFitted(f))))
    expect(toFlat((back as typeof f).predict(queries))).toEqual([0, 1, 2, 3])
  })

  it('gives probabilities that sum to one where the base does', () => {
    for (const f of [
      oneVsRest(naiveBayes()).fit({ x, y }),
      nestedDichotomies(naiveBayes(), { trees: 4 }).fit({ x, y }),
    ]) {
      const p = toArray((f.predictProba as NonNullable<typeof f.predictProba>)(queries)) as number[][]
      for (const row of p) expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
    }
    expect(oneVsOne(linearDiscriminant()).fit({ x, y }).predictProba).toBeUndefined()
  })

  it('solves one binary problem per step, each visible in a session', () => {
    const s = oneVsRest(linearDiscriminant()).session({ x, y }, { key: key(0), logs: 'all' })
    s.advance()
    expect(metricSeries(s.run(), 'solver/problems').values).toEqual([1, 2, 3, 4])
    // After one problem only class 0 has been separated: every query is class 0.
    expect(toFlat(s.fittedAt(0).predict(queries))).toEqual([0, 0, 0, 0])
    expect(s.fittedAt(0).nProblems).toBe(1)
    expect(toFlat(s.fittedAt(3).predict(queries))).toEqual([0, 1, 2, 3])
    expect(Object.keys(s.contract)).toContain('solver/problems')
    expect(s.probe(1, 'model/params')).toMatchObject({ classes: [0, 1, 2, 3] })
  })

  it('refuses a base without probabilities for nested dichotomies, and one class', () => {
    expect(() => nestedDichotomies(outputCode(linearDiscriminant())).fit({ x, y })).toThrow(/probabilities/)
    expect(() => oneVsRest(linearDiscriminant()).fit({ x: [[0], [1]], y: [3, 3] })).toThrow(/two classes/)
  })

  it('has the same random code book for the same key', () => {
    const a = outputCode(linearDiscriminant()).fit({ x, y }, { key: key(5) }).params.plan
    const b = outputCode(linearDiscriminant()).fit({ x, y }, { key: key(5) }).params.plan
    expect(a).toEqual(b)
    expect(a.codeBook).toHaveLength(4)
    expect(a.codeBook[0]).toHaveLength(6)
  })
})
