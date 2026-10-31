import { describe, expect, it } from 'vitest'
import { classificationSet } from '../../testing/supervised.ts'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { Log, messages, metricSeries } from '../trace/index.ts'
import { hoeffdingTreeClassifier } from './hoeffding.ts'
import { leafCount } from './tree.ts'

const moons = classificationSet('moons', { n: 600, noise: 0.15 }, 3)

describe('the Hoeffding tree', () => {
  it('learns a threshold from a stream and splits only when the bound allows', () => {
    const x = Array.from({ length: 300 }, (_, i) => [((i * 37) % 300) / 300, ((i * 11) % 7) / 7])
    const y = x.map(([v = 0]) => (v < 0.4 ? 0 : 1))
    const f = hoeffdingTreeClassifier({ gracePeriod: 25, delta: 1e-3 }).fit({ x, y })
    expect(f.params.feature[0]).toBe(0)
    expect(f.params.threshold[0]).toBeGreaterThan(0.3)
    expect(f.params.threshold[0]).toBeLessThan(0.5)
    expect(f.score?.({ x, y })).toBeGreaterThan(0.9)
    // With a huge grace period no leaf ever asks.
    expect(leafCount(hoeffdingTreeClassifier({ gracePeriod: 10_000 }).fit({ x, y }).params)).toBe(1)
  })

  it('classifies a harder stream and reports prequential accuracy that improves', () => {
    const est = hoeffdingTreeClassifier({ gracePeriod: 40, delta: 1e-3, tieThreshold: 0.2 })
    const f = est.fit(moons, { logs: { 'eval/accuracy': Log(), 'tree/split': Log(), 'tree/leaves': Log() } })
    expect(f.score?.(moons)).toBeGreaterThan(0.8)
    const acc = metricSeries(f.run, 'eval/accuracy').values
    expect(acc.length).toBe(600)
    expect(acc.at(-1)).toBeGreaterThan(0.7)
    const splits = messages(f.run, 'tree/split')
    expect(splits.length).toBeGreaterThan(0)
    expect(metricSeries(f.run, 'tree/leaves').values.at(-1)).toBe(splits.length + 1)
  })

  it('a session seeks: the tree after t examples is a prefix of the final tree', () => {
    const s = hoeffdingTreeClassifier({ gracePeriod: 40, delta: 1e-3, tieThreshold: 0.2 }).session(moons, {
      key: key(0),
      logs: { 'tree/leaves': Log() },
    })
    s.advance({ steps: 300 })
    const early = s.fittedAt(150)
    s.advance()
    const late = s.fittedAt(s.t)
    expect(late.params.left.length).toBeGreaterThanOrEqual(early.params.left.length)
    expect(toFlat(early.predict(moons.x))).toHaveLength(600)
    expect(s.probe(300, 'model/importances')).toHaveLength(2)
  })

  it('splits categorical features one value against the rest', () => {
    const x = Array.from({ length: 200 }, (_, i) => [i % 4, (i * 13) % 10])
    const y = x.map(([c = 0]) => (c === 2 ? 1 : 0))
    const f = hoeffdingTreeClassifier({ gracePeriod: 20, delta: 1e-3, categorical: [0] }).fit({ x, y })
    expect(f.params.categories[0]).toEqual([2])
    expect(f.score?.({ x, y })).toBe(1)
  })
})
