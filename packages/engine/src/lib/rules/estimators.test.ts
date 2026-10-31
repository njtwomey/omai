import { describe, expect, it } from 'vitest'
import { plantedRules } from '../../testing/rules.ts'
import { revive, serialiseFitted } from '../learn/index.ts'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import {
  bayesianRuleListClassifier,
  cn2Classifier,
  corelsClassifier,
  decisionTableClassifier,
  oneRClassifier,
  prismClassifier,
  ripperClassifier,
} from './index.ts'

const p = plantedRules(300, 0.05, 2)
const nominal = p.nominal

const estimators = [
  ['cn2', () => cn2Classifier({ nominal, alpha: 0.01, maxLength: 3 })],
  ['cn2-unordered', () => cn2Classifier({ mode: 'unordered', nominal, alpha: 0.01, maxLength: 3 })],
  ['prism', () => prismClassifier({ nominal })],
  ['ripper', () => ripperClassifier({ nominal })],
  ['corels', () => corelsClassifier({ nominal, minSupport: 0.15, maxCardinality: 2, lambda: 0.08 })],
  ['brl', () => bayesianRuleListClassifier({ nominal, minSupport: 0.1, maxCardinality: 1, steps: 300 })],
  ['one-r', () => oneRClassifier({ nominal })],
  ['decision-table', () => decisionTableClassifier({ nominal })],
] as const

describe.each(estimators)('%s as an estimator', (_name, make) => {
  const est = make()
  const fitted = est.fit(p.data, { key: key(1) })

  it('predicts labels and valid probabilities', () => {
    const labels = Array.from(toFlat(fitted.predict(p.data.x)))
    expect(labels.length).toBe(300)
    expect(labels.every((v) => v === 0 || v === 1)).toBe(true)
    const proba = toFlat(fitted.predictProba(p.data.x))
    for (let i = 0; i < 300; i++) expect((proba[2 * i] ?? 0) + (proba[2 * i + 1] ?? 0)).toBeCloseTo(1, 12)
    expect(fitted.score(p.data)).toBeGreaterThan(0.5)
  })

  it('prints its rules and reports the rule that decides each row', () => {
    expect(fitted.text().split('\n').at(-1)).toMatch(/^ELSE/)
    const applied = fitted.apply(p.data.x)
    expect(applied.length).toBe(300)
    expect(Math.max(...applied)).toBeLessThan(fitted.model.rules.length)
  })

  it('round-trips through serialisation to the same predictions', () => {
    const back = revive(est as never, JSON.parse(JSON.stringify(serialiseFitted(fitted)))) as typeof fitted
    expect(Array.from(toFlat(back.predict(p.data.x)))).toEqual(Array.from(toFlat(fitted.predict(p.data.x))))
  })

  it('is a session: the model at the end is the fitted model, and earlier steps give earlier models', () => {
    const session = est.session(p.data, { mode: 'eager', key: key(1) })
    expect(session.fittedAt(session.t).model).toEqual(fitted.model)
    expect(session.fittedAt(Math.floor(session.t / 2)).model.rules.length).toBeLessThanOrEqual(
      fitted.model.rules.length + 1,
    )
  })
})

describe('contract paths', () => {
  it('CN2 logs the covering tracks and answers the rules probe', () => {
    const est = cn2Classifier({ nominal, alpha: 0.01, maxLength: 3 })
    const session = est.session(p.data, { mode: 'eager', key: key(1), logs: 'all' })
    const run = session.run()
    const names = new Set(run.events.flatMap((e) => ('name' in e ? [String((e as { name: unknown }).name)] : [])))
    for (const path of ['ruleset/rules', 'ruleset/accuracy'])
      expect([...names].some((n) => n.includes(path.split('/')[1] ?? ''))).toBe(true)
    const record = session.probe(session.t, 'model/rules') as { kind: string; rules: unknown[] }
    expect(record.kind).toBe('rules')
    expect(record.rules.length).toBeGreaterThan(0)
  })
})
