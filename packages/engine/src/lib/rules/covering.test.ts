import { describe, expect, it } from 'vitest'
import { CONTACT_LENSES, plantedRules } from '../../testing/rules.ts'
import { toFlat } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import {
  cn2Classifier,
  coveringAlgorithm,
  coveringModel,
  fitCovering,
  modelText,
  prismClassifier,
  ruleLine as ruleText,
} from './index.ts'

const lenses = { x: CONTACT_LENSES.x, y: CONTACT_LENSES.y }

describe('the contact lenses data', () => {
  it("is Cendrowska's: 5 soft, 4 hard and 15 none, and tear production reduced is always none", () => {
    const counts = [0, 0, 0]
    for (const c of CONTACT_LENSES.y) counts[c] = (counts[c] ?? 0) + 1
    expect(counts).toEqual([5, 4, 15])
    CONTACT_LENSES.x.forEach((row, i) => {
      if (row[3] === 0) expect(CONTACT_LENSES.y[i]).toBe(2)
    })
  })
})

describe('PRISM', () => {
  it('learns pure rules that classify the training data perfectly, among them tears = reduced for none', () => {
    const model = fitCovering(lenses, { mode: 'prism', nominal: CONTACT_LENSES.nominal })
    const fitted = prismClassifier({ nominal: CONTACT_LENSES.nominal }).fit(lenses)
    expect(Array.from(toFlat(fitted.predict(CONTACT_LENSES.x)))).toEqual(CONTACT_LENSES.y)
    for (const r of model.rules) expect(r.counts.filter((c) => c > 0).length).toBe(1)
    const none = model.rules.filter((r) => r.label === 2)
    expect(
      none.some((r) => r.literals.length === 1 && r.literals[0]?.feature === 3 && r.literals[0]?.value === 0),
    ).toBe(true)
  })
})

describe('CN2', () => {
  it('ordered CN2 on a clean separable table learns the one rule and a default', () => {
    const data = { x: [[0], [1], [2], [3], [4], [5]], y: [0, 0, 0, 1, 1, 1] }
    expect(modelText(fitCovering(data, { alpha: 1, minCovered: 1 }))).toBe('IF x0 <= 2.5 THEN 0\nELSE 1')
  })

  it('recovers the planted list without noise and predicts every row', () => {
    const p = plantedRules(600, 0)
    const fitted = cn2Classifier({ nominal: p.nominal, beamWidth: 8, alpha: 0.01, maxLength: 3 }).fit(p.data)
    expect(fitted.score(p.data)).toBeGreaterThan(0.99)
  })

  it('stays near the planted accuracy under label noise and generalises', () => {
    const train = plantedRules(800, 0.05, 2)
    const test = plantedRules(800, 0, 3)
    const fitted = cn2Classifier({ nominal: train.nominal, beamWidth: 8, alpha: 0.01, maxLength: 3 }).fit(train.data)
    // 5% of the training labels are flipped; a few spurious rules cost a little accuracy on clean test rows.
    expect(fitted.score(test.data)).toBeGreaterThan(0.9)
  })

  it('unordered CN2 votes among rules and is as accurate on the planted data', () => {
    const train = plantedRules(800, 0.05, 2)
    const test = plantedRules(800, 0, 3)
    const fitted = cn2Classifier({
      mode: 'unordered',
      nominal: train.nominal,
      beamWidth: 8,
      alpha: 0.01,
      maxLength: 3,
    }).fit(train.data)
    expect(fitted.model.kind).toBe('set')
    expect(fitted.score(test.data)).toBeGreaterThan(0.9)
  })

  it('the significance level controls how many rules are kept', () => {
    const p = plantedRules(300, 0.3, 4)
    const strict = fitCovering(p.data, { nominal: p.nominal, alpha: 0.001 }).rules.length
    const lax = fitCovering(p.data, { nominal: p.nominal, alpha: 1, minCovered: 2 }).rules.length
    expect(lax).toBeGreaterThanOrEqual(strict)
  })

  it('beam width 1 is greedy and still finds a rule', () => {
    const p = plantedRules(400, 0)
    expect(fitCovering(p.data, { nominal: p.nominal, beamWidth: 1 }).rules.length).toBeGreaterThan(0)
  })

  it('rejects bad options', () => {
    expect(() => coveringAlgorithm({ x: [[0], [1]], y: [0, 1] }, { beamWidth: 0 })).toThrow()
    expect(() => coveringAlgorithm({ x: [[0], [1]], y: [0, 1] }, { alpha: 0 })).toThrow()
  })
})

describe('the covering loop as an algorithm', () => {
  it('takes one rule per step and the state round-trips through JSON', () => {
    const p = plantedRules(500, 0)
    const alg = coveringAlgorithm(p.data, { nominal: p.nominal, beamWidth: 8, alpha: 0.01, maxLength: 3 })
    const { state, steps } = runAlgorithm(alg, { maxSteps: 50, seed: 0 })
    expect(steps).toBe(state.rules.length + 1)
    const back = alg.deserialise?.(JSON.parse(JSON.stringify(alg.serialise?.(state))))
    expect(back).toEqual(state)
    // Each ordered rule removes rows: the rows in play shrink.
    expect(state.remaining.length).toBeLessThan(500)
    const model = coveringModel(p.data, state, { nominal: p.nominal })
    expect(model.rules.length).toBe(state.rules.length)
  })

  it('a session scrubs through it: fittedAt(t) has t rules', () => {
    const p = plantedRules(500, 0)
    const est = cn2Classifier({ nominal: p.nominal, beamWidth: 8, alpha: 0.01, maxLength: 3 })
    const session = est.session(p.data, { mode: 'eager' })
    expect(session.fittedAt(1).model.rules.length).toBe(1)
    expect(session.fittedAt(session.t).model.rules.length).toBeGreaterThanOrEqual(2)
  })

  it('rules print as text', () => {
    const text = ruleText(
      { literals: [{ feature: 0, op: '==', value: 2 }], label: 1, counts: [] },
      ['no', 'yes'],
      ['a'],
    )
    expect(text).toBe('IF a = 2 THEN yes')
  })
})
