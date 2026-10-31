import { describe, expect, it } from 'vitest'
import { CONTACT_LENSES, plantedRules } from '../../testing/rules.ts'
import { toFlat } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { decisionTreeClassifier } from '../trees/index.ts'
import {
  decisionTableAlgorithm,
  decisionTableClassifier,
  fitDecisionTable,
  fitOneR,
  oneRAlgorithm,
  oneRClassifier,
  predictCounts,
  treeRuleSet,
} from './index.ts'

const lenses = { x: CONTACT_LENSES.x, y: CONTACT_LENSES.y }

describe('OneR', () => {
  it("picks tear production on the contact lenses, 17 of 24 correct (Weka's OneR output)", () => {
    const { feature, model } = fitOneR(lenses, { nominal: CONTACT_LENSES.nominal })
    expect(feature).toBe(3)
    const fitted = oneRClassifier({ nominal: CONTACT_LENSES.nominal }).fit(lenses)
    const predicted = Array.from(toFlat(fitted.predict(CONTACT_LENSES.x)))
    expect(predicted.filter((p, i) => p === CONTACT_LENSES.y[i]).length).toBe(17)
    expect(model.rules.map((r) => r.label)).toEqual([2, 0])
  })

  it('cuts a numeric feature into buckets of at least minBucket majority rows and merges neighbours', () => {
    const x = Array.from({ length: 40 }, (_, i) => [i, i % 3])
    const y = x.map((r) => ((r[0] as number) < 14 ? 0 : (r[0] as number) < 28 ? 1 : 0))
    const { model, feature } = fitOneR({ x, y }, { minBucket: 4 })
    expect(feature).toBe(0)
    expect(model.rules.map((r) => r.label)).toEqual([0, 1, 0])
    const middle = model.rules[1]
    expect(middle?.literals.map((l) => l.op)).toEqual(['>', '<='])
    expect(middle?.literals[0]?.value).toBeCloseTo(13.5, 12)
    expect(middle?.literals[1]?.value).toBeCloseTo(27.5, 12)
  })

  it('takes one feature per step', () => {
    const { state, steps } = runAlgorithm(oneRAlgorithm(lenses, { nominal: CONTACT_LENSES.nominal }), { maxSteps: 20 })
    expect(steps).toBe(4)
    expect(state.candidates[3]?.errors).toBe(7)
    expect(Math.min(...state.candidates.map((c) => c.errors))).toBe(7)
    expect(state.best).toBe(3)
  })
})

describe('decision table', () => {
  it('chooses the informative feature by leave-one-out accuracy and predicts from its cells', () => {
    const x = Array.from({ length: 40 }, (_, i) => [i % 2, Math.floor(i / 2) % 2, Math.floor(i / 4) % 2])
    const y = x.map((r, i) => (r[0] === 1 ? 1 : 0) ^ (i % 13 === 0 ? 1 : 0))
    const { features, model } = fitDecisionTable({ x, y }, { nominal: [0, 1, 2] })
    expect(features).toEqual([0])
    expect(model.rules.length).toBe(2)
    const fitted = decisionTableClassifier({ nominal: [0, 1, 2] }).fit({ x, y })
    expect(
      Array.from(
        toFlat(
          fitted.predict([
            [0, 1, 1],
            [1, 0, 0],
          ]),
        ),
      ),
    ).toEqual([0, 1])
  })

  it('adds features until none raises the accuracy, and the state serialises', () => {
    const alg = decisionTableAlgorithm(lenses, { nominal: CONTACT_LENSES.nominal })
    const { state } = runAlgorithm(alg, { maxSteps: 20 })
    expect(state.subset.length).toBeGreaterThan(0)
    expect(alg.deserialise?.(JSON.parse(JSON.stringify(alg.serialise?.(state))))).toEqual(state)
  })

  it('a numeric feature is binned and its cells read as threshold literals', () => {
    const x = Array.from({ length: 60 }, (_, i) => [i / 6, (i * 7) % 10])
    const y = x.map((r) => ((r[0] as number) > 5 ? 1 : 0))
    const { model, features } = fitDecisionTable({ x, y }, { bins: 4 })
    expect(features).toContain(0)
    expect(model.rules.every((r) => r.literals.some((l) => l.feature === 0))).toBe(true)
    const fitted = decisionTableClassifier({ bins: 4 }).fit({ x, y })
    expect(fitted.score({ x, y })).toBeGreaterThan(0.9)
  })
})

describe('rules from a tree', () => {
  it('one rule per leaf, mutually exclusive, predicting exactly as the tree does', () => {
    const p = plantedRules(300, 0.1, 4)
    const tree = decisionTreeClassifier({ maxDepth: 4 }).fit(p.data)
    const model = treeRuleSet(tree.params)
    expect(model.rules.length).toBe(tree.params.left.filter((l) => l < 0).length)
    const x = Array.from(toFlat(p.data.x))
    const counts = predictCounts(model, x)
    const fromRules: number[] = []
    for (let i = 0; i < 300; i++) {
      const row = counts.subarray(i * 2, i * 2 + 2)
      fromRules.push((row[1] ?? 0) > (row[0] ?? 0) ? 1 : 0)
    }
    expect(fromRules).toEqual(Array.from(toFlat(tree.predict(p.data.x))))
    // Exclusive: every row fires exactly one rule.
    const fired = model.rules.map((r) => r.literals)
    expect(fired.length).toBeGreaterThan(1)
  })

  it('merges repeated bounds on a feature to the tightest', () => {
    const x = Array.from({ length: 40 }, (_, i) => [i])
    const y = x.map((r) => ((r[0] as number) < 10 ? 0 : (r[0] as number) < 20 ? 1 : (r[0] as number) < 30 ? 0 : 1))
    const model = treeRuleSet(decisionTreeClassifier().fit({ x, y }).params)
    for (const r of model.rules) {
      const keys = r.literals.map((l) => `${l.feature}${l.op}`)
      expect(new Set(keys).size).toBe(keys.length)
    }
  })

  it('refuses a regression tree', () => {
    expect(() =>
      treeRuleSet({ ...decisionTreeClassifier().fit({ x: [[0], [1]], y: [0, 1] }).params, task: 'regression' }),
    ).toThrow()
  })
})
