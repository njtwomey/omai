import { describe, expect, it } from 'vitest'
import { plantedRules } from '../../testing/rules.ts'
import { foldIn, key } from '../random/index.ts'
import { nullTracer, runAlgorithm, startRun, stepContext } from '../trace/index.ts'
import {
  bayesianRuleListAlgorithm,
  bayesianRuleListClassifier,
  corelsAlgorithm,
  corelsClassifier,
  fitBayesianRuleList,
  fitCorels,
  mineAntecedents,
  modelText,
  ruleListLogPosterior,
} from './index.ts'

/** Four binary features cycling through all combinations; the label follows x0 = 1 and x1 = 0, with every 11th row flipped. */
function table(n: number): { x: number[][]; y: number[] } {
  const x = Array.from({ length: n }, (_, i) => [
    i % 2,
    Math.floor(i / 2) % 2,
    Math.floor(i / 4) % 2,
    Math.floor(i / 8) % 2,
  ])
  const y = x.map((r, i) => (r[0] === 1 && r[1] === 0 ? 1 : 0) ^ (i % 11 === 0 ? 1 : 0))
  return { x, y }
}

describe('antecedents', () => {
  it('are the conjunctions with enough support, without duplicates of the same coverage', () => {
    const p = plantedRules(200, 0, 1)
    const a = mineAntecedents(p.data, { nominal: p.nominal, minSupport: 0.1, maxCardinality: 2 })
    expect(a.literals.length).toBeGreaterThan(6)
    const ids = new Set(a.masks.map((m) => m.join('')))
    expect(ids.size).toBe(a.masks.length)
    for (const m of a.masks) expect(m.reduce((s, v) => s + v, 0)).toBeGreaterThanOrEqual(20)
  })
})

describe('CORELS', () => {
  it('finds the optimum that exhaustive enumeration of lists finds, and certifies it', () => {
    const data = table(48)
    const options = { nominal: [0, 1, 2, 3], minSupport: 0.2, maxCardinality: 2, lambda: 0.03 }
    const a = mineAntecedents(data, options)
    const n = 48
    const objective = (list: readonly number[]): number => {
      const left = new Array<boolean>(n).fill(true)
      let mistakes = 0
      for (const r of list) {
        const c: [number, number] = [0, 0]
        for (let i = 0; i < n; i++) {
          if (left[i] && a.masks[r]?.[i] === 1) {
            c[data.y[i] === 1 ? 1 : 0]++
            left[i] = false
          }
        }
        mistakes += c[0] + c[1] - Math.max(c[0], c[1])
      }
      const rest: [number, number] = [0, 0]
      for (let i = 0; i < n; i++) if (left[i]) rest[data.y[i] === 1 ? 1 : 0]++
      mistakes += rest[0] + rest[1] - Math.max(rest[0], rest[1])
      return mistakes / n + options.lambda * list.length
    }
    let best = objective([])
    const extend = (list: number[], depth: number): void => {
      if (depth === 0) return
      for (let r = 0; r < a.masks.length; r++) {
        if (list.includes(r)) continue
        const next = [...list, r]
        best = Math.min(best, objective(next))
        extend(next, depth - 1)
      }
    }
    extend([], 3)
    const result = fitCorels(data, options)
    expect(result.certified).toBe(true)
    expect(result.objective).toBeCloseTo(best, 12)
    expect(result.model.rules.length).toBeLessThanOrEqual(3)
  })

  it("prunes: far fewer prefixes are evaluated than there are lists up to the answer's length plus one", () => {
    const data = table(64)
    const options = { nominal: [0, 1, 2, 3], minSupport: 0.1, maxCardinality: 2, lambda: 0.02 }
    const A = mineAntecedents(data, options).masks.length
    const r = fitCorels(data, options)
    expect(r.certified).toBe(true)
    expect(r.nodes).toBeLessThan(A * A * A)
  })

  it('a larger penalty gives a shorter or equal list', () => {
    const data = table(64)
    const base = { nominal: [0, 1, 2, 3], minSupport: 0.1, maxCardinality: 2 }
    const small = fitCorels(data, { ...base, lambda: 0.005 })
    const large = fitCorels(data, { ...base, lambda: 0.1 })
    expect(large.model.rules.length).toBeLessThanOrEqual(small.model.rules.length)
  })

  it('learns the planted list well and predicts held-out rows', () => {
    const train = plantedRules(400, 0.05, 2)
    const test = plantedRules(600, 0, 3)
    const fitted = corelsClassifier({ nominal: train.nominal, minSupport: 0.1, maxCardinality: 2, lambda: 0.04 }).fit(
      train.data,
    )
    expect(fitted.score(test.data)).toBeGreaterThan(0.9)
    expect(modelText(fitted.model).split('\n').at(-1)).toMatch(/^ELSE/)
    // The search takes about five seconds alone: the full suite runs many files at once.
  }, 120_000)

  it('the bound never exceeds the incumbent and the queue empties at the certificate', () => {
    const alg = corelsAlgorithm(table(64), { nominal: [0, 1, 2, 3], minSupport: 0.1, maxCardinality: 2, lambda: 0.02 })
    const { state } = runAlgorithm(alg, { maxSteps: 100000, seed: 0 })
    expect(state.certified).toBe(true)
    expect(state.queue.length).toBe(0)
    expect(state.lowerBound).toBeLessThanOrEqual(state.objective + 1e-12)
  })

  it('rejects a negative penalty', () => {
    expect(() => corelsAlgorithm({ x: [[0], [1]], y: [0, 1] }, { lambda: -1 })).toThrow()
  })
})

describe('Bayesian rule lists', () => {
  it('the chain samples the exact posterior over lists (the Hastings ratios are right)', () => {
    // Two binary features, antecedents of one literal each: four antecedents and 65 ordered lists.
    const x = Array.from({ length: 40 }, (_, i) => [i % 2, Math.floor(i / 2) % 2])
    const y = x.map((r, i) => (r[0] === 1 && i % 5 !== 0 ? 1 : 0))
    const data = { x, y }
    const options = { nominal: [0, 1], minSupport: 0.2, maxCardinality: 1 }
    const A = mineAntecedents(data, options).masks.length
    expect(A).toBe(4)
    const lists: number[][] = [[]]
    for (let m = 0; m < lists.length; m++) {
      for (let a = 0; a < A; a++) if (!lists[m]?.includes(a)) lists.push([...(lists[m] ?? []), a])
    }
    const unique = new Map(lists.map((l) => [l.join(','), l]))
    expect(unique.size).toBe(65)
    const exact = new Map([...unique].map(([id, l]) => [id, Math.exp(ruleListLogPosterior(data, l, options))]))
    const z = [...exact.values()].reduce((sum, v) => sum + v, 0)
    const alg = bayesianRuleListAlgorithm(data, options)
    const run = startRun(nullTracer, { name: 'brl', seed: 0, config: {} })
    let s = alg.init(key(0))
    const visits = new Map<string, number>()
    const steps = 40000
    for (let t = 0; t < steps; t++) {
      s = alg.step(s, stepContext(t, foldIn(key(0), t), run))
      visits.set(s.list.join(','), (visits.get(s.list.join(',')) ?? 0) + 1)
      expect(Math.exp(s.logPosterior)).toBeCloseTo(exact.get(s.list.join(',')) ?? Number.NaN, 12)
    }
    let tv = 0
    for (const [id, p] of exact) tv += Math.abs((visits.get(id) ?? 0) / steps - p / z)
    expect(tv / 2).toBeLessThan(0.03)
  })

  it('finds a better list than the empty one and the planted structure', () => {
    const train = plantedRules(500, 0.05, 2)
    const test = plantedRules(500, 0, 3)
    const fitted = bayesianRuleListClassifier({
      nominal: train.nominal,
      minSupport: 0.05,
      maxCardinality: 2,
      steps: 3000,
    }).fit(train.data, { key: key(3) })
    expect(fitted.model.rules.length).toBeGreaterThan(0)
    expect(fitted.score(test.data)).toBeGreaterThan(0.85)
  })

  it('is reproducible and its best log posterior never falls', () => {
    const p = plantedRules(200, 0.05, 2)
    const a = fitBayesianRuleList(p.data, {
      nominal: p.nominal,
      steps: 300,
      seed: 5,
      minSupport: 0.1,
      maxCardinality: 1,
    })
    const b = fitBayesianRuleList(p.data, {
      nominal: p.nominal,
      steps: 300,
      seed: 5,
      minSupport: 0.1,
      maxCardinality: 1,
    })
    expect(a).toEqual(b)
    const alg = bayesianRuleListAlgorithm(p.data, { nominal: p.nominal, minSupport: 0.1, maxCardinality: 1 })
    const run = startRun(nullTracer, { name: 'brl', seed: 0, config: {} })
    let s = alg.init(key(0))
    let previous = s.bestLogPosterior
    for (let t = 0; t < 200; t++) {
      s = alg.step(s, stepContext(t, foldIn(key(0), t), run))
      expect(s.bestLogPosterior).toBeGreaterThanOrEqual(previous)
      previous = s.bestLogPosterior
    }
  })

  it('rejects non-positive priors', () => {
    expect(() => bayesianRuleListAlgorithm({ x: [[0], [1]], y: [0, 1] }, { alpha: 0 })).toThrow()
  })
})
