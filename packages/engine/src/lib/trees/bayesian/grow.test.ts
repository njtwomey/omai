import { describe, expect, it } from 'vitest'
import { key, normal, uniform } from '../../random/index.ts'
import { add, mul, toArray, toFlat, type Value } from '../../tensor/index.ts'
import { runAlgorithm } from '../../trace/index.ts'
import { bayesGrowAlgorithm, growBayesTree, splitLogOdds, treeAfter } from './grow.ts'
import { dirichletLeaf, normalGammaLeaf, regressionPrior } from './leaf.ts'
import { treePrior } from './prior.ts'
import { searchBayesSplits } from './search.ts'
import { quantileRules, splitTable, treeSpace } from './space.ts'
import { bayesData, bayesLeafCount, treeLogEvidence } from './tree.ts'

const staircase = (n: number, k: number, noise = 0.15): { x: Value; y: Value } => {
  const x = uniform(key(k), [n], { low: 0, high: 4 })
  const level = (v: number): number => (v < 1.3 ? 0 : v < 2.7 ? 1.5 : 3)
  const clean = toFlat(x).map(level)
  return { x, y: add(clean, mul(normal(key(k + 100), [n]), noise)) }
}

describe('growBayesTree', () => {
  it('chooses the known best split on a one-step function', () => {
    const x = [[0], [1], [2], [3], [4], [5], [6], [7]]
    const y = [0, 0.1, -0.1, 0, 5, 5.1, 4.9, 5]
    const grown = growBayesTree(bayesData(x, y, normalGammaLeaf(regressionPrior(y))))
    expect(grown.steps.map((s) => s.threshold)).toEqual([3.5])
    expect(grown.leaves.map((l) => l.value[0] ?? 0)).toEqual([expect.closeTo(0.5, 1), expect.closeTo(4.5, 1)])
  })

  it('finds both steps of a staircase, the larger gain first, then stops', () => {
    const { x, y } = staircase(120, 4)
    const grown = growBayesTree(bayesData(x, y, normalGammaLeaf(regressionPrior(y))))
    expect(grown.steps).toHaveLength(2)
    const cuts = grown.steps.map((s) => s.threshold)
    expect(Math.abs((cuts[0] ?? 0) - 1.3) < 0.15 || Math.abs((cuts[0] ?? 0) - 2.7) < 0.15).toBe(true)
    expect(grown.steps.every((s) => s.logOdds > 0)).toBe(true)
    expect(bayesLeafCount(grown.tree)).toBe(3)
    expect(grown.steps[0]?.gain).toBeGreaterThan(grown.steps[1]?.gain ?? 0)
  })

  it('does not split pure noise', () => {
    const x = uniform(key(8), [60], { low: 0, high: 1 })
    const y = normal(key(9), [60])
    const grown = growBayesTree(bayesData(x, y, normalGammaLeaf(regressionPrior(y))))
    expect(grown.steps).toHaveLength(0)
    expect(grown.tree.kind).toBe('leaf')
  })

  it('each decision raises the exact tree evidence by its gain', () => {
    const { x, y } = staircase(90, 2)
    const data = bayesData(x, y, normalGammaLeaf(regressionPrior(y)))
    const grown = growBayesTree(data)
    let before = treeLogEvidence(data, treeAfter(grown.state, 0))
    grown.steps.forEach((s, k) => {
      expect(s.logEvidence - before).toBeCloseTo(s.gain, 9)
      expect(treeLogEvidence(data, treeAfter(grown.state, k + 1))).toBeCloseTo(s.logEvidence, 9)
      before = s.logEvidence
    })
  })

  it('searches every threshold in the same numbers as the candidate-table route', () => {
    const { x, y } = staircase(40, 6)
    const data = bayesData(x, y, normalGammaLeaf(regressionPrior(y)))
    const space = treeSpace(data, treePrior({ minLeaf: 2 }), quantileRules(x, 100))
    const table = splitTable(space, space.root)
    const viaTable = toFlat(table.left).map((l, c) => l + (toFlat(table.right)[c] ?? 0) - space.root.logEvidence)
    const search = searchBayesSplits(data, undefined, 2)
    const byThreshold = new Map(toFlat(search.thresholds).map((t, i) => [t, toFlat(search.gains)[i] ?? 0] as const))
    space.rules.thresholds.forEach((t, c) => {
      if (space.root.valid.includes(c)) expect(byThreshold.get(t)).toBeCloseTo(viaTable[c] ?? Number.NaN, 9)
    })
    expect(search.available).toBe(space.root.valid.length)
  })

  it('splits a two-class problem on the feature that carries the boundary', () => {
    const x = uniform(key(5), [100, 2], { low: 0, high: 1 })
    const rows = toArray(x) as number[][]
    const labels = rows.map((r) => ((r[1] ?? 0) > 0.6 ? 1 : 0))
    const grown = growBayesTree(bayesData(x, labels, dirichletLeaf(2)))
    expect(grown.steps[0]?.feature).toBe(1)
    expect(Math.abs((grown.steps[0]?.threshold ?? 0) - 0.6)).toBeLessThan(0.08)
    expect(grown.leaves.map((l) => l.value[1] ?? 0).sort()[0]).toBeLessThan(0.1)
  })

  it('respects the depth limit and the minimum leaf size', () => {
    const { x, y } = staircase(120, 4)
    const model = normalGammaLeaf(regressionPrior(y))
    const stump = growBayesTree(bayesData(x, y, model), { prior: treePrior({ maxDepth: 1 }) })
    expect(stump.steps).toHaveLength(1)
    const roomy = growBayesTree(bayesData(x, y, model), { prior: treePrior({ minLeaf: 70 }) })
    expect(roomy.steps).toHaveLength(0)
  })

  it('runs as a stepped algorithm and logs each decision', () => {
    const { x, y } = staircase(90, 3)
    const data = bayesData(x, y, normalGammaLeaf(regressionPrior(y)))
    const run = runAlgorithm(bayesGrowAlgorithm(data), { maxSteps: 50 })
    expect(run.status).toBe('converged')
    expect(run.state.steps.length).toBe(2)
  })

  it('has odds that grow with the Bayes factor and fall with the rules to choose from', () => {
    const prior = treePrior()
    expect(splitLogOdds(prior, 0, 10, 6)).toBeGreaterThan(splitLogOdds(prior, 0, 10, 3))
    expect(splitLogOdds(prior, 0, 100, 3)).toBeLessThan(splitLogOdds(prior, 0, 10, 3))
    expect(splitLogOdds(prior, 3, 10, 50)).toBe(Number.NEGATIVE_INFINITY)
  })

  it('prices each new leaf by the rules it has itself: a child that cannot split is a leaf for certain', () => {
    const prior = treePrior()
    // a child at depth 1 splits with probability 0.5 / 4 = 0.125 when it has a rule, and with probability 0 when it has none
    const leaf = Math.log(1 - 0.125)
    const base = splitLogOdds(prior, 0, 10, 5, [0, 0])
    expect(splitLogOdds(prior, 0, 10, 5, [3, 0]) - base).toBeCloseTo(leaf, 12)
    expect(splitLogOdds(prior, 0, 10, 5, [3, 4]) - base).toBeCloseTo(2 * leaf, 12)
    // without the children's rules it is the odds with a rule each
    expect(splitLogOdds(prior, 0, 10, 5)).toBeCloseTo(splitLogOdds(prior, 0, 10, 5, [1, 1]), 12)
  })

  it('the odds the grower reports are the prior odds under the rules the children really have', () => {
    // six rows: a split at 2.5 leaves three rows on each side, too few for a further split (a leaf needs two on each side)
    const y = [0, 0.1, 0, 5, 5.1, 5]
    const data = bayesData([[0], [1], [2], [3], [4], [5]], y, normalGammaLeaf(regressionPrior(y)))
    const grown = growBayesTree(data)
    const first = grown.steps[0]
    expect(first?.threshold).toBe(2.5)
    const found = searchBayesSplits(data, undefined, 2)
    const p = 0.5
    // no leaf charge for the two children: neither has a rule
    const expected = Math.log(p) - Math.log(1 - p) - Math.log(found.available) + (first?.gain ?? Number.NaN)
    expect(first?.logOdds).toBeCloseTo(expected, 12)
  })
})
