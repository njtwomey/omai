import { describe, expect, it } from 'vitest'
import { toFlat } from '../../tensor/index.ts'
import { enumerateTrees, predictEnumerated, probabilityOfCutIn, topTrees } from './enumerate.ts'
import { dirichletLeaf, normalGammaLeaf, regressionPrior } from './leaf.ts'
import { treePrior } from './prior.ts'
import { quantileRules, scoreTree, treeSpace } from './space.ts'
import { type BayesTree, bayesData, bayesLeaf, bayesSplit, treeSummary } from './tree.ts'

const xs = [[0], [0.5], [1], [1.5], [2], [2.5], [3], [3.5], [4], [4.5], [5], [5.5]]
const y = [0.1, -0.2, 0.3, 0.0, 0.2, 2.1, 1.9, 2.2, 2.0, 4.1, 3.9, 4.2]

function setup(minLeaf = 2, perFeature = 8) {
  const model = normalGammaLeaf(regressionPrior(y))
  const prior = treePrior({ maxDepth: 2, minLeaf })
  return treeSpace(bayesData(xs, y, model), prior, quantileRules(xs, perFeature))
}

/** Every tree of depth at most two over the rules, made the slow way. */
function allTrees(c: number, rules: { features: readonly number[]; thresholds: readonly number[] }): BayesTree[] {
  const node = (i: number): BayesTree =>
    bayesSplit(rules.features[i] ?? 0, rules.thresholds[i] ?? 0, bayesLeaf, bayesLeaf, i)
  const options: BayesTree[] = [bayesLeaf, ...Array.from({ length: c }, (_, i) => node(i))]
  const trees: BayesTree[] = [bayesLeaf]
  for (let r = 0; r < c; r++) {
    for (const l of options) {
      for (const rt of options) {
        trees.push(bayesSplit(rules.features[r] ?? 0, rules.thresholds[r] ?? 0, l, rt, r))
      }
    }
  }
  return trees
}

describe('enumerateTrees', () => {
  it('weights every tree by its exact prior times evidence, summing to one', () => {
    const space = setup()
    const e = enumerateTrees(space)
    const c = space.rules.features.length
    const scores = allTrees(c, space.rules).map((t) => scoreTree(space, t).logPosterior)
    const finite = scores.filter((s) => s > Number.NEGATIVE_INFINITY)
    const logZ = Math.log(finite.reduce((a, s) => a + Math.exp(s), 0))
    expect(e.logEvidence).toBeCloseTo(logZ, 10)
    expect(e.trees).toBe(finite.length)
    const weights = toFlat(e.logWeights).map((w) => Math.exp(w - e.logEvidence))
    expect(weights.reduce((a, w) => a + w, e.stump)).toBeCloseTo(1, 12)
    expect(e.root.reduce((a, w) => a + w, e.stump)).toBeCloseTo(1, 12)
  })

  it('agrees with the score of each tree it lists', () => {
    const space = setup()
    const e = enumerateTrees(space)
    for (const { tree, probability } of topTrees(e, 6)) {
      expect(Math.exp(scoreTree(space, tree).logPosterior - e.logEvidence)).toBeCloseTo(probability, 12)
    }
    expect(e.splits.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
  })

  it('puts the root where the data step', () => {
    const space = setup()
    const e = enumerateTrees(space)
    const best = e.root.indexOf(Math.max(...e.root))
    expect([2.25, 3.25, 4.75].some((cut) => Math.abs((space.rules.thresholds[best] ?? 0) - cut) < 0.3)).toBe(true)
    expect(e.usage.reduce((a, b) => a + b, 0)).toBeGreaterThan(1)
  })

  it('gives the probability that a cut lies in a window, the weight of the trees that have one, as brute force does', () => {
    const space = setup()
    const e = enumerateTrees(space)
    const trees = allTrees(space.rules.features.length, space.rules)
    const weights = trees.map((t) => Math.exp(scoreTree(space, t).logPosterior - e.logEvidence))
    const cutsOf = (t: BayesTree): number[] =>
      t.kind === 'leaf' ? [] : [t.threshold, ...cutsOf(t.left), ...cutsOf(t.right)]
    for (const [lo, hi] of [
      [2.0, 2.6],
      [0, 6],
      [4.4, 5.0],
      [3.3, 3.3],
      [10, 11],
    ] as const) {
      const brute = trees.reduce(
        (a, t, i) => a + (cutsOf(t).some((c) => c >= lo && c <= hi) ? (weights[i] ?? 0) : 0),
        0,
      )
      expect(probabilityOfCutIn(e, lo, hi), `${lo} to ${hi}`).toBeCloseTo(brute, 10)
    }
    // the whole range is every tree but the stump; the sum of the rules' usage over it counts a tree once for each cut
    expect(probabilityOfCutIn(e, -1, 7)).toBeCloseTo(1 - e.stump, 10)
    expect(e.usage.reduce((a, b) => a + b, 0)).toBeGreaterThan(probabilityOfCutIn(e, -1, 7))
    // a window on a feature the data do not have holds no cut
    expect(probabilityOfCutIn(e, -1, 7, 3)).toBe(0)
  })

  it('predicts the weighted average of every tree, exactly', () => {
    const space = setup(2, 6)
    const e = enumerateTrees(space)
    const c = space.rules.features.length
    const grid = [[0.2], [1.7], [2.6], [3.1], [5.2]]
    const trees = allTrees(c, space.rules)
    const weights = trees.map((t) => Math.exp(scoreTree(space, t).logPosterior - e.logEvidence))
    const summaries = trees.map((t, i) => (weights[i] ? toFlat(treeSummary(space.data, t, grid)) : []))
    const p = space.data.model.outputs
    const avg = Array.from({ length: grid.length * p }, (_, k) =>
      summaries.reduce((a, s, i) => a + (weights[i] ?? 0) * (s[k] ?? 0), 0),
    )
    const exact = predictEnumerated(e, grid)
    grid.forEach((_, g) => {
      const mean = avg[g * p] ?? 0
      expect(exact.mean[g]).toBeCloseTo(mean, 10)
      const functionVariance = (avg[g * p + 1] ?? 0) - mean ** 2
      expect((exact.sd?.[g] ?? 0) ** 2).toBeCloseTo(functionVariance, 9)
    })
  })

  it('works for a classification leaf too', () => {
    const labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    const space = treeSpace(bayesData(xs, labels, dirichletLeaf(2)), treePrior({ maxDepth: 2 }), quantileRules(xs, 8))
    const e = enumerateTrees(space)
    const prediction = predictEnumerated(e, [[0.2], [5]])
    expect(prediction.mean[0]).toBeLessThan(0.3)
    expect(prediction.mean[1]).toBeGreaterThan(0.7)
    expect(prediction.probabilities?.[0]?.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
  })
})
