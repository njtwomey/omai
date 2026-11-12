import { describe, expect, it } from 'vitest'
import { key } from '../../random/index.ts'
import { runAlgorithm } from '../../trace/index.ts'
import { enumerateTrees, predictEnumerated } from './enumerate.ts'
import { normalGammaLeaf, regressionPrior } from './leaf.ts'
import { treePrior } from './prior.ts'
import { predictSampled, rulePosterior, sampleTrees, treeSampler } from './sampler.ts'
import { quantileRules, scoreTree, treeSpace } from './space.ts'
import { bayesData } from './tree.ts'

// A noisy staircase on [0, 4): jumps near 1.3 and 2.7.
const xs = Array.from({ length: 36 }, (_, i) => [(i + 0.5) / 9])
const noise = [0.1, -0.2, 0.15, -0.05, 0.2, -0.1, 0.05, -0.15, 0.12]
const y = xs.map(([x], i) => ((x ?? 0) < 1.3 ? 0 : (x ?? 0) < 2.7 ? 1.5 : 3) + (noise[i % 9] ?? 0))

function space(maxDepth = 2) {
  const model = normalGammaLeaf(regressionPrior(y))
  return treeSpace(bayesData(xs, y, model), treePrior({ maxDepth, minLeaf: 2 }), quantileRules(xs, 18))
}

describe('the structure sampler', () => {
  it('agrees with the exact enumeration on the root rule and the predictive curve', () => {
    const s = space()
    const exact = enumerateTrees(s)
    const draws = sampleTrees(s, { key: key(11), chains: 4, iterations: 20000, warmup: 1000, thin: 5 })
    const found = rulePosterior(draws)
    const tv =
      0.5 * exact.root.reduce((a, p, i) => a + Math.abs(p - (found.root[i] ?? 0)), Math.abs(exact.stump - found.stump))
    expect(tv).toBeLessThan(0.03)
    const grid = Array.from({ length: 24 }, (_, i) => [0.08 + i * 0.165])
    const a = predictEnumerated(exact, grid).mean
    const b = predictSampled(draws, grid).mean
    const worst = Math.max(...a.map((v, i) => Math.abs(v - (b[i] ?? 0))))
    expect(worst).toBeLessThan(0.03)
    expect(draws.diagnostics.logPosterior.rhat).toBeLessThan(1.1)
    expect(draws.acceptance).toBeGreaterThan(0.05)
  })

  it('keeps the draws inside the prior support and reports each kind of move', () => {
    const s = space(3)
    const draws = sampleTrees(s, { key: key(3), chains: 2, iterations: 600, warmup: 100, thin: 4 })
    expect(draws.draws.every((t) => scoreTree(s, t).logPosterior > Number.NEGATIVE_INFINITY)).toBe(true)
    for (const kind of ['grow', 'prune', 'change'] as const) expect(draws.moves[kind].proposed).toBeGreaterThan(0)
  })

  it('runs as a stepped algorithm and is reproducible from its key', () => {
    const s = space()
    const a = runAlgorithm(treeSampler(s), { maxSteps: 200, seed: 5 })
    const b = runAlgorithm(treeSampler(s), { maxSteps: 200, seed: 5 })
    expect(JSON.stringify(a.state.tree)).toBe(JSON.stringify(b.state.tree))
    expect(a.state.logPosterior).toBe(scoreTree(s, a.state.tree).logPosterior)
  })
})
