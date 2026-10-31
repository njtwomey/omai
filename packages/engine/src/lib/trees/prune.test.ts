import { describe, expect, it } from 'vitest'
import { iris } from '../datasets/index.ts'
import { key } from '../random/index.ts'
import { dict } from '../space/index.ts'
import { toFlat } from '../tensor/index.ts'
import { defineMethod, event, Log, messages, metricSeries, runAlgorithm, scalar, session } from '../trace/index.ts'
import { decisionTreeClassifier } from './decision-tree.ts'
import {
  collapseTree,
  costComplexityPath,
  pruneTree,
  reducedErrorAlgorithm,
  reducedErrorPrune,
  weakestLinkAlgorithm,
} from './prune.ts'
import { leafCount, predictTree, treeDepth } from './tree.ts'

const flowers = iris()
const full = decisionTreeClassifier().fit(flowers).params

describe('cost-complexity pruning', () => {
  it('walks the weakest-link path from the full tree to the root', () => {
    const path = costComplexityPath(full)
    expect(path.alphas[0]).toBe(0)
    expect(path.leaves[0]).toBe(leafCount(full))
    expect(path.leaves.at(-1)).toBe(1)
    for (let i = 1; i < path.alphas.length; i++) {
      expect(path.alphas[i] ?? 0).toBeGreaterThanOrEqual(path.alphas[i - 1] ?? 0)
      expect(path.impurities[i] ?? 0).toBeGreaterThanOrEqual(path.impurities[i - 1] ?? 0)
      expect(path.leaves[i] ?? 0).toBeLessThan(path.leaves[i - 1] ?? 0)
    }
  })

  it('pruneTree stops at alpha: the larger the alpha the smaller the tree, the root at the end', () => {
    const sizes = [0, 0.005, 0.02, 0.1, 1].map((a) => leafCount(pruneTree(full, a)))
    expect(sizes[0]).toBe(leafCount(full))
    for (let i = 1; i < sizes.length; i++) expect(sizes[i] ?? 0).toBeLessThanOrEqual(sizes[i - 1] ?? 0)
    expect(sizes.at(-1)).toBe(1)
  })

  it('is an algorithm that serialises and takes one link per step', () => {
    const alg = weakestLinkAlgorithm(full)
    expect(alg.serialise).toBeDefined()
    const run = runAlgorithm(alg, { maxSteps: 100, seed: 0 })
    expect(run.status).toBe('converged')
    expect(run.steps).toBe(run.state.alphas.length - 1)
    const back = alg.deserialise?.(JSON.parse(JSON.stringify(alg.serialise?.(run.state))))
    expect(back?.alphas).toEqual(run.state.alphas)
  })

  it('logs prune/alpha, prune/leaves and the prune/pruned event when run in a session', () => {
    const method = defineMethod({
      name: 'prune',
      config: dict({}),
      make: () => weakestLinkAlgorithm(full),
      contract: {
        'prune/alpha': scalar(),
        'prune/leaves': scalar(),
        'prune/trainError': scalar(),
        'prune/pruned': event(),
      },
    })
    const s = session(method, {
      key: key(0),
      mode: 'eager',
      steps: 100,
      logs: { 'prune/alpha': Log(), 'prune/leaves': Log(), 'prune/pruned': Log() },
    })
    const alphas = metricSeries(s.run(), 'prune/alpha').values
    expect(alphas.length).toBeGreaterThan(0)
    expect(alphas.length).toBeLessThanOrEqual(leafCount(full) - 1)
    expect(messages(s.run(), 'prune/pruned').length).toBe(alphas.length)
    expect(metricSeries(s.run(), 'prune/leaves').values.at(-1)).toBe(1)
    expect(s.stateAt(2).alphas).toHaveLength(3)
  })
})

describe('reduced-error pruning', () => {
  it('prunes subtrees that do not help on held-out rows and keeps those that do', () => {
    // A deep tree on noisy training data, pruned with a clean held-out set from the same rule.
    const x = Array.from({ length: 200 }, (_, i) => [(i * 37) % 100, (i * 11) % 13])
    const rule = (v: number[]): number => ((v[0] ?? 0) < 50 ? 0 : 1)
    const noisy = x.map((v, i) => (i % 9 === 0 ? 1 - rule(v) : rule(v)))
    const tree = decisionTreeClassifier().fit({ x: x.slice(0, 120), y: noisy.slice(0, 120) }).params
    const held = { x: x.slice(120), y: x.slice(120).map(rule) }
    const pruned = reducedErrorPrune(tree, held)
    expect(leafCount(pruned)).toBeLessThan(leafCount(tree))
    expect(treeDepth(pruned)).toBeLessThanOrEqual(treeDepth(tree))
    const accuracy = (t: typeof tree): number => {
      const p = toFlat(predictTree(t, held.x))
      return p.filter((v, i) => v === held.y[i]).length / p.length
    }
    expect(accuracy(pruned)).toBeGreaterThanOrEqual(accuracy(tree))
    // The algorithm visits every node once, bottom-up.
    const r = runAlgorithm(reducedErrorAlgorithm(tree, held), { maxSteps: 1000 })
    expect(r.steps).toBe(tree.feature.length)
  })

  it('collapsing no node returns the same tree, renumbered in preorder', () => {
    const none = collapseTree(full, new Array<number>(full.feature.length).fill(0))
    expect(none.feature).toEqual(full.feature)
    expect(none.left).toEqual(full.left)
  })
})
