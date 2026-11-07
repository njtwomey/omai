import { describe, expect, it } from 'vitest'
import { toFlat } from '../tensor/index.ts'
import type { FittedTree } from './tree.ts'
import {
  applyTree,
  decisionPath,
  exportText,
  leafCount,
  nodeRegions,
  predictTree,
  stumpTree,
  treeDepth,
  treeFeatureImportances,
  treeModel,
  treeOutputs,
  treePredict,
  treeToStructure,
} from './tree.ts'

// x0 <= 1 -> leaf 1; else x1 in {2, 5} -> leaf 3 else leaf 4; missing x1 goes left.
const tree: FittedTree = {
  task: 'classification',
  classes: [10, 20],
  nFeatures: 2,
  feature: [0, -2, 1, -2, -2],
  threshold: [1, -2, -2, -2, -2],
  left: [1, -1, 3, -1, -1],
  right: [2, -1, 4, -1, -1],
  value: [
    [4, 6],
    [4, 0],
    [0, 6],
    [0, 3],
    [0, 3],
  ],
  impurity: [0.48, 0, 0, 0, 0],
  nSamples: [10, 4, 6, 3, 3],
  weightedNSamples: [10, 4, 6, 3, 3],
  missingLeft: [false, false, true, false, false],
  categories: [[], [], [2, 5], [], []],
}

describe('tree arrays', () => {
  it('routes numeric splits, categorical subsets and missing values', () => {
    expect(
      Array.from(
        applyTree(tree, [
          [0, 9],
          [3, 2],
          [3, 7],
          [3, Number.NaN],
          [Number.NaN, 0],
        ]),
      ),
    ).toEqual([1, 3, 4, 3, 4])
    expect(decisionPath(tree, [[3, 5]])).toEqual([[0, 2, 3]])
  })

  it('predicts labels and probabilities from leaf counts', () => {
    expect(
      toFlat(
        predictTree(tree, [
          [0, 0],
          [3, 2],
        ]),
      ),
    ).toEqual([10, 20])
    expect(toFlat(treeOutputs(tree, [[0, 0]]))).toEqual([1, 0])
    expect(treeDepth(tree)).toBe(2)
    expect(leafCount(tree)).toBe(3)
  })

  it('gives each node its box, importances that sum to one, a structure and text', () => {
    const r = nodeRegions(tree)
    expect(r[1]?.upper).toEqual([1, Number.POSITIVE_INFINITY])
    expect(r[2]?.lower).toEqual([1, Number.NEGATIVE_INFINITY])
    expect(treeFeatureImportances({ ...tree, impurity: [0.48, 0, 0.3, 0, 0] }).reduce((a, b) => a + b, 0)).toBeCloseTo(
      1,
      12,
    )
    const s = treeToStructure(tree, ['a', 'b'], ['no', 'yes'])
    expect(s.nodes.map((n) => n.label)).toEqual(['a <= 1', 'no', 'b in {2, 5}', 'yes', 'yes'])
    expect(exportText(tree, ['a', 'b'])).toContain('b in {2, 5}')
  })

  it('converts to the scalar-valued model that tree SHAP reads, refusing categorical splits', () => {
    expect(() => treeModel(tree)).toThrow(/categorical/)
    const numeric = stumpTree({ feature: 0, threshold: 0.5, left: 2, right: 4 })
    const model = treeModel(numeric)
    expect(model.value).toEqual([3, 2, 4])
    expect(toFlat(treePredict(model, [[0], [1]]))).toEqual([2, 4])
    expect(toFlat(treePredict(treeModel({ ...numeric, task: 'regression' }), [[1]]))).toEqual([4])
  })
})
