import { describe, expect, it } from 'vitest'
import * as E from './index.ts'

describe('exposition records', () => {
  it('reject mismatched shapes with a shape error naming the field', () => {
    expect(() => E.regularisationPath({ penalties: [1, 2], features: ['a'], coefficients: [[0]] })).toThrow(
      /coefficients/,
    )
    expect(() => E.regularisationPath({ penalties: [1], features: ['a', 'b'], coefficients: [[0]] })).toThrow(
      /coefficients\[0\]/,
    )
    expect(() => E.decisionSurface({ features: ['a', 'b'], x: [0, 1], y: [0], labels: [[0]] })).toThrow(/labels\[0\]/)
    expect(() =>
      E.decisionSurface({ features: ['a', 'b'], x: [0, 1], y: [0], steps: [{ iteration: 0, decision: [[0]] }] }),
    ).toThrow(/steps\[0\]\.decision/)
    expect(() =>
      E.decisionSurface({
        features: ['a', 'b'],
        x: [0],
        y: [0],
        rows: { points: [[0, 0]], features: ['a', 'b'] },
        steps: [{ iteration: 0, rowValues: [1, 2] }],
      }),
    ).toThrow(/rowValues/)
    expect(() => E.inputRows({ points: [[0, 0, 0]], features: ['a', 'b'] })).toThrow(/points/)
    expect(() =>
      E.clusteringSteps({ method: 'cop', assignments: [[0, 1]], constraints: { mustLink: [[0, 2]], cannotLink: [] } }),
    ).toThrow(/constraints\.mustLink\[0\]/)
    expect(() =>
      E.clusteringSteps({
        method: 'cop',
        assignments: [[0, 1]],
        constraints: { mustLink: [], cannotLink: [[0, 1]], brokenCannotLink: [1] },
      }),
    ).toThrow(/brokenCannotLink/)
    expect(() =>
      E.marginRecord({ support: [], dualCoefficients: [], intercept: 0, margins: [], dualObjective: [1], steps: [] }),
    ).toThrow(/steps/)
    expect(() => E.mixtureSteps({ family: 'gaussian', weights: [], means: [], objective: [1] })).toThrow(/weights/)
    expect(() =>
      E.episodeRecord({
        environment: 'e',
        agent: 'a',
        returns: [1],
        lengths: [1],
        recorded: [{ episode: 0, states: [0], actions: [0], rewards: [1] }],
      }),
    ).toThrow(/states/)
  })

  it('check the shapes of curves, densities, state sequences, value grids and confusion matrices', () => {
    expect(() => E.curveSet({ x: [1, 2], xLabel: 'epoch', series: [{ name: 'a', y: [1] }] })).toThrow(/a\.y/)
    expect(() => E.curveSet({ xLabel: 'epoch', series: [{ name: 'a', y: [1] }] })).toThrow(/no x values/)
    expect(() => E.curveSet({ x: [1, 2], xLabel: 'epoch', series: [{ name: 'a', y: [1, 2], spread: [0] }] })).toThrow(
      /a\.spread/,
    )
    expect(() =>
      E.densityCurves({ parameter: 'p', x: [0, 1], curves: [], steps: [{ observed: 0, density: [1] }] }),
    ).toThrow(/steps\[0\]/)
    expect(() => E.stateSequence({ method: 'hmm', states: ['a'], posteriors: [[1]], decoded: [1] })).toThrow(/decoded/)
    expect(() => E.stateSequence({ method: 'hmm', states: ['a', 'b'], posteriors: [[1]] })).toThrow(/posteriors/)
    expect(() =>
      E.valueGrids({
        grids: [{ name: 'g', values: [[1, 2]] }],
        steps: [{ iteration: 0, grids: [[[1, 2, 3]]] }],
      }),
    ).toThrow(/steps\[0\]\.grids\[0\]/)
    expect(() => E.confusionMatrix({ classes: ['a', 'b'], counts: [[1, 0]] })).toThrow(/counts/)
    expect(E.confusionCounts([0, 1, 1], [1, 1, 0], 2)).toEqual([
      [0, 1],
      [1, 1],
    ])
    expect(() => E.confusionCounts([0, 2], [0, 0], 2)).toThrow(/row 1/)
    expect(() => E.learningCurve({ sizes: [1, 2], train: [1, 2], validation: [1] })).toThrow(/validation/)
    expect(() =>
      E.clusteringSteps({ method: 'self-training', assignments: [[0, -1]], marked: { initial: [2] } }),
    ).toThrow(/marked/)
    expect(() =>
      E.clusteringSteps({ method: 'self-training', assignments: [[0, -1]], marked: { added: [[], [1]] } }),
    ).toThrow(/marked\.added/)
  })

  it('reject node lists that are not trees', () => {
    const node = { id: 0, parent: null, depth: 0, rows: [], value: [1], impurity: 0, candidates: [], order: 0 }
    expect(() => E.treeRecord({ criterion: 'gini', features: [], nodes: [node, { ...node, id: 1 }] })).toThrow(
      /one root/,
    )
    expect(() =>
      E.treeRecord({
        criterion: 'gini',
        features: [],
        nodes: [{ ...node, split: { feature: 0, threshold: 0, left: 1, right: 2 } }],
      }),
    ).toThrow(/unknown children/)
  })

  it('reject pipelines whose stages do not chain', () => {
    const stage = (name: string, inCols: number, outCols: number): E.PipelineStage => ({
      name,
      kind: 'transformer',
      inputColumns: [],
      outputColumns: [],
      inputSample: [],
      outputSample: [],
      inputShape: [10, inCols],
      outputShape: [10, outCols],
      learned: [],
    })
    expect(E.pipelineStages({ stages: [stage('scale', 3, 3), stage('pca', 3, 2)] }).stages.length).toBe(2)
    expect(() => E.pipelineStages({ stages: [stage('scale', 3, 3), stage('pca', 2, 2)] })).toThrow(/scale outputs 3/)
  })

  it('round-trip through JSON and are recognised by kind', () => {
    const records: E.Exposition[] = [
      E.regularisationPath({ penalties: [1, 0.1], features: ['a'], coefficients: [[0], [0.5]], chosen: 0.1 }),
      E.clusteringSteps({ method: 'kmeans', assignments: [[0, 1]], centres: [[[0], [1]]], objective: [0] }),
      E.learningCurve({ sizes: [1], train: [1], validation: [0.5] }),
    ]
    for (const r of records) {
      expect(JSON.parse(JSON.stringify(r))).toEqual(r)
      expect(E.isExposition(r)).toBe(true)
    }
    expect(E.recordsOf(records, 'clusteringSteps').length).toBe(1)
    expect(E.isExposition({ kind: 'other' })).toBe(false)
    expect(E.isExposition(null)).toBe(false)
  })
})
