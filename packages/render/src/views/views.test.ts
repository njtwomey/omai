import { describe, expect, it } from 'vitest'
import {
  anomalyScores,
  type CoefficientTable,
  clusteringSteps,
  coefficientTable,
  componentsRecord,
  crossValidationCurve,
  decisionSurface,
  type EpisodeRecord,
  type Exposition,
  embeddingSteps,
  ensembleStages,
  episodeRecord,
  hyperparameterSurface,
  learningCurve,
  likelihoodTable,
  marginRecord,
  mixtureSteps,
  neighbourRecord,
  pipelineStages,
  posteriorGrid,
  pruningPath,
  regularisationPath,
  searchTrials,
  type TreeNode,
  treeRecord,
} from 'whitebox-engine/exposition'
import { componentsView } from './components.ts'
import { episodeView } from './episodes.ts'
import { coefficientsAt, crossValidationView, regularisationPathView } from './paths.ts'
import { stillOf, viewFor } from './registry.ts'
import { ensembleFrames } from './stages.ts'
import {
  clusteringFrames,
  cutClusters,
  dendrogramLayout,
  dendrogramView,
  embeddingFrames,
  mixtureAssignment,
  mixtureFrames,
} from './steps.ts'
import { decisionSurfaceView } from './surfaces.ts'
import { coefficientRows } from './tables.ts'
import { splitCandidatesView, splitText, treeNodeOf, treeScene } from './trees.ts'

type Opt = { series: { name?: string }[]; grid: unknown[]; xAxis: { type: string; inverse?: boolean }[] }
const o = (x: unknown): Opt => x as Opt

const path = regularisationPath({
  penalties: [10, 1, 0.1, 0.01],
  features: ['a', 'b'],
  coefficients: [
    [0, 0],
    [0.5, 0],
    [0.9, -0.2],
    [1, -0.3],
  ],
  chosen: 0.1,
})

describe('path views', () => {
  it('draws a regularisation path as coefficients over the active-set size', () => {
    const opt = o(regularisationPathView(path, { selected: 1 }))
    expect(opt.grid).toHaveLength(2)
    expect(opt.xAxis[0]?.type).toBe('log')
    expect(opt.xAxis[0]?.inverse).toBe(true)
    expect(opt.series.filter((s) => s.name === 'a' || s.name === 'b')).toHaveLength(4)
  })

  it('interpolates the coefficients at a penalty in log penalty', () => {
    expect(coefficientsAt(path, 1)).toEqual([0.5, 0])
    expect(coefficientsAt(path, Math.sqrt(10))[0]).toBeCloseTo(0.25, 12)
  })

  it('draws a cross-validation curve with its band, mean and marks', () => {
    const curve = crossValidationCurve({
      parameter: 'alpha',
      values: [1, 2, 3],
      meanScores: [0.7, 0.8, 0.75],
      standardErrors: [0.05, 0.04, 0.05],
      foldScores: [
        [0.6, 0.8],
        [0.8, 0.8],
        [0.7, 0.8],
      ],
      higherIsBetter: true,
      best: 2,
    })
    const names = o(crossValidationView(curve)).series.map((s) => s.name)
    expect(names.filter((n) => n?.startsWith('__fold'))).toHaveLength(2)
    expect(names).toContain('mean')
    expect(names).toContain('best')
  })
})

describe('the view registry', () => {
  it('names a component and draws a still for the kinds it knows', () => {
    expect(viewFor('regularisationPath')?.component).toBe('PathView')
    const lc = learningCurve({ sizes: [10, 20], train: [1, 0.9], validation: [0.6, 0.8] })
    expect(stillOf(lc)).toBeDefined()
    expect(stillOf(pruningPath({ alphas: [0, 0.1], impurities: [0, 0.2], leaves: [4, 1] }))).toBeDefined()
  })
})

describe('surface and step views', () => {
  const rows = [
    [0, 0],
    [0, 1],
    [3, 3],
    [3, 4],
  ]

  it('draws decision regions with the training rows by class', () => {
    const s = decisionSurface({
      features: ['a', 'b'],
      x: [0, 1],
      y: [0, 1],
      probabilities: [
        [
          [0.9, 0.4],
          [0.6, 0.1],
        ],
        [
          [0.1, 0.6],
          [0.4, 0.9],
        ],
      ],
      classes: ['no', 'yes'],
    })
    const names = o(decisionSurfaceView(s, { points: rows, labels: ['no', 'no', 'yes', 'yes'] })).series.map(
      (x) => x.name,
    )
    expect(names).toContain('no')
    expect(names).toContain('yes')
  })

  it('plays clustering steps with centres and the objective', () => {
    const steps = clusteringSteps({
      method: 'kmeans',
      assignments: [
        [0, 0, 0, 1],
        [0, 0, 1, 1],
      ],
      centres: [
        [
          [1, 1],
          [3, 4],
        ],
        [
          [0, 0.5],
          [3, 3.5],
        ],
      ],
      objective: [10, 1],
    })
    const frames = clusteringFrames(steps, { points: rows })
    expect(frames.count).toBe(2)
    const opt = o(frames.option(1))
    expect(opt.grid).toHaveLength(2)
    expect(opt.series.filter((x) => x.name?.startsWith('centre'))).toHaveLength(2)
  })

  it('lays out and cuts a dendrogram', () => {
    const merges: [number, number, number, number][] = [
      [0, 1, 1, 2],
      [2, 3, 1, 2],
      [4, 5, 5, 4],
    ]
    expect(dendrogramLayout(merges, 4).order).toEqual([0, 1, 2, 3])
    expect(cutClusters(merges, 4, 2)).toEqual([0, 0, 1, 1])
    const steps = clusteringSteps({ method: 'agglomerative', assignments: [[0, 0, 1, 1]], merges })
    expect(o(dendrogramView(steps, { points: rows })).grid).toHaveLength(2)
  })

  it('plays mixture steps with ellipses, objective and weights', () => {
    const steps = mixtureSteps({
      family: 'gaussian-full',
      weights: [[0.5, 0.5]],
      means: [
        [
          [0, 0.5],
          [3, 3.5],
        ],
      ],
      covariances: [
        [
          [1, 0, 0, 1],
          [1, 0, 0, 1],
        ],
      ],
      objective: [-3],
    })
    expect(mixtureAssignment(steps, 0, rows)).toEqual([0, 0, 1, 1])
    expect(o(mixtureFrames(steps, { points: rows }).option(0)).grid).toHaveLength(3)
  })

  it('plays an embedding coloured by label', () => {
    const steps = embeddingSteps({
      method: 'tsne',
      positions: [rows, rows],
      iterations: [0, 50],
      objective: [2, 1],
    })
    const opt = o(embeddingFrames(steps, { labels: ['a', 'a', 'b', 'b'] }).option(1))
    expect(opt.series.map((x) => x.name)).toContain('b')
  })
})

describe('tree, ensemble and component views', () => {
  const tree = treeRecord({
    criterion: 'gini',
    features: ['petal length (cm)'],
    classes: ['a', 'b'],
    nodes: [
      {
        id: 0,
        parent: null,
        depth: 0,
        rows: [0, 1, 2],
        value: [2, 1],
        impurity: 0.44,
        candidates: [
          { feature: 0, threshold: 0.5, impurityLeft: 0, impurityRight: 0, decrease: 0.44 },
          { feature: 0, threshold: 1.5, impurityLeft: 0, impurityRight: 0.5, decrease: 0.11 },
        ],
        split: { feature: 0, threshold: 0.5, left: 1, right: 2 },
        order: 0,
      },
      { id: 1, parent: 0, depth: 1, rows: [0, 1], value: [2, 0], impurity: 0, candidates: [], order: 1 },
      { id: 2, parent: 0, depth: 1, rows: [2], value: [0, 1], impurity: 0, candidates: [], order: 2 },
    ],
  })

  it('draws a tree as cards with hit-testable ids, and grows it', () => {
    const ids = (s: ReturnType<typeof treeScene>): string[] =>
      s.children.flatMap((c) => ('id' in c && typeof c.id === 'string' ? [c.id] : []))
    expect(ids(treeScene(tree))).toEqual(['node:0', 'node:1', 'node:2'])
    expect(ids(treeScene(tree, { grownTo: 0 }))).toEqual(['node:0'])
    expect(splitText(tree, tree.nodes[0] as TreeNode)).toBe('petal length ≤ 0.5')
    expect(treeNodeOf('node:2')).toBe(2)
  })

  it('charts a node’s candidate splits with the chosen one', () => {
    const names = o(splitCandidatesView(tree, 0)).series.map((s) => s.name)
    expect(names).toContain('petal length (cm)')
    expect(names).toContain('chosen')
  })

  it('plays ensemble stages over the training rows', () => {
    const stages = ensembleStages({
      method: 'adaboost',
      weights: [0.8, 0.4],
      trainScores: [0.7, 0.9],
      targets: [
        [0.25, 0.25, 0.25, 0.25],
        [0.1, 0.1, 0.4, 0.4],
      ],
    })
    const frames = ensembleFrames(stages, {
      points: [
        [0, 0],
        [1, 1],
        [2, 2],
        [3, 3],
      ],
      labels: [0, 0, 1, 1],
    })
    expect(frames.count).toBe(2)
    expect(o(frames.option(1)).grid).toHaveLength(3)
  })

  it('draws a decomposition as biplot, scree and loadings', () => {
    const record = componentsRecord({
      method: 'pca',
      features: ['x', 'y'],
      components: [
        [0.6, 0.8],
        [-0.8, 0.6],
      ],
      explainedVarianceRatio: [0.9, 0.1],
      scores: [
        [1, 0],
        [-1, 0.1],
      ],
    })
    expect(o(componentsView(record)).grid).toHaveLength(3)
  })
})

describe('posterior, search, table, margin, anomaly, pipeline and episode views', () => {
  it('draws every kind through the registry', () => {
    const records: Exposition[] = [
      posteriorGrid({
        inputs: [[0], [1], [2]],
        mean: [0, 1, 0],
        variance: [1, 0.1, 1],
        observed: { inputs: [[1]], targets: [1] },
      }),
      hyperparameterSurface({
        parameters: ['a', 'b'],
        x: [0.1, 1, 10],
        y: [0.1, 1],
        values: [
          [-3, -1, -2],
          [-4, -2, -3],
        ],
        path: [
          [0.1, 0.1],
          [1, 0.1],
        ],
        logScale: [true, true],
      }),
      anomalyScores({
        method: 'iforest',
        scores: [0.4, 0.5, 0.9],
        threshold: 0.6,
        evidence: { name: 'path', values: [9, 8, 3] },
      }),
      marginRecord({
        support: [0, 2],
        dualCoefficients: [0.5, -0.5],
        intercept: 0,
        margins: [1, 2, 1],
        dualObjective: [0, -1],
        workingSets: [
          [0, 2],
          [0, 2],
        ],
      }),
      coefficientTable({ names: ['const', 'x'], estimates: [5, 2], standardErrors: [0.1, 0.5], pValues: [0, 0.01] }),
      likelihoodTable({ classes: ['a', 'b'], features: ['x'], logPriors: [-0.7, -0.7], contributions: [[-1], [-2]] }),
      neighbourRecord({ indices: [[1, 2]], distances: [[0.1, 0.4]], metric: 'euclidean' }),
      searchTrials({
        method: 'grid',
        parameters: ['alpha'],
        trials: [
          { params: { alpha: 1 }, value: 0.8, state: 'complete' },
          { params: { alpha: 10 }, value: 0.7, state: 'pruned', intermediate: [0.6, 0.7] },
        ],
        higherIsBetter: true,
        best: 0,
      }),
      pipelineStages({
        stages: [
          {
            name: 'scale',
            kind: 'transformer',
            inputColumns: ['x'],
            outputColumns: ['x'],
            inputSample: [[1]],
            outputSample: [[0]],
            inputShape: [10, 1],
            outputShape: [10, 1],
            learned: [{ name: 'mean', value: [1] }],
          },
        ],
      }),
      episodeRecord({
        environment: 'FrozenLake',
        agent: 'q',
        returns: [0, 1, 1, 1],
        lengths: [5, 8, 6, 6],
        snapshots: [{ episode: 3, values: [0, 0.5, 0.2, 1], policy: [1, 2, 1, 0] }],
      }),
    ]
    for (const r of records) expect(stillOf(r), r.kind).toBeDefined()
    expect(coefficientRows(records[4] as CoefficientTable).rows[0]?.[4]).toBe('<1e-16')
    const grid = o(episodeView(records[9] as EpisodeRecord, { gridShape: [2, 2] }))
    expect(grid.series.some((s) => s.name === '__policy')).toBe(true)
  })
})
