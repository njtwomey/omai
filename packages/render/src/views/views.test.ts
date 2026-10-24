import { describe, expect, it } from 'vitest'
import {
  clusteringSteps,
  crossValidationCurve,
  decisionSurface,
  embeddingSteps,
  learningCurve,
  mixtureSteps,
  pruningPath,
  regularisationPath,
} from 'whitebox-engine/exposition'
import { coefficientsAt, crossValidationView, regularisationPathView } from './paths.ts'
import { stillOf, viewFor } from './registry.ts'
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
