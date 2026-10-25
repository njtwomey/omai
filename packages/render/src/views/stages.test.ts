import { decisionSurface, inputRows, type PipelineStage, pipelineStages, type StageData } from 'mdml-engine/exposition'
import { describe, expect, it } from 'vitest'
import {
  pipelinePanelOf,
  pipelineProjectionView,
  pipelineScatterView,
  pipelineStepSummary,
  pipelineSteps,
  pipelineSurfaceView,
} from './stages.ts'

const data = (columns: string[]): StageData => ({
  columns,
  rows: [columns.map(() => 0), columns.map(() => 1)],
  n: 2,
  labels: [0, 1],
  stats: columns.map(() => ({ mean: 0, std: 1, min: -1, max: 1 })),
})

const stage = (name: string, d0: number, columns: string[], extra: Partial<PipelineStage> = {}): PipelineStage => ({
  name,
  kind: 'transformer',
  inputColumns: Array.from({ length: d0 }, (_, j) => `x${j}`),
  outputColumns: columns,
  inputSample: [],
  outputSample: [],
  inputShape: [2, d0],
  outputShape: [2, columns.length],
  learned: [],
  output: data(columns),
  ...extra,
})

const surface = decisionSurface({
  features: ['x0', 'x1'],
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
  classes: ['a', 'b'],
  rows: inputRows({
    points: [
      [0, 0],
      [1, 1],
    ],
    features: ['x0', 'x1'],
    labels: [0, 1],
  }),
})

const record = pipelineStages({
  input: data(['x0', 'x1']),
  classes: ['a', 'b'],
  stages: [
    stage('poly', 2, ['x0', 'x1', 'x0^2']),
    stage('scale', 3, ['x0', 'x1', 'x0^2']),
    stage('pca', 3, ['pca0', 'pca1'], { explainedVarianceRatio: [0.7, 0.2] }),
    stage('logistic', 2, ['p(a)', 'p(b)'], {
      kind: 'estimator',
      surface,
      loss: {
        name: 'log loss',
        mean: 0.3,
        values: [0.1, 1.2],
        misclassified: [1],
        expected: [
          [0.3, 0.6],
          [0.6, 0.3],
        ],
      },
    }),
  ],
})

describe('pipeline view', () => {
  const steps = pipelineSteps(record)

  it('lists the input and then each stage with the data it took', () => {
    expect(steps.map((s) => s.name)).toEqual(['input', 'poly', 'scale', 'pca', 'logistic'])
    expect(steps[2]?.before?.columns).toEqual(['x0', 'x1', 'x0^2'])
  })

  it('chooses each step’s panel by its output', () => {
    expect(steps.map(pipelinePanelOf)).toEqual(['scatter', 'table', 'table', 'projection', 'surface'])
    // Two columns without a projection's variance are a scatter; an estimator without a surface is a table.
    expect(pipelinePanelOf({ name: 's', kind: 'transformer', data: data(['a', 'b']) })).toBe('scatter')
    expect(pipelinePanelOf({ name: 'one', kind: 'transformer', data: data(['a']) })).toBe('table')
    const bare = stage('m', 2, ['p(a)', 'p(b)'], { kind: 'estimator' })
    expect(pipelinePanelOf({ name: 'm', kind: 'estimator', stage: bare, data: data(['p(a)', 'p(b)']) })).toBe('table')
  })

  it('draws the scatter, projection and both shadings of the surface', () => {
    const series = (o: { series?: unknown }): unknown[] => o.series as unknown[]
    expect(series(pipelineScatterView(steps[0] as (typeof steps)[0], { classes: ['a', 'b'] })).length).toBe(2)
    expect((pipelineProjectionView(steps[3] as (typeof steps)[0]).grid as unknown[]).length).toBe(2)
    const probability = pipelineSurfaceView(steps[4] as (typeof steps)[0])
    const loss = pipelineSurfaceView(steps[4] as (typeof steps)[0], { field: 'loss' })
    expect(JSON.stringify(probability.title)).toContain('1 of 2 misclassified')
    expect(JSON.stringify(loss.title)).toContain('expected log loss')
    expect(series(loss).length).toBeGreaterThan(0)
  })

  it('summarises each step for its card', () => {
    expect(steps.map((s) => pipelineStepSummary(s, record.classes).detail)).toEqual([
      '2 classes',
      '1 feature made',
      'every column: mean 0, std 1',
      'keeps 90.0% of the variance',
      'log loss 0.3',
    ])
    expect(pipelineStepSummary(steps[1] as (typeof steps)[0]).shape).toBe('2 × 2 → 2 × 3')
  })
})
