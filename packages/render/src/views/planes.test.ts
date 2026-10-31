import { complexPlane, pointsSteps, reachabilityRecord, sceneSteps, simplexRecord } from 'mdml-engine/exposition'
import { circle, scene } from 'mdml-engine/scene'
import { describe, expect, it } from 'vitest'
import { complexPlaneLayers, complexPlaneLimits, complexPlaneView } from './complex-plane.ts'
import { pointsStepsLimits, pointsStepsScene, pointsStepsScene3d, pointsStepsView } from './points-steps.ts'
import { reachabilityClusters, reachabilityPlot } from './reachability.ts'
import { stillOf, viewFor } from './registry.ts'
import { sceneStepsScene } from './scene-steps.ts'
import { simplexLayers, simplexPoint, simplexView } from './simplex.ts'

type Series = { name?: string; type?: string; data?: unknown[]; symbol?: string }
const seriesOf = (o: unknown): Series[] => (o as { series: Series[] }).series
const names = (o: unknown): (string | undefined)[] => seriesOf(o).map((s) => s.name)

describe('the points view', () => {
  const grow = pointsSteps({
    dims: 2,
    features: ['re', 'im'],
    steps: [
      { step: 0, points: [[0, 0]], groups: [0] },
      {
        step: 4,
        points: [
          [4, 2],
          [2, -1],
        ],
        groups: [1, 1],
      },
    ],
  })

  it('takes its axes from every step, so they hold still while points arrive', () => {
    expect(pointsStepsLimits(grow)).toEqual([
      [0, 4],
      [-1, 2],
    ])
    const early = pointsStepsView(grow, 1) as { xAxis: { min?: number; max?: number }[] }
    const late = pointsStepsView(grow, 9) as { xAxis: { min?: number; max?: number }[] }
    expect([early.xAxis[0]?.min, early.xAxis[0]?.max]).toEqual([late.xAxis[0]?.min, late.xAxis[0]?.max])
  })

  it('draws the older points and, apart, those the latest step added', () => {
    expect(names(pointsStepsView(grow, 1))).toEqual(['points'])
    expect(names(pointsStepsView(grow, 9)).at(-1)).toBe('latest')
    const moving = pointsSteps({ ...grow, accumulate: false })
    expect(names(pointsStepsView(moving, 9))).not.toContain('latest')
  })

  it('scales a three-dimensional cloud into the unit box whatever the step', () => {
    const r = pointsSteps({
      dims: 3,
      steps: [
        { step: 0, points: [[0, 0, 0]] },
        { step: 1, points: [[10, 20, 30]] },
      ],
    })
    const points = (step: number) =>
      pointsStepsScene3d(r, step).items.filter((i) => i.kind === 'point') as { at: readonly number[] }[]
    expect(points(0)[0]?.at).toEqual([-1, -1, -1])
    expect(points(1)).toHaveLength(2)
    expect(points(1)[1]?.at).toEqual([1, 1, 1])
    expect(pointsStepsScene(r, 1).kind).toBe('scene')
  })
})

describe('the complex plane view', () => {
  const plane = complexPlane({
    domain: 's',
    poles: [
      [-1, 2],
      [-1, -2],
      [0.5, 0],
    ],
    zeros: [[-3, 0]],
    labels: ['p1', 'p2', 'p3'],
    locus: [
      {
        name: 'branch',
        path: [
          [-1, 0],
          [-0.5, 1],
        ],
      },
    ],
  })

  it('draws the boundary, the locus, the zeros and the poles apart by stability', () => {
    const layers = complexPlaneLayers(plane)
    expect(layers.map((l) => l.series[0]?.name)).toEqual(['imaginary axis', 'branch', 'zeros', 'poles (stable)'])
    expect(layers[3]?.series.map((s) => (s as Series).name)).toEqual(['poles (stable)', 'poles (unstable)'])
    const stable = layers[3]?.series[0] as { data: { value: number[] }[] }
    expect(stable.data).toHaveLength(2)
  })

  it('draws the unit circle for a discrete-time system and keeps both axes to the same range', () => {
    const z = complexPlane({ domain: 'z', poles: [[0.5, 0.5]] })
    expect(complexPlaneLayers(z)[0]?.series[0]?.name).toBe('unit circle')
    const limits = complexPlaneLimits(z)
    expect(limits.real).toEqual(limits.imaginary)
    expect(limits.real[1]).toBeGreaterThanOrEqual(1)
    const option = complexPlaneView(plane) as { xAxis: { name?: string }[]; yAxis: { name?: string }[] }
    expect([option.xAxis[0]?.name, option.yAxis[0]?.name]).toEqual(['real', 'imaginary'])
  })

  it('makes room for every pole and the locus', () => {
    const far = complexPlane({
      domain: 's',
      poles: [[-30, 0]],
      locus: [
        {
          path: [
            [0, 12],
            [-3, 0],
          ],
        },
      ],
    })
    const { real, imaginary } = complexPlaneLimits(far)
    expect(real[0]).toBeLessThan(-30)
    expect(imaginary[1]).toBeGreaterThan(12)
  })
})

describe('the simplex view', () => {
  it('places the corners of the triangle and the square', () => {
    expect(simplexPoint('triangle', [1, 0, 0])).toEqual([0, 0])
    expect(simplexPoint('triangle', [0, 1, 0])).toEqual([1, 0])
    const top = simplexPoint('triangle', [0, 0, 1])
    expect(top[0]).toBeCloseTo(0.5, 12)
    expect(top[1]).toBeCloseTo(Math.sqrt(3) / 2, 12)
    const centre = simplexPoint('triangle', [1 / 3, 1 / 3, 1 / 3])
    expect(centre[0]).toBeCloseTo(0.5, 12)
    expect(simplexPoint('square', [0.2, 0.9])).toEqual([0.2, 0.9])
  })

  it('draws outline, names, field, paths with their heads and the fixed points by class', () => {
    const r = simplexRecord({
      shape: 'triangle',
      labels: ['rock', 'paper', 'scissors'],
      paths: [
        {
          name: 'run',
          points: [
            [0.6, 0.2, 0.2],
            [0.4, 0.3, 0.3],
          ],
        },
      ],
      field: {
        at: [
          [0.4, 0.3, 0.3],
          [0.2, 0.2, 0.6],
        ],
        by: [
          [-0.1, 0.05, 0.05],
          [0, 0, 0],
        ],
      },
      fixedPoints: [{ at: [1 / 3, 1 / 3, 1 / 3], stability: 'centre', label: 'mixed' }],
    })
    const layers = simplexLayers(r)
    expect(layers.map((l) => l.series[0]?.name)).toEqual([
      '__outline',
      '__names',
      '__arrows',
      'run',
      '__head0',
      'centre',
    ])
    // The still field has an arrow only where the state changes.
    const field = layers[2]?.series[0] as { data: unknown[] } | undefined
    expect(field?.data).toHaveLength(1)
    const names = layers[1]?.series[0] as { data: { name?: string; value: unknown[] }[] }
    expect(names.data).toHaveLength(3)
    expect(seriesOf(simplexView(r)).length).toBe(6)
  })

  it('draws a square with the two axes named', () => {
    const r = simplexRecord({
      shape: 'square',
      labels: ['row heads', 'column heads'],
      paths: [{ name: 'p', points: [[0.1, 0.9]] }],
    })
    expect(simplexLayers(r)[0]?.series[0]?.data).toHaveLength(5)
  })
})

describe('the reachability view', () => {
  const r = reachabilityRecord({ reach: [null, 0.1, 0.1, 0.9, 0.1, 0.1, 2], threshold: 0.5 })

  it('cuts valleys at the threshold and calls the rest noise', () => {
    expect(reachabilityClusters(r)).toEqual([0, 0, 0, 1, 1, 1, -1])
    expect(reachabilityClusters({ ...r, labels: [1, 1, 1, 1, 1, 1, 1] })).toEqual([1, 1, 1, 1, 1, 1, 1])
    expect(reachabilityClusters({ ...r, threshold: 5 })).toEqual([0, 0, 0, 0, 0, 0, 0])
  })

  it('draws a stacked bar per cluster, an undefined distance as a full bar, and the cut', () => {
    const option = reachabilityPlot(r)
    expect(names(option)).toEqual(['noise', 'cluster 1', 'cluster 2', '__reference'])
    const first = seriesOf(option)[1] as { data: number[] }
    // The undefined first distance is the tallest bar's height (2) plus 8%.
    expect(first.data[0]).toBeCloseTo(2.16, 12)
    expect(first.data[1]).toBeCloseTo(0.1, 12)
    expect(seriesOf(option)).toHaveLength(4)
  })
})

describe('the scene view', () => {
  const a = scene(100, 50, [circle(10, 10, 5)])
  const r = sceneSteps({
    steps: [
      { step: 0, scene: a },
      { step: 5, scene: scene(100, 50, [circle(20, 20, 5)]), caption: 'later' },
    ],
  })

  it('shows the latest scene at or before the step, with its caption added', () => {
    expect(sceneStepsScene(r, 2)).toBe(a)
    const late = sceneStepsScene(r, 6)
    expect(late.children.map((c) => c.kind)).toEqual(['circle', 'text'])
    expect(sceneStepsScene(r).children).toHaveLength(2)
    expect(sceneStepsScene(sceneSteps({ steps: [] })).width).toBe(1)
  })
})

describe('the registry of the second set of kinds', () => {
  it('names a component for each and draws a headless still', () => {
    expect(viewFor('pointsSteps').component).toBe('PointsStepsView')
    expect(viewFor('complexPlane').component).toBe('ComplexPlaneView')
    expect(viewFor('simplex').component).toBe('SimplexView')
    expect(viewFor('reachability').component).toBe('ReachabilityPlotView')
    expect(viewFor('sceneSteps').component).toBe('SceneStepsView')
    expect(stillOf(pointsSteps({ dims: 2, steps: [{ step: 0, points: [[0, 1]] }] }))).toHaveProperty('series')
    expect(stillOf(pointsSteps({ dims: 3, steps: [{ step: 0, points: [[0, 1, 2]] }] }))).toHaveProperty('children')
    expect(stillOf(complexPlane({ domain: 's', poles: [[-1, 0]] }))).toHaveProperty('series')
    expect(stillOf(simplexRecord({ shape: 'square', labels: ['a', 'b'] }))).toHaveProperty('series')
    expect(stillOf(reachabilityRecord({ reach: [null, 1] }))).toHaveProperty('series')
    expect(stillOf(sceneSteps({ steps: [] }))).toHaveProperty('children')
  })
})
