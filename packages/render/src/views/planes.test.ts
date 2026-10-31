import {
  complexPlane,
  imageSteps,
  pointsSteps,
  reachabilityRecord,
  sceneSteps,
  simplexRecord,
} from 'mdml-engine/exposition'
import { circle, scene } from 'mdml-engine/scene'
import { describe, expect, it } from 'vitest'
import { complexPlaneLayers, complexPlaneLimits, complexPlaneView } from './complex-plane.ts'
import { imageGridFigure } from './images.ts'
import { cellShapes, matrixMarkLayers, matrixMarksView } from './matrix-marks.ts'
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
    expect(layers.map((l) => l.series[0]?.name)).toEqual(['__imaginary axis', 'branch', 'zeros', 'poles (stable)'])
    expect(layers[3]?.series.map((s) => (s as Series).name)).toEqual(['poles (stable)', 'poles (unstable)'])
    const stable = layers[3]?.series[0] as { data: { value: number[] }[] }
    expect(stable.data).toHaveLength(2)
  })

  it('draws the unit circle for a discrete-time system and keeps both axes to the same range', () => {
    const z = complexPlane({ domain: 'z', poles: [[0.5, 0.5]] })
    expect(complexPlaneLayers(z)[0]?.series[0]?.name).toBe('__unit circle')
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

describe('the matrix marks', () => {
  type Item = { renderItem: (p: { dataIndex: number }, api: unknown) => Record<string, unknown>; data: number[][] }
  // A fake axis: column c centres at 100 + 20 c, row index y (bottom-up) at 200 - 10 y, cells 20 by 10.
  const api = {
    coord: (p: number[]): number[] => [100 + 20 * (p[0] ?? 0), 200 - 10 * (p[1] ?? 0)],
    size: (): number[] => [20, 10],
    value: (): number => 0,
  }

  it('draws a box on the cell edges in pixels, and a dashed boundary as a line', () => {
    // A 3 by 4 matrix: row 1, and the boundary left of column 2.
    const layers = matrixMarkLayers([{ row: 1 }, { cell: [0, 3] }, { colEdge: 2 }, { rowEdge: 2 }], 3, 4, 'light')
    expect(layers).toHaveLength(1)
    const series = layers[0]?.series[0] as unknown as Item
    expect(series.data).toHaveLength(4)
    // Row 1 of 3 is index 1 from the bottom: its top edge is at y = 1.5, its left edge at x = -0.5, 4 cells wide.
    expect(series.renderItem({ dataIndex: 0 }, api)).toMatchObject({
      type: 'rect',
      shape: { x: 90, y: 185, width: 80, height: 10 },
    })
    expect(series.renderItem({ dataIndex: 1 }, api)).toMatchObject({ shape: { x: 150, y: 175, width: 20, height: 10 } })
    expect(series.renderItem({ dataIndex: 2 }, api)).toMatchObject({
      type: 'line',
      shape: { x1: 130, x2: 130 },
      style: { lineDash: [6, 4] },
    })
    expect(series.renderItem({ dataIndex: 3 }, api)).toMatchObject({ type: 'line', shape: { y1: 195, y2: 195 } })
  })

  it('drops marks outside the matrix and has no layer when nothing is left', () => {
    expect(matrixMarkLayers([{ row: 5 }, { cell: [0, -1] }, { colEdge: 0 }], 3, 3, 'light')).toEqual([])
    expect(cellShapes([], 2, 2, 'light').series).toEqual([])
  })

  it('writes the labels, names every column and puts the marks over the cells', () => {
    const option = matrixMarksView(
      [
        [1, 2],
        [3, 4],
      ],
      { rowLabels: ['u1', 'u2'], colLabels: ['a', 'b'], marks: [{ cell: [1, 0] }] },
    ) as { xAxis: { data: string[]; axisLabel: { interval: number } }[]; series: { name?: string }[] }
    expect(option.xAxis[0]?.data).toEqual(['a', 'b'])
    expect(option.xAxis[0]?.axisLabel.interval).toBe(0)
    expect(option.series.map((s) => s.name)).toEqual(['__matrix', '__colourbar', '__cellShapes'])
  })
})

describe('the image grid figure', () => {
  it('gives the tiles room: the given width, almost no padding, and a height for square tiles', () => {
    const r = imageSteps({
      method: 'm',
      height: 2,
      width: 2,
      groups: ['a', 'b'],
      steps: [
        {
          iteration: 0,
          images: [
            [
              [0, 0, 0, 0],
              [1, 1, 1, 1],
            ],
            [
              [0, 1, 0, 1],
              [1, 0, 1, 0],
            ],
          ],
        },
      ],
    })
    const fig = imageGridFigure(r, 440)
    expect(fig.width).toBe(440)
    expect(fig.padding.left).toBe(8)
    // 2 images and a names' column of 2.2 tiles across, 2 groups down: a tile is (440 - 16) / 4.2 wide.
    expect(fig.height).toBe(Math.round(424 / (4.2 / 2)) + 16)
  })
})

describe('the fixed range of points', () => {
  it('overrides the span of the steps, so the axes hold from the first step', () => {
    const r = pointsSteps({
      dims: 2,
      range: [
        [0, 10],
        [-5, 5],
      ],
      steps: [{ step: 0, points: [[1, 1]] }],
    })
    expect(pointsStepsLimits(r)).toEqual([
      [0, 10],
      [-5, 5],
    ])
    const unit = pointsStepsScene3d(
      pointsSteps({
        dims: 3,
        range: [
          [0, 10],
          [0, 10],
          [0, 10],
        ],
        steps: [{ step: 0, points: [[5, 10, 0]] }],
      }),
      0,
    )
    const point = unit.items.find((i) => i.kind === 'point') as { at: readonly number[] }
    expect(point.at).toEqual([0, 1, -1])
  })
})
