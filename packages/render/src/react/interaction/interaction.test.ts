import { describe, expect, it } from 'vitest'
import { brushSelection } from '../../figure/linking.ts'
import { alongAxis, chain, inBox, onCircle, onGraph, onGrid, onPolyline, onSegment } from './constraints.ts'
import { chartProjection, linearProjection } from './projection.ts'
import { nearestPoint, rectBetween } from './selection.ts'

const projection = linearProjection({
  x: [0, 10],
  y: [0, 5],
  width: 220,
  height: 120,
  margins: { left: 20, top: 10, bottom: 10 },
})

describe('projections', () => {
  it('maps data to pixels with y up, and back', () => {
    expect(projection.toPixel([0, 0])).toEqual([20, 110])
    expect(projection.toPixel([10, 5])).toEqual([220, 10])
    const [x, y] = projection.toData(projection.toPixel([3, 2]))
    expect(x).toBeCloseTo(3, 12)
    expect(y).toBeCloseTo(2, 12)
  })

  it('adapts a chart converter', () => {
    const chart = {
      convertToPixel: (_: { gridIndex: number }, v: number[]) => v.map((u) => u * 2),
      convertFromPixel: (_: { gridIndex: number }, v: number[]) => v.map((u) => u / 2),
    }
    expect(chartProjection(chart).toPixel([1, 2])).toEqual([2, 4])
    expect(chartProjection(chart).toData([2, 4])).toEqual([1, 2])
  })
})

describe('constraints', () => {
  it('keeps points in boxes, on graphs, segments, polylines and circles', () => {
    expect(inBox([0, 1], [0, 1])([2, -1])).toEqual([1, 0])
    expect(onGraph((x) => x * x, [-1, 1])([2, 0])).toEqual([1, 1])
    expect(onSegment([0, 0], [2, 0])([1, 5])).toEqual([1, 0])
    expect(
      onPolyline([
        [0, 0],
        [1, 0],
        [1, 1],
      ])([2, 0.5]),
    ).toEqual([1, 0.5])
    const [cx, cy] = onCircle([0, 0], 2)([3, 4])
    expect(cx).toBeCloseTo(1.2, 12)
    expect(cy).toBeCloseTo(1.6, 12)
  })

  it('snaps to grids, moves along axes and chains', () => {
    expect(onGrid(0.5)([0.74, 1.26])).toEqual([0.5, 1.5])
    expect(alongAxis('x', 2)([5, 9])).toEqual([5, 2])
    expect(chain(onGrid(1), inBox([0, 3], [0, 3]))([7.4, 1.6])).toEqual([3, 2])
  })
})

describe('selection', () => {
  it('gives a dragged rectangle as ranges that brushSelection reads', () => {
    const rect = rectBetween([2, 3], [0, 1])
    expect(rect).toEqual({ x: [0, 2], y: [1, 3] })
    expect(
      brushSelection(
        [
          [1, 2],
          [3, 2],
          [2, 3],
        ],
        rect,
      ),
    ).toEqual([0, 2])
  })

  it('picks the nearest point on screen within a radius', () => {
    const points: [number, number][] = [
      [1, 1],
      [5, 2.5],
    ]
    const pixel = projection.toPixel([5.1, 2.5])
    expect(nearestPoint(points, pixel, projection)).toBe(1)
    expect(nearestPoint(points, [0, 0], projection)).toBe(-1)
  })
})
