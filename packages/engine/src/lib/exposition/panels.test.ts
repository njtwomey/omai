import { describe, expect, it } from 'vitest'
import { scene, structure } from '../scene/index.ts'
import { complexPlane, isStablePole } from './complex-plane.ts'
import { sampleVectorField, vectorFieldRecord } from './fields.ts'
import { imageSteps } from './images.ts'
import { pointsAt, pointsSteps } from './points-steps.ts'
import { reachabilityRecord } from './reachability.ts'
import { isExposition } from './record.ts'
import { sceneStepAt, sceneSteps } from './scene-steps.ts'
import { simplexRecord } from './simplex.ts'
import { stripRecord } from './strips.ts'
import { structureStepAt, structureSteps } from './structure-steps.ts'
import { markedTable } from './tables.ts'

describe('the records the generic panels draw', () => {
  it('check a vector field against its grid and are recognised as records', () => {
    expect(() => vectorFieldRecord({ x: [0, 1], y: [0], u: [[1]], v: [[1, 1]] })).toThrow(/u\[0\]/)
    expect(() => vectorFieldRecord({ x: [0], y: [0, 1], u: [[1]], v: [[1], [1]] })).toThrow(/u has 1/)
    expect(() =>
      vectorFieldRecord({
        x: [0],
        y: [0],
        u: [[1]],
        v: [[1]],
        fixedPoints: [{ at: [Number.NaN, 0], stability: 'stable' }],
      }),
    ).toThrow(/fixedPoints/)
    const r = sampleVectorField((x, y) => [y, -x], [-1, 0, 1], [-1, 0, 1], {
      fixedPoints: [{ at: [0, 0], stability: 'centre' }],
    })
    expect(r.u[0]).toEqual([-1, -1, -1])
    expect(r.v[0]?.map((x) => x + 0)).toEqual([1, 0, -1])
    expect(isExposition(r)).toBe(true)
  })

  it('check a strip: one entry per token, a path of tracks, marks at positions', () => {
    const tokens = ['a', 'b']
    const tracks = [{ name: 'p', values: [0, 1] }]
    expect(() => stripRecord({ tokens, tracks: [{ name: 'p', values: [0] }] })).toThrow(/tracks\[0\]/)
    expect(() => stripRecord({ tokens, tracks, path: [0, 1] })).toThrow(/not a track/)
    expect(() => stripRecord({ tokens, tracks, marks: [2] })).toThrow(/not a position/)
    expect(() => stripRecord({ tokens, tracks, tags: [{ name: 'pos', labels: ['x'] }] })).toThrow(/tags\[0\]/)
    expect(isExposition(stripRecord({ tokens, tracks, path: [0, 0], marks: [1] }))).toBe(true)
  })

  it('check a table: rows as wide as the columns, marks on cells that exist, and keep data tables apart', () => {
    expect(() => markedTable({ columns: ['a', 'b'], cells: [[1]] })).toThrow(/cells\[0\]/)
    expect(() => markedTable({ columns: ['a'], cells: [[1]], rowNames: [] })).toThrow(/rowNames/)
    expect(() => markedTable({ columns: ['a'], cells: [[1]], marks: [{ status: 'ok' }] })).toThrow(/neither/)
    expect(() => markedTable({ columns: ['a'], cells: [[1]], marks: [{ row: 1, status: 'ok' }] })).toThrow(/row 1/)
    expect(() => markedTable({ columns: ['a'], cells: [[1]], marks: [{ col: 3, status: 'ok' }] })).toThrow(/column 3/)
    expect(isExposition(markedTable({ columns: ['a'], cells: [[1]] }))).toBe(true)
    expect(isExposition({ kind: 'table', columns: ['a'], rows: [[1]] })).toBe(false)
  })

  it('check a structure with steps against its nodes and edges, and read the latest step at or before a step', () => {
    const s = structure(
      'graph',
      ['a', 'b'].map((id) => ({ id, label: id, role: 'latent' as const })),
      [{ from: 'a', to: 'b' }],
    )
    expect(() => structureSteps({ structure: s, steps: [{ step: 0, nodes: { c: {} } }] })).toThrow(/node c/)
    expect(() => structureSteps({ structure: s, steps: [{ step: 0, edges: [{ from: 'b', to: 'a' }] }] })).toThrow(
      /edge b to a/,
    )
    expect(() => structureSteps({ structure: s, steps: [{ step: 3 }, { step: 1 }] })).toThrow(/out of order/)
    const r = structureSteps({
      structure: s,
      steps: [
        { step: 0, caption: 'first' },
        { step: 5, caption: 'second' },
      ],
    })
    expect(structureStepAt(r, 4)?.caption).toBe('first')
    expect(structureStepAt(r, 5)?.caption).toBe('second')
    expect(structureStepAt(r, -1)?.caption).toBe('first')
  })

  it('check the captions of image steps against the images', () => {
    const base = { method: 'm', height: 1, width: 1, groups: ['g'] }
    expect(() => imageSteps({ ...base, steps: [{ iteration: 0, images: [[[0], [1]]], labels: [['one']] }] })).toThrow(
      /labels\[0\]/,
    )
    expect(
      imageSteps({ ...base, steps: [{ iteration: 0, images: [[[0], [1]]], labels: [['a', 'b']] }] }).steps[0]?.labels,
    ).toEqual([['a', 'b']])
  })

  it("check points: each point has the record's dimension, groups match, steps run in order; accumulation adds up", () => {
    expect(() => pointsSteps({ dims: 3, steps: [{ step: 0, points: [[1, 2]] }] })).toThrow(/expected 3/)
    expect(() => pointsSteps({ dims: 2, steps: [{ step: 0, points: [[1, 2]], groups: [0, 1] }] })).toThrow(/groups/)
    expect(() =>
      pointsSteps({
        dims: 2,
        steps: [
          { step: 2, points: [] },
          { step: 1, points: [] },
        ],
      }),
    ).toThrow(/order/)
    const grow = pointsSteps({
      dims: 2,
      steps: [
        { step: 0, points: [[0, 0]], groups: [0], caption: 'one' },
        {
          step: 4,
          points: [
            [1, 1],
            [2, 2],
          ],
          groups: [1, 1],
          caption: 'three',
        },
      ],
    })
    expect(pointsAt(grow, 3)).toEqual({ points: [[0, 0]], groups: [0], caption: 'one', fresh: 1 })
    expect(pointsAt(grow, 9).points).toHaveLength(3)
    expect(pointsAt(grow, 9).groups).toEqual([0, 1, 1])
    expect(pointsAt(grow, 9).fresh).toBe(2)
    expect(pointsAt(grow, -5).points).toHaveLength(1)
    const move = pointsSteps({ ...grow, accumulate: false })
    expect(pointsAt(move, 9).points).toEqual([
      [1, 1],
      [2, 2],
    ])
    expect(isExposition(grow)).toBe(true)
  })

  it('check the complex plane: finite points, a label per pole, stability by domain', () => {
    expect(() => complexPlane({ domain: 's', poles: [[Number.NaN, 0]] })).toThrow(/poles\[0\]/)
    expect(() => complexPlane({ domain: 's', poles: [[-1, 0]], labels: [] })).toThrow(/labels/)
    expect(isStablePole('s', [-0.1, 5])).toBe(true)
    expect(isStablePole('s', [0, 1])).toBe(false)
    expect(isStablePole('z', [0.6, 0.7])).toBe(true)
    expect(isStablePole('z', [1, 0])).toBe(false)
    expect(isExposition(complexPlane({ domain: 'z', poles: [[0.5, 0]], zeros: [[0, 0]] }))).toBe(true)
  })

  it("check the simplex: the shape's coordinates, probabilities, a triangle summing to one", () => {
    const labels = ['a', 'b', 'c']
    expect(() => simplexRecord({ shape: 'triangle', labels: ['a', 'b'] })).toThrow(/labels/)
    expect(() => simplexRecord({ shape: 'triangle', labels, paths: [{ name: 'p', points: [[0.5, 0.5]] }] })).toThrow(
      /expected 3/,
    )
    expect(() =>
      simplexRecord({ shape: 'triangle', labels, paths: [{ name: 'p', points: [[0.5, 0.5, 0.5]] }] }),
    ).toThrow(/sum/)
    expect(() =>
      simplexRecord({ shape: 'square', labels: ['x', 'y'], paths: [{ name: 'p', points: [[1.5, 0]] }] }),
    ).toThrow(/probability/)
    expect(() => simplexRecord({ shape: 'triangle', labels, field: { at: [[1, 0, 0]], by: [] } })).toThrow(/directions/)
    expect(
      isExposition(
        simplexRecord({ shape: 'triangle', labels, fixedPoints: [{ at: [0.2, 0.3, 0.5], stability: 'centre' }] }),
      ),
    ).toBe(true)
  })

  it('check a reachability plot: one entry per position, distances that are distances', () => {
    expect(() => reachabilityRecord({ reach: [null, 1], order: [0] })).toThrow(/order/)
    expect(() => reachabilityRecord({ reach: [null, 1], labels: [0] })).toThrow(/labels/)
    expect(() => reachabilityRecord({ reach: [null, -1] })).toThrow(/reach\[1\]/)
    expect(isExposition(reachabilityRecord({ reach: [null, 0.2] }))).toBe(true)
  })

  it('check scenes per step: in order, the latest at or before the step', () => {
    const s = (w: number) => scene(w, w, [])
    expect(() =>
      sceneSteps({
        steps: [
          { step: 3, scene: s(1) },
          { step: 1, scene: s(1) },
        ],
      }),
    ).toThrow(/order/)
    const r = sceneSteps({
      steps: [
        { step: 0, scene: s(10) },
        { step: 5, scene: s(20), caption: 'later' },
      ],
    })
    expect(sceneStepAt(r, 4)?.scene.width).toBe(10)
    expect(sceneStepAt(r, 5)?.caption).toBe('later')
    expect(sceneStepAt(r, -1)?.scene.width).toBe(10)
    expect(sceneStepAt(sceneSteps({ steps: [] }), 0)).toBeUndefined()
    expect(isExposition(r)).toBe(true)
  })
})
