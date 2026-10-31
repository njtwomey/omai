import { describe, expect, it } from 'vitest'
import { structure } from '../scene/index.ts'
import { sampleVectorField, vectorFieldRecord } from './fields.ts'
import { imageSteps } from './images.ts'
import { isExposition } from './record.ts'
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
})
