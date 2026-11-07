import { describe, expect, it } from 'vitest'
import { circle, ellipse, grid, line, mergeStructures, rect, scene, structure, timeline } from './index.ts'

const node = (id: string): { id: string; label: string; role: 'other' } => ({ id, label: id, role: 'other' })

describe('the scene builders check their arguments where the drawing is made', () => {
  it('rejects non-finite coordinates, negative sizes, duplicate ids, a merged dangling edge and a bad grid or timeline', () => {
    const bad: [string, () => unknown][] = [
      ['NaN width', () => scene(Number.NaN, 5, [])],
      ['negative height', () => scene(5, -1, [])],
      ['infinite width', () => scene(Number.POSITIVE_INFINITY, 5, [])],
      ['negative rect width', () => rect(0, 0, -10, 5)],
      ['NaN rect corner', () => rect(Number.NaN, 0, 1, 1)],
      ['negative radius', () => circle(0, 0, -1)],
      ['NaN centre', () => circle(0, Number.NaN, 1)],
      ['negative ellipse radius', () => ellipse(0, 0, 3, -1)],
      ['infinite line end', () => line(0, 0, Number.POSITIVE_INFINITY, 1)],
      ['no columns', () => grid([scene(1, 1, [])], 0, [10, 10])],
      ['fractional columns', () => grid([scene(1, 1, [])], 1.5, [10, 10])],
      ['empty cell', () => grid([scene(1, 1, [])], 1, [0, 10])],
      ['duplicate node ids', () => structure('g', [node('a'), node('a')], [])],
      [
        'merged edge to a missing node',
        () => mergeStructures([structure('g', [node('a')], [{ from: 'a', to: 'zzz' }])]),
      ],
      ['negative fps', () => timeline([], { fps: -1 })],
      ['NaN step', () => timeline([{ kind: 'scene', step: Number.NaN, scene: scene(1, 1, []) }])],
    ]
    for (const [name, make] of bad) expect(make, name).toThrow(expect.objectContaining({ code: 'value' }))
    // Zero sizes are drawings of nothing, not errors; a valid structure and timeline are built as before.
    expect(scene(0, 0, []).width).toBe(0)
    expect(rect(0, 0, 0, 0).width).toBe(0)
    expect(circle(0, 0, 0).r).toBe(0)
    expect(structure('g', [node('a'), node('b')], [{ from: 'a', to: 'b' }]).edges).toHaveLength(1)
    expect(structure('part', [node('a')], [{ from: 'a', to: 'elsewhere' }]).edges).toHaveLength(1)
    expect(timeline([{ kind: 'scene', step: 2, scene: scene(1, 1, []) }]).fps).toBe(10)
  })
})
