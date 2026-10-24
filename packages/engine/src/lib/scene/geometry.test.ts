import { describe, expect, it } from 'vitest'
import { isWhiteboxError } from '../base/index.ts'
import { circle, group, line, path, rect, scene, text } from './build.ts'
import { bounds, fit, grid, invertTransform, rotate, scaleTo, translate } from './geometry.ts'

describe('geometry', () => {
  it('bounds covers every kind of node, through groups', () => {
    expect(bounds([])).toEqual({ x: 0, y: 0, width: 0, height: 0 })
    expect(bounds([line(1, 2, 4, 6), text(10, 0, 'x')])).toEqual({ x: 1, y: 0, width: 9, height: 6 })
    expect(
      bounds([
        path([
          [0, 0],
          [2, 3],
        ]),
      ]),
    ).toEqual({ x: 0, y: 0, width: 2, height: 3 })
    expect(bounds([group([circle(0, 0, 1)], { translate: [5, 5], scale: 2 })])).toEqual({
      x: 3,
      y: 3,
      width: 4,
      height: 4,
    })
  })

  it('bounds of a rotated rectangle uses all four corners', () => {
    const b = bounds([rotate(rect(0, 0, 2, 1), 90)])
    expect(b.x).toBeCloseTo(-1, 12)
    expect(b.y).toBeCloseTo(0, 12)
    expect(b.width).toBeCloseTo(1, 12)
    expect(b.height).toBeCloseTo(2, 12)
  })

  it('translate wraps several nodes in one group', () => {
    const g = translate([rect(0, 0, 1, 1), circle(3, 3, 1)], 1, 1)
    expect(g.children).toHaveLength(2)
    expect(g.translate).toEqual([1, 1])
  })

  it('rotate about a point composes into an existing group', () => {
    const g = rotate(group([rect(0, 0, 1, 1)], { scale: 2 }), 180, [1, 1])
    const b = bounds([g])
    expect(b.x).toBeCloseTo(0, 12)
    expect(b.y).toBeCloseTo(0, 12)
    expect(b.width).toBeCloseTo(2, 12)
  })

  it('scaleTo validates its input and handles degenerate boxes', () => {
    expect(() => scaleTo(rect(0, 0, 1, 1), {})).toThrow()
    expect(scaleTo(line(0, 0, 4, 0), { height: 2 }).scale).toBe(1)
    expect(scaleTo(line(0, 0, 4, 0), { width: 2 }).scale).toBe(0.5)
  })

  it('invertTransform rejects a zero scale', () => {
    try {
      invertTransform({ scale: 0 })
      expect.unreachable()
    } catch (e) {
      expect(isWhiteboxError(e, 'singular')).toBe(true)
    }
  })

  it('fit centres and grid tiles', () => {
    const g = fit([rect(0, 0, 2, 1)], { x: 0, y: 0, width: 10, height: 10 })
    expect(g.scale).toBe(5)
    expect(g.translate).toEqual([0, 2.5])
    const strip = grid([scene(10, 10, []), scene(10, 10, []), scene(10, 10, [])], 2, [5, 5])
    expect([strip.width, strip.height]).toEqual([10, 10])
    expect((strip.children[2] as { translate: readonly number[] }).translate).toEqual([0, 5])
  })
})
