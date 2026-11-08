import { describe, expect, it } from 'vitest'
import { tensor } from '../tensor/index.ts'
import { circle, group, image, isScene, line, polygon, recolour, scene, text, toPixels } from './build.ts'

describe('build', () => {
  it('converts grey tensors with their range, and NaN to transparent', () => {
    const px = toPixels(tensor([[0, 5, 10, Number.NaN]]))
    expect([px.width, px.height]).toEqual([4, 1])
    expect(Array.from(px.rgba)).toEqual([0, 0, 0, 255, 128, 128, 128, 255, 255, 255, 255, 255, 0, 0, 0, 0])
    expect(Array.from(toPixels([[5]], { range: [0, 10] }).rgba)).toEqual([128, 128, 128, 255])
    expect(Array.from(toPixels([[3, 3]]).rgba.slice(0, 4))).toEqual([0, 0, 0, 255])
  })

  it('converts RGB and RGBA arrays in 0-1 or 0-255', () => {
    expect(Array.from(toPixels([[[0, 128, 255]]]).rgba)).toEqual([0, 128, 255, 255])
    expect(Array.from(toPixels([[[1, 1, 1, 0.5]]]).rgba)).toEqual([255, 255, 255, 128])
    expect(Array.from(toPixels(tensor([[[0.5], [1]]])).rgba.slice(4))).toEqual([255, 255, 255, 255])
  })

  it('rejects bad shapes', () => {
    expect(() => toPixels(tensor([1, 2, 3]))).toThrow()
    expect(() => toPixels([[[1, 2]]])).toThrow()
    expect(() => toPixels([[1, 2], [3]])).toThrow()
    expect(() =>
      toPixels([
        [
          [1, 2, 3],
          [1, 2],
        ],
      ]),
    ).toThrow()
    expect(() => toPixels({ shape: [2, 2], data: [1, 2] })).toThrow()
  })

  it('builds image nodes from pixels or arrays, and groups and scenes', () => {
    const node = image(
      0,
      0,
      4,
      4,
      tensor([
        [0, 1],
        [1, 0],
      ]),
      { smoothing: true, id: 'img' },
    )
    expect(node.pixels.height).toBe(2)
    expect(node.smoothing).toBe(true)
    expect(
      group(
        [
          polygon([
            [0, 0],
            [1, 0],
            [0, 1],
          ]),
        ],
        { rotate: 45, opacity: 0.5 },
      ).rotate,
    ).toBe(45)
    expect(isScene(scene(1, 1, [], { background: 'bg' }))).toBe(true)
    expect(isScene(null)).toBe(false)
  })

  it('recolours named nodes: a fill, a line as a stroke, text, a group as all its nodes, the rest left alone', () => {
    const s = scene(100, 100, [
      circle(5, 5, 2, { fill: 'tab0', id: 'dot' }),
      line(0, 0, 9, 9, { stroke: 'fg', id: 'rule' }),
      text(1, 1, 'a', { id: 'label' }),
      group([circle(1, 1, 1, { fill: 'tab1' }), line(0, 0, 1, 1)], { id: 'pair' }),
      circle(7, 7, 2, { fill: 'tab2', id: 'other' }),
    ])
    const r = recolour(s, { dot: 'tab3', rule: 'tab4', label: 'accent', pair: 'tab5', missing: 'tab6' })
    const [dot, rule, label, pair, other] = r.children
    expect(dot?.kind === 'circle' && dot.fill).toBe('tab3')
    expect(rule?.kind === 'line' && rule.stroke).toBe('tab4')
    expect(label?.kind === 'text' && label.fill).toBe('accent')
    expect(
      pair?.kind === 'group' &&
        pair.children.map((c) => (c.kind === 'circle' ? c.fill : c.kind === 'line' ? c.stroke : null)),
    ).toEqual(['tab5', 'tab5'])
    expect(other).toBe(s.children[4])
    expect(recolour(s, {})).toBe(s)
    expect(recolour(s, { missing: 'tab1' })).toBe(s)
    expect((s.children[0] as { fill: string }).fill).toBe('tab0')
  })
})
