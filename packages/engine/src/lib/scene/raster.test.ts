import { describe, expect, it } from 'vitest'
import { arrow, circle, ellipse, group, image, line, path, polygon, rect, scene, text } from './build.ts'
import { palettes, parsePaint } from './paint.ts'
import { camera, layer3d } from './projection.ts'
import { rasterise } from './raster.ts'
import { face3d, scene3d } from './three-d.ts'
import type { Pixels, SceneNode } from './types.ts'

const black = { background: '#000000' } as const

function at(px: Pixels, x: number, y: number): number[] {
  return Array.from(px.rgba.slice(4 * (y * px.width + x), 4 * (y * px.width + x) + 4))
}

function lit(px: Pixels): number {
  let n = 0
  for (let i = 0; i < px.width * px.height; i++) if ((px.rgba[4 * i] ?? 0) > 0) n++
  return n
}

const draw = (nodes: readonly SceneNode[], size = 20, options = {}): Pixels =>
  rasterise(scene(size, size, nodes), { ...black, ...options })

describe('parsePaint', () => {
  it('reads every colour form and theme tokens', () => {
    expect(parsePaint('#ff000080')).toEqual([255, 0, 0, 128 / 255])
    expect(parsePaint('#0f08')).toEqual([0, 255, 0, (8 * 17) / 255])
    expect(parsePaint('#123456')).toEqual([18, 52, 86, 1])
    expect(parsePaint('rgba(300, 0, 0, 2)')).toEqual([255, 0, 0, 1])
    expect(parsePaint('none')).toBeUndefined()
    expect(parsePaint('bg', palettes.dark)).toEqual([10, 10, 10, 1])
    expect(() => parsePaint('#12' as '#12')).toThrow()
    expect(() => parsePaint('rgba(a, b, c)')).toThrow()
  })
})

describe('rasterise', () => {
  it('fills rectangles exactly at pixel centres', () => {
    const px = draw([rect(2, 3, 5, 4, { fill: '#ffffff' })])
    expect(lit(px)).toBe(20)
    expect(at(px, 2, 3)).toEqual([255, 255, 255, 255])
    expect(at(px, 7, 3)).toEqual([0, 0, 0, 255])
  })

  it('fills circles with about pi r^2 pixels and applies the default fill', () => {
    const px = draw([circle(50, 50, 20)], 100)
    expect(Math.abs(lit(px) - Math.PI * 400) / (Math.PI * 400)).toBeLessThan(0.02)
    expect(at(px, 50, 50)).toEqual([10, 10, 10, 255])
    const e = draw([ellipse(50, 50, 30, 10, { fill: '#ffffff' })], 100)
    expect(Math.abs(lit(e) - Math.PI * 300) / (Math.PI * 300)).toBeLessThan(0.03)
  })

  it('strokes lines with their width, dashes and arrow heads', () => {
    const thick = draw([line(0, 10, 20, 10, { stroke: '#ffffff', strokeWidth: 2 })])
    expect(lit(thick)).toBe(40)
    const dashed = draw([line(0, 10, 20, 10, { stroke: '#ffffff', strokeWidth: 2, dash: [5, 5] })])
    expect(lit(dashed)).toBe(20)
    expect(at(dashed, 7, 10)[0]).toBe(0)
    const plain = lit(draw([line(0, 10, 18, 10, { stroke: '#ffffff' })]))
    const headed = lit(draw([arrow(0, 10, 18, 10, { stroke: '#ffffff' })]))
    expect(headed).toBeGreaterThan(plain)
  })

  it('draws open paths as strokes and closed paths as fills, with rounded rects and smooth curves', () => {
    const open = draw([
      path(
        [
          [2, 2],
          [18, 2],
          [18, 18],
        ],
        { strokeWidth: 3, stroke: '#ffffff' },
      ),
    ])
    expect(at(open, 10, 2)[0]).toBe(255)
    expect(at(open, 10, 10)[0]).toBe(0)
    const closed = draw([
      polygon(
        [
          [2, 2],
          [18, 2],
          [18, 18],
        ],
        { fill: '#ffffff' },
      ),
    ])
    expect(at(closed, 15, 5)[0]).toBe(255)
    const rounded = draw([rect(0, 0, 20, 20, { fill: '#ffffff', radius: 6 })])
    expect(at(rounded, 0, 0)[0]).toBe(0)
    expect(at(rounded, 10, 10)[0]).toBe(255)
    const smooth = draw([
      path(
        [
          [0, 10],
          [10, 0],
          [20, 10],
        ],
        { stroke: '#ffffff', smooth: true, strokeWidth: 2 },
      ),
    ])
    expect(lit(smooth)).toBeGreaterThan(20)
    const headed = draw([
      path(
        [
          [0, 10],
          [10, 10],
          [18, 10],
        ],
        { stroke: '#ffffff', arrow: true, fill: '#ff0000' },
      ),
    ])
    expect(lit(headed)).toBeGreaterThan(18)
  })

  it('composites with opacity, groups and literal alpha', () => {
    const px = rasterise(scene(4, 4, [group([rect(0, 0, 4, 4, { fill: '#ff0000', opacity: 0.5 })], { opacity: 1 })]), {
      background: '#ffffff',
    })
    expect(at(px, 1, 1)).toEqual([255, 128, 128, 255])
    const clear = rasterise(scene(4, 4, [rect(0, 0, 2, 2, { fill: 'rgba(0, 0, 255, 0.5)' })]), { background: 'none' })
    expect(at(clear, 0, 0)).toEqual([0, 0, 255, 128])
    expect(at(clear, 3, 3)).toEqual([0, 0, 0, 0])
  })

  it('applies group transforms and scales to the output size', () => {
    const moved = draw([group([rect(0, 0, 2, 2, { fill: '#ffffff' })], { translate: [10, 10], scale: 2 })])
    expect(at(moved, 13, 13)[0]).toBe(255)
    expect(at(moved, 9, 9)[0]).toBe(0)
    const rotated = draw([group([rect(0, 0, 10, 2, { fill: '#ffffff' })], { translate: [10, 0], rotate: 90 })])
    expect(at(rotated, 9, 5)[0]).toBe(255)
    const doubled = rasterise(scene(10, 10, [rect(0, 0, 5, 5, { fill: '#ffffff' })]), {
      ...black,
      width: 20,
      height: 20,
    })
    expect(lit(doubled)).toBe(100)
  })

  it('draws images nearest-neighbour or bilinear', () => {
    const src = { width: 2, height: 1, rgba: new Uint8ClampedArray([255, 0, 0, 255, 0, 0, 255, 255]) }
    const px = draw([image(0, 0, 20, 10, src)])
    expect(at(px, 4, 4)).toEqual([255, 0, 0, 255])
    expect(at(px, 15, 4)).toEqual([0, 0, 255, 255])
    expect(at(px, 4, 15)).toEqual([0, 0, 0, 255])
    const smooth = draw([image(0, 0, 20, 10, src, { smoothing: true })])
    const mid = at(smooth, 10, 4)
    expect(mid[0]).toBeGreaterThan(0)
    expect(mid[2]).toBeGreaterThan(0)
  })

  it('projects 3-D layers, skips text and follows the theme', () => {
    const face = face3d(
      [
        [-1, 0, -1],
        [1, 0, -1],
        [1, 0, 1],
        [-1, 0, 1],
      ],
      { fill: '#ffffff' },
    )
    const cam = camera({ azimuth: -90, elevation: 0, projection: 'orthographic', extent: 2 })
    const px = draw([layer3d(scene3d([face]), cam, { x: 0, y: 0, width: 20, height: 20 })])
    expect(lit(px)).toBe(100)
    expect(lit(draw([text(10, 10, 'hello', { fill: '#ffffff' })]))).toBe(0)
    expect(at(rasterise(scene(2, 2, []), { mode: 'dark' }), 0, 0)).toEqual([10, 10, 10, 255])
    expect(at(rasterise(scene(2, 2, [], { background: 'tab0' })), 0, 0)).toEqual([31, 119, 180, 255])
    const custom = rasterise(scene(2, 2, [rect(0, 0, 2, 2, { fill: 'accent' })]), { palette: { accent: '#00ff00' } })
    expect(at(custom, 0, 0)).toEqual([0, 255, 0, 255])
  })

  it('supersamples edges and rejects bad sizes', () => {
    const px = draw([rect(0, 0, 10.5, 20, { fill: '#ffffff' })], 20, { samples: 2 })
    expect(at(px, 10, 5)[0]).toBe(128)
    expect(() => rasterise(scene(0, 0, []))).toThrow()
    expect(() => rasterise(scene(4, 4, []), { width: 2.5 })).toThrow()
  })

  it('is deterministic', () => {
    const s = scene(30, 30, [circle(15, 15, 9, { fill: 'tab2', stroke: 'fg', strokeWidth: 2 }), line(0, 0, 30, 30)])
    expect(rasterise(s, { samples: 2 }).rgba).toEqual(rasterise(s, { samples: 2 }).rgba)
  })
})
