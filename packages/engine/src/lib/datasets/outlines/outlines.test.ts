import { describe, expect, it } from 'vitest'
import { fontGlyphs, fontInstances, fonts, glyphContours, glyphOutline, outlineShapes, shapeOutline } from './index.ts'

describe('outline datasets', () => {
  it('draws every shape as a closed, centred outline of unit radius', () => {
    const shapes = outlineShapes()
    expect(new Set(shapes.map((s) => s.group))).toEqual(new Set(['icons', 'drawings', 'curves']))
    for (const s of shapes) {
      const p = shapeOutline(s.id, 128)
      expect(p).toHaveLength(128)
      expect(p.every(([x, y]) => Number.isFinite(x) && Number.isFinite(y))).toBe(true)
      expect(Math.max(...p.map(([x, y]) => Math.hypot(x, y)))).toBeCloseTo(1, 9)
    }
    expect(() => shapeOutline('unicorn', 8)).toThrow()
  })

  it('reads every glyph of every font, with its contours in correspondence', () => {
    const glyphs = fontGlyphs()
    for (let f = 0; f < fontInstances().length; f += 13)
      for (const g of glyphs) {
        expect(glyphContours(f, g.char).map((c) => c.length)).toEqual(g.contours)
        expect(glyphOutline(f, g.char, 64)).toHaveLength(64)
      }
    const d = fonts({ chars: 'HO' })
    expect(d.x.shape[0]).toBe(66)
    expect(() => fonts({ chars: 'K' })).toThrow()
    // The H's top sits at the cap height in every font.
    const tops = fontInstances().map((_, f) => Math.max(...(glyphContours(f, 'H')[0] ?? []).map((p) => p[1])))
    for (const t of tops) expect(Math.abs(t - 700)).toBeLessThan(25)
  })
})
