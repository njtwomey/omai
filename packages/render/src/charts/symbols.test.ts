import { describe, expect, it } from 'vitest'
import { points } from './points.ts'
import { SYMBOLS, symbolOf } from './symbols.ts'

// biome-ignore lint/suspicious/noExplicitAny: tests inspect untyped ECharts option trees
type Opt = any
const opt = (o: unknown): Opt => o as Opt

describe('named symbols', () => {
  it('resolves a name of the table and passes everything else through', () => {
    expect(symbolOf('plane')).toBe(SYMBOLS.plane)
    expect(symbolOf('cross')).toBe(SYMBOLS.cross)
    expect(symbolOf('diamond')).toBe('diamond')
    expect(symbolOf('path://M0,0L1,1Z')).toBe('path://M0,0L1,1Z')
    // A name that is a property of every object is not a symbol.
    expect(symbolOf('toString')).toBe('toString')
  })

  it('every path is a closed filled shape inside a 24 unit box (plane) or about the origin (cross)', () => {
    for (const path of Object.values(SYMBOLS)) {
      expect(path.startsWith('path://M')).toBe(true)
      expect(path.endsWith('Z')).toBe(true)
    }
    const numbers = SYMBOLS.plane.slice('path://'.length).match(/-?\d+(\.\d+)?/g) ?? []
    expect(numbers.length).toBeGreaterThan(20)
    expect(Math.min(...numbers.map(Number))).toBeGreaterThanOrEqual(0)
    expect(Math.max(...numbers.map(Number))).toBeLessThanOrEqual(24)
  })
})

describe('points: a named symbol, rotation and a label offset', () => {
  it('draws a named symbol as its path, with its aspect kept so a rotated plane is not squashed', () => {
    const s = opt(points([{ x: [0], y: [0], symbol: 'plane', size: 22 }]).series[0])
    expect(s.symbol).toBe(SYMBOLS.plane)
    expect(s.symbolKeepAspect).toBe(true)
    const plain = opt(points([{ x: [0], y: [0], symbol: 'diamond' }]).series[0])
    expect(plain.symbol).toBe('diamond')
    expect(plain.symbolKeepAspect).toBeUndefined()
  })

  it('rotates every marker by one angle, or each by its own (a heading per aircraft)', () => {
    const all = opt(points([{ x: [0, 1], y: [0, 1], symbol: 'plane', symbolRotate: 90 }]).series[0])
    expect(all.symbolRotate).toBe(-90) // ECharts turns counterclockwise; ours is a clockwise heading
    expect(all.data[0].symbolRotate).toBeUndefined()
    const each = opt(points([{ x: [0, 1, 2], y: [0, 1, 2], symbol: 'plane', symbolRotate: [10, 20] }]).series[0])
    expect(each.symbolRotate).toBeUndefined()
    expect(each.data.map((d: Opt) => d.symbolRotate)).toEqual([-10, -20, -0])
    expect(opt(points([{ x: [0], y: [0] }]).series[0]).symbolRotate).toBeUndefined()
  })

  it('rotates the points of each group by their own index in the set', () => {
    const l = points([{ x: [0, 1, 2], y: [0, 1, 2], groups: ['a', 'b', 'a'], symbolRotate: [10, 20, 30] }])
    const [a, b] = l.series.map(opt)
    expect(a.data.map((d: Opt) => d.symbolRotate)).toEqual([-10, -30])
    expect(b.data.map((d: Opt) => d.symbolRotate)).toEqual([-20])
  })

  it('moves a label from where its position puts it, for the series and for each labelled point', () => {
    const s = opt(points([{ x: [0], y: [0], labels: ['a'], labelPosition: 'top', labelOffset: [0, -12] }]).series[0])
    expect(s.label.offset).toEqual([0, -12])
    expect(s.data[0].label).toMatchObject({ show: true, position: 'top', offset: [0, -12] })
    const none = opt(points([{ x: [0], y: [0], labels: ['a'] }]).series[0])
    expect(none.label.offset).toBeUndefined()
    expect(none.data[0].label.offset).toBeUndefined()
  })
})
