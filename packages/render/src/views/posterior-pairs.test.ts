import { OmaiError } from 'omai-engine/base'
import { chainDiagnostics, type PairDraws, posteriorPairs } from 'omai-engine/exposition'
import { describe, expect, it } from 'vitest'
import { pairEdge, pairGrid, thinEvenly } from './_view.ts'
import { pairView } from './chain-charts.ts'
import { pairsFigureOptions, pairsMethodColour, posteriorPairsView } from './posterior-pairs.ts'

type Datum = [number, number] | { value: [number, number] }
type Series = { name?: string; type?: string; data?: Datum[]; markLine?: unknown }
const series = (o: { series?: unknown }): Series[] => (o.series ?? []) as Series[]
const xy = (d: Datum | undefined): [number, number] | undefined => (Array.isArray(d) ? d : d?.value)

/** A deterministic correlated cloud of `n` draws of three coordinates around a centre. */
function cloud(n: number, centre: number, rho: number): PairDraws['draws'] {
  const a: number[] = []
  const b: number[] = []
  const c: number[] = []
  for (let i = 0; i < n; i++) {
    const u = Math.sin(i * 12.9898) * 43758.5453
    const w = Math.sin(i * 78.233) * 12345.6789
    const z1 = (u - Math.floor(u) - 0.5) * 3.4
    const z2 = (w - Math.floor(w) - 0.5) * 3.4
    a.push(centre + z1)
    b.push(centre + rho * z1 + Math.sqrt(1 - rho * rho) * z2)
    c.push(z2 * 2)
  }
  return { a, b, c }
}

const record = posteriorPairs(
  [
    {
      name: 'nuts',
      draws: cloud(300, 0, 0.8),
      divergent: Array.from({ length: 300 }, (_, i) => (i % 100 === 7 ? 1 : 0)),
    },
    { name: 'vi', draws: cloud(120, 0.4, 0.1) },
  ],
  { frames: 6, gridPoints: 30, maxPoints: 100, truth: { a: 0, b: 0 } },
)

describe('posteriorPairsView', () => {
  it('lays out every cell of the grid for the coordinates shown', () => {
    expect((posteriorPairsView(record).grid as unknown[]).length).toBe(9)
    expect((posteriorPairsView(record, { coordinates: ['c', 'a'] }).grid as unknown[]).length).toBe(4)
  })

  it('scatters the points the cursor has reached in the lower triangle, a series per method', () => {
    const end = series(posteriorPairsView(record)).filter((s) => s.name === 'nuts' && s.type === 'scatter')
    // Three coordinates: three lower panels.
    expect(end).toHaveLength(3)
    expect(end[0]?.data).toHaveLength(100)
    const early = series(posteriorPairsView(record, { cursor: record.cursors[1] as number })).filter(
      (s) => s.name === 'nuts' && s.type === 'scatter',
    )
    const n = (record.methods[0]?.points.steps ?? []).filter((s) => s <= (record.cursors[1] as number)).length
    expect(early[0]?.data).toHaveLength(n)
    expect(n).toBeLessThan(100)
    // The first point is the first draw of the method: the lower panel at row 1, column 0 has x = a and y = b.
    const first = xy(
      series(posteriorPairsView(record)).find((s) => s.name === 'nuts' && s.type === 'scatter')?.data?.[0],
    )
    expect(first).toEqual([record.methods[0]?.points.values[0]?.[0], record.methods[0]?.points.values[1]?.[0]])
  })

  it('draws a density per method on each diagonal panel, and none before the first frame', () => {
    const lines = (o: ReturnType<typeof posteriorPairsView>) =>
      series(o).filter((s) => s.type === 'line' && (s.name === 'nuts' || s.name === 'vi'))
    expect(lines(posteriorPairsView(record))).toHaveLength(6)
    expect(lines(posteriorPairsView(record, { cursor: -1 }))).toHaveLength(0)
    // A method's curve is on the record's grid.
    expect(lines(posteriorPairsView(record))[0]?.data).toHaveLength(30)
  })

  it('shows only the methods named, keeping each one its colour, and says if none is in the record', () => {
    const only = series(posteriorPairsView(record, { methods: ['vi'] }))
    expect(only.some((s) => s.name === 'nuts')).toBe(false)
    expect(only.some((s) => s.name === 'vi')).toBe(true)
    expect(
      JSON.stringify(
        series(posteriorPairsView(record, { methods: ['vi'], mode: 'light' })).find((s) => s.name === 'vi'),
      ),
    ).toContain('#ff7f0e')
    expect(() => posteriorPairsView(record, { methods: ['nope'] })).toThrow(OmaiError)
    expect(() => posteriorPairsView(record, { coordinates: ['a'] })).toThrow(expect.objectContaining({ code: 'value' }))
  })

  it('marks the divergent transitions that had happened, and the truth where the pair has one', () => {
    const div = series(posteriorPairsView(record)).filter((s) => s.name === 'divergent')
    expect(div).toHaveLength(3)
    expect(div[0]?.data).toHaveLength(3)
    expect(series(posteriorPairsView(record, { cursor: 5 })).filter((s) => s.name === 'divergent')).toHaveLength(0)
    // Truth for a and b: one diamond at the pair (a, b) and none for the pair with c.
    const diamonds = series(posteriorPairsView(record)).filter((s) => s.name === 'truth')
    expect(diamonds).toHaveLength(1)
    expect(xy(diamonds[0]?.data?.[0])).toEqual([0, 0])
    expect(JSON.stringify(posteriorPairsView(record))).toContain('markLine')
  })

  it("writes each method's correlation in the upper triangle, or draws its ellipses", () => {
    const o = posteriorPairsView(record) as { graphic?: { style?: { text?: string } }[] }
    const texts = (o.graphic ?? []).map((g) => g.style?.text)
    // Three upper panels, two methods each.
    expect(texts).toHaveLength(6)
    expect(texts.some((t) => t?.includes('0.8') || t?.includes('0.7'))).toBe(true)
    const e = posteriorPairsView(record, { upper: 'ellipses' })
    expect(((e as { graphic?: unknown[] }).graphic ?? []).length).toBe(0)
    const loops = series(e).filter((s) => s.type === 'line' && s.name?.startsWith('__') && s.name.endsWith(' 50'))
    expect(loops.length).toBe(6)
    // A closed ellipse of 49 points.
    expect(loops[0]?.data).toHaveLength(49)
  })

  it('keeps the axes of a coordinate the same at every step, from the record', () => {
    const limits = (o: ReturnType<typeof posteriorPairsView>): unknown =>
      JSON.stringify([
        (o.xAxis as { min?: number; max?: number }[]).map((a) => [a.min, a.max]),
        (o.yAxis as { min?: number; max?: number }[]).map((a) => [a.min, a.max]),
      ])
    expect(limits(posteriorPairsView(record, { cursor: record.cursors[1] as number }))).toBe(
      limits(posteriorPairsView(record)),
    )
  })

  it('names the methods in one legend, in the dark mode too', () => {
    for (const mode of ['light', 'dark'] as const) {
      const o = posteriorPairsView(record, { mode }) as { legend?: { data?: string[]; show?: boolean } }
      expect(o.legend?.data).toEqual(['nuts', 'vi', 'divergent', 'truth'])
      expect(o.legend?.show).toBe(true)
    }
  })

  it('gives the methods tab colours without the red of the divergent transitions, and larger plots more width', () => {
    expect([0, 1, 2, 3, 4, 8, 9].map(pairsMethodColour)).toEqual([
      'tab0',
      'tab1',
      'tab2',
      'tab4',
      'tab5',
      'tab9',
      'tab0',
    ])
    expect(pairsFigureOptions(8).width).toBeGreaterThan(pairsFigureOptions(3).width)
  })
})

describe('the pair helpers pairView shares', () => {
  it('number the cells of the lower triangle, or all of them, and label the outer edges', () => {
    expect(pairGrid(3).cells.map((c) => c.id)).toEqual(['p0_0', 'p1_0', 'p1_1', 'p2_0', 'p2_1', 'p2_2'])
    expect(pairGrid(3, true).cells).toHaveLength(9)
    expect(pairEdge(['a', 'b', 'c'], 2, 1)).toEqual({ x: { label: 'b' } })
    expect(pairEdge(['a', 'b', 'c'], 2, 0)).toEqual({ x: { label: 'a' }, y: { label: 'c' } })
    expect(pairEdge(['a', 'b', 'c'], 0, 0)).toEqual({})
  })

  it('thin evenly, keeping the ends', () => {
    expect(thinEvenly([1, 2, 3, 4, 5, 6, 7], 3)).toEqual([1, 4, 7])
    expect(thinEvenly([1, 2], 3)).toEqual([1, 2])
  })

  it('still draw pairView as before (a lower triangle, with the edge labels)', () => {
    const a = [
      [0, 1, 0, 1, 2],
      [1, 0, 1, 0, 2],
    ]
    const o = pairView(chainDiagnostics({ a, b: a, c: a }))
    expect((o.grid as unknown[]).length).toBe(6)
    expect(JSON.stringify(o)).toContain('"name":"c"')
  })
})
