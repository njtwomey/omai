import { describe, expect, it } from 'vitest'
import { plugInMutualInformation } from '../information/index.ts'
import { key, normal } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { mrmr, mrmrAlgorithm } from './mrmr.ts'

/** x0 and x2 are near copies; y depends on x0 and x1; six pure-noise columns. */
function planted(n: number, seed: number): { x: number[][]; y: number[] } {
  const z = toFlat(normal(key(seed), [n * 9]))
  const x = Array.from({ length: n }, (_, i) => {
    const row = Array.from({ length: 9 }, (_, j) => z[i * 9 + j] ?? 0)
    row[2] = (row[0] ?? 0) + 0.1 * (row[2] ?? 0)
    return row
  })
  const e = toFlat(normal(key(seed + 1), [n]))
  return { x, y: x.map((r, i) => 1.5 * (r[0] ?? 0) + 0.8 * (r[1] ?? 0) + 0.3 * (e[i] ?? 0)) }
}

describe('mRMR', () => {
  it('avoids the redundant copy that plain relevance ranking takes second', () => {
    const { x, y } = planted(600, 1)
    const r = mrmr(x, y, 2)
    // Relevance alone ranks x0 and its near copy x2 on top.
    const byRelevance = Array.from(r.relevance.keys()).sort((a, b) => (r.relevance[b] ?? 0) - (r.relevance[a] ?? 0))
    expect(byRelevance.slice(0, 2).sort()).toEqual([0, 2])
    // mRMR takes one of the pair, then the independent informative feature.
    expect([0, 2]).toContain(r.selected[0])
    expect(r.selected[1]).toBe(1)
    for (const criterion of ['mid', 'miq'] as const) {
      const c = mrmr(x, y, 2, { criterion })
      expect(c.selected.includes(1)).toBe(true)
      expect(c.selected.includes(0) && c.selected.includes(2)).toBe(false)
    }
  })

  it('each step takes the argmax of relevance minus mean redundancy, recomputed independently', () => {
    const { x, y } = planted(300, 3)
    const alg = mrmrAlgorithm(x, y, 3, { bins: 6 })
    const res = runAlgorithm(alg, { maxSteps: 3 })
    const bin = (col: number[]): number[] => {
      const order = col.map((v, i) => [v, i] as const).sort((a, b) => a[0] - b[0])
      const out = new Array<number>(col.length).fill(0)
      order.forEach(([, i], r) => {
        out[i] = Math.min(5, Math.floor((r * 6) / col.length))
      })
      return out
    }
    const cols = Array.from({ length: 9 }, (_, j) => bin(x.map((r) => r[j] ?? 0)))
    const yb = bin(y)
    const rel = cols.map((c) => plugInMutualInformation(c, yb))
    const sel = res.state.selected
    expect(sel[0]).toBe(rel.indexOf(Math.max(...rel)))
    for (let step = 1; step < 3; step++) {
      const chosen = sel.slice(0, step)
      const scores = cols.map((c, j) =>
        chosen.includes(j)
          ? Number.NEGATIVE_INFINITY
          : (rel[j] ?? 0) - chosen.reduce((a, s) => a + plugInMutualInformation(c, cols[s] ?? []), 0) / step,
      )
      expect(sel[step]).toBe(scores.indexOf(Math.max(...scores)))
      expect(res.state.scores[step]).toBeCloseTo(Math.max(...scores), 10)
    }
  })

  it('works for class labels with the binned and the nearest-neighbour estimators', () => {
    const { x, y } = planted(300, 5)
    const labels = y.map((v) => (v > 0 ? 1 : 0))
    expect(mrmr(x, labels, 2).selected.includes(1)).toBe(true)
    const ksg = mrmr(x, labels, 2, { estimator: 'ksg' })
    expect(ksg.selected.includes(1)).toBe(true)
    expect([0, 2].some((j) => ksg.selected.includes(j))).toBe(true)
  })

  it('rejects an impossible k and serialises its state', () => {
    const { x, y } = planted(60, 7)
    expect(() => mrmr(x, y, 0)).toThrow()
    expect(() => mrmr(x, y, 10)).toThrow()
    const alg = mrmrAlgorithm(x, y, 2)
    const r = runAlgorithm(alg, { maxSteps: 2 })
    expect(alg.deserialise?.(JSON.parse(JSON.stringify(alg.serialise?.(r.state))))?.selected).toEqual(r.state.selected)
  })
})
