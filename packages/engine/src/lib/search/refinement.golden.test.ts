import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../testing/golden.ts'
import {
  conjunctions,
  refinementBeamSearch,
  refinementBestFirst,
  refinementExhaustive,
  type Selector,
} from './index.ts'

/** attribute=value selectors for every value of every column, as pysubgroup's create_selectors makes them. */
function selectorsOf(columns: Readonly<Record<string, readonly string[]>>): Selector[] {
  return Object.entries(columns).flatMap(([attribute, values]) =>
    [...new Set(values)].sort().map((v) => ({
      name: `${attribute}=${v}`,
      attribute,
      covers: values.map((x) => (x === v ? 1 : 0)),
    })),
  )
}

/** Weighted relative accuracy and pysubgroup's optimistic estimate (the quality of the subgroup's positives alone). */
function wracc(target: readonly number[]): {
  quality: (c: Uint8Array) => number
  estimate: (c: Uint8Array) => number
} {
  const N = target.length
  const P = target.reduce((a, b) => a + b, 0)
  const counts = (c: Uint8Array): [number, number] => {
    let n = 0
    let p = 0
    c.forEach((v, i) => {
      n += v
      p += v * (target[i] ?? 0)
    })
    return [n, p]
  }
  return {
    quality: (c) => {
      const [n, p] = counts(c)
      return p / N - (n * P) / (N * N)
    },
    estimate: (c) => {
      const [, p] = counts(c)
      return (p / N) * (1 - P / N)
    },
  }
}

describe('refinement search against pysubgroup', () => {
  for (const c of loadFixture('search/refinement').cases) {
    it(c.name, () => {
      const p = c.params
      const target = p['target'] as number[]
      const { quality, estimate } = wracc(target)
      const selectors = selectorsOf(p['columns'] as Record<string, string[]>)
      const options = { k: p['k'] as number, maxDepth: p['depth'] as number }
      const expected = p['qualities'] as number[]
      const pruned = conjunctions(selectors, quality, { optimisticEstimate: estimate })
      const plain = conjunctions(selectors, quality)
      for (const r of [
        refinementExhaustive(pruned, options),
        refinementBestFirst(pruned, options),
        refinementExhaustive(plain, options),
      ]) {
        expect(r.status).toBe('complete')
        const got = r.top.map((t) => t.quality)
        expect(got.length).toBe(expected.length)
        got.forEach((q, i) => {
          expect(q).toBeCloseTo(expected[i] ?? 0, 12)
        })
      }
      // Pruning saves work without changing the answer; the beam finds no better patterns than exhaustive search.
      expect(refinementExhaustive(pruned, options).evaluated).toBeLessThanOrEqual(
        refinementExhaustive(plain, options).evaluated,
      )
      const beam = refinementBeamSearch(plain, { ...options, width: 3 })
      beam.top.forEach((t, i) => {
        expect(t.quality).toBeLessThanOrEqual((expected[i] ?? 0) + 1e-12)
      })
    })
  }
})
