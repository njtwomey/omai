import { describe, expect, it } from 'vitest'
import { decode, loadFixture } from '../../testing/golden.ts'
import { toArray } from '../tensor/index.ts'
import { type KgModel, kgScore, rankKg } from './knowledge-graph.ts'

const rows = (x: unknown): Float64Array[] =>
  (toArray(decode(x) as never) as number[][]).map((r) => Float64Array.from(r))

describe('embedding/knowledge-graph against numpy and scipy', () => {
  for (const c of loadFixture('embedding/knowledge_graph').cases) {
    it(c.name, () => {
      if (c.fn === 'kgScore') {
        const e = rows(c.inputs[0])
        const r = rows(c.inputs[1])
        const triples = rows(c.inputs[2])
        const expected = toArray(decode(c.output) as never) as number[]
        for (const [i, t] of triples.entries()) {
          const got = kgScore(
            c.params['model'] as KgModel,
            e[t[0] as number] as Float64Array,
            r[t[1] as number] as Float64Array,
            e[t[2] as number] as Float64Array,
            c.params['norm'] as 1 | 2,
          )
          expect(Math.abs(got - (expected[i] ?? 0))).toBeLessThanOrEqual(c.atol + c.rtol * Math.abs(expected[i] ?? 0))
        }
      } else {
        const s = toArray(decode(c.inputs[0]) as never) as number[]
        const known = c.params['known'] as number[]
        const emb = {
          entities: s.map(() => new Float64Array(1)),
          score: (_h: number, _r: number, t: number) => s[t] as number,
        }
        const got = rankKg(
          emb,
          [[0, 0, c.params['true'] as number]],
          known.map((t) => [0, 0, t] as [number, number, number]).concat([[0, 0, c.params['true'] as number]]),
          { heads: false },
        ).ranks[0]
        expect(got).toBe(c.output)
      }
    })
  }
})
