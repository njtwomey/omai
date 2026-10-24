import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { type Tensor, tensor, toArray } from '../tensor/index.ts'
import { analogy, cooccurrence, lsa, nearestWords, pmi, svdWordVectors } from './index.ts'

const rows = (x: unknown): Float64Array[] =>
  (toArray(decode(x) as Tensor) as number[][]).map((r) => Float64Array.from(r))
const flat = (rs: readonly Float64Array[]): number[][] => rs.map((r) => Array.from(r))

describe('text/representations against numpy and gensim', () => {
  for (const c of loadFixture('text/representations').cases) {
    it(c.name, () => {
      switch (c.fn) {
        case 'cooccurrence': {
          const m = cooccurrence(c.inputs[0] as string[][], {
            window: c.params['window'] as number,
            weighting: c.params['harmonic'] === true ? 'harmonic' : 'uniform',
          })
          expect(m.words).toEqual(c.params['words'])
          expectMatches(tensor(flat(m.rows)), c.output, 0, 1e-12, c.name)
          return
        }
        case 'pmi': {
          const m = rows(c.inputs[0])
          const p = pmi(
            { words: m.map((_, i) => String(i)), rows: m },
            {
              alpha: c.params['alpha'] as number,
              positive: c.params['positive'] as boolean,
              shift: c.params['shift'] as number,
            },
          )
          expectMatches(tensor(flat(p.rows)), c.output, c.rtol, c.atol, c.name)
          return
        }
        case 'lsa': {
          const r = lsa(rows(c.inputs[0]), c.params['k'] as number)
          expectMatches(
            tensor([...r.singularValues, ...flat(r.documents).flat(), ...flat(r.terms).flat()]),
            c.output,
            c.rtol,
            c.atol,
            c.name,
          )
          return
        }
        case 'svdWordVectors': {
          const m = rows(c.inputs[0])
          const v = svdWordVectors({ words: m.map((_, i) => String(i)), rows: m }, c.params['k'] as number)
          expectMatches(tensor(flat(v.rows)), c.output, c.rtol, c.atol, c.name)
          return
        }
        default: {
          const words = c.inputs[0] as string[]
          const v = { words, rows: rows(c.inputs[1]) }
          const expected = c.output as Record<string, [string, number][]>
          const check = (
            got: readonly { word: string; similarity: number }[],
            want: [string, number][] | undefined,
          ): void => {
            expect(got.map((g) => g.word)).toEqual(want?.map((w) => w[0]))
            // gensim stores vectors in float32.
            for (const [i, g] of got.entries()) {
              const w = want?.[i]?.[1] ?? 0
              expect(Math.abs(g.similarity - w)).toBeLessThan(1e-6 * Math.max(1, Math.abs(w)))
            }
          }
          check(nearestWords(v, 'king', 4), expected['near'])
          check(analogy(v, 'man', 'woman', 'king', { topn: 3 }), expected['add'])
          check(analogy(v, 'man', 'woman', 'king', { topn: 3, method: '3cosmul' }), expected['mul'])
          check(analogy(v, 'paris', 'france', 'rome', { topn: 3 }), expected['add2'])
        }
      }
    })
  }
})
