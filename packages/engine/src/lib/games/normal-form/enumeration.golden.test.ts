import { describe, expect, it } from 'vitest'
import { decode, loadFixture } from '../../../testing/golden.ts'
import { type Tensor, toArray } from '../../tensor/index.ts'
import { bimatrixGame, type Equilibrium, isNash, lemkeHowson, supportEnumeration, vertexEnumeration } from '../index.ts'

type Pair = number[][]

const close = (a: Pair, b: Pair, atol: number): boolean =>
  a.every((s, p) => s.every((v, k) => Math.abs(v - (b[p]?.[k] ?? Number.NaN)) <= atol))

/** Whether every equilibrium of `want` matches a distinct one of `got` within `atol` (and, if `exact`, no more). */
function expectCovers(got: readonly Equilibrium[], want: readonly Pair[], atol: number, exact: boolean): void {
  if (exact) expect(got.length).toBe(want.length)
  const left = got.map((e) => e.strategies.map((s) => [...s]))
  for (const w of want) {
    const i = left.findIndex((g) => close(g, w, atol))
    expect(i, `missing ${JSON.stringify(w)}`).toBeGreaterThanOrEqual(0)
    left.splice(i, 1)
  }
}

describe('bimatrix equilibria against nashpy', () => {
  for (const c of loadFixture('games/normal_form/enumeration').cases) {
    it(c.name, () => {
      const [a, b] = c.inputs.map((x) => toArray(decode(x) as Tensor) as number[][])
      const game = bimatrixGame(a ?? [], b)
      if (c.fn === 'lemkeHowson') {
        for (const [label, want] of (c.params['labels'] as Pair[]).entries()) {
          const r = lemkeHowson(game, { initialLabel: label })
          expect(r.status).toBe('equilibrium')
          expect(isNash(game, r.strategies, 1e-9)).toBe(true)
          for (const [p, s] of want.entries())
            for (const [k, v] of s.entries()) expect(r.strategies[p]?.[k]).toBeCloseTo(v, 9)
        }
        return
      }
      const found = c.fn === 'supportEnumeration' ? supportEnumeration(game) : vertexEnumeration(game)
      if (c.fn === 'supportEnumeration') {
        expectCovers(found, c.params['equilibria'] as Pair[], c.atol, false)
        expectCovers(found, c.params['all'] as Pair[], 1e-8, true)
      } else expectCovers(found, c.params['equilibria'] as Pair[], c.atol, true)
      for (const e of found) expect(isNash(game, e.strategies, 1e-9)).toBe(true)
    })
  }
})
