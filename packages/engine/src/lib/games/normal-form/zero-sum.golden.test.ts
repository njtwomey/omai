import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import { type Tensor, tensor, toArray } from '../../tensor/index.ts'
import { bimatrixGame, maximin, solveZeroSum } from '../index.ts'

describe('zero-sum games against scipy linprog', () => {
  for (const c of loadFixture('games/normal_form/zero_sum').cases) {
    it(c.name, () => {
      const mats = c.inputs.map((x) => toArray(decode(x) as Tensor) as number[][])
      if (c.fn === 'maximin') {
        const game = bimatrixGame(mats[0] ?? [], mats[1])
        expectMatches(tensor([maximin(game, 0).value, maximin(game, 1).value]), c.output, 0, c.atol, c.name)
        return
      }
      const r = solveZeroSum(mats[0] ?? [])
      expectMatches(r.value, c.output, 0, c.atol, c.name)
      expect(r.value).toBeCloseTo(-(c.params['negated'] as number), 8)
      expect(Math.abs(r.gap)).toBeLessThan(1e-9)
      for (const [i, v] of (c.params['row'] as number[]).entries()) expect(r.row[i]).toBeCloseTo(v, 7)
      for (const [j, v] of (c.params['column'] as number[]).entries()) expect(r.column[j]).toBeCloseTo(v, 7)
    })
  }
})
