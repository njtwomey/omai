import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import type { Tensor } from '../../tensor/index.ts'
import { correlatedEquilibrium, isCorrelatedEquilibrium, normalFormGame } from '../index.ts'

describe('correlated equilibria against scipy linprog', () => {
  for (const c of loadFixture('games/normal_form/correlated').cases) {
    it(c.name, () => {
      const game = normalFormGame(c.inputs.map((x) => decode(x) as Tensor))
      const coarse = c.params['coarse'] as boolean
      const r = correlatedEquilibrium(game, { coarse, objective: c.params['objective'] as 'welfare' | 'egalitarian' })
      expectMatches(r.value, c.output, 0, c.atol, c.name)
      expect(isCorrelatedEquilibrium(game, r.distribution, { coarse, tol: 1e-9 })).toBe(true)
    })
  }
})
