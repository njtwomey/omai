import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import type { Tensor } from '../tensor/index.ts'
import { lqr } from './control.ts'

describe('LQR against scipy.linalg.solve_discrete_are', () => {
  for (const c of loadFixture('methods/rl/agents/control').cases) {
    it(c.name, () => {
      const [a, b, q, r] = c.inputs.map((x) => decode(x) as Tensor)
      const sol = lqr(a as Tensor, b as Tensor, q as Tensor, r as Tensor)
      expectMatches(c.fn === 'lqrP' ? sol.P : sol.K, c.output, c.rtol, c.atol, c.name)
    })
  }
})
