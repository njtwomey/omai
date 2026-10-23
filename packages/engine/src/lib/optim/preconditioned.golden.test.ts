import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { trajectory } from '../../testing/trajectories.ts'
import type { Tensor } from '../tensor/index.ts'
import * as O from './index.ts'

describe('preconditioners against scipy.linalg', () => {
  for (const c of loadFixture('optim/preconditioned').cases) {
    it(c.name, () => {
      if (c.fn === 'matrixInversePthRoot') {
        const got = O.matrixInversePthRoot(
          decode(c.inputs[0]) as Tensor,
          c.params['p'] as number,
          c.params['ridge'] as number,
        )
        expectMatches(got, c.output, c.rtol, c.atol, c.name)
      } else {
        const opt = O.shampoo({ learningRate: c.params['learningRate'] as number, eps: c.params['eps'] as number })
        expectMatches(trajectory(c, opt).path, c.output, c.rtol, c.atol, c.name)
      }
    })
  }
})
