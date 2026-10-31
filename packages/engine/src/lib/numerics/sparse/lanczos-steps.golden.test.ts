import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import { type Tensor, tensor } from '../../tensor/index.ts'
import * as S from './index.ts'

describe('stepped Lanczos against a dense full-reorthogonalisation Lanczos and scipy.linalg', () => {
  for (const c of loadFixture('numerics/sparse/lanczos').cases) {
    it(c.name, () => {
      const t = c.inputs.map((v) => decode(v) as Tensor)
      const steps = c.params['steps'] as number
      const state = S.lanczosSteps(S.csr(t[0] as Tensor), steps, { v0: Array.from((t[1] as Tensor).data) })
      const actual = c.fn === 'alpha' ? state.alpha : c.fn === 'beta' ? state.beta : state.ritzValues
      expectMatches(tensor([...actual]), c.output, c.rtol, c.atol, c.name)
    })
  }
})
