import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { type Tensor, tensor } from '../tensor/index.ts'
import * as O from './index.ts'

type Out = Readonly<Record<string, unknown>>

describe('linear SVM dual coordinate descent against liblinear (scikit-learn LinearSVC)', () => {
  for (const c of loadFixture('optim/dual_cd').cases) {
    for (const shrinking of [true, false]) {
      it(`${c.name}${shrinking ? ', shrinking' : ''}`, () => {
        const r = O.linearSvmDualCd(
          decode(c.inputs[0]) as Tensor,
          decode(c.inputs[1]) as Tensor,
          c.params['C'] as number,
          {
            loss: c.params['loss'] as O.DualCdLoss,
            tol: 1e-10,
            shrinking,
            maxSweeps: 100000,
          },
        )
        if (!r.converged) throw new Error('dual coordinate descent did not converge')
        const out = c.output as Out
        expectMatches(tensor(Array.from(r.coef)), out['coef'], c.rtol, c.atol, `${c.name} coef`)
        expectMatches(r.intercept, out['intercept'], c.rtol, c.atol, `${c.name} intercept`)
      })
    }
  }
})
