import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { type Tensor, tensor, toFlat } from '../tensor/index.ts'
import { oneClassSvm, svdd } from './one-class.ts'

const cases = loadFixture('svm/one_class').cases

describe('oneClassSvm against scikit-learn (libsvm)', () => {
  for (const c of cases) {
    it(c.name, () => {
      const out = c.output as Readonly<Record<string, unknown>>
      const x = decode(c.inputs[0]) as Tensor
      const xt = decode(c.inputs[1]) as Tensor
      const p = c.params as Record<string, unknown>
      const fitted = oneClassSvm
        .with({
          nu: p['nu'] as number,
          gamma: p['gamma'] as number,
          tol: 1e-8,
          ...(p['kernel'] === undefined ? {} : { kernel: p['kernel'] as 'poly' }),
          ...(p['degree'] === undefined ? {} : { degree: p['degree'] as number }),
          ...(p['coef0'] === undefined ? {} : { coef0: p['coef0'] as number }),
        })
        .fit({ x }, { explain: false })
      expectMatches(tensor(fitted.params.supportIndex), out['support'], 0, 0, `${c.name} support_`)
      expectMatches(tensor([fitted.params.coef]), out['dualCoef'], c.rtol, c.atol, `${c.name} dual_coef_`)
      expectMatches(tensor([-fitted.params.rho]), out['intercept'], c.rtol, c.atol, `${c.name} intercept_`)
      expectMatches(fitted.decisionFunction(xt), out['decision'], c.rtol, c.atol, `${c.name} decision_function`)
      expectMatches(fitted.predict(xt), out['predict'], 0, 0, `${c.name} predict`)
    })
  }
})

describe('svdd', () => {
  it('is the one-class machine on a kernel of unit diagonal, with C = 1 / (nu n)', () => {
    for (const c of cases.filter((k) => k.params['kernel'] === undefined)) {
      const x = decode(c.inputs[0]) as Tensor
      const xt = decode(c.inputs[1]) as Tensor
      const n = x.shape[0] ?? 0
      const nu = c.params['nu'] as number
      const gamma = c.params['gamma'] as number
      const ball = svdd.with({ C: 1 / (nu * n), gamma, tol: 1e-8 }).fit({ x }, { explain: false })
      const machine = oneClassSvm.with({ nu, gamma, tol: 1e-8 }).fit({ x }, { explain: false })
      expect(Array.from(toFlat(ball.predict(xt)))).toEqual(Array.from(toFlat(machine.predict(xt))))
      expect(ball.params.supportIndex).toEqual(machine.params.supportIndex)
    }
  })
})
