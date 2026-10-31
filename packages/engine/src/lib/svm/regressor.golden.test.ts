import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { tensor } from '../tensor/index.ts'
import { type SvrHyper, svr } from './regressor.ts'

describe('svr against scikit-learn (libsvm)', () => {
  for (const c of loadFixture('svm/regressor').cases) {
    it(c.name, () => {
      const out = c.output as Readonly<Record<string, unknown>>
      const x = decode(c.inputs[0]) as never
      const y = decode(c.inputs[1]) as never
      const xt = decode(c.inputs[2]) as never
      const weights = c.inputs[3] === undefined ? undefined : (decode(c.inputs[3]) as never)
      const p = c.params as Record<string, unknown>
      const hyper: Partial<SvrHyper> = {
        form: p['form'] as 'epsilon' | 'nu',
        tol: 1e-8,
        C: p['C'] as number,
        ...(p['epsilon'] === undefined ? {} : { epsilon: p['epsilon'] as number }),
        ...(p['nu'] === undefined ? {} : { nu: p['nu'] as number }),
        kernel: p['kernel'] as 'rbf',
        ...(p['gamma'] === undefined ? {} : { gamma: p['gamma'] as number | 'scale' }),
        ...(p['degree'] === undefined ? {} : { degree: p['degree'] as number }),
        ...(p['coef0'] === undefined ? {} : { coef0: p['coef0'] as number }),
      }
      const fitted = svr.with(hyper).fit({ x, y, ...(weights === undefined ? {} : { weights }) }, { explain: false })
      expectMatches(tensor(fitted.params.supportIndex), out['support'], 0, 0, `${c.name} support_`)
      expectMatches(tensor([fitted.params.coef]), out['dualCoef'], c.rtol, c.atol, `${c.name} dual_coef_`)
      expectMatches(tensor([fitted.params.intercept]), out['intercept'], c.rtol, c.atol, `${c.name} intercept_`)
      expectMatches(fitted.predict(xt), out['predict'], c.rtol, c.atol, `${c.name} predict`)
    })
  }
})
