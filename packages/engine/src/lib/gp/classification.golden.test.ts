import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { constantKernel, maternKernel, productKernel, rbfKernel, theta } from '../kernels/index.ts'
import { type Tensor, tensor, toFlat } from '../tensor/index.ts'
import { laplaceMarginal } from './approximations.ts'
import { gpClassifier } from './classification.ts'

describe('gpClassifier against scikit-learn (Laplace) and a GPML reference (EP)', () => {
  for (const c of loadFixture('gp/classification').cases) {
    it(c.name, () => {
      const out = c.output as Readonly<Record<string, unknown>>
      const x = decode(c.inputs[0]) as Tensor
      const y = decode(c.inputs[1]) as Tensor
      const xt = decode(c.inputs[2]) as Tensor
      const p = c.params as { constant: number; lengthScale: number; family: string; optimise?: boolean }
      const base =
        p.family === 'rbf'
          ? rbfKernel({ lengthScale: p.lengthScale })
          : maternKernel({ lengthScale: p.lengthScale, nu: 2.5 })
      const kernel = productKernel(constantKernel({ value: p.constant }), base)
      const ep = c.name.startsWith('ep/')
      const est = gpClassifier.with({
        kernel,
        method: ep ? 'ep' : 'laplace',
        likelihood: ep ? 'probit' : 'logit',
        optimise: p.optimise === true,
        tol: ep ? 1e-8 : 1e-10,
      })
      const fitted = est.fit({ x, y }, { explain: false })
      expectMatches(fitted.params.logMarginalLikelihood, out['lml'], c.rtol, c.atol, `${c.name} lml`)
      expectMatches(fitted.predictProba(xt), out['proba'], c.rtol, c.atol, `${c.name} proba`)
      if (out['gradient'] !== undefined) {
        const t = Array.from(toFlat(theta(kernel)))
        const labels = Array.from(toFlat(y), (v) => (v === toFlat(y)[0] ? 0 : 1))
        const g = laplaceMarginal(kernel, x, labels, { theta: t })
        expectMatches(tensor(Array.from(g.gradient)), out['gradient'], c.rtol, c.atol, `${c.name} gradient`)
      }
      if (out['latentMean'] !== undefined) {
        const { mean, variance } = fitted.latent(xt)
        expectMatches(mean, out['latentMean'], c.rtol, c.atol, `${c.name} latent mean`)
        expectMatches(variance, out['latentVariance'], c.rtol, c.atol, `${c.name} latent variance`)
      }
      if (out['theta'] !== undefined)
        expectMatches(tensor(fitted.params.theta), out['theta'], c.rtol, 1e-2, `${c.name} theta`)
      if (out['tau'] !== undefined && fitted.params.posterior !== undefined && 'tau' in fitted.params.posterior) {
        expectMatches(tensor(fitted.params.posterior.tau), out['tau'], c.rtol, c.atol, `${c.name} tau`)
        expectMatches(tensor(fitted.params.posterior.nu), out['nu'], c.rtol, c.atol, `${c.name} nu`)
      }
    })
  }
})
