import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import { item, type Tensor, tensor, toFlat } from '../../tensor/index.ts'
import { betaBernoulli, dirichletCategorical, gammaPoisson, normalNormal } from './exact.ts'

describe('probprog/validation/exact against scipy', () => {
  for (const c of loadFixture('probprog/validation/exact').cases) {
    it(c.name, () => {
      const input = c.inputs.map((x) => toFlat(decode(x) as Tensor))
      const y = input[0] ?? []
      let got: number[]
      if (c.fn === 'normalNormal') {
        const e = normalNormal({
          priorMean: c.params['priorMean'] as number,
          priorSd: c.params['priorSd'] as number,
          sigma: c.params['sigma'] as number,
          y,
        })
        got = [e.logEvidence, item(e.posterior.mean()), item(e.posterior.stddev())]
      } else if (c.fn === 'betaBernoulli') {
        const e = betaBernoulli({ a: c.params['a'] as number, b: c.params['b'] as number, y })
        got = [e.logEvidence, item(e.posterior.mean()), item(e.posterior.stddev())]
      } else if (c.fn === 'gammaPoisson') {
        const e = gammaPoisson({ shape: c.params['shape'] as number, rate: c.params['rate'] as number, counts: y })
        got = [e.logEvidence, item(e.posterior.mean()), item(e.posterior.stddev())]
      } else if (c.fn === 'dirichletCategorical') {
        const e = dirichletCategorical({ alpha: y, y: input[1] ?? [] })
        got = [e.logEvidence, ...toFlat(e.posterior.mean())]
      } else throw new Error(`no function ${c.fn}`)
      expectMatches(tensor(got), c.output, c.rtol, c.atol, c.name)
    })
  }
})
