import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import { categorical, normal } from '../../dists/index.ts'
import { key } from '../../random/index.ts'
import { mul, type Tensor, take } from '../../tensor/index.ts'
import { observe, param, plate, sample } from '../primitives.ts'
import { em } from './em.ts'

const gmm = (y: Tensor): void => {
  const w = param('w', [0.5, 0.5], { support: { kind: 'simplex' } })
  const mu = param('mu', [-1, 1])
  const sigma = param('sigma', [1, 1], { support: { kind: 'positive' } })
  plate('n', y.shape[0] ?? 0, () => {
    const z = sample('z', categorical({ probs: w }))
    observe('y', normal(take(mu, z), take(sigma, z)), y)
  })
}

describe('probprog/engines/em against scikit-learn', () => {
  for (const c of loadFixture('probprog/engines/em').cases) {
    it(c.name, () => {
      const y = decode(c.inputs[0]) as Tensor
      const post = em(gmm, y, { key: key(0), maxIter: 5000, tol: 1e-14 })
      if (c.fn === 'gmmLogLikelihood') {
        expectMatches(post.diagnostics['objective'] as number, c.output, c.rtol, c.atol, c.name)
        return
      }
      const out = c.output as Record<string, unknown>
      expectMatches(post.mean('w'), out['weights'], c.rtol, c.atol, `${c.name}.weights`)
      expectMatches(post.mean('mu'), out['means'], c.rtol, c.atol, `${c.name}.means`)
      const s = post.mean('sigma')
      expectMatches(mul(s, s) as Tensor, out['variances'], c.rtol, c.atol, `${c.name}.variances`)
    })
  }
})
