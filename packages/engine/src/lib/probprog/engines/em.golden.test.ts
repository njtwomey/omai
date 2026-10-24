import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import { categorical, normal } from '../../dists/index.ts'
import { key } from '../../random/index.ts'
import { add, mul, sqrt, type Tensor, take } from '../../tensor/index.ts'
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

/** One-factor analysis: x_i = mu + w z_i + noise with diagonal variances psi. */
const factorModel = (x: Tensor): void => {
  const w = param('w', [0.5, 0.5, 0.5])
  const mu = param('mu', [0, 0, 0])
  const psi = param('psi', [1, 1, 1], { support: { kind: 'positive' } })
  plate(
    'n',
    x.shape[0] ?? 0,
    () => {
      const z = sample('z', normal(0, 1))
      observe('x', normal(add(mu, mul(z, w)), sqrt(psi)), x)
    },
    { dim: -2 },
  )
}

describe('probprog/engines/em against scikit-learn', () => {
  for (const c of loadFixture('probprog/engines/em').cases) {
    it(c.name, () => {
      if (c.fn === 'factorAnalysis') {
        const x = decode(c.inputs[0]) as Tensor
        const post = em(factorModel, x, { key: key(0), maxIter: 500, tol: 1e-9 })
        expect(post.diagnostics['eStep']).toBe('gaussian')
        // The factors' posterior given the estimates (a multivariate normal over the site's flattened coordinates).
        expect(post.mean('z').shape).toEqual([x.shape[0] ?? 0])
        expect(post.sample(key(1), 3)['z']?.shape).toEqual([3, x.shape[0] ?? 0, 1])
        expectMatches(post.diagnostics['objective'] as number, c.output, c.rtol, c.atol, c.name)
        return
      }
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
