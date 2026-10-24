import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { type Tensor, tensor, toFlat } from '../tensor/index.ts'
import { discountedReturns, gae, lambdaReturns, nStepReturns, retrace, vtrace } from './returns.ts'

describe('returns and advantages against numpy, scipy and the stable-baselines3 and rlax recurrences', () => {
  for (const c of loadFixture('rl/returns').cases) {
    it(c.name, () => {
      const x = c.inputs.map((v) => toFlat(decode(v) as Tensor))
      const [a = [], b = [], d = [], e = [], f = []] = x
      const gamma = (c.params['gamma'] as number | undefined) ?? 0.97
      const lam = (c.params['lam'] as number | undefined) ?? 0.95
      const n = a.length
      let got: number[]
      switch (c.fn) {
        case 'discountedReturns':
          got = discountedReturns(a, gamma)
          break
        case 'discountedReturnsDones':
          got = discountedReturns(a, gamma, { dones: b, bootstrap: c.params['bootstrap'] as number })
          break
        case 'gae':
          got = gae(a, b.slice(0, n), d, b[n] ?? 0, gamma, lam).advantages
          break
        case 'lambdaReturns':
          got = lambdaReturns(a, b.slice(1), d, gamma, lam)
          break
        case 'nStepReturns':
          got = nStepReturns(a, b, d, gamma, c.params['n'] as number)
          break
        case 'vtraceVs':
        case 'vtracePg': {
          const r = vtrace(a, b, d.slice(0, n), d[n] ?? 0, e, { lambda: lam })
          got = c.fn === 'vtraceVs' ? r.vs : r.pgAdvantages
          break
        }
        default:
          got = retrace(a, b, d, e, f)
      }
      expectMatches(tensor(got), c.output, Math.max(c.rtol, 1e-12), Math.max(c.atol, 1e-12), c.name)
    })
  }
})
