import { describe, it, vi } from 'vitest'
import { categoricalPolicy, gaussianPolicy } from '../../../lib/rl/index.ts'
import { type Tensor, tensor, toFlat } from '../../../lib/tensor/index.ts'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import { policyKl } from './_on-policy.ts'
import { ppoClipObjective, ppoKlObjective, trpoStep } from './policy-gradient.ts'

// Learning runs take seconds alone; the timeout only guards against a heavily loaded machine.
vi.setConfig({ testTimeout: 60_000 })

const matvec = (m: Tensor) => {
  const n = m.shape[0] ?? 0
  const a = toFlat(m)
  return (v: readonly number[]): number[] =>
    Array.from({ length: n }, (_, i) => {
      let s = 0
      for (let j = 0; j < n; j++) s += (a[i * n + j] ?? 0) * (v[j] ?? 0)
      return s
    })
}

describe('policy-gradient objectives against torch', () => {
  for (const c of loadFixture('methods/rl/agents/policy_gradient').cases) {
    it(c.name, () => {
      const xs = c.inputs.map((v) => decode(v) as Tensor)
      const [a, b, d, e, f, g] = xs as [Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
      const beta = c.params['beta'] as number
      let got: Tensor
      switch (c.fn) {
        case 'ppoClipObjective':
          got = ppoClipObjective(a, b, d, c.params['clipRange'] as number) as Tensor
          break
        case 'ppoKlCategorical': {
          const kl = policyKl({ kind: 'categorical', logits: a }, { kind: 'categorical', logits: b })
          const lp = categoricalPolicy(b).logProb(d)
          const old = categoricalPolicy(a).logProb(d)
          got = ppoKlObjective(lp, old, e, kl, beta) as Tensor
          break
        }
        case 'klCategorical':
          got = policyKl({ kind: 'categorical', logits: a }, { kind: 'categorical', logits: b }) as Tensor
          break
        case 'ppoKlGaussian': {
          const kl = policyKl({ kind: 'gaussian', mean: a, logStd: b }, { kind: 'gaussian', mean: d, logStd: e })
          const lp = gaussianPolicy(d, e).logProb(f)
          const old = gaussianPolicy(a, b).logProb(f)
          got = ppoKlObjective(lp, old, g, kl, beta) as Tensor
          break
        }
        case 'klGaussian':
          got = policyKl({ kind: 'gaussian', mean: a, logStd: b }, { kind: 'gaussian', mean: d, logStd: e }) as Tensor
          break
        default: {
          const opts = { maxKl: c.params['maxKl'] as number, damping: c.params['damping'] as number, cgIterations: 6 }
          const r = trpoStep(matvec(a), toFlat(b), opts)
          got = tensor(c.fn === 'trpoDirection' ? r.direction : r.step)
        }
      }
      expectMatches(got, c.output, c.rtol, c.atol, c.name)
    })
  }
})
