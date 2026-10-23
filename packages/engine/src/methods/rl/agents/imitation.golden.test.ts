import { describe, it } from 'vitest'
import type { TabularMdp } from '../../../lib/rl/index.ts'
import { type Tensor, tensor, toFlat } from '../../../lib/tensor/index.ts'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import { maxEntIrl, softValueIteration, stateVisitationFrequencies } from './imitation.ts'

const rows = (t: Tensor, n: number): number[][] => {
  const flat = toFlat(t)
  return Array.from({ length: flat.length / n }, (_, i) => flat.slice(i * n, (i + 1) * n))
}

describe('maximum-entropy IRL against numpy', () => {
  for (const c of loadFixture('methods/rl/agents/imitation').cases) {
    it(c.name, () => {
      const [p, t, mu0, reward, extra] = c.inputs.map((v) => decode(v) as Tensor) as [
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
      ]
      const [nS = 0, nA = 0] = p.shape
      const gamma = c.params['gamma'] as number
      const mdp: TabularMdp = {
        nStates: nS,
        nActions: nA,
        transitions: Float64Array.from(toFlat(p)),
        terminal: Float64Array.from(toFlat(t)),
        rewards: new Float64Array(nS * nA),
        initial: Float64Array.from(toFlat(mu0)),
        gamma,
      }
      const r = toFlat(reward)
      let got: Tensor
      switch (c.fn) {
        case 'softV':
          got = tensor(softValueIteration(mdp, r).V)
          break
        case 'softPolicy':
          got = tensor(softValueIteration(mdp, r).policy)
          break
        case 'svfFinite':
          got = tensor(stateVisitationFrequencies(mdp, rows(extra, nA), { horizon: c.params['horizon'] as number }))
          break
        case 'svfDiscounted':
          got = tensor(stateVisitationFrequencies(mdp, rows(extra, nA)))
          break
        default:
          got = tensor([...maxEntIrl(mdp, rows(extra, 3), { maxIter: 1, learningRate: 0.1 }).theta])
      }
      expectMatches(got, c.output, c.rtol, c.atol, c.name)
    })
  }
})
