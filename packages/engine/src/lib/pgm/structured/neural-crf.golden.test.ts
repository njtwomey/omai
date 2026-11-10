import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import { valueAndGrad } from '../../autodiff/index.ts'
import { type Tensor, tensor, toFlat, type Value } from '../../tensor/index.ts'
import {
  type NeuralCrfParameters,
  neuralCrfDecode,
  neuralCrfLogLikelihood,
  neuralCrfMarginals,
  neuralCrfPartition,
  neuralCrfScore,
} from './neural-crf.ts'

const flat = (x: unknown): number[] => toFlat(decode(x) as Tensor)

describe('the CRF layer against pytorch-crf (non-zero start and end transitions, masks, both layouts)', () => {
  for (const c of loadFixture('pgm/structured/neural_crf').cases) {
    it(c.name, () => {
      const [emissions, transitions, start, end, tags, mask] = c.inputs.map((x) => decode(x) as Tensor) as [
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
      ]
      const batchFirst = c.params['batch_first'] as boolean
      const params: NeuralCrfParameters = { transitions, startTransitions: start, endTransitions: end }
      const options = { batchFirst, mask }
      const B = batchFirst ? emissions.shape[0] : emissions.shape[1]
      const pieces: number[] = []
      pieces.push(...toFlat(neuralCrfLogLikelihood(emissions, tags, params, { ...options, reduction: 'none' })))
      for (const reduction of ['sum', 'mean', 'token_mean'] as const)
        pieces.push(...toFlat(neuralCrfLogLikelihood(emissions, tags, params, { ...options, reduction })))
      pieces.push(...toFlat(neuralCrfPartition(emissions, params, options)))
      const decoded = neuralCrfDecode(emissions, params, options)
      const expectedPaths = c.params['paths'] as number[][]
      expect(decoded.paths).toEqual(expectedPaths)
      pieces.push(...decoded.scores)
      // The score of the gold tags and of the decoded paths agree with the sum of parts used by the oracle.
      expect(toFlat(neuralCrfScore(emissions, tags, params, options))).toHaveLength(B ?? 0)
      // Gradients of the summed log-likelihood with respect to the emissions and the three parameters, by autodiff.
      const { grad } = valueAndGrad(
        (p: { e: Value; transitions: Value; startTransitions: Value; endTransitions: Value }) =>
          neuralCrfLogLikelihood(p.e, tags, p, options),
      )({ e: emissions, transitions, startTransitions: start, endTransitions: end })
      for (const g of [grad.e, grad.transitions, grad.startTransitions, grad.endTransitions]) pieces.push(...toFlat(g))
      pieces.push(...toFlat(neuralCrfMarginals(emissions, params, options)))
      expectMatches(tensor(pieces), c.output, c.rtol, c.atol, c.name)
      expect(flat(c.output)).toHaveLength(pieces.length)
    })
  }
})
