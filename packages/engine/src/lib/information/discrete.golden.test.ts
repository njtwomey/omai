import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import type { Tensor, Value } from '../tensor/index.ts'
import * as I from './index.ts'

type Params = Readonly<Record<string, unknown>>
type Call = (xs: readonly Tensor[], p: Params) => Value
const at = (xs: readonly Tensor[], i: number): Tensor => xs[i] as Tensor

const CALLS: Readonly<Record<string, Call>> = {
  entropy: (x, p) => I.entropy(at(x, 0), p),
  binaryEntropy: (x, p) => I.binaryEntropy(at(x, 0), p),
  relativeEntropy: (x, p) => I.relativeEntropy(at(x, 0), at(x, 1), p),
  crossEntropy: (x, p) => I.crossEntropy(at(x, 0), at(x, 1), p),
  jensenShannonDivergence: (x, p) => I.jensenShannonDivergence(at(x, 0), at(x, 1), p),
  jensenShannonDistance: (x, p) => I.jensenShannonDistance(at(x, 0), at(x, 1), p),
  totalVariation: (x, p) => I.totalVariation(at(x, 0), at(x, 1), p),
  mutualInformation: (x, p) => I.mutualInformation(at(x, 0), p),
  jointEntropy: (x, p) => I.jointEntropy(at(x, 0), p),
  conditionalEntropy: (x, p) => I.conditionalEntropy(at(x, 0), p['given'] as number | number[], p),
  conditionalMutualInformation: (x, p) => I.conditionalMutualInformation(at(x, 0), p),
  interactionInformation: (x, p) => I.interactionInformation(at(x, 0), p),
  informationGain: (x, p) => I.informationGain(at(x, 0), p),
  gainRatio: (x) => I.gainRatio(at(x, 0)),
}

describe('information/discrete against scipy, sklearn and pyitlib', () => {
  for (const c of loadFixture('information/discrete').cases) {
    it(c.name, () => {
      const call = CALLS[c.fn]
      if (call === undefined) throw new Error(`no function ${c.fn}`)
      expectMatches(
        call(
          c.inputs.map((x) => decode(x) as Tensor),
          c.params,
        ),
        c.output,
        c.rtol,
        c.atol,
        c.name,
      )
    })
  }
})
