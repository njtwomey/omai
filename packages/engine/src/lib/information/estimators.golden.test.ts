import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import type { Tensor, Value } from '../tensor/index.ts'
import * as I from './index.ts'

type Params = Readonly<Record<string, unknown>>
const at = (xs: readonly Tensor[], i: number): Tensor => xs[i] as Tensor

const CALLS: Readonly<Record<string, (xs: readonly Tensor[], p: Params) => Value | number>> = {
  plugInEntropy: (x, p) => I.plugInEntropy(at(x, 0), p),
  plugInMutualInformation: (x, p) => I.plugInMutualInformation(at(x, 0), at(x, 1), p),
  kdeEntropy: (x, p) => I.kdeEntropy(at(x, 0), p),
  kdeMutualInformation: (x, p) => I.kdeMutualInformation(at(x, 0), at(x, 1), p),
  transferEntropy: (x, p) => I.transferEntropy(at(x, 0), at(x, 1), p),
  infoNce: (x) => I.infoNce(at(x, 0)),
  nwjBound: (x) => I.nwjBound(at(x, 0)),
  mineBound: (x) => I.mineBound(at(x, 0)),
}

describe('information/estimators against scipy, sklearn, pyitlib and NPEET', () => {
  for (const c of loadFixture('information/estimators').cases) {
    it(c.name, () => {
      const call = CALLS[c.fn]
      if (call === undefined) throw new Error(`no function ${c.fn}`)
      const inputs = c.inputs.map((x) => decode(x) as Tensor)
      expectMatches(call(inputs, c.params), c.output, c.rtol, c.atol, c.name)
    })
  }
})
