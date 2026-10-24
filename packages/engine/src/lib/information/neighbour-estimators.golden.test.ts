import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import type { Tensor } from '../tensor/index.ts'
import * as I from './index.ts'

type Params = Readonly<Record<string, unknown>>
const at = (xs: readonly Tensor[], i: number): Tensor => xs[i] as Tensor

const CALLS: Readonly<Record<string, (xs: readonly Tensor[], p: Params) => number>> = {
  kozachenkoLeonenkoEntropy: (x, p) => I.kozachenkoLeonenkoEntropy(at(x, 0), p),
  ksgMutualInformation: (x, p) => I.ksgMutualInformation(at(x, 0), at(x, 1), p),
  ksgConditionalMutualInformation: (x, p) => I.ksgConditionalMutualInformation(at(x, 0), at(x, 1), at(x, 2), p),
  knnKlDivergence: (x, p) => I.knnKlDivergence(at(x, 0), at(x, 1), p),
  mixedMutualInformation: (x, p) => I.mixedMutualInformation(at(x, 0), at(x, 1), p),
}

describe('information/neighbour-estimators against NPEET and sklearn', () => {
  for (const c of loadFixture('information/neighbour_estimators').cases) {
    it(c.name, () => {
      const call = CALLS[c.fn]
      if (call === undefined) throw new Error(`no function ${c.fn}`)
      const inputs = c.inputs.map((x) => decode(x) as Tensor)
      expectMatches(call(inputs, c.params), c.output, c.rtol, c.atol, c.name)
    })
  }
})
