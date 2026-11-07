// Bit-for-bit pins of the sample streams, recorded from the code before `uniformStream` computed its Philox blocks
// lazily: for a given key the draws must never change. Each row is a checksum (FNV-1a over the 32-bit words of the
// float64 values) of the whole array, and its first value.
import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { type Tensor, tensor, toFlat } from '../tensor/index.ts'
import * as D from './index.ts'

function checksum(values: ArrayLike<number>): string {
  const words = new Uint32Array(Float64Array.from(values).buffer)
  let h = 0x811c9dc5
  for (const w of words) h = Math.imul(h ^ w, 0x01000193) >>> 0
  return h.toString(16).padStart(8, '0')
}

const FAMILIES: Readonly<Record<string, () => D.Distribution>> = {
  gamma: () => D.gamma(tensor([0.4, 2.5, 30]), 2),
  beta: () => D.beta(tensor([0.5, 2]), 3),
  studentT: () => D.studentT(tensor([3, 10]), 1, 2),
  chi2: () => D.chi2(tensor([1, 4, 9])),
  dirichlet: () => D.dirichlet(tensor([0.5, 1, 3])),
  poisson: () => D.poisson(tensor([0.5, 4, 60])),
  binomial: () => D.binomial(tensor([10, 100]), tensor([0.3, 0.6])),
  geometric: () => D.geometric(tensor([0.2, 0.7])),
  negativeBinomial: () => D.negativeBinomial(tensor([3, 8]), tensor([0.3, 0.7])),
  fDistribution: () => D.fDistribution(tensor([3, 8]), 5),
}

const PINS: readonly (readonly [string, string, number, number])[] = [
  ['gamma(0)', '2a678921', 75, 0.8596460295376894],
  ['beta(0)', '99ca12d4', 50, 0.16876078138864833],
  ['studentT(0)', 'b1f31719', 50, 3.2133095782581784],
  ['chi2(0)', 'ed4748ba', 75, 3.984458794539752],
  ['dirichlet(0)', '6bf2fc77', 75, 0.2172620814294898],
  ['poisson(0)', '0e6a72dd', 75, 0],
  ['binomial(0)', '2934c095', 50, 5],
  ['geometric(0)', '67de4095', 50, 4],
  ['negativeBinomial(0)', 'b2c04095', 50, 13],
  ['fDistribution(0)', 'a3f3b92f', 50, 1.3974022807142097],
  ['gamma(7)', 'e82ff008', 75, 0.026327441547930486],
  ['beta(7)', '9b37cf5c', 50, 0.0019937898731295844],
  ['studentT(7)', '4b32d8b3', 50, -2.1597614994870513],
  ['chi2(7)', '67abe961', 75, 0.2220458964492546],
  ['dirichlet(7)', 'f909639b', 75, 0.021410071519533405],
  ['poisson(7)', '34ecb2dd', 75, 0],
  ['binomial(7)', '01e70095', 50, 2],
  ['geometric(7)', 'a95a4095', 50, 6],
  ['negativeBinomial(7)', 'cfb24095', 50, 3],
  ['fDistribution(7)', '4d7c0855', 50, 0.8396332853280354],
  ['gamma(12345)', 'daed2d5e', 75, 0.4290027431157897],
  ['beta(12345)', 'b9d42690', 50, 0.05618313835589769],
  ['studentT(12345)', '86272cea', 50, 1.4807299502698412],
  ['chi2(12345)', 'e6f83142', 75, 2.148919909988445],
  ['dirichlet(12345)', '753238fc', 75, 0.22340878496614208],
  ['poisson(12345)', 'f87a32dd', 75, 1],
  ['binomial(12345)', '496a8095', 50, 2],
  ['geometric(12345)', '55ec4095', 50, 7],
  ['negativeBinomial(12345)', '34164095', 50, 13],
  ['fDistribution(12345)', 'd231205f', 50, 3.787505615478361],
]

describe('distribution samples are bit-for-bit stable', () => {
  it.each(PINS)('%s', (name, hash, length, first) => {
    const [family = '', seed = ''] = name.slice(0, -1).split('(')
    const make = FAMILIES[family]
    if (make === undefined) throw new Error(`no family named ${family}`)
    const values = toFlat(make().sample(key(Number(seed)), [25]) as Tensor)
    expect(values.length).toBe(length)
    expect(values[0]).toBe(first)
    expect(checksum(values)).toBe(hash)
  })
})
