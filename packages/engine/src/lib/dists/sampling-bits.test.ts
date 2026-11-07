// Bit-for-bit pins of the sample streams, recorded from the code before `uniformStream` computed its Philox blocks
// lazily: for a given key the draws must never change. Each row is a checksum of the whole array (see `checksum`) and its first value.
import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { type Tensor, tensor, toFlat } from '../tensor/index.ts'
import * as D from './index.ts'

/**
 * The checksum of a sample: FNV-1a over the 32-bit words of its float64 values. The raw stream and integer-valued draws
 * are hashed bit for bit. The others are built from `log`, `pow` and `sqrt`, whose last digit differs between platforms
 * (macOS against the Linux runner), so their values are rounded to 9 significant digits first: a change of the stream
 * changes them at the first digit, a platform's rounding at the sixteenth.
 */
function checksum(values: ArrayLike<number>, rounded: boolean): string {
  const source = rounded ? Array.from(values, (v) => Number(v.toPrecision(9))) : Array.from(values)
  const words = new Uint32Array(Float64Array.from(source).buffer)
  let h = 0x811c9dc5
  for (const w of words) h = Math.imul(h ^ w, 0x01000193) >>> 0
  return h.toString(16).padStart(8, '0')
}

/** Whether a pinned sample comes through libm (so it is compared to a tolerance): not a raw stream, not integer-valued. */
const roundedPin = (name: string, values: ArrayLike<number>): boolean =>
  !name.startsWith('stream') && !Array.from(values).every((v) => Number.isInteger(v))

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
  ['gamma(0)', '9dd9f069', 75, 0.8596460295376894],
  ['beta(0)', '45f9ded7', 50, 0.16876078138864833],
  ['studentT(0)', '392656b9', 50, 3.2133095782581784],
  ['chi2(0)', 'fb834c2c', 75, 3.984458794539752],
  ['dirichlet(0)', 'eaf28009', 75, 0.2172620814294898],
  ['poisson(0)', '0e6a72dd', 75, 0],
  ['binomial(0)', '2934c095', 50, 5],
  ['geometric(0)', '67de4095', 50, 4],
  ['negativeBinomial(0)', 'b2c04095', 50, 13],
  ['fDistribution(0)', '38aa8be1', 50, 1.3974022807142097],
  ['gamma(7)', '6f492f64', 75, 0.026327441547930486],
  ['beta(7)', '5a106792', 50, 0.0019937898731295844],
  ['studentT(7)', '0166da57', 50, -2.1597614994870513],
  ['chi2(7)', 'efec4f95', 75, 0.2220458964492546],
  ['dirichlet(7)', 'a599e7e5', 75, 0.021410071519533405],
  ['poisson(7)', '34ecb2dd', 75, 0],
  ['binomial(7)', '01e70095', 50, 2],
  ['geometric(7)', 'a95a4095', 50, 6],
  ['negativeBinomial(7)', 'cfb24095', 50, 3],
  ['fDistribution(7)', '923dc12b', 50, 0.8396332853280354],
  ['gamma(12345)', '47c655f0', 75, 0.4290027431157897],
  ['beta(12345)', '9a8b2784', 50, 0.05618313835589769],
  ['studentT(12345)', 'f91879f4', 50, 1.4807299502698412],
  ['chi2(12345)', '1121420e', 75, 2.148919909988445],
  ['dirichlet(12345)', '61e572f8', 75, 0.22340878496614208],
  ['poisson(12345)', 'f87a32dd', 75, 1],
  ['binomial(12345)', '496a8095', 50, 2],
  ['geometric(12345)', '55ec4095', 50, 7],
  ['negativeBinomial(12345)', '34164095', 50, 13],
  ['fDistribution(12345)', '022db667', 50, 3.787505615478361],
]

describe('distribution samples are bit-for-bit stable', () => {
  it.each(PINS)('%s', (name, hash, length, first) => {
    const [family = '', seed = ''] = name.slice(0, -1).split('(')
    const make = FAMILIES[family]
    if (make === undefined) throw new Error(`no family named ${family}`)
    const values = toFlat(make().sample(key(Number(seed)), [25]) as Tensor)
    expect(values.length).toBe(length)
    const rounded = roundedPin(name, values)
    if (rounded) expect(values[0]).toBeCloseTo(first, 10)
    else expect(values[0]).toBe(first)
    expect(checksum(values, rounded)).toBe(hash)
  })
})
