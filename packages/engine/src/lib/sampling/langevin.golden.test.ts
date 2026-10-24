import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { fromStorage, type Tensor, type Value } from '../tensor/index.ts'
import { medianBandwidth, svgdDirection } from './langevin.ts'

type Params = Readonly<Record<string, unknown>>

// The oracle's target: a Gaussian with mean (1, -0.5) and precision [[2, 0.6], [0.6, 1]].
function grads(p: Float64Array): Float64Array {
  const out = new Float64Array(p.length)
  for (let i = 0; i < p.length / 2; i++) {
    const a = (p[2 * i] ?? 0) - 1
    const b = (p[2 * i + 1] ?? 0) + 0.5
    out[2 * i] = -(2 * a + 0.6 * b)
    out[2 * i + 1] = -(0.6 * a + b)
  }
  return out
}

const CALLS: Readonly<Record<string, (x: Tensor, p: Params) => Value | number>> = {
  medianBandwidth: (x) => medianBandwidth(x.data as Float64Array, 2),
  svgdStep: (x, p) => {
    const particles = x.data as Float64Array
    const phi = svgdDirection(particles, grads(particles), 2, p['lengthScale'] as number)
    const eps = p['stepSize'] as number
    return fromStorage(
      particles.map((v, i) => v + eps * (phi[i] ?? 0)),
      [...x.shape],
      'float64',
    )
  },
}

describe('sampling/langevin against BlackJAX', () => {
  for (const c of loadFixture('sampling/langevin').cases) {
    it(c.name, () => {
      const call = CALLS[c.fn]
      if (call === undefined) throw new Error(`no function ${c.fn}`)
      expectMatches(call(decode(c.inputs[0]) as Tensor, c.params), c.output, c.rtol, c.atol, c.name)
    })
  }
})
