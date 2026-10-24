import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { decodeParams, type Params } from '../../testing/suite.ts'
import { key } from '../random/index.ts'
import { type Tensor, toFlat, type Value } from '../tensor/index.ts'
import {
  stableCdf,
  stablePdf,
  stableSample,
  varianceGammaCharacteristicFunction,
  varianceGammaDensity,
} from './levy.ts'

const stable = (p: Params) => ({
  alpha: p['alpha'] as number,
  beta: p['beta'] as number,
  scale: p['scale'] as number,
  loc: p['loc'] as number,
})
const vg = (p: Params) => ({ theta: p['theta'] as number, sigma: p['sigma'] as number, nu: p['nu'] as number })

const FNS: Readonly<Record<string, (x: Tensor, p: Params) => Value>> = {
  'stable-pdf': (x, p) => stablePdf(x, stable(p)),
  'stable-cdf': (x, p) => stableCdf(x, stable(p)),
  'vg-density': (x, p) => varianceGammaDensity(x, p['t'] as number, vg(p)),
  'vg-cf-re': (u, p) => varianceGammaCharacteristicFunction(u, p['t'] as number, vg(p)).re,
  'vg-cf-im': (u, p) => varianceGammaCharacteristicFunction(u, p['t'] as number, vg(p)).im,
}

describe('stochastic/levy against scipy.stats.levy_stable and scipy.integrate.quad', () => {
  for (const c of loadFixture('stochastic/levy').cases) {
    const p = decodeParams(c.params)
    const x = decode(c.inputs[0]) as Tensor
    if (c.fn === 'stable-quantiles') {
      // The sampler cannot reproduce scipy's stream: the empirical CDF at scipy's quantiles must match the levels.
      it(`${c.name} (sampled)`, () => {
        const q = toFlat(decode(c.output) as Tensor)
        const n = 20000
        const draws = toFlat(stableSample(key(c.name.length), [n], stable(p)))
        toFlat(x).forEach((level, i) => {
          const hit = draws.filter((d) => d <= (q[i] ?? 0)).length / n
          expect(Math.abs(hit - level)).toBeLessThan(4 * Math.sqrt((level * (1 - level)) / n))
        })
      })
      continue
    }
    it(c.name, () => {
      const f = FNS[c.fn]
      if (f === undefined) throw new Error(`no function ${c.fn}`)
      expectMatches(f(x, p), c.output, c.rtol, c.atol, c.name)
    })
  }
})
