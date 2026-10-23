import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import type { Tensor, Value } from '../tensor/index.ts'
import * as D from './index.ts'

type Factory = (params: Tensor[]) => D.Distribution
const FAMILIES: Readonly<Record<string, Factory>> = {
  normal: ([loc, scale]) => D.normal(loc as Tensor, scale as Tensor),
  logNormal: ([mu, sigma]) => D.logNormal(mu as Tensor, sigma as Tensor),
  halfNormal: ([scale]) => D.halfNormal(scale as Tensor),
  truncatedNormal: ([loc, scale, lo, hi]) =>
    D.truncatedNormal(loc as Tensor, scale as Tensor, lo as Tensor, hi as Tensor),
  gamma: ([a, r]) => D.gamma(a as Tensor, r as Tensor),
  exponential: ([r]) => D.exponential(r as Tensor),
  beta: ([a, b]) => D.beta(a as Tensor, b as Tensor),
  uniform: ([lo, hi]) => D.uniform(lo as Tensor, hi as Tensor),
  bernoulli: ([p]) => D.bernoulli({ probs: p as Tensor }),
  categorical: ([p]) => D.categorical({ probs: p as Tensor }),
}
const ARITY: Readonly<Record<string, number>> = {
  normal: 2,
  logNormal: 2,
  halfNormal: 1,
  truncatedNormal: 4,
  gamma: 2,
  exponential: 1,
  beta: 2,
  uniform: 2,
  bernoulli: 1,
  categorical: 1,
}
const MOMENTS = new Set(['mean', 'variance', 'entropy', 'mode'])

for (const file of ['continuous', 'discrete']) {
  describe(`dists/${file} against scipy.stats`, () => {
    for (const c of loadFixture(`dists/${file}`).cases) {
      it(c.name, () => {
        const [family, method] = c.fn.split('.') as [string, string]
        const inputs = c.inputs.map((x) => decode(x) as Tensor)
        const n = ARITY[family] ?? 0
        const dist = (FAMILIES[family] as Factory)(inputs.slice(0, n))
        const call = dist[method as keyof D.Distribution] as unknown as (x?: Tensor) => Value
        const got = MOMENTS.has(method) ? call() : call(inputs[n])
        expectMatches(got, c.output, c.rtol, c.atol, c.name)
      })
    }
  })
}
