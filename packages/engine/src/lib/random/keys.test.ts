import { describe, expect, it } from 'vitest'
import { uniform as uniformLaw } from '../dists/index.ts'
import { kstest } from '../stats/tests/index.ts'
import { toFlat } from '../tensor/index.ts'
import * as R from './index.ts'

const flat = (v: Parameters<typeof toFlat>[0]): number[] => Array.from(toFlat(v))

describe('keys give independent streams', () => {
  const samplers: readonly (readonly [string, (k: R.Key, n: number) => number[]])[] = [
    ['uniform', (k, n) => flat(R.uniform(k, [n]))],
    ['normal', (k, n) => flat(R.normal(k, [n]))],
    ['truncatedNormal', (k, n) => flat(R.truncatedNormal(k, [n], -1, 2))],
    ['exponential', (k, n) => flat(R.exponential(k, [n]))],
    ['cauchy', (k, n) => flat(R.cauchy(k, [n]))],
    ['gamma(0.5)', (k, n) => flat(R.gamma(k, [n], 0.5))],
    ['gamma(20)', (k, n) => flat(R.gamma(k, [n], 20))],
    ['beta', (k, n) => flat(R.beta(k, [n], 2, 3))],
    ['studentT', (k, n) => flat(R.studentT(k, [n], 5))],
    ['vonMises', (k, n) => flat(R.vonMises(k, [n], 0, 3))],
    ['randint', (k, n) => flat(R.randint(k, [n], 0, 10))],
    ['bernoulli', (k, n) => flat(R.bernoulli(k, 0.4, [n]))],
    ['categorical', (k, n) => flat(R.categorical(k, [0.2, 0.5, 0.3], [n]))],
    ['poisson(4)', (k, n) => flat(R.poisson(k, 4, [n]))],
    ['poisson(60)', (k, n) => flat(R.poisson(k, 60, [n]))],
    ['binomial', (k, n) => flat(R.binomial(k, 100, 0.6, [n]))],
    ['geometric', (k, n) => flat(R.geometric(k, 0.2, [n]))],
  ]
  it('keys from split, foldIn, keyFromPath and seeds are distinct and give uncorrelated, uniform streams; the same key repeats and a draw is a prefix of a longer one', () => {
    const root = R.key(0)
    const groups: readonly (readonly [string, R.Key[]])[] = [
      ['split', R.split(root, 64)],
      ['split of a child', R.split(R.split(root)[0] as R.Key, 64)],
      ['foldIn', Array.from({ length: 64 }, (_, i) => R.foldIn(root, i))],
      ['keyFromPath', Array.from({ length: 64 }, (_, i) => R.keyFromPath(0, 'layer', i))],
      ['seeds', Array.from({ length: 64 }, (_, i) => R.key(i))],
    ]
    const keys = groups.flatMap(([, ks]) => ks)
    expect(new Set(keys.map((k) => k.words.join(':'))).size, 'no two keys share their words').toBe(keys.length)

    // Standardised streams of 1,000 uniform draws: the correlation of a pair is their dot product, and for independent
    // streams it has standard deviation 1 / sqrt(1000), so 6 standard deviations (0.19) is passed by chance 2e-9 of pairs.
    const n = 1000
    const streams = keys.map((k) => {
      const x = flat(R.uniform(k, [n]))
      const m = x.reduce((a, b) => a + b, 0) / n
      const norm = Math.sqrt(x.reduce((a, b) => a + (b - m) ** 2, 0))
      return x.map((v) => (v - m) / norm)
    })
    let worst = 0
    for (let i = 0; i < streams.length; i++)
      for (let j = i + 1; j < streams.length; j++) {
        const a = streams[i] as number[]
        const b = streams[j] as number[]
        let r = 0
        for (let t = 0; t < n; t++) r += (a[t] as number) * (b[t] as number)
        worst = Math.max(worst, Math.abs(r))
      }
    expect(worst, 'largest correlation between two streams').toBeLessThan(6 / Math.sqrt(n))

    // The first draw of 20,000 consecutive children, and of 20,000 consecutive seeds, is uniform: no pattern in the keys
    // survives into the output.
    const firsts = (make: (i: number) => R.Key): number[] =>
      Array.from({ length: 20000 }, (_, i) => flat(R.uniform(make(i), [1]))[0] as number)
    for (const [name, make] of [
      ['foldIn', (i: number) => R.foldIn(root, i)],
      ['seeds', (i: number) => R.key(i)],
      ['split', (i: number) => R.split(R.key(i))[1] as R.Key],
    ] as const)
      expect(
        kstest(firsts(make), uniformLaw(0, 1), { method: 'asymp' }).pValue,
        `${name}: first draws`,
      ).toBeGreaterThan(1e-3)

    // Within one stream, successive draws are uncorrelated (lag 1 and lag 2).
    const x = flat(R.uniform(root, [20000]))
    for (const lag of [1, 2]) {
      const m = x.reduce((a, b) => a + b, 0) / x.length
      let num = 0
      let den = 0
      for (let t = 0; t < x.length; t++) {
        den += ((x[t] as number) - m) ** 2
        if (t >= lag) num += ((x[t] as number) - m) * ((x[t - lag] as number) - m)
      }
      expect(Math.abs(num / den), `lag ${lag}`).toBeLessThan(5 / Math.sqrt(x.length))
    }
    for (const [name, draw] of samplers) {
      const a = draw(R.key(3), 50)
      expect(draw(R.key(3), 50), `${name}: the same key twice`).toEqual(a)
      expect(draw(R.key(4), 50), `${name}: another seed`).not.toEqual(a)
      expect(draw(R.split(R.key(3))[0] as R.Key, 50), `${name}: a split child`).not.toEqual(a)
      expect(draw(R.foldIn(R.key(3), 0), 50), `${name}: a folded key`).not.toEqual(a)
      expect(draw(R.key(3), 200).slice(0, 50), `${name}: shape [50] is the start of shape [200]`).toEqual(a)
    }
  })
})
