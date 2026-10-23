import { describe, expect, it } from 'vitest'
import { toFlat } from '../tensor/index.ts'
import {
  aliasSample,
  aliasTable,
  bernoulli,
  beta,
  binomial,
  categorical,
  choice,
  dirichlet,
  exponential,
  foldIn,
  Generator,
  gamma,
  halton,
  key,
  keyFromPath,
  normal,
  permutation,
  philox4x32,
  poisson,
  randint,
  randomSplit,
  split,
  truncatedNormal,
  uniform,
  vonMises,
} from './index.ts'

const mean = (xs: ArrayLike<number>): number => Array.from(xs).reduce((a, b) => a + b, 0) / xs.length
const variance = (xs: ArrayLike<number>): number => {
  const m = mean(xs)
  return Array.from(xs).reduce((a, b) => a + (b - m) ** 2, 0) / (xs.length - 1)
}

describe('Philox4x32-10', () => {
  it('matches the Random123 known-answer vectors', () => {
    expect(philox4x32([0, 0, 0, 0], [0, 0])).toEqual([0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8])
    expect(philox4x32([0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff], [0xffffffff, 0xffffffff])).toEqual([
      0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd,
    ])
    expect(philox4x32([0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344], [0xa4093822, 0x299f31d0])).toEqual([
      0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1,
    ])
  })
})

describe('keys', () => {
  it('are reproducible and independent', () => {
    expect(toFlat(uniform(key(7), [5]))).toEqual(toFlat(uniform(key(7), [5])))
    const [a, b] = split(key(7))
    expect(toFlat(uniform(a as never, [3]))).not.toEqual(toFlat(uniform(b as never, [3])))
    expect(foldIn(key(1), 2).words).toEqual(foldIn(key(1), 2).words)
    expect(keyFromPath(3, 'a', 1).words).not.toEqual(keyFromPath(3, 'a', 2).words)
  })

  it('rejects invalid seeds', () => {
    expect(() => key(-1)).toThrow(/seed/)
    expect(() => key(1.5)).toThrow(/seed/)
  })
})

describe('samplers have the right moments', () => {
  const n = 40000
  it('uniform and normal', () => {
    const u = uniform(key(1), [n]).data
    expect(mean(u)).toBeCloseTo(0.5, 2)
    expect(variance(u)).toBeCloseTo(1 / 12, 2)
    const z = normal(key(2), [n], { loc: 1, scale: 2 }).data
    expect(mean(z)).toBeCloseTo(1, 1)
    expect(Math.sqrt(variance(z))).toBeCloseTo(2, 1)
  })

  it('exponential, gamma and beta', () => {
    expect(mean(exponential(key(3), [n], 2).data)).toBeCloseTo(0.5, 2)
    const g = gamma(key(4), [n], 3).data
    expect(mean(g)).toBeCloseTo(3, 1)
    expect(variance(g)).toBeCloseTo(3, 0)
    expect(mean(gamma(key(5), [n], 0.3).data)).toBeCloseTo(0.3, 1)
    expect(mean(beta(key(6), [n], 2, 6).data)).toBeCloseTo(0.25, 2)
  })

  it('truncated normal stays in range with the right mean', () => {
    const t = truncatedNormal(key(7), [n], 2, Infinity).data
    expect(Math.min(...Array.from(t.slice(0, 1000)))).toBeGreaterThanOrEqual(2)
    // E[Z | Z > 2] = phi(2) / (1 - Phi(2)) = 2.373215
    expect(mean(t)).toBeCloseTo(2.3732, 1)
  })

  it('discrete samplers', () => {
    expect(mean(poisson(key(8), 3, [n]).data)).toBeCloseTo(3, 1)
    expect(mean(poisson(key(9), 80, [n]).data)).toBeCloseTo(80, 0)
    expect(mean(binomial(key(10), 20, 0.3, [n]).data)).toBeCloseTo(6, 1)
    expect(mean(binomial(key(11), 20, 0.8, [n]).data)).toBeCloseTo(16, 1)
    expect(mean(bernoulli(key(12), 0.3, [n]).data)).toBeCloseTo(0.3, 2)
    const c = categorical(key(13), [1, 2, 7], [n]).data
    expect(mean(Array.from(c, (v) => (v === 2 ? 1 : 0)))).toBeCloseTo(0.7, 2)
    const lc = categorical(key(14), [0, Math.log(3)], [n], { logits: true }).data
    expect(mean(lc)).toBeCloseTo(0.75, 2)
    const a = aliasSample(key(15), aliasTable([1, 2, 7]), [n]).data
    expect(mean(Array.from(a, (v) => (v === 2 ? 1 : 0)))).toBeCloseTo(0.7, 2)
    expect(randint(key(16), [n], -2, 3).data.every((v) => v >= -2 && v < 3)).toBe(true)
  })

  it('dirichlet rows sum to one and von Mises concentrates', () => {
    const d = dirichlet(key(17), [0.5, 1, 2], 1000)
    for (let r = 0; r < 1000; r++) {
      expect((d.data[r * 3] ?? 0) + (d.data[r * 3 + 1] ?? 0) + (d.data[r * 3 + 2] ?? 0)).toBeCloseTo(1, 12)
    }
    expect(mean(Array.from(vonMises(key(18), [n], 1, 50).data))).toBeCloseTo(1, 1)
  })
})

describe('permutations and choices', () => {
  it('permutation is a permutation', () => {
    expect(Array.from(permutation(key(1), 50).data).sort((a, b) => a - b)).toEqual(
      Array.from({ length: 50 }, (_, i) => i),
    )
  })

  it('choice without replacement is distinct; weighted choice respects zero weights', () => {
    expect(new Set(choice(key(2), 10, 10, { replace: false }).data).size).toBe(10)
    const w = choice(key(3), 4, 3, { replace: false, weights: [1, 0, 1, 1] })
    expect(Array.from(w.data)).not.toContain(1)
  })

  it('randomSplit partitions', () => {
    const parts = randomSplit(key(4), 10, [0.7, 0.3])
    expect(parts.map((p) => p.shape[0])).toEqual([7, 3])
    expect(new Set([...(parts[0]?.data ?? []), ...(parts[1]?.data ?? [])]).size).toBe(10)
  })

  it('Generator is reproducible', () => {
    expect(toFlat(new Generator(5).normal([4]))).toEqual(toFlat(new Generator(5).normal([4])))
  })

  it('halton starts with the radical inverses', () => {
    expect(Array.from(halton(3, 1).data)).toEqual([0.5, 0.25, 0.75])
  })
})
