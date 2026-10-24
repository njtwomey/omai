import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../testing/check.ts'
import { meanOf, varianceOf } from '../base/index.ts'
import { key } from '../random/index.ts'
import { item, softmax, tensor, toFlat, type Value } from '../tensor/index.ts'
import {
  birthDeathGenerator,
  birthDeathStationary,
  extinctionByGeneration,
  extinctionProbability,
  galtonWatson,
  galtonWatsonMoments,
  linearBirthDeath,
  offspringPgf,
  simulateBirthDeath,
  yulePmf,
} from './branching.ts'
import { ctmcStationary } from './ctmc.ts'

const pmf = [0.25, 0.25, 0.5]

describe('Galton-Watson processes', () => {
  it('simulated generations have the closed-form mean and variance', () => {
    const z = galtonWatson(key(1), pmf, 4, { paths: 4000 })
    const last = toFlat(z).filter((_, i) => i % 5 === 4)
    const { mean, variance } = galtonWatsonMoments(pmf, 4)
    expect(Math.abs(meanOf(last) - mean)).toBeLessThan(4 * Math.sqrt(variance / 4000))
    expect(Math.abs(varianceOf(last) - variance)).toBeLessThan(0.25 * variance)
  })

  it('go extinct with the probability of the fixed point of the generating function', () => {
    const z = galtonWatson(key(2), pmf, 25, { paths: 3000, maxPopulation: 1e5 })
    const last = toFlat(z).filter((_, i) => i % 26 === 25)
    const extinct = last.filter((v) => v === 0).length / 3000
    const q = item(extinctionProbability(pmf))
    expect(Math.abs(extinct - q)).toBeLessThan(4 * Math.sqrt((q * (1 - q)) / 3000))
    const byGen = toFlat(extinctionByGeneration(pmf, 40))
    expect(byGen.at(-1)).toBeCloseTo(q, 4)
    for (let n = 1; n < byGen.length; n++) expect(byGen[n]).toBeGreaterThanOrEqual(byGen[n - 1] ?? 0)
  })

  it('handles critical, subcritical and degenerate laws', () => {
    expect(item(extinctionProbability([0.5, 0, 0.5]))).toBe(1)
    expect(item(extinctionProbability([0.7, 0.3]))).toBe(1)
    expect(item(extinctionProbability([0, 1]))).toBe(0)
    expect(item(extinctionProbability([0, 0.5, 0.5]))).toBe(0)
    expect(galtonWatsonMoments([0.5, 0, 0.5], 3)).toEqual({ mean: 1, variance: 3 })
    expect(() => extinctionProbability([0.5, 0.6])).toThrow()
    // a population cap freezes exploding paths and terminates
    const z = galtonWatson(key(3), [0, 0, 0, 1], 30, { maxPopulation: 1000 })
    expect(toFlat(z).at(-1)).toBeGreaterThan(1000)
  })

  it('differentiates the extinction probability in the offspring law', () => {
    expectConformant((logits: Value) => extinctionProbability(softmax(logits, -1)), [tensor([0.1, -0.3, 0.8])], {
      vmap: false,
    })
    expectConformant((s: Value) => offspringPgf(pmf, s), [tensor([0.2, 0.7])])
  })
})

describe('birth-death processes', () => {
  const rates = { birth: (n: number) => 3 / (n + 1), death: (n: number) => 1.5 * n }

  it('the product-form stationary law matches the generator', () => {
    const pi = toFlat(birthDeathStationary(rates, 15))
    const viaQ = toFlat(ctmcStationary(birthDeathGenerator(rates, 15)))
    for (let i = 0; i < 16; i++) expect(pi[i]).toBeCloseTo(viaQ[i] ?? 0, 12)
    expect(() => birthDeathGenerator({ birth: () => -1, death: () => 1 }, 3)).toThrow()
  })

  it('linear birth-death simulations match extinction and growth', () => {
    const bd = linearBirthDeath({ lambda: 1.2, mu: 1 })
    const runs = Array.from({ length: 600 }, (_, j) =>
      simulateBirthDeath(key(100 + j), bd.rates, 1, 2, { maxEvents: 1e5 }),
    )
    const dead = runs.filter((r) => r.sizes.at(-1) === 0).length / 600
    const want = item(bd.extinctionBy(2))
    expect(Math.abs(dead - want)).toBeLessThan(4 * Math.sqrt((want * (1 - want)) / 600))
    const sizes = runs.map((r) => r.sizes.at(-1) ?? 0)
    expect(Math.abs(meanOf(sizes) - item(bd.mean(2)))).toBeLessThan(4 * Math.sqrt(varianceOf(sizes) / 600))
    expect(item(bd.ultimateExtinction(2))).toBeCloseTo((1 / 1.2) ** 2, 12)
    expect(item(linearBirthDeath({ lambda: 1, mu: 2 }).ultimateExtinction())).toBe(1)
  })

  it('a pure birth process is geometric (Yule)', () => {
    const k = Array.from({ length: 200 }, (_, i) => i + 1)
    expect(toFlat(yulePmf(k, 0.5, 1)).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
    expectConformant((l: Value) => yulePmf([1, 3, 6], 0.8, l), [tensor(0.9)])
    expectConformant((l: Value) => linearBirthDeath({ lambda: l, mu: 0.5 }).extinctionBy(1.5), [tensor(0.9)], {
      vmap: false,
    })
  })

  it('stops at an event cap or when no event can happen', () => {
    expect(() =>
      simulateBirthDeath(key(0), { birth: (n) => 5 * n, death: () => 0 }, 5, 100, { maxEvents: 50 }),
    ).toThrow()
    expect(Array.from(simulateBirthDeath(key(0), { birth: () => 0, death: () => 0 }, 4, 1).sizes)).toEqual([4])
  })
})
