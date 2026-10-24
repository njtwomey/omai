import { describe, expect, it } from 'vitest'
import { key, normal, uniform } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import * as I from './index.ts'

describe('complexity measures', () => {
  it('compression distance is small for related and large for unrelated objects, with any compressor', () => {
    const a = 'colourless green ideas sleep furiously '.repeat(10)
    const b = 'colourless green ideas sleep soundly '.repeat(10)
    const noise = Uint8Array.from(toFlat(uniform(key(3), [400])), (v) => Math.floor(v * 256))
    const related = I.normalisedCompressionDistance(a, b)
    const unrelated = I.normalisedCompressionDistance(a, noise)
    expect(related).toBeLessThan(unrelated)
    expect(unrelated).toBeGreaterThan(0.8)
    for (const compressor of ['gzip', 'deflate'] as const)
      expect(I.normalisedCompressionDistance(a, b, { compressor })).toBeLessThan(0.5)
    // A custom compressor: the length of the LZW code sequence.
    const lzw = (bytes: Uint8Array): number => I.lzwEncode(bytes).length
    expect(I.normalisedCompressionDistance(a, a, { compressor: lzw })).toBeLessThan(
      I.normalisedCompressionDistance(a, noise, { compressor: lzw }),
    )
    const m = I.compressionDistanceMatrix([a, b, noise])
    expect(m[0]?.[0]).toBe(0)
    expect(m[0]?.[1]).toBeCloseTo(related, 12)
    expect(m[1]?.[2]).toBeGreaterThan(0.8)
  })

  it('Lempel-Ziv complexity grows with randomness and its normalised form estimates the entropy rate', () => {
    expect(I.lempelZivComplexity([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])).toBe(3)
    expect(I.lempelZivComplexity('ABCDEFGHIJKLMNOPQRSTUVWXYZ', { normalise: true })).toBe(1)
    expect(I.lempelZivComplexity([0, 0, 0, 0, 0, 0])).toBe(2)
    const coin = toFlat(uniform(key(4), [4000])).map((v) => (v < 0.5 ? 1 : 0))
    expect(I.lempelZivComplexity(coin, { normalise: true })).toBeGreaterThan(0.85)
    expect(I.lempelZivComplexity(coin, { normalise: true })).toBeLessThan(1.15)
    expect(I.lempelZivComplexity(['a', 'b', 'a', 'b'])).toBe(3)
  })

  it('entropy-rate estimates approach the rate of the Markov chain that produced the data', () => {
    const p = [
      [0.9, 0.1],
      [0.4, 0.6],
    ]
    const u = toFlat(uniform(key(5), [20000]))
    const chain = [0]
    for (let t = 1; t < u.length; t++) chain.push((u[t] ?? 0) < (p[chain[t - 1] ?? 0]?.[1] ?? 0) ? 1 : 0)
    const rate = I.markovEntropyRate(p)
    expect(Math.abs(I.entropyRate(chain) - rate)).toBeLessThan(0.01)
    expect(I.entropyRate(chain, { order: 0 })).toBeGreaterThan(rate)
    expect(I.entropyRate(chain, { order: 3 })).toBeCloseTo(I.entropyRate(chain), 2)
    expect(I.blockEntropy(chain, 1)).toBeCloseTo(I.entropyRate(chain, { order: 0 }), 12)
    expect(
      I.markovEntropyRate([
        [0, 1],
        [1, 0],
      ]),
    ).toBe(0)
    expect(() =>
      I.markovEntropyRate([
        [0.5, 0.6],
        [0.5, 0.5],
      ]),
    ).toThrow()
    expect(() => I.markovEntropyRate([1, 0])).toThrow()
    expect(() => I.entropyRate([0, 1], { order: 2 })).toThrow()
    expect(() => I.blockEntropy([0, 1], 3)).toThrow()
  })

  it('approximate and sample entropy separate regular from irregular series', () => {
    const t = Array.from({ length: 300 }, (_, i) => Math.sin((2 * Math.PI * i) / 20))
    const noise = toFlat(normal(key(6), [300]))
    expect(I.approximateEntropy(t)).toBeLessThan(I.approximateEntropy(noise))
    expect(I.sampleEntropy(t)).toBeLessThan(I.sampleEntropy(noise))
    expect(I.sampleEntropy(noise, { order: 2, tolerance: 0 })).toBeNaN()
    expect(I.sampleEntropy([0, 0, 5, 0, 0, 9], { tolerance: 0.5 })).toBe(Number.POSITIVE_INFINITY)
    expect(I.approximateEntropy(noise, { order: 1 })).toBeGreaterThan(0)
    expect(() => I.sampleEntropy([1, 2])).toThrow()
    expect(() => I.approximateEntropy(noise, { tolerance: -1 })).toThrow()
  })
})
