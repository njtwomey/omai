import { describe, expect, it } from 'vitest'
import { key, normal } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { recorder } from '../trace/index.ts'
import * as I from './index.ts'

describe('LDPC codes and belief propagation', () => {
  const h = I.gallagerParityCheck(96, 3, 6, key(1))
  const code = I.ldpcCode(h)
  const message = Array.from({ length: code.k }, (_, i) => ((i * 5 + 1) % 3 === 0 ? 1 : 0))
  const word = I.ldpcEncode(code, message)

  it('builds regular Gallager codes whose codewords satisfy every check', () => {
    expect(h.length).toBe(48)
    expect(h.every((row) => row.reduce((a, b) => a + b, 0) === 6)).toBe(true)
    for (let j = 0; j < 96; j++) expect(h.reduce((s, row) => s + (row[j] ?? 0), 0)).toBe(3)
    expect(code.k).toBeGreaterThanOrEqual(48)
    expect(I.syndrome(code, word).every((v) => v === 0)).toBe(true)
    expect(code.information.map((i) => word[i])).toEqual(message)
  })

  it('decodes a noisy BPSK codeword by sum-product and by min-sum', () => {
    const sigma = 0.6
    const noise = toFlat(normal(key(2), [96]))
    const y = word.map((b, i) => 1 - 2 * b + sigma * (noise[i] ?? 0))
    const llr = I.bpskLlr(y, sigma * sigma)
    const hard = llr.map((l) => (l < 0 ? 1 : 0))
    expect(hard.some((b, i) => b !== word[i])).toBe(true)
    for (const method of ['sumProduct', 'minSum'] as const) {
      const result = I.ldpcDecode(code, llr, { method, ...(method === 'minSum' ? { scale: 0.8 } : {}) })
      expect(result.converged).toBe(true)
      expect(result.codeword).toEqual(word)
      expect(result.message).toEqual(message)
    }
  })

  it('corrects flipped bits from a binary symmetric channel and traces the unsatisfied checks', () => {
    const received = word.map((b, i) => (i === 5 || i === 40 || i === 77 ? 1 - b : b))
    const rec = recorder()
    const result = I.ldpcDecode(code, I.bscLlr(received, 0.03), { tracer: rec })
    expect(result.codeword).toEqual(word)
    const unsatisfied = rec.events.filter((e) => e.type === 'metric' && e.name === 'unsatisfied')
    expect((unsatisfied[0] as { value: number }).value).toBeGreaterThan(0)
    expect((unsatisfied[unsatisfied.length - 1] as { value: number }).value).toBe(0)
    // A clean word converges at once.
    expect(I.ldpcDecode(code, I.bscLlr(word, 0.03)).iterations).toBe(0)
  })

  it('reports failure to converge and rejects bad input', () => {
    const garbage = I.bscLlr(
      word.map((b, i) => (i % 2 === 0 ? 1 - b : b)),
      0.4,
    )
    expect(I.ldpcDecode(code, garbage, { maxIter: 3 }).converged).toBe(false)
    expect(() => I.gallagerParityCheck(10, 3, 4, key(0))).toThrow()
    expect(() => I.gallagerParityCheck(12, 4, 4, key(0))).toThrow()
    expect(() =>
      I.ldpcCode([
        [1, 0],
        [0, 1],
      ]),
    ).toThrow()
    expect(() => I.ldpcEncode(code, [1])).toThrow()
    expect(() => I.beliefPropagationAlgorithm(h, [1, 2])).toThrow()
    expect(() => I.bpskLlr([1], 0)).toThrow()
    expect(() => I.bscLlr([1], 0.7)).toThrow()
  })
})
