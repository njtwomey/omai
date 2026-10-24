import { describe, expect, it } from 'vitest'
import { policies, recorder, runAlgorithm } from '../trace/index.ts'
import * as I from './index.ts'

describe('prefix codes', () => {
  it('Kraft sums, prefix-freeness and unique decodability', () => {
    expect(I.kraftSum([1, 1, 1], 3)).toBe(1)
    expect(I.kraftSum([0, 1, 1])).toBe(1)
    expect(() => I.kraftSum([1], 1)).toThrow()
    expect(I.isPrefixFree([[0], [0]])).toBe(false)
    expect(I.isPrefixFree([[], [0], [1]])).toBe(true)
    // {1, 10, 100} is uniquely decodable (a suffix code) but not prefix-free.
    expect(I.isUniquelyDecodable([[1], [1, 0], [1, 0, 0]])).toBe(true)
    expect(I.isPrefixFree([[1], [1, 0], [1, 0, 0]])).toBe(false)
    expect(I.isUniquelyDecodable([[1], [0, 1, 1], [0, 1, 1, 1, 0], [1, 1, 1, 0], [1, 0, 0, 1, 1]])).toBe(false)
    expect(I.isUniquelyDecodable([[0], [0]])).toBe(false)
    expect(I.isUniquelyDecodable([[0], [1, 0], [1, 1]])).toBe(true)
  })

  it('Huffman codes are optimal and complete', () => {
    const p = [0.4, 0.2, 0.2, 0.1, 0.1]
    const code = I.huffman(p)
    expect(I.expectedLength(code, p)).toBeCloseTo(2.2, 14)
    expect(I.kraftSum(code.lengths)).toBe(1)
    expect(I.isPrefixFree(code.codewords)).toBe(true)
    expect(I.huffman([1]).codewords).toEqual([[0]])
    expect(I.huffman([1], { arity: 3 }).codewords).toEqual([[0]])
    expect(I.expectedLength(I.huffman([0.5, 0.5, 0]), [0.5, 0.5, 0])).toBe(1.5)
    const ternary = I.huffman([0.25, 0.25, 0.2, 0.15, 0.15], { arity: 3 })
    expect(I.kraftSum(ternary.lengths, 3)).toBeLessThanOrEqual(1)
    expect(ternary.codewords.flat().every((d) => d >= 0 && d < 3)).toBe(true)
    expect(I.isPrefixFree(ternary.codewords)).toBe(true)
    expect(() => I.huffman([-1, 2])).toThrow()
    expect(() => I.huffman([0, 0])).toThrow()
    expect(() => I.huffman([1, 2], { arity: 1 })).toThrow()
  })

  it('records the Huffman construction as a trace of merges', () => {
    const rec = recorder()
    I.huffman([3, 1, 1, 2], { tracer: rec, policy: policies.teaching })
    const events = rec.events
    const merges = events.filter((e) => e.type === 'message')
    expect(merges.length).toBe(3)
    const algorithm = I.huffmanAlgorithm([3, 1, 1, 2])
    const result = runAlgorithm(algorithm, { maxSteps: 10 })
    expect(result.status).toBe('converged')
    expect(result.state.nodes.length).toBe(7)
    expect(result.state.nodes[result.state.queue[0] ?? 0]?.weight).toBeCloseTo(1, 14)
  })

  it('canonical codes depend only on the lengths and reject Kraft violations', () => {
    expect(I.canonicalCode([3, 3, 3, 3, 3, 2, 4, 4]).codewords).toEqual([
      [0, 1, 0],
      [0, 1, 1],
      [1, 0, 0],
      [1, 0, 1],
      [1, 1, 0],
      [0, 0],
      [1, 1, 1, 0],
      [1, 1, 1, 1],
    ])
    expect(I.canonicalCode([1, 0, 1]).codewords).toEqual([[0], [], [1]])
    expect(I.canonicalCode([1, 1, 1], { arity: 3 }).codewords).toEqual([[0], [1], [2]])
    expect(() => I.canonicalCode([1, 1, 1])).toThrow()
  })

  it('Shannon and Fano codes are prefix-free and within a digit of the entropy', () => {
    const p = [0.3, 0.25, 0.2, 0.15, 0.1]
    const h = -p.reduce((s, v) => s + v * Math.log2(v), 0)
    for (const code of [I.shannonFano(p), I.shannonCode(p)]) {
      expect(I.isPrefixFree(code.codewords)).toBe(true)
      const l = I.expectedLength(code, p)
      expect(l).toBeGreaterThanOrEqual(h - 1e-12)
      expect(l).toBeLessThan(h + 1)
    }
    const ternary = I.shannonCode(p, { arity: 3 })
    expect(I.isPrefixFree(ternary.codewords)).toBe(true)
    expect(I.shannonCode([1, 0]).codewords).toEqual([[0], []])
    expect(I.shannonFano([1]).codewords).toEqual([[0]])
  })

  it('encodes and decodes, rejecting invalid input', () => {
    const code = I.huffman([0.5, 0.3, 0.2])
    const symbols = [0, 1, 2, 2, 1, 0, 0]
    expect(I.prefixDecode(code, I.prefixEncode(code, symbols))).toEqual(symbols)
    expect(() => I.prefixEncode(code, [3])).toThrow()
    expect(() => I.prefixEncode(I.shannonCode([1, 0]), [1])).toThrow()
    expect(() => I.prefixDecode({ arity: 2, codewords: [[0], [0, 1]] }, [0])).toThrow()
    const partial = I.canonicalCode([1, 2, 0])
    expect(() => I.prefixDecode(partial, [1, 1])).toThrow()
    expect(() => I.prefixDecode(partial, [1])).toThrow()
    expect(() => I.expectedLength([1, 2], [1])).toThrow()
  })
})
