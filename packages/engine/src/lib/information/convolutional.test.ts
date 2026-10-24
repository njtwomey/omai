import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { RUNS } from '../../testing/arbitrary.ts'
import { policies, recorder } from '../trace/index.ts'
import * as I from './index.ts'

const code = I.convolutionalCode([0o7, 0o5])

/** The least Hamming distance from `received` to the codeword of any message of the given length. */
function bruteForce(c: I.ConvolutionalCode, received: readonly number[], length: number): number {
  let best = Number.POSITIVE_INFINITY
  for (let m = 0; m < 2 ** length; m++) {
    const word = I.convolutionalEncode(
      c,
      Array.from({ length }, (_, i) => (m >> i) & 1),
    )
    best = Math.min(
      best,
      word.reduce((s, b, i) => s + (b === received[i] ? 0 : 1), 0),
    )
  }
  return best
}

describe('convolutional codes and Viterbi decoding', () => {
  it('Viterbi is maximum likelihood: its metric equals the brute-force minimum distance (property)', () => {
    fc.assert(
      fc.property(
        fc.array(fc.integer({ min: 0, max: 1 }), { minLength: 1, maxLength: 8 }),
        fc.nat(),
        (message, seed) => {
          for (const c of [code, I.convolutionalCode([0o13, 0o15, 0o17])]) {
            const word = I.convolutionalEncode(c, message)
            const received = word.map((b, i) => ((seed >> (i % 30)) & 1 && i % 3 === 0 ? 1 - b : b))
            const result = I.viterbiDecode(c, received)
            expect(result.metric).toBe(bruteForce(c, received, message.length))
            expect(result.bits.length).toBe(message.length)
          }
        },
      ),
      { numRuns: RUNS },
    )
  })

  it('corrects errors up to half the free distance and soft decisions beat hard ones', () => {
    // The (7, 5) code has free distance 5: two well-separated errors are always corrected.
    const message = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0]
    const word = I.convolutionalEncode(code, message)
    const received = word.map((b, i) => (i === 3 || i === 17 ? 1 - b : b))
    expect(I.viterbiDecode(code, received).bits).toEqual(message)
    // Soft input: BPSK with one strongly and one weakly corrupted symbol per error.
    const soft = word.map((b, i) => (1 - 2 * b) * (i === 3 ? -0.2 : i === 10 ? -0.3 : 0.9))
    expect(I.viterbiDecode(code, soft, { decision: 'soft' }).bits).toEqual(message)
    const open = I.convolutionalEncode(code, message, { terminate: false })
    expect(open.length).toBe(2 * message.length)
    expect(I.viterbiDecode(code, open, { terminated: false }).bits).toEqual(message)
  })

  it('records path metrics stage by stage', () => {
    const rec = recorder()
    I.viterbiDecode(code, I.convolutionalEncode(code, [1, 1, 0]), { tracer: rec, policy: policies.teaching })
    expect(rec.events.filter((e) => e.type === 'tensor').length).toBe(6)
  })

  it('rejects malformed codes and words', () => {
    expect(() => I.convolutionalCode([])).toThrow()
    expect(() => I.convolutionalCode([0o7], 2)).toThrow()
    expect(() => I.convolutionalCode([1])).toThrow()
    expect(() => I.convolutionalEncode(code, [2])).toThrow()
    expect(() => I.viterbiDecode(code, [1, 1, 1])).toThrow()
  })
})
