import { describe, expect, it } from 'vitest'
import { policies, recorder, runAlgorithm } from '../trace/index.ts'
import { lempelZivComplexity, lempelZivComplexityAlgorithm } from './complexity.ts'
import { lz77Algorithm, lz77Decode, lz77Encode, lzContract } from './dictionary-coding.ts'
import { lz77Windows, reedSolomonFields } from './dictionary-coding-options.ts'
import { reedSolomon, rsContract, rsDecode, rsDecodeAlgorithm, rsDecodeResult, rsEncode } from './reed-solomon.ts'

const viaJson = <T>(x: T): T => JSON.parse(JSON.stringify(x)) as T
const text = 'abracadabra abracadabra alakazam abracadabra'

describe('lz77Algorithm', () => {
  it('emits one token per step, decodes back to the input and equals lz77Encode', () => {
    const run = runAlgorithm(lz77Algorithm(text), { maxSteps: 100 })
    expect(run.status).toBe('converged')
    expect(run.steps).toBe(run.state.tokens.length)
    expect(run.state.tokens).toEqual(lz77Encode(text))
    expect(lz77Decode(run.state.tokens)).toEqual(Array.from(new TextEncoder().encode(text)))
    // Matches never look past the window.
    const small = runAlgorithm(lz77Algorithm(text, { window: 4 }), { maxSteps: 100 })
    for (const t of small.state.tokens) expect(t.offset).toBeLessThanOrEqual(4)
    expect(small.state.tokens.length).toBeGreaterThan(run.state.tokens.length)
  })

  it('serialises and resumes, logging the position, match and phrase counts', () => {
    const alg = lz77Algorithm(text)
    const whole = runAlgorithm(alg, { maxSteps: 100 })
    const part = runAlgorithm(alg, { maxSteps: 6 }).state
    const rest = runAlgorithm(alg, {
      maxSteps: 100,
      start: { state: viaJson(alg.serialise?.(part) ?? null), step: 6, serialised: true },
    })
    expect(rest.state).toEqual(whole.state)
    const rec = recorder()
    runAlgorithm(alg, { maxSteps: 100, tracer: rec, policy: { ...policies.minimal, metrics: true, detail: 'full' } })
    const metric = (name: string): number[] =>
      rec.events.flatMap((e) => (e.type === 'metric' && e.name === name ? [e.value] : []))
    expect(metric('lz/position').at(-1)).toBe(text.length)
    expect(metric('lz/phrase').at(-1)).toBe(whole.steps)
    expect(Object.keys(lzContract)).toHaveLength(5)
    expect(() => lz77Algorithm(text, { window: 0 })).toThrow(/>= 1/)
  })
})

describe('lempelZivComplexityAlgorithm', () => {
  it('finds the phrases one per step and agrees with lempelZivComplexity', () => {
    const run = runAlgorithm(lempelZivComplexityAlgorithm('1001111011000010'), { maxSteps: 100 })
    expect(run.status).toBe('converged')
    expect(run.state.complexity).toBe(6)
    expect(run.state.phrases).toEqual([1, 2, 4, 8, 12, 16])
    expect(lempelZivComplexity('1001111011000010')).toBe(6)
    expect(lempelZivComplexity('1001111011000010', { normalise: true })).toBeCloseTo(1.5, 12)
    expect(lempelZivComplexity('aaaaaaaa')).toBe(2)
  })

  it('has as many phrases as the complexity, and resumes from JSON', () => {
    const seq = Array.from({ length: 200 }, (_, i) => (i * i + (i >> 2)) % 3)
    const alg = lempelZivComplexityAlgorithm(seq)
    const whole = runAlgorithm(alg, { maxSteps: 400 })
    expect(whole.state.phrases).toHaveLength(whole.state.complexity)
    expect(whole.state.phrases.at(-1)).toBe(200)
    const part = runAlgorithm(alg, { maxSteps: 5 }).state
    const rest = runAlgorithm(alg, {
      maxSteps: 400,
      start: { state: viaJson(alg.serialise?.(part) ?? null), step: 5, serialised: true },
    })
    expect(rest.state).toEqual(whole.state)
  })
})

describe('rsDecodeAlgorithm', () => {
  const code = reedSolomon(4)
  const message = [1, 2, 3, 4, 5, 6]

  it('corrects errors one Berlekamp-Massey step at a time, as rsDecode does', () => {
    const word = rsEncode(code, message)
    word[1] = 99
    word[7] = 3
    const run = runAlgorithm(rsDecodeAlgorithm(code, word), { maxSteps: 20 })
    expect(run.status).toBe('converged')
    // nsym Berlekamp-Massey steps and the correction step.
    expect(run.steps).toBe(5)
    const result = rsDecodeResult(run.state, code)
    expect(result.message).toEqual(message)
    expect(result.errata).toEqual([1, 7])
    expect(rsDecode(code, word)).toEqual(result)
    // The locator's degree is the number of errors.
    expect(run.state.lambda.length - 1).toBe(2)
  })

  it('uses erasures, passes a clean word untouched and fails on too many errors', () => {
    const word = rsEncode(code, message)
    const erased = [...word]
    erased[2] = 0
    erased[3] = 0
    erased[9] = 0
    expect(
      rsDecodeResult(
        runAlgorithm(rsDecodeAlgorithm(code, erased, { erasures: [2, 3, 9] }), { maxSteps: 20 }).state,
        code,
      ).message,
    ).toEqual(message)
    const clean = runAlgorithm(rsDecodeAlgorithm(code, word), { maxSteps: 20 })
    expect([clean.status, clean.steps]).toEqual(['converged', 0])
    const bad = [...word]
    for (const i of [0, 2, 4]) bad[i] = ((bad[i] ?? 0) + 7) % 256
    const failed = runAlgorithm(rsDecodeAlgorithm(code, bad), { maxSteps: 20 })
    expect(failed.status).toBe('failed')
    expect(() => rsDecode(code, bad)).toThrow(/too many errors/)
    expect(() => rsDecodeAlgorithm(code, word, { erasures: [99] })).toThrow(/out of range/)
  })

  it('serialises mid-decode, resumes, and logs the locator and discrepancy', () => {
    const word = rsEncode(code, message)
    word[0] = 200
    word[5] = 17
    const alg = rsDecodeAlgorithm(code, word)
    const whole = runAlgorithm(alg, { maxSteps: 20 })
    const part = runAlgorithm(alg, { maxSteps: 2 }).state
    const rest = runAlgorithm(alg, {
      maxSteps: 20,
      start: { state: viaJson(alg.serialise?.(part) ?? null), step: 2, serialised: true },
    })
    expect(rest.state).toEqual(whole.state)
    const rec = recorder()
    runAlgorithm(alg, { maxSteps: 20, tracer: rec, policy: { ...policies.minimal, metrics: true, detail: 'full' } })
    const metric = (name: string): number[] =>
      rec.events.flatMap((e) => (e.type === 'metric' && e.name === name ? [e.value] : []))
    expect(metric('rs/discrepancy')).toHaveLength(4)
    expect(metric('rs/errors').at(-1)).toBe(2)
    expect(Object.keys(rsContract)).toHaveLength(4)
  })
})

describe('information option tables', () => {
  it('build LZ77 windows and Reed-Solomon codes', () => {
    expect(lz77Windows.names).toEqual(['tiny', 'standard', 'deflate'])
    expect(lz77Windows.make({ kind: 'tiny' })).toEqual({ window: 16, maxLength: 32 })
    expect(reedSolomonFields.names).toEqual(['gf16', 'gf256', 'gf1024'])
    const code = reedSolomonFields.make({ kind: 'gf16', nsym: 4 })
    expect(code.field.size).toBe(16)
    expect(rsDecode(code, rsEncode(code, [1, 2, 3])).message).toEqual([1, 2, 3])
  })
})
