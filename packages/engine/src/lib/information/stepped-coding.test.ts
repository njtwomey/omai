import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { iterate, runAlgorithm } from '../trace/index.ts'
import {
  arithmeticDecode,
  arithmeticDecoderAlgorithm,
  arithmeticEncode,
  arithmeticEncoderAlgorithm,
  expectedLength,
  frequencyTable,
  huffman,
  markovSource,
  memorylessSource,
  prefixEncode,
  prefixEncoderAlgorithm,
  rangeDecoderAlgorithm,
  rangeEncode,
  rangeEncoderAlgorithm,
} from './index.ts'

const probs = [0.6, 0.25, 0.1, 0.05]
const spec = { kind: 'static', frequencies: frequencyTable(probs) } as const
const adaptive = { kind: 'adaptive', alphabetSize: 4 } as const
const sample = memorylessSource(key(3), probs, 400)

describe('sources with their truth', () => {
  it('a memoryless source has the entropy of its distribution and its sample averages to it', () => {
    const s = memorylessSource(key(0), [0.5, 0.25, 0.25], 10)
    expect(s.entropy).toBe(1.5)
    const big = memorylessSource(key(1), probs, 20000)
    expect(Math.abs(big.ideal / 20000 - big.entropy)).toBeLessThan(0.02)
    expect(big.entropy).toBeCloseTo(1.49, 2)
    expect(() => memorylessSource(key(0), [0, 0], 3)).toThrow(/non-negative/)
  })

  it('a Markov source has a stationary law and an entropy rate below the stationary entropy', () => {
    const m = markovSource(
      key(0),
      [
        [0.9, 0.1],
        [0.1, 0.9],
      ],
      30000,
    )
    expect(m.stationary[0]).toBeCloseTo(0.5, 12)
    expect(m.entropy).toBeCloseTo(-(0.9 * Math.log2(0.9) + 0.1 * Math.log2(0.1)), 12)
    expect(m.entropy).toBeLessThan(1)
    expect(Math.abs(m.ideal / 30000 - m.entropy)).toBeLessThan(0.02)
    const skew = markovSource(
      key(1),
      [
        [0.5, 0.5, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
      ],
      5,
    )
    expect(skew.stationary.map((x) => Math.round(x * 1e9) / 1e9)).toEqual(
      [1 / 3, 1 / 3, 1 / 3].map((x) => Math.round(x * 1e9) / 1e9),
    )
    expect(() =>
      markovSource(
        key(0),
        [
          [0.5, 0.4],
          [0.5, 0.5],
        ],
        3,
      ),
    ).toThrow(/sum to 1/)
    expect(() => markovSource(key(0), [[1, 0]], 3)).toThrow(/square/)
  })
})

describe('arithmetic encoder as an algorithm', () => {
  it('writes exactly the code of arithmeticEncode, one symbol per step and one to finish', () => {
    for (const model of [spec, adaptive]) {
      const run = runAlgorithm(arithmeticEncoderAlgorithm(sample.symbols, model), { seed: 0, maxSteps: 1000 })
      expect(run.state.output).toEqual(arithmeticEncode(sample.symbols, model))
      expect(run.steps).toBe(401)
      expect(run.status).toBe('completed')
      expect(arithmeticDecode(run.state.output as number[], 400, model)).toEqual(sample.symbols)
    }
  })

  it('stays within a few bits of the information content and tracks the interval', () => {
    const states = Array.from(
      iterate(arithmeticEncoderAlgorithm(sample.symbols, spec), { seed: 0, maxSteps: 401 }),
      (x) => x.state,
    )
    const last = states[400]
    // The model's probabilities, not the source's: the information content of the sample under the integer frequencies.
    const f = spec.frequencies
    const total = f.reduce((a, b) => a + b, 0)
    const ideal = sample.symbols.reduce((a, s) => a - Math.log2((f[s] ?? 1) / total), 0)
    expect(last?.ideal).toBeCloseTo(ideal, 8)
    const length = (last?.output.length ?? 0) as number
    // The decoder reads missing trailing bits as zeros, so the code can end a little short of -log2 of the width.
    expect(length).toBeGreaterThanOrEqual(ideal - 3)
    expect(length).toBeLessThan(ideal + 2.5)
    for (const s of states.slice(0, 40)) {
      // The working interval is a sub-interval of [0, 1), and the whole message's interval has width 2^-ideal.
      expect(s.low).toBeLessThan(s.high)
      expect(s.width).toBeCloseTo(2 ** -s.ideal, 12)
      expect(s.start).toBeGreaterThanOrEqual(0)
      expect(s.start + s.width).toBeLessThanOrEqual(1 + 1e-12)
    }
  })

  it('approaches the entropy where a Huffman code cannot', () => {
    const source = memorylessSource(key(5), [0.95, 0.05], 20000)
    const model = { kind: 'static', frequencies: frequencyTable([0.95, 0.05]) } as const
    const run = runAlgorithm(arithmeticEncoderAlgorithm(source.symbols, model), { seed: 0, maxSteps: 30000 })
    const perSymbol = run.state.output.length / 20000
    expect(perSymbol).toBeLessThan(source.entropy + 0.02)
    expect(perSymbol).toBeGreaterThan(source.entropy - 0.02)
    // A prefix code spends at least one whole digit per symbol.
    expect(expectedLength(huffman([0.95, 0.05]), [0.95, 0.05])).toBe(1)
    expect(source.entropy).toBeCloseTo(0.2864, 3)
  })

  it('an adaptive model pays about (m - 1) / 2 log2 n bits more than the entropy to learn the source', () => {
    const source = memorylessSource(key(6), probs, 4000)
    const run = runAlgorithm(arithmeticEncoderAlgorithm(source.symbols, adaptive), { seed: 0, maxSteps: 5000 })
    const extra = run.state.output.length - 4000 * source.entropy
    expect(extra).toBeGreaterThan(-50)
    expect(extra).toBeLessThan(((4 - 1) / 2) * Math.log2(4000) + 40)
    // Order-0 statistics miss the dependence of a Markov source and pay the entropy of its stationary law.
    const markov = markovSource(
      key(7),
      [
        [0.95, 0.05],
        [0.05, 0.95],
      ],
      4000,
    )
    const coded = runAlgorithm(arithmeticEncoderAlgorithm(markov.symbols, { kind: 'adaptive', alphabetSize: 2 }), {
      seed: 0,
      maxSteps: 5000,
    })
    expect(coded.state.output.length / 4000).toBeGreaterThan(0.9)
    expect(markov.entropy).toBeLessThan(0.3)
  })

  it('resumes exactly from a checkpoint, adaptive model included', () => {
    const alg = arithmeticEncoderAlgorithm(sample.symbols, adaptive)
    const straight = runAlgorithm(alg, { seed: 0, maxSteps: 1000 })
    const early = runAlgorithm(alg, { seed: 0, maxSteps: 150 })
    const resumed = runAlgorithm(alg, {
      seed: 0,
      maxSteps: 1000,
      start: { state: alg.serialise?.(early.state) ?? null, step: early.steps, serialised: true },
    })
    expect(resumed.state.output).toEqual(straight.state.output)
    expect(resumed.state.counts).toEqual(straight.state.counts)
  })

  it('rejects a symbol the model gives no probability', () => {
    expect(() => arithmeticEncoderAlgorithm([0, 3], { kind: 'static', frequencies: [3, 1, 1, 0] })).toThrow(
      /zero probability/,
    )
  })
})

describe('arithmetic decoder as an algorithm', () => {
  it('decodes one symbol per step and mirrors the encoder', () => {
    for (const model of [spec, adaptive]) {
      const bits = arithmeticEncode(sample.symbols, model)
      const run = runAlgorithm(arithmeticDecoderAlgorithm(bits, 400, model), { seed: 0, maxSteps: 1000 })
      expect(run.state.output).toEqual(sample.symbols)
      expect(run.steps).toBe(400)
      expect(run.state.position).toBeLessThanOrEqual(bits.length + 40)
    }
  })

  it('resumes exactly', () => {
    const bits = arithmeticEncode(sample.symbols, adaptive)
    const alg = arithmeticDecoderAlgorithm(bits, 400, adaptive)
    const early = runAlgorithm(alg, { seed: 0, maxSteps: 120 })
    const resumed = runAlgorithm(alg, {
      seed: 0,
      maxSteps: 1000,
      start: { state: alg.serialise?.(early.state) ?? null, step: early.steps, serialised: true },
    })
    expect(resumed.state.output).toEqual(sample.symbols)
  })
})

describe('range coder as algorithms', () => {
  it('writes the bytes of rangeEncode and decodes them back', () => {
    for (const model of [spec, adaptive]) {
      const enc = runAlgorithm(rangeEncoderAlgorithm(sample.symbols, model), { seed: 0, maxSteps: 1000 })
      expect(enc.state.output).toEqual(Array.from(rangeEncode(sample.symbols, model)))
      expect(enc.steps).toBe(401)
      const dec = runAlgorithm(rangeDecoderAlgorithm(Uint8Array.from(enc.state.output), 400, model), {
        seed: 0,
        maxSteps: 1000,
      })
      expect(dec.state.output).toEqual(sample.symbols)
    }
  })

  it('costs within a few bytes of the information content', () => {
    const enc = runAlgorithm(rangeEncoderAlgorithm(sample.symbols, spec), { seed: 0, maxSteps: 1000 })
    expect(enc.state.output.length * 8).toBeLessThan(enc.state.ideal + 40)
    expect(enc.state.output.length * 8).toBeGreaterThanOrEqual(enc.state.ideal - 8)
  })
})

describe('prefix encoder as an algorithm', () => {
  it('writes the codewords and falls up to a digit per symbol behind the information content', () => {
    const code = huffman(probs)
    const run = runAlgorithm(prefixEncoderAlgorithm(code, sample.symbols, probs), { seed: 0, maxSteps: 1000 })
    expect(run.state.output).toEqual(prefixEncode(code, sample.symbols))
    expect(run.state.output.length).toBeGreaterThanOrEqual(run.state.ideal - 1e-9)
    expect(run.state.output.length).toBeLessThan(run.state.ideal + 400)
    expect(run.steps).toBe(400)
  })

  it('rejects a symbol without a codeword', () => {
    expect(() => prefixEncoderAlgorithm({ arity: 2, codewords: [[0], []] }, [1])).toThrow(/no codeword/)
  })
})
