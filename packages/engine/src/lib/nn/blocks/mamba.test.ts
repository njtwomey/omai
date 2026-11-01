import { describe, expect, it } from 'vitest'
import { checkPrimitive } from '../../autodiff/index.ts'
import { key, normal } from '../../random/index.ts'
import { concat, fromStorage, sliceAxis, toFlat, type Value, zeros } from '../../tensor/index.ts'
import type { Params } from '../module/module.ts'
import { causalDepthwiseConv, Mamba2Block, MambaBlock, mambaModel } from './mamba.ts'
import type { MixerState } from './mixer.ts'
import { SequenceModel, type SequenceModelState } from './sequence-model.ts'

const n = (seed: number, shape: number[]): Value => normal(key(seed), shape)
const close = (a: Value, b: Value | number[], digits = 9): void => {
  const x = toFlat(a)
  const y = Array.isArray(b) ? b : toFlat(b)
  expect(x.length).toBe(y.length)
  for (const [i, v] of x.entries()) expect(v).toBeCloseTo(y[i] ?? Number.NaN, digits)
}

/** Feed a sequence one token at a time, threading the state. */
function generate(
  step: (x: Value, state: MixerState | undefined) => { output: Value; state: MixerState },
  x: Value,
): Value {
  let state: MixerState | undefined
  const outs: Value[] = []
  for (let t = 0; t < (x.shape[1] ?? 0); t++) {
    const r = step(sliceAxis(x, 1, t, t + 1), state)
    outs.push(r.output)
    state = r.state
  }
  return concat(outs, 1)
}

describe('the causal depthwise convolution', () => {
  it('continues from its window: two halves equal the whole', () => {
    const x = n(1, [2, 3, 9])
    const w = n(2, [3, 1, 4])
    const whole = causalDepthwiseConv(x, w)
    const head = causalDepthwiseConv(sliceAxis(x, 2, 0, 5), w)
    const tail = causalDepthwiseConv(sliceAxis(x, 2, 5, 9), w, undefined, head.window)
    close(concat([head.y, tail.y], 2), whole.y, 12)
    close(tail.window, whole.window, 12)
  })
})

describe('the Mamba block', () => {
  const block = new MambaBlock(6, { stateSize: 4, key: key(3) })

  it('has the parameters of the paper and the right shapes', () => {
    const names = Object.keys(block.stateDict())
    expect(names).toEqual(
      expect.arrayContaining([
        'in_proj.weight',
        'x_proj.weight',
        'dt_proj.weight',
        'dt_proj.bias',
        'out_proj.weight',
        'aLog',
        'd',
      ]),
    )
    const x = n(4, [2, 7, 6])
    const r = block.forward(x)
    expect(r.output.shape).toEqual([2, 7, 6])
    expect(r.state['conv']).toBeDefined()
    expect((r.state['ssm'] as Value).shape).toEqual([2, 12, 4])
    // A = -exp(aLog) is negative with the S4D-Real spectrum 1 .. N.
    const a = toFlat(block.stateMatrix())
    expect(a.slice(0, 4)).toEqual([-1, -2, -3, -4].map((v) => expect.closeTo(v, 12)))
  })

  it('is causal', () => {
    const x = n(5, [1, 8, 6])
    const y = block.forward(x).output
    const x2 = concat([sliceAxis(x, 1, 0, 5), n(6, [1, 3, 6])], 1)
    close(sliceAxis(block.forward(x2).output, 1, 0, 5), sliceAxis(y, 1, 0, 5), 12)
  })

  it('generation token by token reproduces the parallel pass to 1e-9', () => {
    const x = n(7, [2, 10, 6])
    const par = block.forward(x).output
    const gen = generate((t, s) => block.forward({ x: t, ...(s === undefined ? {} : { state: s }) }), x)
    close(gen, par, 9)
  })

  it('with the step pushed to infinity it is memoryless in time (the convolution reduced to one tap)', () => {
    const m = new MambaBlock(4, { stateSize: 3, convKernel: 3, key: key(8) })
    const p = m.params as Record<string, unknown>
    const inner = 8
    const conv = new Float64Array(inner * 3)
    for (let c = 0; c < inner; c++) conv[c * 3 + 2] = 1
    const dtp = p['dt_proj'] as Params
    m.setParams({
      ...p,
      convWeight: fromStorage(conv, [inner, 1, 3], 'float64'),
      convBias: zeros([inner]),
      dt_proj: { weight: zeros([inner, 1]), bias: fromStorage(new Float64Array(inner).fill(40), [inner], 'float64') },
    })
    void dtp
    const x = n(9, [1, 6, 4])
    const y = m.forward(x).output
    // Replace everything before the last position: its output must not move.
    const x2 = concat([n(10, [1, 5, 4]), sliceAxis(x, 1, 5, 6)], 1)
    close(sliceAxis(m.forward(x2).output, 1, 5, 6), sliceAxis(y, 1, 5, 6), 9)
  })

  it('with A -> 0 it keeps everything it read (a cumulative linear map); with large decay it forgets', () => {
    const make = (aLog: number): MambaBlock => {
      const m = new MambaBlock(4, { stateSize: 3, key: key(11) })
      m.setParams({ ...m.params, aLog: fromStorage(new Float64Array(8 * 3).fill(aLog), [8, 3], 'float64') })
      return m
    }
    const x = n(12, [1, 12, 4])
    const x2 = concat([n(13, [1, 1, 4]), sliceAxis(x, 1, 1, 12)], 1) // change only the first token
    const lastDifference = (m: MambaBlock): number => {
      const a = toFlat(sliceAxis(m.forward(x).output, 1, 11, 12))
      const b = toFlat(sliceAxis(m.forward(x2).output, 1, 11, 12))
      return Math.max(...a.map((v, i) => Math.abs(v - (b[i] ?? 0))))
    }
    // A = -exp(aLog): -60 gives A of order 1e-26 (no decay), +10 gives A = -22026.
    expect(lastDifference(make(-60))).toBeGreaterThan(1e-6)
    expect(lastDifference(make(10))).toBeLessThan(1e-12)
  })

  it('passes the gradient check', () => {
    const m = new MambaBlock(3, { stateSize: 2, convKernel: 2, key: key(14) })
    const f = (x: Value): Value => m.apply(m.params, x).output
    expect(checkPrimitive(f, [n(15, [1, 3, 3])], { rtol: 1e-5, atol: 1e-7 }).failures).toEqual([])
  })
})

describe('the Mamba-2 block', () => {
  const opts = { stateSize: 4, headDim: 4, key: key(21), chunk: 3 }

  it('the recurrent, quadratic and chunked forms of the duality give the same layer', () => {
    const x = n(22, [2, 8, 4])
    const rec = new Mamba2Block(4, { ...opts, form: 'recurrent' }).forward(x)
    const quad = new Mamba2Block(4, { ...opts, form: 'quadratic' }).forward(x)
    const chunked = new Mamba2Block(4, { ...opts, form: 'chunked' }).forward(x)
    close(quad.output, rec.output, 10)
    close(chunked.output, rec.output, 10)
    close(quad.state['ssm'] as Value, rec.state['ssm'] as Value, 10)
  })

  it('generation token by token reproduces the parallel pass, and the layer is causal', () => {
    const m = new Mamba2Block(4, { ...opts, form: 'quadratic' })
    const x = n(23, [1, 9, 4])
    const par = m.forward(x).output
    close(
      generate((t, s) => m.forward({ x: t, ...(s === undefined ? {} : { state: s }) }), x),
      par,
      9,
    )
    const x2 = concat([sliceAxis(x, 1, 0, 4), n(24, [1, 5, 4])], 1)
    close(sliceAxis(m.forward(x2).output, 1, 0, 4), sliceAxis(par, 1, 0, 4), 12)
  })

  it('has per-head scalar decays and the gated norm of the paper', () => {
    const m = new Mamba2Block(4, opts)
    expect(m.heads).toBe(2)
    expect(toFlat(m.decayRates()).every((a) => a <= -1 && a >= -16)).toBe(true)
    expect(Object.keys(m.stateDict())).toContain('norm.weight')
  })

  it('passes the gradient check', () => {
    const m = new Mamba2Block(2, { stateSize: 2, headDim: 2, convKernel: 2, form: 'quadratic', key: key(25) })
    const f = (x: Value): Value => m.apply(m.params, x).output
    expect(checkPrimitive(f, [n(26, [1, 3, 2])], { rtol: 1e-5, atol: 1e-7 }).failures).toEqual([])
  })
})

describe('the sequence model', () => {
  const lm = mambaModel(11, 8, 2, { stateSize: 4, key: key(31) })

  it('is a language model: logits per position, tied embeddings, parallel equals generation', () => {
    const tokens = [[1, 4, 2, 9, 3, 0, 7]]
    const par = lm.forward(tokens)
    expect(par.shape).toEqual([1, 7, 11])
    expect(lm.tieEmbeddings).toBe(true)
    let state: SequenceModelState | undefined
    const outs: Value[] = []
    for (let t = 0; t < 7; t++) {
      const r = lm.run(lm.params, [[tokens[0]?.[t] ?? 0]], state)
      outs.push(r.output)
      state = r.state
    }
    close(concat(outs, 1), par, 9)
  })

  it('is a sequence classifier or regressor over features with pooling', () => {
    const cls = new SequenceModel(8, 1, (d, k) => new MambaBlock(d, { stateSize: 4, key: k }), {
      input: 'features',
      inputSize: 3,
      outputSize: 5,
      pooling: 'mean',
      key: key(32),
    })
    expect(cls.forward(n(33, [4, 6, 3])).shape).toEqual([4, 5])
    const last = new SequenceModel(8, 1, (d, k) => new Mamba2Block(d, { stateSize: 4, headDim: 8, key: k }), {
      input: 'features',
      inputSize: 3,
      outputSize: 1,
      pooling: 'last',
      ffn: 'gated',
      key: key(34),
    })
    expect(last.forward(n(35, [2, 6, 3])).shape).toEqual([2, 1])
  })

  it('trains end to end under grad', () => {
    const tokens = [[1, 4, 2, 9]]
    const f = (x: Value): Value => lm.apply(lm.params, x)
    void f
    const loss = (embed: Value): Value => {
      const params = { ...lm.params, embed: { weight: embed } }
      return lm.apply(params, tokens)
    }
    const w = (lm.params['embed'] as Params)['weight'] as Value
    expect(checkPrimitive(loss, [w], { rtol: 1e-4, atol: 1e-6 }).failures).toEqual([])
  })
})
