import { describe, expect, it } from 'vitest'
import { checkPrimitive } from '../../autodiff/index.ts'
import { key, normal } from '../../random/index.ts'
import { concat, fromStorage, sliceAxis, toFlat, type Value, zeros } from '../../tensor/index.ts'
import { discretise, discretiseDiagonal, lru, ssmKernel } from '../functional/state-space.ts'
import type { Params } from '../module/module.ts'
import { nplrHippo } from './hippo.ts'
import { LRUMixer, S4, S4D, S5 } from './state-space.ts'

const n = (seed: number, shape: number[]): Value => normal(key(seed), shape)
const close = (a: Value, b: Value | number[], digits = 9): void => {
  const x = toFlat(a)
  const y = Array.isArray(b) ? b : toFlat(b)
  expect(x.length).toBe(y.length)
  for (const [i, v] of x.entries()) expect(v).toBeCloseTo(y[i] ?? Number.NaN, digits)
}
const state = (s: Readonly<Record<string, unknown>>, k: string): Value => s[k] as Value

describe('S4D', () => {
  for (const init of ['lin', 'inv', 'real'] as const)
    for (const discretisation of ['zoh', 'bilinear'] as const)
      it(`the convolution and the recurrence agree to 1e-9 (${init}, ${discretisation})`, () => {
        const opts = { stateSize: 8, init, discretisation, key: key(1) }
        const conv = new S4D(3, { ...opts, mode: 'convolution' })
        const rec = new S4D(3, { ...opts, mode: 'recurrence' })
        const x = n(2, [2, 12, 3])
        const a = conv.forward(x)
        const b = rec.forward(x)
        close(a.output, b.output, 9)
        close(state(a.state, 're'), state(b.state, 're'), 9)
        close(state(a.state, 'im'), state(b.state, 'im'), 9)
      })

  it('generation one token at a time reproduces the parallel pass', () => {
    const m = new S4D(2, { stateSize: 6, key: key(3) })
    const x = n(4, [1, 9, 2])
    const whole = m.forward(x)
    let st = m.initialState(1)
    const outs: Value[] = []
    for (let t = 0; t < 9; t++) {
      const r = m.forward({ x: sliceAxis(x, 1, t, t + 1), state: st })
      outs.push(r.output)
      st = r.state
    }
    close(concat(outs, 1), whole.output, 9)
    close(state(st, 're'), state(whole.state, 're'), 9)
  })

  it('with real eigenvalues its kernel is the one of the functional SSM', () => {
    const m = new S4D(1, { stateSize: 4, init: 'real', key: key(5) })
    const p: Params = { ...m.params, cIm: zeros([1, 2]), d: zeros([1]) }
    m.setParams(p)
    const impulse = zeros([1, 7, 1]).data.slice()
    impulse[0] = 1
    const y = toFlat(m.forward(fromStorage(impulse, [1, 7, 1], 'float64')).output)
    const dt = Math.exp(toFlat(p['logDt'] as Value)[0] ?? 0)
    const lam = toFlat(p['logNegRe'] as Value).map((v) => -Math.exp(v))
    const d = discretiseDiagonal(lam, [1, 1], dt)
    const cRe = toFlat(p['cRe'] as Value).map((v) => 2 * v)
    close(ssmKernel(d, cRe, 7), y, 10)
  })

  it('is memoryless when the step is large: y = K_0 u + D u', () => {
    const m = new S4D(2, { stateSize: 4, init: 'real', key: key(6) })
    m.setParams({ ...m.params, logDt: fromStorage(Float64Array.of(8, 8), [2], 'float64') })
    const x = n(7, [1, 5, 2])
    const y = toFlat(m.forward(x).output)
    const p = m.params
    const lam = toFlat(p['logNegRe'] as Value).map((v) => -Math.exp(v))
    const cRe = toFlat(p['cRe'] as Value)
    const d = toFlat(p['d'] as Value)
    const xs = toFlat(x)
    for (let t = 0; t < 5; t++)
      for (let c = 0; c < 2; c++) {
        // K_0 = 2 sum_n C_n (1 - exp(dt lambda)) / (-lambda) with exp(dt lambda) ~ 0.
        let k0 = 0
        for (let j = 0; j < 2; j++) k0 += 2 * (cRe[c * 2 + j] ?? 0) * (-1 / (lam[c * 2 + j] ?? 1))
        expect(y[t * 2 + c]).toBeCloseTo(k0 * (xs[t * 2 + c] ?? 0) + (d[c] ?? 0) * (xs[t * 2 + c] ?? 0), 6)
      }
  })

  it('passes the gradient check in both modes', () => {
    for (const mode of ['convolution', 'recurrence'] as const) {
      const m = new S4D(2, { stateSize: 4, mode, key: key(8) })
      const f = (x: Value): Value => m.apply(m.params, x).output
      expect(checkPrimitive(f, [n(9, [1, 4, 2])], { rtol: 1e-5, atol: 1e-7 }).failures).toEqual([])
    }
  })
})

describe('S4', () => {
  for (const measure of ['legs', 'legt', 'lagt'] as const)
    it(`the convolution and the recurrence agree to 1e-9 (${measure})`, () => {
      const opts = { stateSize: 6, measure, key: key(11) }
      const conv = new S4(2, { ...opts, mode: 'convolution' })
      const rec = new S4(2, { ...opts, mode: 'recurrence' })
      const x = n(12, [2, 10, 2])
      const a = conv.forward(x)
      const b = rec.forward(x)
      close(a.output, b.output, 9)
      close(state(a.state, 're'), state(b.state, 're'), 9)
      close(state(a.state, 'im'), state(b.state, 'im'), 9)
    })

  it('its eigenbasis system is the dense HiPPO system: same bilinear kernel as the real SSM', () => {
    const size = 4
    const f = nplrHippo('legs', size)
    const m = new S4(1, { stateSize: size, key: key(13) })
    const dt = 0.05
    // C~ = C V for a real C.
    const c = [0.3, -1, 0.7, 0.2]
    const cRe = new Float64Array(size)
    const cIm = new Float64Array(size)
    for (let j = 0; j < size; j++)
      for (let q = 0; q < size; q++) {
        cRe[j] = (cRe[j] ?? 0) + (c[q] ?? 0) * (f.v.re.data[q * size + j] ?? 0)
        cIm[j] = (cIm[j] ?? 0) + (c[q] ?? 0) * (f.v.im.data[q * size + j] ?? 0)
      }
    m.setParams({
      ...m.params,
      logDt: fromStorage(Float64Array.of(Math.log(dt)), [1], 'float64'),
      cRe: fromStorage(cRe, [1, size], 'float64'),
      cIm: fromStorage(cIm, [1, size], 'float64'),
      d: zeros([1]),
    })
    const impulse = new Float64Array(8)
    impulse[0] = 1
    const y = m.forward(fromStorage(impulse, [1, 8, 1], 'float64')).output
    close(ssmKernel(discretise(f.a, f.b, dt, 'bilinear'), c, 8), y, 8)
  })

  it('passes the gradient check', () => {
    const m = new S4(2, { stateSize: 4, key: key(14) })
    const f = (x: Value): Value => m.apply(m.params, x).output
    expect(checkPrimitive(f, [n(15, [1, 3, 2])], { rtol: 1e-5, atol: 1e-7 }).failures).toEqual([])
  })
})

describe('S5', () => {
  it('the parallel scan and the recurrence agree to 1e-9, with blocks', () => {
    const opts = { stateSize: 8, blocks: 2, key: key(21) }
    const par = new S5(3, { ...opts, mode: 'convolution' })
    const rec = new S5(3, { ...opts, mode: 'recurrence' })
    const x = n(22, [2, 11, 3])
    const a = par.forward(x)
    const b = rec.forward(x)
    close(a.output, b.output, 9)
    close(state(a.state, 're'), state(b.state, 're'), 9)
  })

  it('resuming from the state reproduces the whole pass', () => {
    const m = new S5(2, { stateSize: 4, key: key(23) })
    const x = n(24, [1, 8, 2])
    const whole = m.forward(x)
    const head = m.forward(sliceAxis(x, 1, 0, 5))
    const tail = m.forward({ x: sliceAxis(x, 1, 5, 8), state: head.state })
    close(concat([head.output, tail.output], 1), whole.output, 9)
    close(state(tail.state, 'im'), state(whole.state, 'im'), 9)
  })

  it('passes the gradient check', () => {
    const m = new S5(2, { stateSize: 4, key: key(25) })
    const f = (x: Value): Value => m.apply(m.params, x).output
    expect(checkPrimitive(f, [n(26, [1, 5, 2])], { rtol: 1e-5, atol: 1e-7 }).failures).toEqual([])
  })
})

describe('the linear recurrent unit as a mixer', () => {
  it('equals the functional lru, is stable, and resumes from its state', () => {
    const m = new LRUMixer(3, { stateSize: 4, key: key(31) })
    const x = n(32, [2, 6, 3])
    const out = m.forward(x)
    close(out.output, lru(m.params as never, x), 12)
    const modulus = toFlat(m.eigenvalues().re).map((v, i) => Math.hypot(v, toFlat(m.eigenvalues().im)[i] ?? 0))
    for (const v of modulus) expect(v).toBeLessThan(1)
    const head = m.forward(sliceAxis(x, 1, 0, 2))
    const tail = m.forward({ x: sliceAxis(x, 1, 2, 6), state: head.state })
    close(concat([head.output, tail.output], 1), out.output, 12)
  })

  it('re-initialising gives the weights a fresh module has', () => {
    const a = new LRUMixer(3, { stateSize: 4, key: key(33) })
    const b = new LRUMixer(3, { stateSize: 4, key: key(34) })
    b.resetParameters(key(33))
    close(b.params['cRe'] as Value, a.params['cRe'] as Value, 12)
  })
})
