import { describe, expect, it } from 'vitest'
import { checkPrimitive } from '../../autodiff/index.ts'
import { key, normal, uniform } from '../../random/index.ts'
import { add, concat, fromStorage, ones, sliceAxis, toFlat, type Value } from '../../tensor/index.ts'
import { elu } from './activations.ts'
import { linearAttention } from './attention.ts'
import { decayedLinearAttention, wkv } from './linear-recurrent.ts'

const n = (seed: number, shape: number[]): Value => normal(key(seed), shape)
const close = (a: Value, b: Value | number[], digits = 10): void => {
  const x = toFlat(a)
  const y = Array.isArray(b) ? b : toFlat(b)
  expect(x.length).toBe(y.length)
  for (const [i, v] of x.entries()) expect(v).toBeCloseTo(y[i] ?? Number.NaN, digits)
}

const [bs, heads, len, dk, dv] = [2, 2, 7, 3, 4]
const q = n(1, [bs, heads, len, dk])
const k = n(2, [bs, heads, len, dk])
const v = n(3, [bs, heads, len, dv])
const s0 = n(4, [bs, heads, dk, dv])
const scalar = uniform(key(5), [bs, heads, len], { low: -1, high: 0 })
const vector = uniform(key(6), [bs, heads, len, dk], { low: -1, high: 0 })

describe('attention with a decaying state', () => {
  for (const [name, logDecay] of [
    ['no decay', undefined],
    ['one decay per head', scalar],
    ['one decay per key dimension', vector],
  ] as const)
    it(`the recurrent, parallel and chunked forms agree (${name})`, () => {
      const input = { q, k, v, ...(logDecay === undefined ? {} : { logDecay }), state: s0 }
      const rec = decayedLinearAttention(input, 'recurrent')
      for (const [form, chunk] of [
        ['parallel', 8],
        ['chunked', 3],
        ['chunked', 1],
      ] as const) {
        const r = decayedLinearAttention(input, form, chunk)
        close(r.output, rec.output)
        close(r.state, rec.state)
      }
    })

  it('without decay it is the causal linear attention of the library, with the normaliser as an extra value', () => {
    const fq = add(elu(q), 1)
    const fk = add(elu(k), 1)
    const aug = concat([v, ones([bs, heads, len, 1])], 3)
    const r = decayedLinearAttention({ q: fq, k: fk, v: aug }, 'parallel').output
    const num = sliceAxis(r, 3, 0, dv)
    const den = sliceAxis(r, 3, dv, dv + 1)
    const want = linearAttention(q, k, v, { causal: true, eps: 0 })
    const got = toFlat(num).map((x, i) => x / (toFlat(den)[Math.floor(i / dv)] ?? 1))
    close(want, got, 9)
  })

  it('a constant decay is retention: the weight of token s on query t is gamma^(t-s)', () => {
    const gamma = 0.8
    const L = 4
    const x = ones([1, 1, L, 1])
    const logDecay = fromStorage(new Float64Array(L).fill(Math.log(gamma)), [1, 1, L], 'float64')
    const out = toFlat(decayedLinearAttention({ q: x, k: x, v: x, logDecay }, 'parallel').output)
    for (let t = 0; t < L; t++) {
      let want = 0
      for (let s = 0; s <= t; s++) want += gamma ** (t - s)
      expect(out[t]).toBeCloseTo(want, 12)
    }
  })

  it('passes the gradient check in every form', () => {
    const small = {
      q: n(7, [1, 1, 3, 2]),
      k: n(8, [1, 1, 3, 2]),
      v: n(9, [1, 1, 3, 2]),
    }
    for (const form of ['recurrent', 'parallel', 'chunked'] as const)
      for (const g of [
        uniform(key(10), [1, 1, 3], { low: -1, high: 0 }),
        uniform(key(11), [1, 1, 3, 2], { low: -1, high: 0 }),
      ])
        expect(
          checkPrimitive(
            (a, b, c, d) => decayedLinearAttention({ q: a, k: b, v: c, logDecay: d }, form, 2).output,
            [small.q, small.k, small.v, g],
            { rtol: 1e-5, atol: 1e-7 },
          ).failures,
        ).toEqual([])
  })
})

describe('the WKV recurrence', () => {
  const [b2, l2, d2] = [2, 6, 3]
  const kk = n(21, [b2, l2, d2])
  const vv = n(22, [b2, l2, d2])
  const w = uniform(key(23), [d2], { low: 0.05, high: 1.5 })
  const u = n(24, [d2])

  it('equals its definition as a weighted average', () => {
    const out = toFlat(wkv(kk, vv, w, u).output)
    const [K, V, W, U] = [toFlat(kk), toFlat(vv), toFlat(w), toFlat(u)]
    for (let t = 0; t < l2; t++)
      for (let c = 0; c < d2; c++) {
        let num = Math.exp((U[c] ?? 0) + (K[(1 * l2 + t) * d2 + c] ?? 0)) * (V[(1 * l2 + t) * d2 + c] ?? 0)
        let den = Math.exp((U[c] ?? 0) + (K[(1 * l2 + t) * d2 + c] ?? 0))
        for (let i = 0; i < t; i++) {
          const wt = Math.exp(-(t - 1 - i) * (W[c] ?? 0) + (K[(1 * l2 + i) * d2 + c] ?? 0))
          num += wt * (V[(1 * l2 + i) * d2 + c] ?? 0)
          den += wt
        }
        expect(out[(1 * l2 + t) * d2 + c]).toBeCloseTo(num / den, 10)
      }
  })

  it('resumes from its state and does not overflow on large keys', () => {
    const whole = wkv(kk, vv, w, u)
    const head = wkv(sliceAxis(kk, 1, 0, 2), sliceAxis(vv, 1, 0, 2), w, u)
    const tail = wkv(sliceAxis(kk, 1, 2, 6), sliceAxis(vv, 1, 2, 6), w, u, head.state)
    close(concat([head.output, tail.output], 1), whole.output, 12)
    const big = wkv(add(kk, 500), vv, w, u).output
    expect(toFlat(big).every((x) => Number.isFinite(x))).toBe(true)
  })

  it('passes the gradient check', () => {
    expect(
      checkPrimitive(
        (a, b, c, d) => wkv(a, b, c, d).output,
        [n(25, [1, 4, 2]), n(26, [1, 4, 2]), uniform(key(27), [2], { low: 0.1, high: 1 }), n(28, [2])],
        {
          rtol: 1e-5,
          atol: 1e-7,
        },
      ).failures,
    ).toEqual([])
  })
})
