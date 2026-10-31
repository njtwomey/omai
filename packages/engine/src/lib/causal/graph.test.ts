import { describe, expect, it } from 'vitest'
import { dag } from '../pgm/index.ts'
import { key as makeKey, normal, uniform } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { backdoorEffect, frontdoorEffect, type NamedData } from './index.ts'

// The classic front-door model: an unobserved u drives both treatment x and outcome y; x acts on y only through m.
const n = 6000
const u = Array.from(toFlat(uniform(makeKey(1), [n])), (v) => (v < 0.5 ? 1 : 0))
const ux = Array.from(toFlat(uniform(makeKey(2), [n])))
const um = Array.from(toFlat(uniform(makeKey(3), [n])))
const noise = Array.from(toFlat(normal(makeKey(4), [n])))
const x = u.map((v, i) => ((ux[i] ?? 1) < 0.15 + 0.7 * v ? 1 : 0))
const m = x.map((v, i) => ((um[i] ?? 1) < 0.1 + 0.7 * v ? 1 : 0))
const y = m.map((v, i) => 2 * v + 3 * (u[i] ?? 0) + 0.5 * (noise[i] ?? 0))
const data: NamedData = { u, x, m, y }
const withU = dag([
  ['u', 'x'],
  ['u', 'y'],
  ['x', 'm'],
  ['m', 'y'],
])
// The planted effect of x on y is 0.7 * 2 = 1.4.

describe('effects through a graph', () => {
  it('the naive contrast is confounded by the hidden cause', () => {
    const t1 = y.filter((_, i) => x[i] === 1)
    const t0 = y.filter((_, i) => x[i] === 0)
    const naive = t1.reduce((a, b) => a + b, 0) / t1.length - t0.reduce((a, b) => a + b, 0) / t0.length
    expect(naive - 1.4).toBeGreaterThan(1)
  })

  it('back-door adjustment finds the observed confounder and recovers the effect', () => {
    const r = backdoorEffect(data, withU, 'x', 'y')
    expect(r.adjustment).toEqual(['u'])
    expect(Math.abs(r.effect.estimate.value - 1.4)).toBeLessThan(0.15)
    const dr = backdoorEffect(data, withU, 'x', 'y', { estimator: 'aipw' })
    expect(Math.abs(dr.effect.estimate.value - 1.4)).toBeLessThan(0.15)
  })

  it('with the confounder hidden the back-door is closed and the front-door recovers the effect', () => {
    expect(() => backdoorEffect(data, withU, 'x', 'y', { latent: ['u'] })).toThrow(/back-door/)
    const r = frontdoorEffect(data, withU, 'x', 'y', { latent: ['u'], resamples: 100 })
    expect(r.adjustment).toEqual(['m'])
    expect(Math.abs(r.effect.estimate.value - 1.4)).toBeLessThan(0.2)
    expect(r.effect.estimate.standardError).toBeGreaterThan(0)
    expect(r.effect.estimate.interval?.[0]).toBeLessThan(1.4)
    expect(r.effect.estimate.interval?.[1]).toBeGreaterThan(1.4)
  })

  it('an empty back-door set is a difference of means, and a numeric treatment gets a linear effect', () => {
    const g = dag([['t', 'y']])
    const t = Array.from({ length: 8 }, (_, i) => i % 2)
    const r = backdoorEffect({ t, y: t.map((v, i) => 3 * v + 0.01 * i) }, g, 't', 'y')
    expect(r.adjustment).toEqual([])
    expect(r.effect.estimate.value).toBeCloseTo(3, 1)
    const s = Array.from({ length: 200 }, (_, i) => Math.sin(i))
    const tn = s.map((v, i) => v + 0.5 * Math.cos(3 * i))
    const lin = backdoorEffect(
      { s, t: tn, y: s.map((v, i) => 2 * v + 1.5 * (tn[i] ?? 0)) },
      dag([
        ['s', 't'],
        ['s', 'y'],
        ['t', 'y'],
      ]),
      't',
      'y',
    )
    expect(lin.effect.estimate.value).toBeCloseTo(1.5, 8)
  })

  it('refuses when the graph has no front-door set', () => {
    expect(() => frontdoorEffect(data, dag([['x', 'y']]), 'x', 'y')).toThrow(/front-door/)
  })
})
