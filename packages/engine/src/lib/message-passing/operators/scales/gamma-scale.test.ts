import { describe, expect, it } from 'vitest'
import { add, arange, div, mul, toFlat } from '../../../tensor/index.ts'
import { messageArg } from '../../core/args.ts'
import { gamma, gammaEta } from '../../core/families.ts'
import { lgammaOf } from '../../numerics/_scalar.ts'
import { gammaProduct, gammaRate } from './gamma-scale.ts'

const N = 900
const grid = (m: number, sd: number): number[] => toFlat(add(m - 9 * sd, div(mul(arange(N), 18 * sd), N - 1)))
const logq = (t: number, shape: number, rate: number): number => shape * t - rate * Math.exp(t)

/** The tilted E log and E of each variable under a weight on a grid of (log x, log y). */
function tilted(axes: readonly [readonly number[], readonly number[]], logw: (tx: number, ty: number) => number) {
  let z = 0
  let lx = 0
  let ex = 0
  let ly = 0
  let ey = 0
  let lp = 0
  let ep = 0
  for (const tx of axes[0])
    for (const ty of axes[1]) {
      const w = Math.exp(logw(tx, ty))
      z += w
      lx += w * tx
      ex += w * Math.exp(tx)
      ly += w * ty
      ey += w * Math.exp(ty)
      lp += w * (tx + ty)
      ep += w * Math.exp(tx + ty)
    }
  return { x: [lx / z, ex / z], y: [ly / z, ey / z], p: [lp / z, ep / z], z }
}

const moments = (cavity: readonly number[], message: readonly number[]): readonly number[] =>
  gamma.moments(cavity.map((v, i) => v + (message[i] ?? 0)))

describe('gamma-rate', () => {
  it('the messages to x and r match a two-dimensional grid of the tilted density', () => {
    const [a, c] = [0.7, 1.6]
    const qx = [2.5, 1.2] as const
    const qr = [3.1, 2.4] as const
    // Axes in the log of each variable: the cavity's own range.
    const axis = (shape: number, rate: number): number[] => grid(Math.log(shape / rate), 1.2 / Math.sqrt(shape))
    const ref = tilted([axis(...qx), axis(...qr)], (tx, tr) => {
      const x = Math.exp(tx)
      const r = Math.exp(tr)
      // The factor Gamma(x; a, c r) in (log x, log r) coordinates, with the cavities' densities.
      return logq(tx, ...qx) + logq(tr, ...qr) + a * Math.log(c * r) + (a - 1) * tx - c * r * x - lgammaOf(a)
    })
    const args = [messageArg(gamma, gammaEta(...qx)), messageArg(gamma, gammaEta(...qr))]
    const toR = gammaRate.ep?.({ mode: 'ep', to: 1, args, params: { shape: a, scale: c }, power: 1 }).message ?? []
    const toX = gammaRate.ep?.({ mode: 'ep', to: 0, args, params: { shape: a, scale: c }, power: 1 }).message ?? []
    const mr = moments(gammaEta(...qr), toR)
    const mx = moments(gammaEta(...qx), toX)
    expect(mr[0]).toBeCloseTo(ref.y[0] as number, 3)
    expect(mr[1]).toBeCloseTo(ref.y[1] as number, 3)
    expect(mx[0]).toBeCloseTo(ref.x[0] as number, 3)
    expect(mx[1]).toBeCloseTo(ref.x[1] as number, 3)
  })

  it('a flat cavity of x says nothing about the rate, and the variational messages are the conjugate ones', () => {
    const args = [messageArg(gamma, gamma.flat), messageArg(gamma, gammaEta(3, 2))]
    const toR = gammaRate.ep?.({ mode: 'ep', to: 1, args, params: { shape: 0.5 }, power: 1 }).message ?? []
    expect(Math.abs(toR[0] ?? 1)).toBeLessThan(1e-6)
    expect(Math.abs(toR[1] ?? 1)).toBeLessThan(1e-6)
    const vargs = [messageArg(gamma, gammaEta(2, 3)), messageArg(gamma, gammaEta(4, 2))]
    // E[r] = 2, E[x] = 2 / 3: the message to x has rate c E[r] and to r the shape a and rate c E[x].
    expect(
      gammaRate.vmp?.({ mode: 'vmp', to: 0, args: vargs, params: { shape: 0.5, scale: 3 }, power: 1 }).message,
    ).toEqual([-0.5, -6])
    const toRv =
      gammaRate.vmp?.({ mode: 'vmp', to: 1, args: vargs, params: { shape: 0.5, scale: 3 }, power: 1 }).message ?? []
    expect(toRv[0]).toBeCloseTo(0.5, 12)
    expect(toRv[1]).toBeCloseTo(-3 * (2 / 3), 12)
  })
})

describe('gamma-product', () => {
  it('the messages to p, u and tau match a two-dimensional grid, with an informative cavity of p', () => {
    const qu = [2.2, 1.1] as const
    const qt = [3.4, 2.7] as const
    const qp = [2.8, 1.9] as const
    const axis = (shape: number, rate: number): number[] => grid(Math.log(shape / rate), 1.2 / Math.sqrt(shape))
    const ref = tilted(
      [axis(...qu), axis(...qt)],
      (tu, tt) => logq(tu, ...qu) + logq(tt, ...qt) + (qp[0] - 1) * (tu + tt) - qp[1] * Math.exp(tu + tt),
    )
    const args = [
      messageArg(gamma, gammaEta(...qp)),
      messageArg(gamma, gammaEta(...qu)),
      messageArg(gamma, gammaEta(...qt)),
    ]
    const out = (to: number) => gammaProduct.ep?.({ mode: 'ep', to, args, params: {}, power: 1 }).message ?? []
    const mu = moments(gammaEta(...qu), out(1))
    const mt = moments(gammaEta(...qt), out(2))
    const mp = moments(gammaEta(...qp), out(0))
    expect(mu[0]).toBeCloseTo(ref.x[0] as number, 3)
    expect(mu[1]).toBeCloseTo(ref.x[1] as number, 3)
    expect(mt[0]).toBeCloseTo(ref.y[0] as number, 3)
    expect(mt[1]).toBeCloseTo(ref.y[1] as number, 3)
    expect(mp[0]).toBeCloseTo(ref.p[0] as number, 3)
    expect(mp[1]).toBeCloseTo(ref.p[1] as number, 3)
  })

  it('with a flat cavity of p the message to p is the pushforward: E log p = E log u + E log tau, E p = E u E tau', () => {
    const qu = gammaEta(3, 2)
    const qt = gammaEta(5, 4)
    const args = [messageArg(gamma, gamma.flat), messageArg(gamma, qu), messageArg(gamma, qt)]
    const m = gamma.moments(gammaProduct.ep?.({ mode: 'ep', to: 0, args, params: {}, power: 1 }).message ?? [])
    const mu = gamma.moments(qu)
    const mt = gamma.moments(qt)
    expect(m[0]).toBeCloseTo((mu[0] ?? 0) + (mt[0] ?? 0), 6)
    expect(m[1]).toBeCloseTo((mu[1] ?? 0) * (mt[1] ?? 0), 6)
  })
})
