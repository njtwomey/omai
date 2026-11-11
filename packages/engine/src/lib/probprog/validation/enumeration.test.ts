// The enumeration oracle of the spike and slab regression against closed forms and a brute-force grid.
import { describe, expect, it } from 'vitest'
import { spikeSlabPosterior } from '../../bayesian-examples/ep-atoms-mixtures.ts'
import { spikeSlabRegression } from './enumeration.ts'

describe('spikeSlabRegression', () => {
  it('one coefficient over several rows is the two-component closed form on the sufficient statistics', () => {
    const x = [0.5, -1, 2, 1.2]
    const y = [0.9, -1.4, 2.3, 1.1]
    const sigma = 0.6
    const sxx = x.reduce((a, v) => a + v * v, 0)
    const yEff = x.reduce((a, v, i) => a + v * (y[i] as number), 0) / sxx
    const closed = spikeSlabPosterior(0.3, 2, 0.1, yEff, sigma / Math.sqrt(sxx))
    const got = spikeSlabRegression({
      X: x.map((v) => [v]),
      y,
      inclusion: 0.3,
      slab: 2,
      spike: 0.1,
      noise: { sd: sigma },
    })
    expect(got.inclusion[0]).toBeCloseTo(closed.p1, 10)
    expect(got.mean[0]).toBeCloseTo(closed.selected.mean, 10)
    expect(got.sd[0]).toBeCloseTo(closed.selected.sd, 10)
    expect(got.subsets).toBe(2)
  })

  it('an unknown precision (a gamma prior on a grid) agrees with a two-dimensional brute-force sum', () => {
    const x = Array.from({ length: 12 }, (_, i) => Math.sin(i + 1) * 1.5)
    const y = x.map((v, i) => 0.9 * v + 0.4 * Math.cos(5 * i))
    const [shape, rate, pi, slab, spike] = [2, 0.5, 0.4, 1.5, 0.2]
    const got = spikeSlabRegression({
      X: x.map((v) => [v]),
      y,
      inclusion: pi,
      slab,
      spike,
      noise: { shape, rate },
      gridPoints: 121,
    })
    // The sum over (w, log tau) of prior x likelihood, on a fine grid.
    let z = 0
    let m = 0
    let t = 0
    let inc = 0
    const norm = (w: number, s: number): number => Math.exp(-0.5 * (w / s) ** 2) / (s * Math.sqrt(2 * Math.PI))
    for (let a = 0; a <= 400; a++) {
      const lt = -6 + (a * 12) / 400
      const tau = Math.exp(lt)
      const prior = Math.exp(shape * lt - rate * tau) // gamma density in log tau, up to a constant
      for (let b = 0; b <= 800; b++) {
        const w = -6 + (b * 12) / 800
        const lik = x.reduce((s, xi, i) => s + ((y[i] as number) - xi * w) ** 2, 0)
        const like = tau ** (x.length / 2) * Math.exp(-0.5 * tau * lik)
        const slabW = pi * norm(w, slab)
        const wt = prior * like * (slabW + (1 - pi) * norm(w, spike))
        z += wt
        m += wt * w
        t += wt * tau
        inc += prior * like * slabW
      }
    }
    expect(got.mean[0]).toBeCloseTo(m / z, 2)
    expect(got.inclusion[0]).toBeCloseTo(inc / z, 2)
    expect(got.precision).toBeCloseTo(t / z, 1)
  })

  it('finds the active coefficients of a small sparse problem', () => {
    const X = Array.from({ length: 30 }, (_, i) => [Math.sin(i), Math.cos(2 * i), Math.sin(3 * i + 1), Math.cos(i / 2)])
    const y = X.map((r, i) => 2 * (r[0] as number) - 1.5 * (r[2] as number) + 0.05 * Math.sin(7 * i))
    const got = spikeSlabRegression({ X, y, inclusion: 0.3, slab: 3, spike: 0.05, noise: { shape: 2, rate: 0.1 } })
    expect(got.inclusion.map((v) => v > 0.5)).toEqual([true, false, true, false])
    expect(got.mean[0]).toBeCloseTo(2, 1)
  })
})
