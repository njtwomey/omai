// EP on a mixture of two separated clusters with everything unknown (the weights, the means, one shared spread, a label per row), the
// model of the Gaussian mixture example on its defaults: 150 rows of two normals with unit spread 3 apart, standardised. From flat
// messages and no damping the first sweeps commit every row to a side, and the run falls into the symmetric fixed point (both
// components the same, one wide blob, spread 1.8 in the data's units where it is 1). A few damped first sweeps (`warmup`, `auto` on a
// graph with exchangeable labels) let the components separate first, and the run settles on the clusters.
import { describe, expect, it } from 'vitest'
import { compileGraph, defaultOperators, hasExchangeableLabels } from '../../message-passing/index.ts'
import { normal as drawNormal, uniform as drawUniform, foldIn, key } from '../../random/index.ts'
import { toFlat } from '../../tensor/index.ts'
import { session } from '../../trace/index.ts'
import * as d from '../model/dists/index.ts'
import { toFactorGraph } from '../model/factors.ts'
import { defineModel } from '../model/index.ts'
import { gaussianMixtureModel } from '../models/mixture.ts'
import { epMultiStartMethod } from './ep-multistart.ts'
import { epSpecMethod } from './mp.ts'

/** Two clusters at `-separation / 2` and `separation / 2`, unit spread, equal shares, standardised: the data and what the truth is on that scale. */
function clusters(rows: number, seed: number, separation = 3) {
  const k = foldIn(key(seed), 0)
  const u = toFlat(drawUniform(foldIn(k, 0), [rows]))
  const e = toFlat(drawNormal(foldIn(k, 1), [rows]))
  const raw = Array.from(u, (v, i) => (v < 0.5 ? -separation / 2 : separation / 2) + (e[i] as number))
  const mean = raw.reduce((a, b) => a + b, 0) / rows
  const scale = Math.sqrt(raw.reduce((a, b) => a + (b - mean) ** 2, 0) / rows)
  return {
    y: raw.map((v) => (v - mean) / scale),
    means: [(-separation / 2 - mean) / scale, (separation / 2 - mean) / scale],
    sd: 1 / scale,
  }
}

const data = clusters(150, 1)
const bound = (ordered: boolean) => gaussianMixtureModel({ components: 2, ordered }).bind({ y: data.y })

/** The posterior summary of a run of `epSpecMethod`: the means, sorted, and the spread. */
function fit(ordered: boolean, config: Record<string, unknown>) {
  const method = epSpecMethod(bound(ordered))
  const run = session(method, { steps: 200, mode: 'eager', config: { tol: 1e-6, ...config } })
  const rows = method.posterior?.(run.state, run.config).summary() ?? []
  return {
    mu: rows
      .filter((r) => r.site === 'mu')
      .map((r) => r.mean)
      .sort((a, b) => a - b),
    sigma: rows.find((r) => r.site === 'sigma')?.mean ?? Number.NaN,
  }
}

describe('EP on a two-cluster mixture with unknown weights, means and spread', () => {
  it('a graph with a latent label that selects a component has exchangeable labels, one without does not', () => {
    const g = compileGraph(toFactorGraph(bound(true)).graph as never, defaultOperators())
    expect(hasExchangeableLabels(g)).toBe(true)
    const plain = defineModel('plain', (m) => {
      const x = m.latent('x', d.normal(0, 1))
      m.observe('y', d.normal(x, 1), m.data('y'))
    }).bind({ y: 0.3 })
    expect(hasExchangeableLabels(compileGraph(toFactorGraph(plain).graph as never, defaultOperators()))).toBe(false)
  })

  it('without the warm-up the run falls into the symmetric fixed point: both means near zero, the spread of the whole data', () => {
    const off = fit(true, { warmup: 'off' })
    expect(Math.abs((off.mu[1] ?? 0) - (off.mu[0] ?? 0))).toBeLessThan(0.5)
    expect(off.sigma).toBeGreaterThan(0.9)
  })

  it('with the warm-up (the default) it recovers the clusters: the means and the spread within a sd of the truth', () => {
    const on = fit(true, {})
    expect(Math.abs((on.mu[0] ?? 0) - (data.means[0] as number))).toBeLessThan(0.15)
    expect(Math.abs((on.mu[1] ?? 0) - (data.means[1] as number))).toBeLessThan(0.15)
    expect(Math.abs(on.sigma - data.sd)).toBeLessThan(0.1)
  })

  it('free means: one labelling of the two clusters, with the same means', () => {
    const free = fit(false, {})
    expect(Math.abs((free.mu[0] ?? 0) - (data.means[0] as number))).toBeLessThan(0.2)
    expect(Math.abs((free.mu[1] ?? 0) - (data.means[1] as number))).toBeLessThan(0.2)
  })

  // Four starts of EP on 150 rows: 10 s here, over 30 s on the CI runner.
  it('the multi-start method reports the clusters too, not the symmetric reading that has an evidence', {
    timeout: 120_000,
  }, () => {
    const method = epMultiStartMethod(bound(true))
    const run = session(method, { steps: 4, mode: 'eager', config: { starts: 4, tol: 1e-6, maxSweeps: 200 } })
    const post = method.posterior?.(run.state, run.config)
    const mu = (post?.summary() ?? []).filter((r) => r.site === 'mu').map((r) => r.mean)
    expect(Math.abs((mu[0] ?? 0) - (data.means[0] as number))).toBeLessThan(0.15)
    expect(Math.abs((mu[1] ?? 0) - (data.means[1] as number))).toBeLessThan(0.15)
  })
})
