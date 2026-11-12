// Stochastic volatility with the level, the persistence and the innovation precision all unknown: one Gaussian per parameter is
// far too narrow for the level and the persistence (they are coupled through the whole chain), and EP inside a grid over the
// persistence (the level and the precision inside each EP run) is not, run as a walk outwards from the middle of the range with
// each node started from its neighbour's messages. The references are a NUTS run of the same model on the same data (two chains,
// 400 warm-up and 400 draws) and a brute-force posterior: a point-mass filter for the likelihood of each (level, persistence,
// precision) on a 26 by 30 by 24 grid, with the prior (120 states over -6.5 to 4.5, a few minutes in a scratch script).
import { describe, expect, it } from 'vitest'
import { session } from '../../trace/index.ts'
import * as d from '../model/dists/index.ts'
import { defineModel, ops } from '../model/index.ts'
import { epGridMethod } from './ep-grid.ts'
import { epSpecMethod } from './mp.ts'

const T = 60
const TRUTH = { mu: -1, phi: 0.9, tau: 11.1 }

/** A uniform in (0, 1) from a hashed index, and a standard normal from two of them (Box-Muller). */
const unit = (i: number, s: number): number => {
  const x = Math.sin(i * 12.9898 + s * 78.233) * 43758.5453
  return (x - Math.floor(x)) * 0.999998 + 1e-6
}
const gauss = (i: number, s: number): number =>
  Math.sqrt(-2 * Math.log(unit(i, s))) * Math.cos(2 * Math.PI * unit(i + 7919, s * 3 + 1))

/** Returns of a log-variance that follows an AR(1) around `mu` (seed 1). */
function simulate(seed: number): number[] {
  const sigma = TRUTH.tau ** -0.5
  let h = TRUTH.mu + (sigma / Math.sqrt(1 - TRUTH.phi ** 2)) * gauss(0, seed)
  const y: number[] = []
  for (let t = 0; t < T; t++) {
    if (t > 0) h = TRUTH.mu + TRUTH.phi * (h - TRUTH.mu) + sigma * gauss(2 * t, seed)
    y.push(Math.exp(h / 2) * gauss(2 * t + 1, seed))
  }
  return y
}

const model = defineModel('sv-learned', (b) => {
  const y = b.data('y', { dims: ['time'] })
  const mu = b.latent('mu', d.normal(-1, 1))
  const phi = b.latent('phi', d.normal(0.8, 0.3))
  const tau = b.latent('tau', d.gamma(3, 0.3))
  b.scan('h', {
    over: 'time',
    init: d.normal(-1, 1),
    step: ({ prev }) => d.normal(prev.sub(mu).mul(phi).add(mu), tau.pow(-0.5)),
    emit: ({ state, observe }) => observe('y', d.normal(0, ops.exp(state.mul(0.5))), y),
  })
})

/** The posterior of the three parameters in a NUTS run of the same data. */
const NUTS = { mu: { mean: -0.918, sd: 0.425 }, phi: { mean: 0.731, sd: 0.199 }, tau: { mean: 11.511, sd: 6.077 } }

/** The brute-force posterior of the same data (seed 1): the NUTS run agrees with it to a fifth of a sd. */
const GRID = { mu: { mean: -0.913, sd: 0.402 }, phi: { mean: 0.74, sd: 0.192 }, tau: { mean: 10.921, sd: 5.661 } }

describe('stochastic volatility with unknown level, persistence and precision', () => {
  const bound = model.bind({ y: simulate(1) })

  it('plain EP is far too narrow about the level (a fifth of the sampler sd or less): the parameters are coupled through the chain', () => {
    const ep = epSpecMethod(bound)
    const run = session(ep, { config: { tol: 1e-4, damping: 0.5 }, seed: 1, steps: 400, mode: 'eager' })
    const level = ep
      .posterior?.(run.state, run.config)
      .summary()
      .find((r) => r.site === 'mu')
    expect(level?.sd ?? Number.POSITIVE_INFINITY).toBeLessThan(0.4 * NUTS.mu.sd)
  }, 60_000)

  it('EP in a walk over the persistence has the brute-force means and the spread of the persistence and the precision, in a dozen nodes', () => {
    const grid = epGridMethod(bound, { over: ['phi'] })
    const run = session(grid, {
      config: {
        points: 21,
        tol: 1e-4,
        damping: 0.2,
        maxSweeps: 120,
        order: 'walk',
        window: 10,
        warm: true,
        maxNodes: 40,
      },
      seed: 1,
      steps: 40,
      mode: 'eager',
    })
    expect(run.state.nodes.length).toBeLessThan(16)
    const post = grid.posterior?.(run.state, run.config)
    const row = (name: string) => post?.summary().find((r) => r.site === name)
    for (const name of ['mu', 'phi', 'tau'] as const) {
      const ref = GRID[name]
      expect(Math.abs((row(name)?.mean ?? Number.NaN) - ref.mean), `${name} mean`).toBeLessThan(0.15 * ref.sd)
    }
    // The persistence and the precision are as wide as the truth (within a fifth and a tenth); the level is wider (about 1.5 times:
    // the EP belief of the level inside a node is wider than the exact one), never narrower.
    expect((row('phi')?.sd ?? Number.NaN) / GRID.phi.sd).toBeGreaterThan(0.75)
    expect((row('phi')?.sd ?? Number.NaN) / GRID.phi.sd).toBeLessThan(1.15)
    expect(Math.abs((row('tau')?.sd ?? Number.NaN) / GRID.tau.sd - 1)).toBeLessThan(0.1)
    expect((row('mu')?.sd ?? Number.NaN) / GRID.mu.sd).toBeGreaterThan(1)
    expect((row('mu')?.sd ?? Number.NaN) / GRID.mu.sd).toBeLessThan(2)
    // and the truth is inside each 90% interval
    for (const name of ['mu', 'phi', 'tau'] as const) {
      const r = row(name)
      expect(r?.lower ?? Number.NaN, `${name} lower`).toBeLessThan(TRUTH[name])
      expect(r?.upper ?? Number.NaN, `${name} upper`).toBeGreaterThan(TRUTH[name])
    }
  }, 60_000)
})
