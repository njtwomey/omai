import { describe, expect, it } from 'vitest'
import { expectPosteriorMoments } from '../../../testing/probprog.ts'
import { key } from '../../random/index.ts'
import type { TensorLike } from '../../tensor/index.ts'
import * as d from '../model/dists/index.ts'
import { defineModel, ops, toProgram } from '../model/index.ts'
import type { Model } from '../primitives.ts'
import { gibbs } from './gibbs.ts'

/** The mean of a normal centre with a N(0, 3) prior, from observations that share a spread: the conjugate posterior. */
function centre(y: readonly number[], sd: number): { mean: number; sd: number } {
  const precision = 1 / 9 + y.length / sd ** 2
  return { mean: y.reduce((a, b) => a + b, 0) / sd ** 2 / precision, sd: Math.sqrt(1 / precision) }
}

/** How many times a program is run in a Gibbs run: a closed-form update runs it once per sweep, a slice update dozens of times. */
function runsPerSweep(model: Model, init: Record<string, TensorLike>, sweeps: number): number {
  let runs = 0
  const counted: Model = () => {
    runs += 1
    return model()
  }
  gibbs(counted, undefined, { key: key(0), numDraws: sweeps, numWarmup: 0, init })
  return runs / sweeps
}

describe('Gibbs sampling of mixture components with a spread per group', { timeout: 60_000 }, () => {
  const clusters = [
    [-2.4, -1.7, -2.1, -1.9],
    [1.8, 2.5, 2.2, 2.0],
    [6.1, 5.7, 6.4, 5.9],
  ]
  const spreads = [0.5, 1, 0.25]
  // `byRow`: the spread is a value for each row, alternating between 0.5 and 1 inside every group, and not one for each group
  const build = (groups: number, byRow: boolean): Model => {
    const rows = clusters.slice(0, groups).flat()
    const per = spreads.slice(0, groups)
    return toProgram(
      defineModel('spread-per-group', (m) => {
        const row = m.range('row')
        const k = m.range('group', { size: groups })
        const obs = m.data('y', { dims: [row] })
        // a spread for each group, looked up by the group of the row
        const spread = m.data('sd', { dims: [k] })
        const rowSpread = m.data('rowSd', { dims: [row] })
        const mu = m.latent('mu', d.normal(0, 3), { dims: [k] })
        const z = m.latent('z', d.categorical({ probs: per.map(() => 1 / groups) }), { dims: [row] })
        m.observe('y', d.normal(mu.at(z), byRow ? rowSpread : spread.at(z)), obs)
      }).bind({ y: rows, sd: per, rowSd: rows.map((_, i) => 0.5 + (i % 2) * 0.5) }, { sizes: { group: groups } }),
    )
  }

  for (const groups of [2, 3]) {
    it(`draws ${groups} centres from the conjugate posterior of each group's own spread`, () => {
      const model = build(groups, false)
      // the start: each group at its cluster, so that the groups keep their order (the posterior has one mode per labelling)
      const init = {
        mu: clusters.slice(0, groups).map((c) => c.reduce((a, b) => a + b, 0) / c.length),
        z: clusters.slice(0, groups).flatMap((c, g) => c.map(() => g)),
      }
      const post = gibbs(model, undefined, { key: key(3), numDraws: 1500, numWarmup: 50, init })
      for (let g = 0; g < groups; g++) {
        const truth = centre(clusters[g] ?? [], spreads[g] ?? 1)
        expectPosteriorMoments(post, 'mu', truth, { index: g, z: 5 })
      }
      // the centres are updated in closed form, group by group, and not by slice steps over the whole vector
      expect(runsPerSweep(model, init, 20)).toBeLessThan(12)
    })
  }

  it('falls back to a correct update when the spread differs between the rows of a group', () => {
    const model = build(2, true)
    const init = { mu: [-2, 2], z: [0, 0, 0, 0, 1, 1, 1, 1] }
    const post = gibbs(model, undefined, { key: key(4), numDraws: 2500, numWarmup: 100, init })
    // the rows of a group alternate between a spread of 0.5 and one of 1: the weights of the conjugate posterior
    for (let g = 0; g < 2; g++) {
      const y = clusters[g] ?? []
      const w = y.map((_, i) => 1 / (0.5 + (i % 2) * 0.5) ** 2)
      const precision = 1 / 9 + w.reduce((a, b) => a + b, 0)
      const mean = y.reduce((a, b, i) => a + b * (w[i] ?? 0), 0) / precision
      expectPosteriorMoments(post, 'mu', { mean, sd: Math.sqrt(1 / precision) }, { index: g, z: 5 })
    }
  })
})

describe('Gibbs sampling of a latent with a feature axis and a group axis', { timeout: 60_000 }, () => {
  // two features, rows in three clusters (labels 0, 1, 2); a row's centre is the entry of its group in each feature
  const y = [
    [-2.4, -1.7, -2.1, 1.8, 2.5, 2.2, 6.1, 5.7, 6.4],
    [0.1, 0.3, -0.2, 3.1, 2.8, 3.3, -1.9, -2.2, -2.0],
  ]
  const build = (groups: number): Model => {
    const rows = 3 * groups
    return toProgram(
      defineModel('feature-by-group', (m) => {
        const row = m.range('row')
        const feature = m.range('feature', { size: 2 })
        const group = m.range('group', { size: groups })
        const obs = m.data('y', { dims: [feature, row] })
        // the centres: one for each feature and group, picked along the group axis by the row's group
        const mu = m.latent('mu', d.normal(0, 3), { dims: [feature, group] })
        const z = m.latent('z', d.categorical({ probs: Array.from({ length: groups }, () => 1 / groups) }), {
          dims: [row],
        })
        m.observe('y', d.normal(ops.pick(mu, z), 0.5), obs)
      }).bind({ y: y.map((r) => r.slice(0, rows)) }, { sizes: { group: groups, feature: 2 } }),
    )
  }

  for (const groups of [2, 3]) {
    it(`draws the centres of ${groups} groups from their conjugate posterior`, () => {
      const model = build(groups)
      const start = Array.from({ length: 2 }, (_, f) =>
        Array.from({ length: groups }, (_, g) => {
          const c = (y[f] ?? []).slice(3 * g, 3 * g + 3)
          return c.reduce((a, b) => a + b, 0) / 3
        }),
      )
      // each group starts at its cluster (rows 3g to 3g + 2), so that the groups keep their order
      const init = { mu: start, z: Array.from({ length: 3 * groups }, (_, i) => Math.floor(i / 3)) }
      const post = gibbs(model, undefined, { key: key(5), numDraws: 1500, numWarmup: 50, init })
      expect(post.draws?.['mu']?.shape).toEqual([1500, 2, groups])
      for (let f = 0; f < 2; f++)
        for (let g = 0; g < groups; g++)
          expectPosteriorMoments(post, 'mu', centre((y[f] ?? []).slice(3 * g, 3 * g + 3), 0.5), {
            index: f * groups + g,
            z: 5,
          })
      expect(runsPerSweep(model, init, 20)).toBeLessThan(12)
    })
  }
})

describe('Gibbs sampling of a latent with a feature axis and a spread for each row', { timeout: 60_000 }, () => {
  // two features and two groups of three rows; the spread is a value for each row, so it has the row axis only (not y's shape)
  const y = [
    [-2.4, -1.7, -2.1, 1.8, 2.5, 2.2],
    [0.1, 0.3, -0.2, 3.1, 2.8, 3.3],
  ]
  const build = (spread: number[]): Model =>
    toProgram(
      defineModel('feature-by-group-row-spread', (m) => {
        const row = m.range('row')
        const feature = m.range('feature', { size: 2 })
        const group = m.range('group', { size: 2 })
        const obs = m.data('y', { dims: [feature, row] })
        const sd = m.data('sd', { dims: [row] })
        const mu = m.latent('mu', d.normal(0, 3), { dims: [feature, group] })
        const z = m.latent('z', d.categorical({ probs: [0.5, 0.5] }), { dims: [row] })
        m.observe('y', d.normal(ops.pick(mu, z), sd), obs)
      }).bind({ y, sd: spread }, { sizes: { group: 2, feature: 2 } }),
    )
  const init = {
    mu: [
      [-2, 2],
      [0, 3],
    ],
    z: [0, 0, 0, 1, 1, 1],
  }
  const check = (spread: number[]): void => {
    const post = gibbs(build(spread), undefined, { key: key(6), numDraws: 2500, numWarmup: 100, init })
    for (let f = 0; f < 2; f++)
      for (let g = 0; g < 2; g++) {
        const rows = [0, 1, 2].map((i) => 3 * g + i)
        const w = rows.map((i) => 1 / (spread[i] ?? 1) ** 2)
        const precision = 1 / 9 + w.reduce((a, b) => a + b, 0)
        const mean = rows.reduce((a, i, k) => a + ((y[f] ?? [])[i] ?? 0) * (w[k] ?? 0), 0) / precision
        expectPosteriorMoments(post, 'mu', { mean, sd: Math.sqrt(1 / precision) }, { index: f * 2 + g, z: 5 })
      }
  }

  it('takes a spread that is one value for each group of rows as a parameter of the group: a closed form for each group', () => {
    const spread = [0.5, 0.5, 0.5, 1, 1, 1]
    check(spread)
    expect(runsPerSweep(build(spread), init, 20)).toBeLessThan(12)
  })

  it('weighs each row by its own spread when the spreads differ inside a group, as the conjugate posterior does', () => {
    check([0.5, 1, 0.5, 1, 0.5, 1])
  })
})

