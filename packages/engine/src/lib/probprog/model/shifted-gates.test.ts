// A Gaussian shifted by a known function of a discrete variable (a wrapped normal's copies a period apart): the shift on the mean
// (`at(mean + shift, j)`) and the shift on the observed value (`N(v[j]; mean, s)` moved across to the mean) compile to the same gate
// of shifted forms (the same arithmetic; EP's projections agree to about 1e-3), against the exact posterior (the discrete states summed out, the means conjugate) by brute force.
import { describe, expect, it } from 'vitest'
import {
  compileGraph,
  epAlgorithm,
  defaultOperators as factorOperators,
  type MpState,
  marginalsOf,
} from '../../message-passing/index.ts'
import { runAlgorithm } from '../../trace/index.ts'
import * as d from './dists/index.ts'
import { defineModel, ops, toFactorGraph } from './index.ts'

const SHIFT = [-6, 0, 6]
const LOC = [1, 4]
const PRIOR_SD = 1.2
const SD = 0.8

/** Two components of three copies each; the joint state j = 3 k + c picks the component k and the copy c. */
function model(form: 'mean' | 'data') {
  return defineModel(`shifted-${form}`, (m) => {
    const row = m.range('row')
    const comp = m.range('comp', { size: 2 })
    const state = m.range('state', { size: 6 })
    const compOf = m.data('compOf', { dims: [state] })
    const shiftOf = m.data('shiftOf', { dims: [state] })
    const mu = m.latent('mu', d.normal(m.data('loc', { dims: [comp] }), PRIOR_SD), { dims: [comp] })
    const j = m.latent('j', d.categorical({ probs: [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6] }), { dims: [row] })
    const hours = m.data('hours', { dims: [row] })
    if (form === 'mean') m.observe('hours', d.normal(ops.at(ops.add(ops.at(mu, compOf), shiftOf), j), SD), hours)
    else m.observe('hours', d.normal(ops.at(ops.at(mu, compOf), j), SD), ops.sub(hours, ops.at(shiftOf, j)))
  })
}

const data = (hours: number[]) => ({
  hours,
  loc: LOC,
  compOf: [0, 0, 0, 1, 1, 1],
  shiftOf: [...SHIFT, ...SHIFT],
})

function ep(form: 'mean' | 'data', hours: number[]) {
  const c = toFactorGraph(model(form).bind(data(hours)))
  expect(c.issues).toEqual([])
  const g = compileGraph(c.graph as never, factorOperators())
  const r = runAlgorithm(epAlgorithm(g, { tol: 1e-12 }), { maxSteps: 400 })
  return { c, marginals: marginalsOf(g, r.state as MpState) }
}

/** One reading: the joint state's posterior, and the posterior means of the two component means, by summing the six states. */
function exact(y: number): { probs: number[]; mean: number[] } {
  const probs: number[] = []
  const mean = [0, 0]
  let total = 0
  for (let j = 0; j < 6; j++) {
    const k = Math.floor(j / 3)
    const v = PRIOR_SD ** 2 + SD ** 2
    const target = y - (SHIFT[j % 3] as number)
    const p = Math.exp((-0.5 * (target - (LOC[k] as number)) ** 2) / v) / Math.sqrt(v)
    probs.push(p)
    total += p
    mean[k] = (mean[k] as number) + (p * ((LOC[k] as number) * SD ** 2 + target * PRIOR_SD ** 2)) / v
  }
  // The mean of component k is its conditional mean when it is the one that generated the reading and its prior mean otherwise.
  const pk = [0, 1].map((k) => probs.slice(3 * k, 3 * k + 3).reduce((a, b) => a + b, 0) / total)
  return {
    probs: probs.map((p) => p / total),
    mean: [0, 1].map((k) => (mean[k] as number) / total + (1 - (pk[k] as number)) * (LOC[k] as number)),
  }
}

describe('a value shifted by a known function of a discrete variable', () => {
  it('the shift on the mean and the shift on the data give the same gate of shifted forms: the same marginals for one reading', () => {
    const a = ep('mean', [5.2])
    const b = ep('data', [5.2])
    expect(a.c.graph?.factors.filter((f) => f.operator === 'select').length).toBe(1)
    expect(b.c.graph?.factors.filter((f) => f.operator === 'select').length).toBe(1)
    const pa = (a.marginals['j[0]'] as { probs: number[] }).probs
    const pb = (b.marginals['j[0]'] as { probs: number[] }).probs
    for (let k = 0; k < 6; k++) expect(Math.abs((pa[k] as number) - (pb[k] as number))).toBeLessThan(0.01)
    for (const i of [0, 1])
      expect(
        Math.abs(
          (a.marginals[`mu[${i}]`] as { mean: number }).mean - (b.marginals[`mu[${i}]`] as { mean: number }).mean,
        ),
      ).toBeLessThan(0.05)
  })

  it('the state posterior and the means of one reading are near the exact ones (EP projects the gate onto a Gaussian)', () => {
    const y = 5.2
    const want = exact(y)
    const { marginals } = ep('data', [y])
    const probs = (marginals['j[0]'] as { probs: number[] }).probs
    const dominant = want.probs.indexOf(Math.max(...want.probs))
    expect(probs.indexOf(Math.max(...probs))).toBe(dominant)
    for (let k = 0; k < 6; k++) expect(Math.abs((probs[k] as number) - (want.probs[k] as number))).toBeLessThan(0.1)
    for (const i of [0, 1])
      expect(Math.abs((marginals[`mu[${i}]`] as { mean: number }).mean - (want.mean[i] as number))).toBeLessThan(0.15)
  })

  it('a copy a period away is chosen when the reading sits near the shifted mean: 0.1 is the copy 6 below component 0 at 1', () => {
    // 0.1 reads as component 1 (mean 4) shifted down... or component 0 (mean 1) on time: the evidence is on the on-time copies.
    const { marginals } = ep('mean', [-4.9])
    const probs = (marginals['j[0]'] as { probs: number[] }).probs
    // -4.9 + 6 = 1.1 is a reading of component 0 on the late-minus copy (shift -6): state 0.
    expect(probs.indexOf(Math.max(...probs))).toBe(0)
  })

  it('several readings: the two forms agree closely and both recover which copy each reading is', () => {
    const hours = [1.1, 0.7, -4.9, 4.2, 10.1, 3.8]
    const a = ep('mean', hours)
    const b = ep('data', hours)
    for (const i of [0, 1])
      expect(
        Math.abs(
          (a.marginals[`mu[${i}]`] as { mean: number }).mean - (b.marginals[`mu[${i}]`] as { mean: number }).mean,
        ),
      ).toBeLessThan(0.05)
    const states = hours.map((_, r) => {
      const p = (b.marginals[`j[${r}]`] as { probs: number[] }).probs
      return p.indexOf(Math.max(...p))
    })
    expect(states).toEqual([1, 1, 0, 4, 5, 4])
  })

  it('a shift that is not a function of the variable the mean is gated by is refused with its reason, not mis-compiled', () => {
    const bad = defineModel('shifted-bad', (m) => {
      const row = m.range('row')
      const mu = m.latent('mu', d.normal(0, 1))
      const c = m.latent('c', d.normal(0, 1), { dims: [row] })
      m.observe('hours', d.normal(mu, SD), ops.add(m.data('hours', { dims: [row] }), c))
    })
    const g = toFactorGraph(bad.bind({ hours: [1, 2] }))
    expect(g.graph).toBeUndefined()
    expect(g.issues.length).toBeGreaterThan(0)
  })
})
