import { describe, expect, it } from 'vitest'
import { key } from '../../random/index.ts'
import { runAlgorithm } from '../../trace/index.ts'
import * as P from '../index.ts'

/** The exact mean energy and magnetisation per site of a small lattice by summing over every configuration. */
function exact(L: number, q: number, beta: number): { energy: number; magnetisation: number } {
  const n = L * L
  const spins = new Int32Array(n)
  let z = 0
  let e = 0
  let m = 0
  const total = q ** n
  for (let code = 0; code < total; code++) {
    let c = code
    for (let i = 0; i < n; i++) {
      spins[i] = c % q
      c = Math.floor(c / q)
    }
    const conf = { L, q, spins }
    const energy = P.pottsEnergyPerSite(conf)
    const w = Math.exp(-beta * energy * n)
    z += w
    e += w * energy
    m += w * P.pottsMagnetisation(conf)
  }
  return { energy: e / z, magnetisation: m / z }
}

/** The mean of a recorded series and its standard error by batch means. */
function meanAndError(x: readonly number[]): { mean: number; se: number } {
  const batches = 20
  const size = Math.floor(x.length / batches)
  const means = Array.from(
    { length: batches },
    (_, b) => x.slice(b * size, (b + 1) * size).reduce((a, v) => a + v, 0) / size,
  )
  const mean = means.reduce((a, v) => a + v, 0) / batches
  const variance = means.reduce((a, v) => a + (v - mean) ** 2, 0) / (batches - 1)
  return { mean, se: Math.sqrt(variance / batches) }
}

function run(
  update: P.PottsUpdate,
  opts: P.PottsSamplerOptions,
  steps: number,
  seed = 1,
): { energy: number[]; magnetisation: number[] } {
  const alg = P.pottsSamplerAlgorithm(update, opts)
  let state = alg.init(key(seed))
  const energy: number[] = []
  const magnetisation: number[] = []
  const sweepKey = key(seed + 1000)
  for (let t = 0; t < steps; t++) {
    state = alg.step(state, { t, key: foldKey(sweepKey, t) } as never)
    if (t >= 200) {
      energy.push(state.energy)
      magnetisation.push(state.magnetisation)
    }
  }
  return { energy, magnetisation }
}

import { foldIn } from '../../random/index.ts'

const foldKey = foldIn

describe('critical values', () => {
  it('has the Onsager critical coupling and critical energy', () => {
    expect(P.pottsCriticalCoupling(2)).toBeCloseTo(0.5 * Math.log(1 + Math.SQRT2) * 2, 12)
    expect(P.pottsCriticalCoupling(3)).toBeCloseTo(Math.log(1 + Math.sqrt(3)), 12)
    expect(P.ISING_CRITICAL_ENERGY_PER_SITE).toBeCloseTo(-1.7071067811865475, 12)
    expect(P.onsagerMagnetisation(P.pottsCriticalCoupling(2) - 0.01)).toBe(0)
    expect(P.onsagerMagnetisation(1.0)).toBeCloseTo((1 - Math.sinh(1) ** -4) ** 0.125, 12)
  })
})

describe('detailed balance against exact enumeration', () => {
  const cases = [
    { L: 4, q: 2, beta: 0.6 },
    { L: 3, q: 3, beta: 0.8 },
  ]
  for (const { L, q, beta } of cases) {
    const truth = exact(L, q, beta)
    for (const update of ['swendsen-wang', 'wolff', 'metropolis'] as const) {
      it(`${update} reproduces the exact energy and magnetisation of the ${L}x${L} ${q}-state lattice`, () => {
        const r = run(update, { L, q, beta, start: 'random' }, update === 'metropolis' ? 30000 : 12000)
        const e = meanAndError(r.energy)
        const m = meanAndError(r.magnetisation)
        expect(Math.abs(e.mean - truth.energy)).toBeLessThan(5 * e.se + 1e-3)
        expect(Math.abs(m.mean - truth.magnetisation)).toBeLessThan(5 * m.se + 1e-3)
      })
    }
  }
})

describe('the critical lattice', () => {
  const beta = P.pottsCriticalCoupling(2)
  it('has the Onsager energy at the critical coupling (Swendsen-Wang, L = 32)', () => {
    const r = run('swendsen-wang', { L: 32, beta, start: 'random' }, 1500)
    expect(Math.abs(meanAndError(r.energy).mean - P.ISING_CRITICAL_ENERGY_PER_SITE)).toBeLessThan(0.01)
  })
  it('has the Onsager spontaneous magnetisation below the critical temperature (Wolff, L = 32, beta = 1)', () => {
    const r = run('wolff', { L: 32, beta: 1.0 }, 800)
    expect(Math.abs(meanAndError(r.magnetisation).mean - P.onsagerMagnetisation(1.0))).toBeLessThan(0.01)
  })
  it('decorrelates far faster than single-spin Metropolis, and the gap widens with the lattice (critical slowing down)', () => {
    const tau = (u: P.PottsUpdate, L: number): number =>
      P.integratedAutocorrelationTime(run(u, { L, beta, start: 'random' }, 4000, 7).magnetisation)
    const metropolis32 = tau('metropolis', 32)
    const swendsenWang32 = tau('swendsen-wang', 32)
    const wolff32 = tau('wolff', 32)
    expect(swendsenWang32).toBeLessThan(6)
    expect(wolff32).toBeLessThan(3)
    expect(metropolis32).toBeGreaterThan(5 * swendsenWang32)
    expect(metropolis32).toBeGreaterThan(20 * wolff32)
    // Metropolis slows down with L (about L^2); the cluster update barely does.
    expect(metropolis32).toBeGreaterThan(3 * tau('metropolis', 8))
    expect(swendsenWang32).toBeLessThan(3 * tau('swendsen-wang', 8))
  })
  it('serialises the sampler state', () => {
    const alg = P.pottsSamplerAlgorithm('swendsen-wang', { L: 6, beta })
    const r = runAlgorithm(alg, { maxSteps: 5, seed: 3 })
    expect(alg.deserialise?.(alg.serialise?.(r.state) as never)).toEqual(r.state)
  })
})

describe('updates', () => {
  it('Wolff flips a cluster to a different spin and leaves the rest', () => {
    const c = { L: 4, q: 3, spins: new Int32Array(16) }
    const size = P.wolffUpdate(c, 0, () => 0)
    // At beta = 0 the cluster is the seed alone.
    expect(size).toBe(1)
    expect(Array.from(c.spins).filter((s) => s !== 0)).toHaveLength(1)
  })
  it('Swendsen-Wang at infinite coupling keeps each ordered lattice as one cluster', () => {
    const c = { L: 5, q: 4, spins: new Int32Array(25).fill(2) }
    expect(P.swendsenWangUpdate(c, 80, () => 0.3)).toBe(1)
    expect(new Set(c.spins).size).toBe(1)
  })
})
