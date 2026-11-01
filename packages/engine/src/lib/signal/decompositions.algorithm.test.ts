import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { policies, recorder, runAlgorithm, stateToJson } from '../trace/index.ts'
import {
  ceemdan,
  ceemdanAlgorithm,
  eemd,
  eemdAlgorithm,
  emd,
  emdAlgorithm,
  emdContract,
  fixedSiftStop,
  sdStop,
} from './decompositions.ts'
import { siftStops } from './emd-options.ts'

const n = 200
const x = Array.from({ length: n }, (_, i) => Math.sin(i / 3) + 0.5 * Math.sin(i / 20) + i / 100)
const energy = (a: ArrayLike<number>): number => Array.from(a).reduce((p, q) => p + q * q, 0)
const total = (a: ArrayLike<number>): number => Array.from(a).reduce((p, q) => p + q, 0)

describe('emd, eemd and ceemdan give what they gave before, exactly', () => {
  // Taken from the previous implementation: squared sums, residue sums and samples of the results.
  it('emd', () => {
    const d = emd(x)
    expect(d.imfs.shape).toEqual([3, n])
    expect(energy(toFlat(d.imfs))).toBeCloseTo(1789.1876646256826, 6)
    expect(toFlat(d.imfs)[50]).toBeCloseTo(-0.858790290736344, 6)
    const d2 = emd(x, { maxImfs: 2, sdThreshold: 0.05, maxSifts: 7 })
    expect(d2.imfs.shape).toEqual([2, n])
    expect(energy(toFlat(d2.imfs))).toBeCloseTo(1114.2978722472908, 6)
    expect(total(toFlat(d2.residue))).toBeCloseTo(-51.22662065418421, 6)
  })

  it('eemd and ceemdan', () => {
    const e = eemd(x, { key: key(1), trials: 6, maxImfs: 3 })
    expect([e.imfs.shape, energy(toFlat(e.imfs)), total(toFlat(e.residue))]).toEqual([
      [3, n],
      expect.closeTo(487.3300803722296, 6),
      expect.closeTo(81.61295402567936, 6),
    ])
    const auto = eemd(x, { key: key(1), trials: 4 })
    expect([auto.imfs.shape, energy(toFlat(auto.imfs))]).toEqual([[4, n], expect.closeTo(605.3293728220918, 6)])
    const c = ceemdan(x, { key: key(2), trials: 5, maxImfs: 3 })
    expect([c.imfs.shape, energy(toFlat(c.imfs)), total(toFlat(c.residue))]).toEqual([
      [3, n],
      expect.closeTo(5.70839657211073, 6),
      expect.closeTo(225.32438020685416, 6),
    ])
    const cAuto = ceemdan(x, { key: key(2), trials: 4 })
    expect([cAuto.imfs.shape, energy(toFlat(cAuto.imfs))]).toEqual([[8, n], expect.closeTo(485.8370893905951, 6)])
  })
})

describe('the EMD algorithms: state, paths, invariants', () => {
  it('emd: a state that is JSON resumes to the same decomposition', () => {
    const alg = emdAlgorithm(x)
    const whole = runAlgorithm(alg, { maxSteps: 1000 })
    const half = runAlgorithm(alg, { maxSteps: 6 }).state
    const stored = JSON.parse(JSON.stringify(alg.serialise?.(half)))
    const rest = runAlgorithm(alg, { maxSteps: 994, start: { state: stored, step: 6, serialised: true } })
    expect(stateToJson(rest.state)).toEqual(stateToJson(whole.state))
    expect(whole.status).toBe('converged')
  })

  it('emd: logs the candidate, sifts, SD, residue and envelopes each step', () => {
    const rec = recorder()
    runAlgorithm(emdAlgorithm(x), {
      maxSteps: 5,
      tracer: rec,
      policy: { ...policies.minimal, metrics: true, tensors: true, roles: ['output', 'state'], detail: 'full' },
    })
    const metric = (m: string): number[] =>
      rec.events.flatMap((e) => (e.type === 'metric' && e.name === m ? [e.value] : []))
    expect(metric('emd/sift')).toHaveLength(5)
    expect(metric('emd/sd')).toHaveLength(5)
    for (const p of ['emd/imf', 'emd/residue', 'emd/envelope'])
      expect(rec.events.filter((e) => e.type === 'tensor' && e.name === p)).toHaveLength(5)
    expect(Object.keys(emdContract)).toContain('emd/envelope')
  })

  it('emd: the modes and the residue sum to the signal at every step, at every step', () => {
    const alg = emdAlgorithm(x)
    let s = alg.init(undefined as never)
    const ctx = { t: 0, wants: () => false, log: () => undefined, emit: () => undefined } as never
    for (let i = 0; i < 12 && !s.finished; i++) {
      s = alg.step(s, ctx)
      const sum = s.residue.map((r, k) => r + s.imfs.reduce((a, m) => a + (m[k] ?? 0), 0))
      for (const [k, v] of sum.entries()) expect(v).toBeCloseTo(x[k] ?? 0, 10)
    }
  })

  it('eemd: a state that is JSON resumes, and the average is built one member at a time', () => {
    const alg = eemdAlgorithm(x, { key: key(1), trials: 6, maxImfs: 3 })
    const whole = runAlgorithm(alg, { maxSteps: 10 })
    const half = runAlgorithm(alg, { maxSteps: 2 }).state
    expect(half.trial).toBe(2)
    const stored = JSON.parse(JSON.stringify(alg.serialise?.(half)))
    const rest = runAlgorithm(alg, { maxSteps: 8, start: { state: stored, step: 2, serialised: true } })
    expect(stateToJson(rest.state)).toEqual(stateToJson(whole.state))
    expect(whole.state.trial).toBe(6)
  })

  it('ceemdan: a state that is JSON resumes (the noise cache is recomputed), one mode a step', () => {
    const make = () => ceemdanAlgorithm(x, { key: key(2), trials: 4, maxImfs: 3 })
    const alg = make()
    const whole = runAlgorithm(alg, { maxSteps: 10 })
    const half = runAlgorithm(alg, { maxSteps: 1 }).state
    expect(half.imfs).toHaveLength(1)
    const stored = JSON.parse(JSON.stringify(alg.serialise?.(half)))
    const rest = runAlgorithm(make(), { maxSteps: 9, start: { state: stored, step: 1, serialised: true } })
    expect(stateToJson(rest.state)).toEqual(stateToJson(whole.state))
    expect(whole.state.imfs).toHaveLength(3)
  })
})

describe('siftStops: the slot', () => {
  it('each rule runs and gives a complete decomposition', () => {
    for (const kind of siftStops.names) {
      const stopping = siftStops.make({ kind })
      const s = runAlgorithm(emdAlgorithm(x, { stopping }), { maxSteps: 5000 }).state
      const sum = s.residue.map((r, k) => r + s.imfs.reduce((a, m) => a + (m[k] ?? 0), 0))
      for (const [k, v] of sum.entries()) expect(v).toBeCloseTo(x[k] ?? 0, 9)
      expect(s.imfs.length).toBeGreaterThan(0)
    }
  })

  it('a fixed rule spends exactly that many sifts on each mode', () => {
    const alg = emdAlgorithm(x, { stopping: fixedSiftStop(3), maxImfs: 1 })
    const steps = runAlgorithm(alg, { maxSteps: 100 }).steps
    expect(steps).toBe(3)
    expect(sdStop(0.2).name).toBe('sd')
    expect(siftStops.where((p) => p.fixedCount)).toEqual(['fixed'])
  })
})
