import { describe, expect, it } from 'vitest'
import { toFlat } from '../../tensor/index.ts'
import { policies, recorder, runAlgorithm } from '../../trace/index.ts'
import { arnoldi, arnoldiAlgorithm, eigsh, eigshAlgorithm, krylovContract, svds, svdsAlgorithm } from './eigen.ts'
import { spectrumTargets } from './eigen-options.ts'

const viaJson = <T>(x: T): T => JSON.parse(JSON.stringify(x)) as T
const diag = (d: number[]): number[][] => d.map((v, i) => d.map((_, j) => (i === j ? v : 0)))
const spectrum = Array.from({ length: 30 }, (_, i) => 1 + i * 0.5)
const matrix = diag(spectrum)

describe('arnoldiAlgorithm', () => {
  it('takes one Krylov vector per step and agrees with arnoldi', () => {
    const a = [
      [1, 2, 0, 1],
      [0, 3, 1, 0],
      [1, 0, 2, 1],
      [0, 1, 0, 4],
    ]
    const run = runAlgorithm(arnoldiAlgorithm(a, 3), { maxSteps: 10 })
    expect([run.steps, run.state.basis.length, run.state.columns.length]).toEqual([3, 4, 3])
    const f = arnoldi(a, 3)
    expect(f.hessenberg.shape).toEqual([4, 3])
    // The basis is orthonormal and A Q_m = Q_{m+1} H_m.
    const q = run.state.basis
    for (const [i, u] of q.entries())
      for (const [j, v] of q.entries())
        expect(u.reduce((s, x, k) => s + x * (v[k] ?? 0), 0)).toBeCloseTo(i === j ? 1 : 0, 10)
    const h = toFlat(f.hessenberg)
    for (let j = 0; j < 3; j++) {
      const aq = a.map((row) => row.reduce((s, x, k) => s + x * (q[j]?.[k] ?? 0), 0))
      for (let r = 0; r < 4; r++) {
        let rhs = 0
        for (let i = 0; i <= j + 1; i++) rhs += (h[i * 3 + j] ?? 0) * (q[i]?.[r] ?? 0)
        expect(aq[r]).toBeCloseTo(rhs, 10)
      }
    }
  })

  it('stops with a breakdown on an invariant subspace, resumes from JSON, and logs', () => {
    const run = runAlgorithm(arnoldiAlgorithm(diag([1, 1, 1]), 3, { v0: [1, 0, 0] }), { maxSteps: 10 })
    expect([run.status, run.steps]).toEqual(['converged', 1])
    const alg = arnoldiAlgorithm(matrix, 8)
    const whole = runAlgorithm(alg, { maxSteps: 20 })
    const part = runAlgorithm(alg, { maxSteps: 3 }).state
    const rest = runAlgorithm(alg, {
      maxSteps: 20,
      start: { state: viaJson(alg.serialise?.(part) ?? null), step: 3, serialised: true },
    })
    expect(rest.state.columns).toEqual(whole.state.columns)
    const rec = recorder()
    runAlgorithm(alg, { maxSteps: 20, tracer: rec, policy: { ...policies.minimal, metrics: true, detail: 'full' } })
    const names = new Set(rec.events.flatMap((e) => (e.type === 'metric' ? [e.name] : [])))
    expect(names.has('krylov/basis')).toBe(true)
    expect(Object.keys(krylovContract)).toHaveLength(5)
  })
})

describe('eigshAlgorithm', () => {
  it('locks pairs round by round until k converge, agreeing with eigsh', () => {
    const run = runAlgorithm(eigshAlgorithm(matrix, 3), { maxSteps: 50 })
    expect(run.status).toBe('converged')
    expect(run.steps).toBe(run.state.locked.length > 0 ? run.steps : 0)
    expect(run.state.locked).toHaveLength(3)
    const r = eigsh(matrix, 3)
    const top = [...spectrum]
      .sort((x, y) => y - x)
      .slice(0, 3)
      .sort((x, y) => x - y)
    for (const [i, v] of toFlat(r.values).entries()) expect(v).toBeCloseTo(top[i] ?? 0, 8)
    for (const res of r.residuals) expect(res).toBeLessThan(1e-8)
  })

  it('restarts when a round does not converge and serialises mid-run', () => {
    const alg = eigshAlgorithm(matrix, 2, { ncv: 5, which: 'SA' })
    const whole = runAlgorithm(alg, { maxSteps: 400 })
    expect(whole.status).toBe('converged')
    expect(whole.state.restarts).toBeGreaterThan(0)
    const part = runAlgorithm(alg, { maxSteps: 2 }).state
    const rest = runAlgorithm(alg, {
      maxSteps: 400,
      start: { state: viaJson(alg.serialise?.(part) ?? null), step: 2, serialised: true },
    })
    expect(rest.state.restarts).toBe(whole.state.restarts)
    for (const [i, v] of rest.state.lockedValues.entries()) expect(v).toBeCloseTo(whole.state.lockedValues[i] ?? 0, 9)
    expect(whole.state.lockedValues.sort((x, y) => x - y)[0]).toBeCloseTo(1, 8)
  })

  it('fails past the restart limit (and eigsh throws) and rejects a bad k', () => {
    const run = runAlgorithm(eigshAlgorithm(matrix, 2, { ncv: 4, maxRestarts: 1, tol: 1e-14 }), { maxSteps: 50 })
    expect(run.status).toBe('failed')
    expect(() => eigsh(matrix, 2, { ncv: 4, maxRestarts: 1, tol: 1e-14 })).toThrow(/restart/)
    expect(() => eigshAlgorithm(matrix, 30)).toThrow(/k must/)
  })

  it('logs the Ritz values, residuals, restarts and locked pairs', () => {
    const rec = recorder()
    runAlgorithm(eigshAlgorithm(matrix, 2, { ncv: 12 }), {
      maxSteps: 400,
      tracer: rec,
      policy: { ...policies.minimal, metrics: true, tensors: true, roles: ['output'], detail: 'full' },
    })
    const locked = rec.events.flatMap((e) => (e.type === 'metric' && e.name === 'krylov/locked' ? [e.value] : []))
    expect(locked.at(-1)).toBe(2)
    expect(locked.every((v, i) => i === 0 || v >= (locked[i - 1] ?? 0))).toBe(true)
  })
})

describe('svdsAlgorithm', () => {
  it('finds the squares of the singular values and agrees with svds', () => {
    const a = [
      [3, 0, 0],
      [0, 4, 0],
      [0, 0, 1],
      [0, 0, 0],
    ]
    const run = runAlgorithm(svdsAlgorithm(a, 1), { maxSteps: 50 })
    expect(run.status).toBe('converged')
    expect(run.state.lockedValues[0]).toBeCloseTo(16, 8)
    expect(toFlat(svds(a, 1).s)[0]).toBeCloseTo(4, 8)
  })
})

describe('spectrumTargets', () => {
  it('names the ends and builds eigsh options', () => {
    expect(spectrumTargets.names).toEqual(['largest-magnitude', 'largest', 'smallest'])
    expect(spectrumTargets.make({ kind: 'smallest', tol: 1e-8 })).toMatchObject({ which: 'SA', tol: 1e-8 })
  })
})
