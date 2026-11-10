import { describe, expect, it } from 'vitest'
import { expectResumable } from '../../testing/mp-variants.ts'
import { runAlgorithm } from '../trace/index.ts'
import * as M from './index.ts'

const close = (a: number, b: number, tol: number): void => {
  expect(Math.abs(a - b), `${a} vs ${b}`).toBeLessThanOrEqual(tol * (1 + Math.abs(b)))
}

const posterior = (g: M.CompiledGraph, state: M.MpState): { mean: number[]; covariance: number[][] } => {
  const x = M.marginalsOf(g, state)['x'] as Record<string, never>
  return { mean: x['mean'] as unknown as number[], covariance: x['covariance'] as unknown as number[][] }
}

describe('expectation consistent inference: its fixed point is that of EP', () => {
  // Three coordinates coupled by a Gaussian of full covariance, with a probit label on two and a logistic label on one.
  const precision = [2, -0.6, 0.2, -0.6, 1.5, -0.4, 0.2, -0.4, 1.2]
  const g = M.expectationConsistentGraph({
    precision,
    shift: [0.1, -0.2, 0.3],
    sites: [
      { kind: 'probit', label: 1 },
      { kind: 'probit', label: 0, noise: 1.5 },
      { kind: 'logistic', label: 1 },
    ],
  })
  it('the parallel damped iteration and the sequential undamped one reach the same posterior and the same evidence', () => {
    const ec = runAlgorithm(M.ecAlgorithm(g, { tol: 1e-12, maxSweeps: 2000 }), { maxSteps: 2000 })
    const ep = runAlgorithm(M.epAlgorithm(g, { tol: 1e-12, maxSweeps: 2000 }), { maxSteps: 2000 })
    expect(ec.status).toBe('converged')
    expect(ep.status).toBe('converged')
    const a = posterior(g, ec.state)
    const b = posterior(g, ep.state)
    for (let i = 0; i < 3; i++) {
      close(a.mean[i] as number, b.mean[i] as number, 1e-8)
      for (let j = 0; j < 3; j++)
        close((a.covariance[i] as number[])[j] as number, (b.covariance[i] as number[])[j] as number, 1e-8)
    }
    const settings: M.EngineSettings = { ...M.epDefaults, projection: M.momentProjection, tol: 1e-12 }
    close(M.evidenceOf(g, settings, ec.state), M.evidenceOf(g, settings, ep.state), 1e-8)
  })
})

describe('adaptive TAP: an Ising model as a Gaussian times spins', () => {
  const n = 8
  const J = Array.from({ length: n * n }, (_, k) => {
    const i = Math.floor(k / n)
    const j = k % n
    return i === j ? 0 : 0.17 * Math.sin(1 + 3 * Math.min(i, j) + 5 * Math.max(i, j))
  })
  const h = Array.from({ length: n }, (_, i) => 0.4 * Math.cos(2 + 3 * i))
  // exact by enumeration of the 2^8 configurations
  let logZ = Number.NEGATIVE_INFINITY
  const exactMean = new Array<number>(n).fill(0)
  {
    const weights: number[] = []
    const spins: number[][] = []
    for (let a = 0; a < 1 << n; a++) {
      const s = Array.from({ length: n }, (_, i) => ((a >> i) & 1 ? 1 : -1))
      let e = 0
      for (let i = 0; i < n; i++) {
        e += (h[i] as number) * (s[i] as number)
        for (let j = 0; j < n; j++) e += 0.5 * (J[i * n + j] as number) * (s[i] as number) * (s[j] as number)
      }
      weights.push(e)
      spins.push(s)
    }
    const top = Math.max(...weights)
    const z = weights.reduce((acc, w) => acc + Math.exp(w - top), 0)
    logZ = top + Math.log(z)
    weights.forEach((w, a) => {
      for (let i = 0; i < n; i++)
        exactMean[i] = (exactMean[i] as number) + (Math.exp(w - top) / z) * ((spins[a] as number[])[i] as number)
    })
  }
  // The coupling as a Gaussian: P = c I - J (positive definite), and the constant exp(c n / 2) from x_i^2 = 1.
  const c =
    1 +
    Math.max(
      ...Array.from({ length: n }, (_, i) =>
        Array.from({ length: n }, (__, j) => Math.abs(J[i * n + j] as number)).reduce((a, b) => a + b, 0),
      ),
    )
  const precision = J.map((v, k) => (k % (n + 1) === 0 ? c : -v))
  const g = M.expectationConsistentGraph({ precision, sites: h.map((field) => ({ kind: 'spin' as const, field })) })
  const run = runAlgorithm(M.ecAlgorithm(g, { tol: 1e-12, maxSweeps: 3000 }), { maxSteps: 3000 })
  const ec = (M.marginalsOf(g, run.state)['x'] as Record<string, never>)['mean'] as unknown as number[]

  it('converges, and the magnetisations are within 0.01 of the exact ones', () => {
    expect(run.status).toBe('converged')
    for (let i = 0; i < n; i++) expect(Math.abs((ec[i] as number) - (exactMean[i] as number))).toBeLessThan(0.01)
  })
  it('beats naive mean field, which has no Onsager correction', () => {
    let m = h.map((v) => Math.tanh(v))
    for (let it = 0; it < 2000; it++) {
      m = m.map((_, i) => {
        let field = h[i] as number
        for (let j = 0; j < n; j++) field += (J[i * n + j] as number) * (m[j] as number)
        return 0.5 * Math.tanh(field) + 0.5 * (m[i] as number)
      })
    }
    const err = (xs: readonly number[]): number => Math.max(...xs.map((v, i) => Math.abs(v - (exactMean[i] as number))))
    expect(err(ec)).toBeLessThan(err(m) / 3)
  })
  it('the evidence, with the constant, is the log partition function within 0.01', () => {
    const settings: M.EngineSettings = { ...M.epDefaults, projection: M.momentProjection, tol: 1e-12 }
    const evidence = M.evidenceOf(g, settings, run.state)
    const normaliser = M.gaussianMv(n).logPartition([...new Array<number>(n).fill(0), ...precision])
    expect(Math.abs(evidence + normaliser + (c * n) / 2 - logZ)).toBeLessThan(0.01)
  })
  it('the spin site is a closed form: its message matches the brute-force moments of the two-point tilted distribution', () => {
    const cavity = M.gaussianEta(0.3, 1.7)
    const m = M.spin.ep?.({
      mode: 'ep',
      to: 0,
      args: [M.messageArg(M.gaussian, cavity)],
      params: { field: 0.5 },
      power: 1,
    })
    const dens = (x: number): number => Math.exp((-0.5 * (x - 0.3) ** 2) / 1.7)
    const wp = Math.exp(0.5) * dens(1)
    const wm = Math.exp(-0.5) * dens(-1)
    const mean = (wp - wm) / (wp + wm)
    const projected = M.gaussian.fromMoments([mean, 1])
    close((m?.message[0] as number) + (cavity[0] as number) - (projected[0] as number), 0, 1e-12)
    close((m?.message[1] as number) + (cavity[1] as number) - (projected[1] as number), 0, 1e-12)
  })
})

describe('expectation consistent inference: the contract', () => {
  const g = M.expectationConsistentGraph({
    precision: [2, -0.3, -0.3, 2],
    sites: [
      { kind: 'spin', field: 0.6 },
      { kind: 'probit', label: 1 },
    ],
  })
  it('serialises and resumes to the same state', () => {
    expectResumable(M.ecAlgorithm(g, { granularity: 'message', tol: 1e-12 }), 4)
  })
  it('runs anytime in slices that reach the same marginals as the uninterrupted run', () => {
    const alg = M.ecAlgorithm(g, { granularity: 'message', tol: 1e-12 })
    let now = 0
    const first = M.runAnytime(g, alg, { budgetMs: 8, snapshotEveryMs: 2, clock: () => (now += 1) })
    expect(first.status).toBe('budget')
    const second = M.runAnytime(g, alg, { budgetMs: 1e9, clock: () => (now += 1), from: first.slice })
    const whole = runAlgorithm(alg, { maxSteps: 100000 })
    expect(second.snapshot.marginals).toEqual(M.mpSnapshot(g, whole.state).marginals)
  })
  it('is a method with its own knobs, the EP tape, and a graph builder that checks its sizes', () => {
    const method = M.ecMethod(g)
    expect(method.name).toBe('ec')
    expect(Object.keys(method.config.spaces)).toEqual(
      expect.arrayContaining(['schedule', 'damping', 'tol', 'maxSweeps']),
    )
    expect(Object.keys(method.contract as object)).toContain('mp/evidence')
    expect(method.make({ schedule: 'flooding' } as never, { kind: 'key', words: [0, 0] }).name).toBe('ec')
    expect(() =>
      M.expectationConsistentGraph({ precision: [1, 2, 3], sites: [{ kind: 'spin' }, { kind: 'spin' }] }),
    ).toThrow(/precision/)
    expect(() =>
      M.expectationConsistentGraph({
        precision: [1, 0, 0, 1],
        shift: [1],
        sites: [{ kind: 'spin' }, { kind: 'spin' }],
      }),
    ).toThrow(/shift/)
  })
})
