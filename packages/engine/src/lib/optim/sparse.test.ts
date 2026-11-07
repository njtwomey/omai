import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../testing/golden.ts'
import { key, normal } from '../random/index.ts'
import { div, fromStorage, matmul, type Tensor, toFlat } from '../tensor/index.ts'
import { messages, metricSeries, recorder, tracePolicy } from '../trace/index.ts'
import * as O from './index.ts'

const fixture = { cases: ['optim/sparse', 'optim/dictionary'].flatMap((f) => loadFixture(f).cases) }
const cases = (fn: string) =>
  fixture.cases.filter((c) => c.fn === fn).map((c) => ({ ...c, inputs: c.inputs.map((v) => decode(v) as Tensor) }))

/** A Gaussian measurement matrix with unit-norm columns and an s-sparse signal. */
function compressed(seed: number, m: number, k: number, s: number): { d: Tensor; x: Float64Array; y: Tensor } {
  const raw = normal(key(seed), [m, k])
  const d = div(raw, Math.sqrt(m)) as Tensor
  const x = new Float64Array(k)
  for (let j = 0; j < s; j++) x[(j * 7 + 3) % k] = (j % 2 === 0 ? 1 : -1) * (1 + j)
  const y = matmul(d, fromStorage(x, [k, 1], 'float64')) as Tensor
  return { d, x, y: fromStorage(Float64Array.from(y.data), [m], 'float64') }
}
const round = (xs: readonly number[], p = 6): number[] => xs.map((v) => Math.round(v * 10 ** p) / 10 ** p + 0)

describe('sparse approximation against sklearn and linprog', () => {
  for (const c of cases('omp')) {
    it(c.name, () => {
      const [d, y] = c.inputs as [Tensor, Tensor]
      const r = O.orthogonalMatchingPursuit(d, y, c.params as O.PursuitOptions)
      expectMatches(r.coef, c.output, c.rtol, c.atol, c.name)
    })
  }
  for (const c of cases('sparseEncode')) {
    it(c.name, () => {
      const [x, dict] = c.inputs as [Tensor, Tensor]
      expectMatches(
        O.sparseEncode(x, dict, { ...(c.params as O.SparseEncodeOptions), tol: 1e-12 }),
        c.output,
        c.rtol,
        c.atol,
        c.name,
      )
    })
  }
  for (const c of cases('bpdn')) {
    it(c.name, () => {
      const [d, y] = c.inputs as [Tensor, Tensor]
      const lam = c.params['lam'] as number
      expectMatches(O.basisPursuitDenoising(d, y, lam, { tol: 1e-14 }).coef, c.output, c.rtol, c.atol, c.name)
      // The same lasso by ADMM and FISTA.
      expectMatches(O.admmLasso(d, y, lam, { absTol: 1e-13, relTol: 1e-13 }).z, c.output, 1e-6, 1e-7, c.name)
    })
  }
  for (const c of cases('elasticNetCd')) {
    it(c.name, () => {
      const [d, y] = c.inputs as [Tensor, Tensor]
      const r = O.lassoCoordinateDescent(d, y, c.params['lam'] as number, { l2: c.params['l2'] as number, tol: 1e-14 })
      expectMatches(r.coef, c.output, c.rtol, c.atol, c.name)
    })
  }
  for (const c of cases('basisPursuit')) {
    it(c.name, () => {
      const [d, y] = c.inputs as [Tensor, Tensor]
      const r = O.basisPursuit(d, y)
      expect(r.converged).toBe(true)
      expectMatches(r.coef, c.output, c.rtol, c.atol, c.name)
    })
  }
})

describe('greedy pursuits and thresholding', () => {
  it('matching pursuit conserves energy with unit atoms and traces its selections', () => {
    const { d, y } = compressed(1, 20, 30, 3)
    const rec = recorder()
    const r = O.matchingPursuit(d, y, { maxIterations: 50, tracer: rec, policy: tracePolicy({ messages: true }) })
    const h = r.history
    expect(h.every((v, i) => i === 0 || v <= (h[i - 1] ?? Infinity) + 1e-12)).toBe(true)
    expect(messages(rec.run(), 'select').length).toBeGreaterThan(0)
    expect(O.matchingPursuit(d, y, { nNonzeroCoefs: 2 }).support.length).toBe(2)
  })

  it('OMP, CoSaMP and IHT recover a sparse signal from noiseless measurements', () => {
    const { d, x, y } = compressed(2, 40, 80, 4)
    const want = round(Array.from(x))
    expect(round(toFlat(O.orthogonalMatchingPursuit(d, y, { nNonzeroCoefs: 4 }).coef))).toEqual(want)
    const co = O.cosamp(d, y, 4)
    expect(co.converged).toBe(true)
    expect(round(toFlat(co.coef))).toEqual(want)
    const rec = recorder()
    const iht = O.iterativeHardThresholding(d, y, 4, { tracer: rec, maxIterations: 2000 })
    expect(round(toFlat(iht.coef), 5)).toEqual(round(want, 5))
    expect(metricSeries(rec.run(), 'residualNorm').values.length).toBeGreaterThan(1)
    const bp = O.basisPursuit(d, y)
    expect(round(toFlat(bp.coef), 5)).toEqual(round(want, 5))
  })

  // The linear-dependence test was absolute (`alpha^2 < eps`): a dictionary or signal in small units selected nothing
  // and reported success. It is the cosine of the atom with the signal that is negligible, not the correlation.
  it('OMP selects the same atoms and coefficients at the scales 1e-9, 1 and 1e9 of the data', () => {
    for (const k of [1e-9, 1, 1e9]) {
      const d = [
        [1, 0],
        [0, 1],
      ].map((row) => row.map((v) => v * k))
      const y = [1, 2].map((v) => v * k)
      const r = O.orthogonalMatchingPursuit(d, y, { nNonzeroCoefs: 2 })
      expect(r.support, `scale ${k}`).toEqual([1, 0])
      expect(round(toFlat(r.coef)), `scale ${k}`).toEqual([1, 2])
      expect(r.iterations).toBe(2)
      // A repeated atom is a linear dependence at any scale: the run stops early, with the atoms it has.
      const twin = O.orthogonalMatchingPursuit(
        [
          [k, k],
          [0, 0],
        ],
        [k, 0],
        { nNonzeroCoefs: 2 },
      )
      expect(twin.support, `scale ${k}`).toEqual([0])
    }
  })

  it('plain IHT with a fixed step below 1 / |D|^2 also converges', () => {
    const { d, x, y } = compressed(3, 40, 60, 2)
    const r = O.iterativeHardThresholding(d, y, 2, {
      normalized: false,
      stepSize: 0.2,
      maxIterations: 5000,
      tol: 1e-12,
    })
    expect(round(toFlat(r.coef), 4)).toEqual(round(Array.from(x), 4))
  })
})

describe('dictionaries', () => {
  // Samples that are sparse combinations of three hidden unit atoms in R^4.
  const atoms = [
    [1, 0, 0, 0],
    [0, 0.6, 0.8, 0],
    [0, 0, 0, 1],
  ]
  const samples = Array.from({ length: 12 }, (_, i) => {
    const a = atoms[i % 3] as number[]
    const b = atoms[(i + 1) % 3] as number[]
    const s = 1 + (i % 4)
    return a.map((v, f) => s * v + (i % 2 === 0 ? 0 : 0.5 * (b[f] ?? 0)))
  })

  it('K-SVD and MOD reduce the reconstruction error and keep unit atoms', () => {
    const perturbed = atoms.map((a, j) => a.map((v, f) => v + 0.05 * Math.sin(3 * j + f)))
    for (const learn of [O.ksvd, O.mod]) {
      const r = learn(samples, 3, 2, { key: key(4), maxIterations: 30 })
      expect((r.history.at(-1) ?? Infinity) <= (r.history[0] ?? 0) + 1e-12).toBe(true)
      const d = toFlat(r.dictionary)
      for (let j = 0; j < 3; j++) expect(Math.hypot(...d.slice(4 * j, 4 * j + 4))).toBeCloseTo(1, 10)
      // Started near the hidden atoms, the learner closes in on them (linearly, and slowly).
      const near = learn(samples, 3, 2, { init: perturbed, maxIterations: 100 })
      expect(near.error).toBeLessThan(0.05 * (near.history[0] ?? 0))
    }
  })

  it('online dictionary learning lowers the error over minibatches', () => {
    const r = O.onlineDictionaryLearning(samples, 3, { key: key(5), alpha: 0.05, maxIterations: 60 })
    expect(r.error).toBeLessThan(r.history[0] ?? 0)
    const d = toFlat(r.dictionary)
    for (let j = 0; j < 3; j++) expect(Math.hypot(...d.slice(4 * j, 4 * j + 4))).toBeLessThanOrEqual(1 + 1e-12)
  })

  it('coherence, Babel function and RIP estimates', () => {
    const eye = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ]
    expect(O.mutualCoherence(eye)).toBe(0)
    expect(O.coherenceSparsityBound(0)).toBe(Infinity)
    const d = [
      [1, 0, 1],
      [0, 1, 1],
    ]
    expect(O.babelFunction(d, 2)).toBeCloseTo(Math.SQRT2, 12)
    const { d: g } = compressed(6, 30, 50, 1)
    const rip = O.ripEstimate(g, 3, { key: key(7), samples: 50 })
    expect(rip.lower).toBeLessThanOrEqual(rip.upper)
    expect(rip.delta).toBeGreaterThan(0)
    expect(rip.delta).toBeLessThan(1.5)
    expect(() => O.ripEstimate(g, 0)).toThrow()
  })
})
