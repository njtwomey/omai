import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { metricSeries, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { nmf, nmfAlgorithm, nmfDivergence, nndsvda } from './nmf.ts'

const topics = [
  [3, 0, 0, 1],
  [0, 2, 2, 0],
]
const x = Array.from(
  { length: 12 },
  (_, i) => topics[0]?.map((v, j) => v * ((i % 4) / 2) + (topics[1]?.[j] ?? 0) * ((i % 3) / 2 + 0.2)) ?? [],
)

describe('NMF', () => {
  for (const [solver, loss] of [
    ['mu', 'frobenius'],
    ['mu', 'kullback-leibler'],
    ['hals', 'frobenius'],
  ] as const)
    it(`${solver}/${loss}: the error falls with every update, the factors stay non-negative`, () => {
      const f = nmf.with({ nComponents: 2, solver, betaLoss: loss, maxIter: 40, tol: 0 }).fit({ x }, { logs: 'all' })
      const err = metricSeries(f.run, 'solver/error').values
      for (let i = 1; i < err.length; i++) expect(err[i] ?? 0).toBeLessThanOrEqual((err[i - 1] ?? 0) + 1e-9)
      expect(f.params.components.flat().every((v) => v >= 0)).toBe(true)
      expect(f.params.weights.flat().every((v) => v >= 0)).toBe(true)
      expect(f.params.reconstructionErr).toBeLessThan(0.5)
      const w = f.transform(x)
      expect(w.shape).toEqual([12, 2])
      expect(toFlat(f.inverseTransform(w)).length).toBe(48)
    })

  it('rejects negative data and HALS with the KL loss; random init is keyed', () => {
    expect(() => nmf.fit({ x: [[-1, 2]] })).toThrow(/non-negative/)
    expect(() => nmfAlgorithm([[1]], [[1]], [[1]], { solver: 'hals', loss: 'kullback-leibler' })).toThrow()
    const a = nmf.with({ nComponents: 2, init: 'random', maxIter: 5 }).fit({ x }, { key: key(1) }).params.components
    const b = nmf.with({ nComponents: 2, init: 'random', maxIter: 5 }).fit({ x }, { key: key(1) }).params.components
    expect(a).toEqual(b)
  })

  it('a session probes the factors early; states round-trip; NNDSVD is non-negative', () => {
    const s = nmf.with({ nComponents: 2 }).session({ x }, { key: key(0), logs: 'all' })
    s.advance({ steps: 2 })
    expect((s.probe(2, 'model/embedding') as number[][]).length).toBe(12)
    const { w, h } = nndsvda(x, 2)
    expect([...w.flat(), ...h.flat()].every((v) => v >= 0)).toBe(true)
    const state = runAlgorithm(nmfAlgorithm(x, w, h), { maxSteps: 3 }).state
    expect(stateFromJson(stateToJson(state))).toEqual(state)
    expect(nmfDivergence(x, w, h, 'frobenius')).toBeGreaterThan(state.error)
  })
})
