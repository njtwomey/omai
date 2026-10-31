import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { sparseDictionary } from '../datasets/index.ts'
import { key } from '../random/index.ts'
import { fromStorage, toFlat } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { dictionaryRecovery, saeAlgorithm, saeDecode, saeEncode } from './index.ts'

const plant = generate(sparseDictionary, { key: key(4), knobs: { n: 800, features: 6, dim: 8, sparsity: 0.8 } })
const truth = plant.truth.extra?.['dictionary'] as number[][]

describe('sparse autoencoder', () => {
  it('recovers a planted dictionary with the L1 penalty', { timeout: 30000 }, () => {
    const alg = saeAlgorithm(plant.data.x, { features: 6, l1Coefficient: 0.02, steps: 600, learningRate: 0.02 })
    const r = runAlgorithm(alg, { maxSteps: 600, seed: 1 })
    const rec = dictionaryRecovery(truth, r.state.params.decoder)
    expect(rec.mean).toBeGreaterThan(0.97)
    expect(Math.min(...rec.similarity)).toBeGreaterThan(0.9)
    expect(r.state.reconstruction).toBeLessThan(0.05)
    // Every learned direction is unit norm.
    const dec = toFlat(r.state.params.decoder)
    for (let j = 0; j < 6; j++) expect(Math.hypot(...Array.from(dec.slice(j * 8, (j + 1) * 8)))).toBeCloseTo(1, 10)
  })
  it('recovers it with a top-k code, and decodes what it encodes', { timeout: 30000 }, () => {
    const alg = saeAlgorithm(plant.data.x, { features: 6, sparsity: 'topk', k: 3, steps: 600, learningRate: 0.02 })
    const r = runAlgorithm(alg, { maxSteps: 600, seed: 2 })
    expect(dictionaryRecovery(truth, r.state.params.decoder).mean).toBeGreaterThan(0.97)
    expect(r.state.l0).toBeLessThanOrEqual(3)
    const f = saeEncode(r.state.params, plant.data.x, { sparsity: 'topk', k: 3 })
    const xhat = toFlat(saeDecode(r.state.params, f))
    const x = toFlat(plant.data.x)
    let err = 0
    for (let i = 0; i < x.length; i++) err += ((x[i] as number) - (xhat[i] as number)) ** 2
    expect(err / 800).toBeCloseTo(r.state.reconstruction, 8)
  })
  it('counts dead features and resampling revives them', { timeout: 60000 }, () => {
    const x = fromStorage(Float64Array.from(toFlat(plant.data.x).slice(0, 200 * 8)), [200, 8], 'float64')
    const base = { features: 24, sparsity: 'topk' as const, k: 1, steps: 300, learningRate: 0.02 }
    const plain = runAlgorithm(saeAlgorithm(x, base), { maxSteps: 300, seed: 3 }).state
    const resampled = runAlgorithm(saeAlgorithm(x, { ...base, resampleEvery: 50 }), { maxSteps: 300, seed: 3 }).state
    expect(plain.dead).toBeGreaterThan(0)
    expect(plain.dead).toBeLessThan(24)
    expect(resampled.dead).toBeLessThan(plain.dead)
    expect(resampled.reconstruction).toBeLessThanOrEqual(plain.reconstruction * 1.05)
  })
  it('lowers the loss and logs its paths', () => {
    const alg = saeAlgorithm(plant.data.x, { features: 6, steps: 60, batchSize: 200 })
    const first = runAlgorithm(alg, { maxSteps: 1, seed: 5 }).state
    const later = runAlgorithm(alg, { maxSteps: 60, seed: 5 }).state
    expect(later.reconstruction).toBeLessThan(first.reconstruction)
    const back = alg.deserialise?.(alg.serialise?.(later) as never)
    expect(back?.step).toBe(60)
    expect(back?.dead).toBe(later.dead)
  })
})
