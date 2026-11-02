import { describe, expect, it } from 'vitest'
import { type Generated, generate, type RegressionData } from '../data/index.ts'
import { fieldImage, signedDistance } from '../datasets/index.ts'
import { adam } from '../optim/index.ts'
import { key } from '../random/index.ts'
import type { SpaceValue } from '../space/index.ts'
import { toFlat, treeLeaves } from '../tensor/index.ts'
import { metricSeries, recorder, runAlgorithm } from '../trace/index.ts'
import { type FieldState, fieldContract, fitField, fitFieldAlgorithm, neuralField } from './index.ts'

const fullMse = (f: ReturnType<typeof neuralField>, g: Generated<RegressionData>): number => {
  const p = toFlat(f.forward(g.data.x))
  const y = toFlat(g.data.y)
  let s = 0
  for (let i = 0; i < p.length; i++) s += ((p[i] as number) - (y[i] as number)) ** 2
  return s / p.length
}

describe('fitField', () => {
  it('fits a 1-D signal and reduces the loss', () => {
    const f = neuralField({ inputDim: 1, outputDim: 1, width: 16, depth: 2, key: key(0) })
    const x = Array.from({ length: 40 }, (_, i) => [-1 + i / 19.5])
    const y = x.map(([t]) => Math.sin(3 * (t as number)))
    const r = fitField(f, { x: x as never, y: y as never }, { optimizer: adam({ learningRate: 0.02 }), steps: 200 })
    expect(r.losses.length).toBe(200)
    expect(r.losses.at(-1)).toBeLessThan((r.losses[0] as number) / 20)
    expect(r.status).toBe('completed')
  })

  it('has Fourier features and a SIREN beat a plain ReLU MLP on the fine-texture image at equal steps', () => {
    const g = generate(fieldImage, { key: key(1), knobs: { side: 16, pattern: 'mixed', texture: 0.5, frequency: 6 } })
    const run = (encoding: SpaceValue, backbone: SpaceValue, learningRate: number): number => {
      const f = neuralField({ inputDim: 2, outputDim: 3, encoding, backbone, width: 24, depth: 2, key: key(0) })
      fitField(f, g.data, { optimizer: adam({ learningRate }), steps: 150, key: key(2) })
      return fullMse(f, g)
    }
    const relu = run({ kind: 'none' }, { kind: 'mlp' }, 3e-3)
    const fourier = run({ kind: 'fourier', features: 32, sigma: 6 }, { kind: 'mlp' }, 3e-3)
    const siren = run({ kind: 'none' }, { kind: 'siren', omega0: 30 }, 1e-3)
    const hash = run({ kind: 'hash', levels: 5, log2TableSize: 9, finestResolution: 32 }, { kind: 'mlp' }, 1e-2)
    // Measured at these settings: relu 0.052, fourier 0.0008, siren 0.0015 (30 to 60 times lower), hash grid below the relu's too.
    expect(hash).toBeLessThan(relu / 5)
    expect(fourier).toBeLessThan(relu / 5)
    expect(siren).toBeLessThan(relu / 5)
  })

  it('fits a signed distance field with the eikonal term: the gradient norm moves towards one', () => {
    const g = generate(signedDistance, { key: key(0), knobs: { shape: 'sphere', dim: 2, n: 96, nearSurface: 0.5 } })
    const f = neuralField({
      inputDim: 2,
      outputDim: 1,
      backbone: { kind: 'siren', omega0: 10 },
      width: 16,
      depth: 2,
      key: key(0),
    })
    const alg = fitFieldAlgorithm(f, g.data, {
      optimizer: adam({ learningRate: 5e-3 }),
      eikonal: { weight: 0.5, extraPoints: 64 },
      batchSize: 96,
    })
    const first = runAlgorithm(alg, { maxSteps: 1, key: key(3) }).state
    const last = runAlgorithm(alg, { maxSteps: 120, key: key(3) }).state
    expect(first.eikonal).toBeGreaterThan(0)
    expect(last.loss).toBeLessThan(first.loss)
    expect(last.mse).toBeLessThan(first.mse)
    expect(Number.isFinite(last.eikonal)).toBe(true)
  })

  it('logs the contract’s scalar paths, and its probes decode the field and measure the whole data', () => {
    const g = generate(signedDistance, { key: key(0), knobs: { shape: 'box', dim: 2, n: 32 } })
    const f = neuralField({ inputDim: 2, outputDim: 1, width: 8, depth: 1, key: key(0) })
    const options = { optimizer: adam({ learningRate: 0.01 }), decode: { resolution: 6 } }
    const alg = fitFieldAlgorithm(f, g.data, options)
    const rec = recorder()
    const state = runAlgorithm(alg, { maxSteps: 5, tracer: rec }).state
    for (const path of ['field/loss', 'field/mse', 'field/psnr', 'field/eikonal', 'field/gradNorm'])
      expect(metricSeries(rec.run(), path).values.length).toBe(5)
    const contract = fieldContract(f, g.data, options)
    expect(Object.keys(contract).sort()).toEqual([
      'field/eikonal',
      'field/fullMse',
      'field/gradNorm',
      'field/loss',
      'field/mse',
      'field/psnr',
      'field/render',
    ])
    const render = (contract['field/render'] as { compute: (s: FieldState) => unknown }).compute(state) as {
      values: number[]
    }
    expect(render.values.length).toBe(36)
    const full = (contract['field/fullMse'] as { compute: (s: FieldState) => unknown }).compute(state) as number
    expect(full).toBeGreaterThan(0)
  })

  it('resumes from a serialised state to exactly the uninterrupted run', () => {
    const g = generate(fieldImage, { key: key(0), knobs: { side: 6 } })
    const f = neuralField({
      inputDim: 2,
      outputDim: 3,
      encoding: { kind: 'frequency', octaves: 3 },
      width: 8,
      depth: 1,
      key: key(0),
    })
    const alg = fitFieldAlgorithm(f, g.data, { optimizer: adam({ learningRate: 0.01 }), batchSize: 16 })
    const straight = runAlgorithm(alg, { maxSteps: 12, key: key(4) })
    const half = runAlgorithm(alg, { maxSteps: 5, key: key(4) })
    const json = JSON.parse(JSON.stringify(alg.serialise?.(half.state))) as never
    const resumed = runAlgorithm(alg, { maxSteps: 7, key: key(4), start: { state: json, step: 5, serialised: true } })
    expect(resumed.steps).toBe(12)
    expect(resumed.state.loss).toBe(straight.state.loss)
    const leaves = (s: FieldState): number[][] => treeLeaves(s.params).map((v) => Array.from(toFlat(v)))
    expect(leaves(resumed.state)).toEqual(leaves(straight.state))
  })
})
