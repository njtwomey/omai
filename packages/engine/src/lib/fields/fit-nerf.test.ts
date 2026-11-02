import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { multiViewImages } from '../datasets/index.ts'
import { pinholeCamera } from '../graphics/index.ts'
import { adam } from '../optim/index.ts'
import { key } from '../random/index.ts'
import { type Tensor, toFlat, treeLeaves } from '../tensor/index.ts'
import { Log, metricSeries, recorder, runAlgorithm, session } from '../trace/index.ts'
import {
  type FitNerfOptions,
  fitNerf,
  fitNerfAlgorithm,
  fitNerfMethod,
  type NerfImage,
  type NerfMethodConfig,
  type NerfRayWeights,
  type NerfState,
  nerfContract,
  nerfData,
  nerfModel,
  nerfParams,
  nerfProblem,
  renderView,
} from './index.ts'

const encodings = {
  positionEncoding: { kind: 'frequency', octaves: 5, includeInput: true },
  directionEncoding: { kind: 'frequency', octaves: 2, includeInput: true },
} as const

const psnrOf = (a: ArrayLike<number>, b: ArrayLike<number>): number => {
  let s = 0
  for (let i = 0; i < a.length; i++) s += ((a[i] as number) - (b[i] as number)) ** 2
  return -10 * Math.log10(s / a.length)
}

describe('fitNerf', () => {
  const generated = generate(multiViewImages, {
    key: key(1),
    knobs: { views: 8, heldOut: 3, resolution: 8, objects: 1 },
  })
  const data = nerfData(generated)

  it('reduces the loss and renders a novel view closer to the truth than the untrained field', () => {
    const model = nerfModel({ width: 24, depth: 2, ...encodings, key: key(0) })
    const sampling = { near: data.near, far: data.far, samples: 8, whiteBackground: true }
    // A camera in none of the views: the truth renders it exactly.
    const novel = pinholeCamera({ eye: [2.9, 2.4, 1.6], fov: 40 })
    const renderTruth = generated.truth.extra?.['renderView'] as (c: typeof novel, w: number, h: number) => Tensor
    const truth = renderTruth(novel, 8, 8)
    const before = renderView(model, nerfParams(model), novel, [8, 8], sampling)
    const r = fitNerf(model, data, {
      optimizer: adam({ learningRate: 5e-3 }),
      batchRays: 64,
      samples: 8,
      steps: 200,
      key: key(2),
    })
    const after = renderView(model, r.params, novel, [8, 8], sampling)
    const start = r.losses.slice(0, 10).reduce((a, b) => a + b, 0) / 10
    const end = r.losses.slice(-10).reduce((a, b) => a + b, 0) / 10
    // Measured: the batch loss falls from 0.21 to 0.036 and the novel view gains 8 decibels (6.0 to 14.2).
    expect(end).toBeLessThan(start / 3)
    expect(psnrOf(after.rgb, toFlat(truth))).toBeGreaterThan(psnrOf(before.rgb, toFlat(truth)) + 4)
    expect(after.alpha.length).toBe(64)
    expect(after.depth.every(Number.isFinite)).toBe(true)
  })

  it('trains a coarse and a fine network and resumes from a serialised state to the uninterrupted run', () => {
    const model = nerfModel({ width: 8, depth: 1, ...encodings, fine: true, key: key(0) })
    const options: FitNerfOptions = {
      optimizer: adam({ learningRate: 5e-3 }),
      batchRays: 16,
      samples: 4,
      fineSamples: 4,
    }
    const alg = fitNerfAlgorithm(model, data, options)
    const straight = runAlgorithm(alg, { maxSteps: 6, key: key(4) })
    const half = runAlgorithm(alg, { maxSteps: 3, key: key(4) })
    const json = JSON.parse(JSON.stringify(alg.serialise?.(half.state))) as never
    const resumed = runAlgorithm(alg, { maxSteps: 3, key: key(4), start: { state: json, step: 3, serialised: true } })
    expect(resumed.state.loss).toBe(straight.state.loss)
    const leaves = (s: NerfState): number[][] => treeLeaves(s.params).map((v) => Array.from(toFlat(v)))
    expect(leaves(resumed.state)).toEqual(leaves(straight.state))
    expect(Object.keys(straight.state.params)).toEqual(['coarse', 'fine'])
    expect(straight.state.step).toBe(6)
    // Both networks moved.
    expect(leaves(straight.state)).not.toEqual(leaves(alg.init(key(0))))
  })

  it('logs its scalar paths and offers views, ray weights and the held-out error as probes', () => {
    const model = nerfModel({ width: 8, depth: 1, ...encodings, fine: true, key: key(0) })
    const options: FitNerfOptions = {
      optimizer: adam({ learningRate: 5e-3 }),
      batchRays: 16,
      samples: 4,
      fineSamples: 4,
    }
    const rec = recorder()
    const state = runAlgorithm(fitNerfAlgorithm(model, data, options), { maxSteps: 4, tracer: rec }).state
    for (const path of ['nerf/loss', 'nerf/mse', 'nerf/psnr', 'nerf/gradNorm'])
      expect(metricSeries(rec.run(), path).values.length).toBe(4)
    const contract = nerfContract(model, data, options)
    const compute = (path: string, args?: object): unknown =>
      (contract[path as keyof typeof contract] as { compute: (s: NerfState, a?: never) => unknown }).compute(
        state,
        args as never,
      )
    const view = compute('nerf/render', { view: 2 }) as NerfImage
    expect([view.kind, view.width, view.height, view.rgb.length, view.psnr !== undefined]).toEqual([
      'nerfImage',
      8,
      8,
      192,
      true,
    ])
    const held = compute('nerf/render', { heldOut: 1 }) as NerfImage
    expect(held.rgb.length).toBe(192)
    const ray = compute('nerf/rayWeights', { view: 0, x: 4, y: 4 }) as NerfRayWeights
    expect(ray.t.length).toBe(8)
    expect(ray.weights.length).toBe(8)
    expect(ray.coarse?.t.length).toBe(4)
    expect(ray.weights.reduce((a, b) => a + b, 0)).toBeCloseTo(ray.alpha, 10)
    expect(Number.isFinite(compute('nerf/heldOutPsnr') as number)).toBe(true)
  })
})

describe('fitNerfMethod', () => {
  const config: Partial<NerfMethodConfig> = {
    views: 3,
    heldOut: 1,
    resolution: 4,
    objects: 1,
    width: 8,
    depth: 1,
    samples: 4,
    fineSamples: 4,
    batchRays: 8,
  }

  it('runs as a session with a view probed at any step', () => {
    const s = session(fitNerfMethod, {
      config,
      logs: { 'nerf/loss': Log(), 'nerf/psnr': Log() },
      mode: 'eager',
      steps: 6,
    })
    expect(metricSeries(s.run(), 'nerf/loss').values.length).toBe(6)
    const view = s.probe(3, 'nerf/render', { view: 1 }) as NerfImage
    expect(view.width).toBe(4)
    expect(view.rgb.length).toBe(48)
  })

  it('builds the problem from the slots, with a fine network when hierarchical', () => {
    const full = {
      ...(config as NerfMethodConfig),
      positionEncoding: { kind: 'hash', levels: 2, log2TableSize: 5, bound: 1.5 },
      directionEncoding: { kind: 'none' },
      densityActivation: 'relu',
      hierarchical: true,
      optimiser: { kind: 'adam' },
      learningRate: 0.01,
    }
    const p = nerfProblem(full as NerfMethodConfig, key(0))
    expect(p.model.fine).toBeDefined()
    expect(p.data.images.shape).toEqual([3, 4, 4, 3])
    expect(p.model.coarse.densityActivation).toBe('relu')
  })
})
