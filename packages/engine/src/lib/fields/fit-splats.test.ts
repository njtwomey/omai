import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { hemisphereCameras, multiViewImages } from '../datasets/index.ts'
import { adam } from '../optim/index.ts'
import { key } from '../random/index.ts'
import { fromStorage, toFlat, treeLeaves } from '../tensor/index.ts'
import { iterate, Log, metricSeries, recorder, runAlgorithm, session } from '../trace/index.ts'
import {
  type FitSplatsOptions,
  fitSplats,
  fitSplatsAlgorithm,
  fitSplatsMethod,
  type GaussianSet,
  gaussianCount,
  gaussianOpacity,
  gaussiansFromPoints,
  type NerfData,
  nerfData,
  randomGaussians,
  renderSplats,
  type SplatEllipses,
  type SplatRender,
  type SplatState,
  type SplatsMethodConfig,
  splatContract,
  splatOptimiser,
} from './index.ts'

const slot: FitSplatsOptions['optimiser'] = (lr) => adam({ learningRate: lr })

/** Images of a planted set of Gaussians on a black background, as the data of a fit. */
function planted(truth: GaussianSet, views = 6, side = 16): NerfData {
  const cameras = hemisphereCameras(key(0), views, { radius: 4, fov: 40 })
  const all = new Float64Array(views * side * side * 3)
  for (const [v, camera] of cameras.entries())
    all.set(toFlat(renderSplats(truth, camera, side, side).colour), v * side * side * 3)
  return {
    images: fromStorage(all, [views, side, side, 3], 'float64'),
    cameras,
    near: 2,
    far: 6,
    whiteBackground: false,
  }
}

const scene = nerfData(
  generate(multiViewImages, { key: key(1), knobs: { views: 8, heldOut: 3, resolution: 8, objects: 1 } }),
)
const leaves = (s: SplatState): number[][] => treeLeaves(s.params).map((v) => Array.from(toFlat(v)))

describe('fitSplats', () => {
  it('reduces the loss and recovers the position of a single coloured blob from 16 by 16 views', () => {
    const at = [0.4, -0.2, 0.3]
    const data = planted(gaussiansFromPoints([at], [[0.9, 0.1, 0.1]], { scale: 0.3, opacity: 0.95 }))
    // A handful of grey Gaussians near the centre, none on the blob.
    const start = randomGaussians(key(3), 5, { boxHalfWidth: 0.25, scale: 0.3, opacity: 0.4 })
    const r = fitSplats(start, data, { optimiser: slot, steps: 150, key: key(4) })
    const mean = (xs: readonly number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length
    // Measured: the loss falls from 0.045 to 0.0015 and the fitted centre is 0.012 from the blob.
    expect(mean(r.losses.slice(-5))).toBeLessThan(mean(r.losses.slice(0, 5)) / 10)
    const o = toFlat(gaussianOpacity(r.params))
    const m = toFlat(r.params.means)
    const total = o.reduce((a, b) => a + b, 0)
    const centre = [0, 1, 2].map((k) => o.reduce((s, w, i) => s + w * (m[3 * i + k] as number), 0) / total)
    expect(Math.hypot(...centre.map((v, k) => v - (at[k] as number)))).toBeLessThan(0.05)
  })

  it('renders a scene’s held-out views better after fitting than before', () => {
    const start = randomGaussians(key(0), 30, { boxHalfWidth: 0.8, scale: 0.25, opacity: 0.3 })
    const held = scene.heldOut
    const psnr = (set: GaussianSet): number => {
      const values = (held?.cameras ?? []).map((camera, v) => {
        const target = toFlat(held?.images as never).slice(v * 192, (v + 1) * 192)
        const rgb = toFlat(renderSplats(set, camera, 8, 8, { background: [1, 1, 1] }).colour)
        return -10 * Math.log10(rgb.reduce((s, x, i) => s + (x - (target[i] as number)) ** 2, 0) / rgb.length)
      })
      return values.reduce((a, b) => a + b, 0) / values.length
    }
    const r = fitSplats(start, scene, { optimiser: slot, steps: 150, key: key(2) })
    // Measured: about 11.7 decibels before and 20 after 200 steps (the same views a NeRF reaches 14 on).
    expect(psnr(r.params)).toBeGreaterThan(psnr(start) + 5)
  })

  it('uses a learning rate per parameter group: a group at rate zero does not move', () => {
    const start = randomGaussians(key(0), 6, { scale: 0.3 })
    const r = fitSplats(start, scene, {
      optimiser: slot,
      learningRates: { colours: 0, opacities: 0 },
      steps: 5,
      key: key(2),
    })
    expect(toFlat(r.params.colours)).toEqual(toFlat(start.colours))
    expect(toFlat(r.params.opacityLogits)).toEqual(toFlat(start.opacityLogits))
    expect(toFlat(r.params.means)).not.toEqual(toFlat(start.means))
    expect(splatOptimiser(slot).name).toContain('means')
  })

  it('adds Gaussians under the densification schedule, announces it, and prunes the nearly transparent', () => {
    const start = randomGaussians(key(0), 12, { boxHalfWidth: 0.6, scale: 0.2, opacity: 0.3 })
    // One Gaussian is all but invisible: the first densification prunes it.
    const faint: GaussianSet = {
      ...start,
      opacityLogits: fromStorage(
        Float64Array.from({ length: 12 }, (_, i) => (i === 0 ? -12 : -0.85)),
        [12],
        'float64',
      ),
    }
    const alg = fitSplatsAlgorithm(faint, scene, {
      optimiser: slot,
      densify: { from: 5, until: 20, every: 5, gradientThreshold: 1e-5, maxGaussians: 60 },
    })
    const counts: number[] = []
    for (const { state } of iterate(alg, { key: key(2), maxSteps: 25 })) counts.push(gaussianCount(state.params))
    // The set changes size only at the schedule's steps 5, 10, 15 and 20 (the first also prunes the faint Gaussian).
    const changes = counts.flatMap((c, t) => (t > 0 && c !== counts[t - 1] ? [t] : []))
    expect(changes.every((t) => [5, 10, 15, 20].includes(t))).toBe(true)
    expect(changes.length).toBeGreaterThanOrEqual(2)
    expect(counts.at(-1)).toBeGreaterThan(12)
    expect(counts.at(-1)).toBeLessThanOrEqual(60)
    const state = runAlgorithm(alg, { maxSteps: 25, key: key(2) }).state
    // Outside the window nothing changes size: the count at step 25 is the count after the step-20 densification.
    expect(state.gradSum.shape[0]).toBe(gaussianCount(state.params))
    expect(counts.at(-1)).toBe(gaussianCount(state.params))
  })

  it('resumes from a serialised state, across a densification, to exactly the uninterrupted run', () => {
    const start = randomGaussians(key(0), 10, { boxHalfWidth: 0.6, scale: 0.2, opacity: 0.3 })
    const alg = fitSplatsAlgorithm(start, scene, {
      optimiser: slot,
      batchPixels: 40,
      densify: { from: 4, until: 12, every: 4, gradientThreshold: 1e-5, maxGaussians: 40, opacityResetEvery: 8 },
    })
    const straight = runAlgorithm(alg, { maxSteps: 12, key: key(5) })
    const half = runAlgorithm(alg, { maxSteps: 6, key: key(5) })
    const json = JSON.parse(JSON.stringify(alg.serialise?.(half.state))) as never
    const resumed = runAlgorithm(alg, { maxSteps: 6, key: key(5), start: { state: json, step: 6, serialised: true } })
    expect(gaussianCount(resumed.state.params)).toBe(gaussianCount(straight.state.params))
    expect(leaves(resumed.state)).toEqual(leaves(straight.state))
    expect(resumed.state.loss).toBe(straight.state.loss)
  })

  it('logs its paths and offers a rendered view, the Gaussians as ellipses and the held-out error as probes', () => {
    const start = randomGaussians(key(0), 8, { scale: 0.3 })
    const alg = fitSplatsAlgorithm(start, scene, { optimiser: slot })
    const rec = recorder()
    const state = runAlgorithm(alg, { maxSteps: 4, key: key(1), tracer: rec }).state
    for (const path of ['splats/loss', 'splats/mse', 'splats/psnr', 'splats/count', 'splats/gradNorm'])
      expect(metricSeries(rec.run(), path).values.length).toBe(4)
    const contract = splatContract(scene)
    const compute = (path: string, args?: object): unknown =>
      (contract[path as keyof typeof contract] as { compute: (s: SplatState, a?: never) => unknown }).compute(
        state,
        args as never,
      )
    const image = compute('splats/render', { view: 3 }) as SplatRender
    expect([image.kind, image.width, image.rgb.length, image.psnr !== undefined]).toEqual(['splatImage', 8, 192, true])
    expect((compute('splats/render', { heldOut: 0 }) as SplatRender).alpha.length).toBe(64)
    const ellipses = compute('splats/ellipses', { view: 0 }) as SplatEllipses
    expect(ellipses.ellipses.length).toBeGreaterThan(0)
    expect(ellipses.ellipses.every((e, i, all) => i === 0 || e.depth <= (all[i - 1]?.depth as number))).toBe(true)
    expect(ellipses.ellipses.every((e) => e.major >= e.minor && e.minor > 0)).toBe(true)
    expect(Number.isFinite(compute('splats/heldOutPsnr') as number)).toBe(true)
  })
})

describe('fitSplatsMethod', () => {
  const config: Partial<SplatsMethodConfig> = { views: 3, heldOut: 1, resolution: 4, objects: 1, gaussians: 8 }

  it('runs as a session with the count on the tape and a view probed at any step', () => {
    const s = session(fitSplatsMethod, {
      config,
      logs: { 'splats/loss': Log(), 'splats/count': Log() },
      mode: 'eager',
      steps: 5,
    })
    expect(metricSeries(s.run(), 'splats/count').values).toEqual([8, 8, 8, 8, 8])
    const view = s.probe(2, 'splats/render', { view: 1 }) as SplatRender
    expect(view.rgb.length).toBe(48)
  })

  it('densifies when asked, and the start can have view-dependent colour', () => {
    const s = session(fitSplatsMethod, {
      config: { ...config, densify: true, densifyFrom: 4, densifyEvery: 4, gradientThreshold: 1e-5, shDegree: 1 },
      logs: { 'splats/count': Log(), 'splats/densify': Log() },
      mode: 'eager',
      steps: 12,
    })
    const counts = metricSeries(s.run(), 'splats/count').values
    expect(counts.at(-1)).toBeGreaterThan(8)
    // The event carries what the densification did, and is recorded on the steps (counting from 0) that did it: after 4, 8 and 12.
    const fired = s.run().events.flatMap((e) => (e.type === 'message' && e.kind === 'splats/densify' ? [e] : []))
    expect(fired.map((e) => e.step)).toEqual([3, 7, 11])
    expect(Object.keys(fired[0]?.payload ?? {})).toEqual(['step', 'cloned', 'split', 'pruned', 'count'])
    expect((s.state as SplatState).params.sh).toBeDefined()
  })
})
