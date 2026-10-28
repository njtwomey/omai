/**
 * Perona-Malik anisotropic diffusion watched, as a method of nothing but its configuration: a noisy synthetic scene
 * (polygons at a few grey levels, so the clean image is known) smoothed one diffusion step per step by the library's
 * algorithm. The image is on the tape with its largest change and its peak signal-to-noise ratio to the clean scene
 * beside the noisy start's, and the conduction (how much flow each pixel lets through, little across an edge) is a
 * probe.
 */

import type { DecisionSurface } from '../../lib/exposition/index.ts'
import { peakSignalNoiseRatio } from '../../lib/metrics/index.ts'
import { choice, dict, int, real } from '../../lib/space/index.ts'
import {
  type Algorithm,
  type Config,
  defineMethod,
  derived,
  Log,
  type Method,
  scalar,
  tensor,
} from '../../lib/trace/index.ts'
import { anisotropicDiffusionAlgorithm, type DiffusionState } from '../../lib/vision/index.ts'
import { picture } from './_picture.ts'
import { grid, scene } from './_scene.ts'

/** The conduction g(|∇I|) at every pixel, by central differences. */
function conductionOf(image: ArrayLike<number>, size: number, kappa: number, quadratic: boolean): number[] {
  const g = (d: number): number => (quadratic ? 1 / (1 + (d / kappa) ** 2) : Math.exp(-((d / kappa) ** 2)))
  const at = (y: number, x: number): number =>
    image[Math.min(size - 1, Math.max(0, y)) * size + Math.min(size - 1, Math.max(0, x))] ?? 0
  const out = new Array<number>(size * size)
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++) {
      const dx = 0.5 * (at(y, x + 1) - at(y, x - 1))
      const dy = 0.5 * (at(y + 1, x) - at(y - 1, x))
      out[y * size + x] = g(Math.hypot(dx, dy))
    }
  return out
}

/**
 * Perona-Malik anisotropic diffusion of a noisy scene, one diffusion step per step.
 *
 * Configuration: `kappa` (the edge threshold on the gradient: differences well below it diffuse, well above it are
 * kept), `lambda` (the time step, stable up to 0.25), `conduction` (exponential or quadratic), `noise` (the pixel
 * noise of the scene) and `size` (the image side). Paths: `image/pixels` (the image, full), `image/change` (the
 * largest pixel change of the step), `quality/psnr` (to the clean scene, in dB), `quality/psnrStart` (the noisy
 * start's, for comparison) and the probe `image/conduction` (the flow each pixel lets through: 1 free, 0 across an
 * edge).
 */
export const peronaMalik: Method<DiffusionState, Config> = defineMethod<DiffusionState, Config>({
  name: 'vision/perona-malik',
  config: dict({
    kappa: real(0.02, 0.5, { default: 0.1, log: true, label: 'edge threshold κ' }),
    lambda: real(0.05, 0.25, { default: 0.2, label: 'time step λ' }),
    conduction: choice(['exponential', 'quadratic'] as const, { default: 'exponential' }),
    noise: real(0, 0.4, { default: 0.15, label: 'pixel noise σ' }),
    size: int(16, 96, { default: 64, label: 'image side' }),
  }),
  make: (c, key) => {
    const size = c['size'] as number
    const sc = scene(key, { size, noise: c['noise'] as number })
    const clean = grid(sc.clean, size)
    const inner: Algorithm<DiffusionState> = anisotropicDiffusionAlgorithm(grid(sc.noisy, size), {
      kappa: c['kappa'] as number,
      lambda: c['lambda'] as number,
      conduction: c['conduction'] as 'exponential' | 'quadratic',
    })
    const psnr = (image: ArrayLike<number>): number =>
      peakSignalNoiseRatio(clean, grid(image, size), { dataRange: 1 }).estimate.value
    const psnrStart = psnr(sc.noisy)
    return {
      name: 'perona-malik',
      config: inner.config,
      init: (k) => inner.init(k),
      step: (s, ctx) => {
        const next = inner.step(s, ctx)
        ctx.log('image/pixels', grid(next.image, size))
        ctx.log('image/change', next.change)
        ctx.log('quality/psnr', psnr(next.image))
        ctx.log('quality/psnrStart', psnrStart)
        return next
      },
      serialise: (s) => ({ image: Array.from(s.image), change: s.change }),
      deserialise: (j) => {
        const o = j as { image: number[]; change: number }
        return { image: Float64Array.from(o.image), change: o.change }
      },
    }
  },
  contract: (c) => {
    const size = c['size'] as number
    const kappa = c['kappa'] as number
    const quadratic = c['conduction'] === 'quadratic'
    return {
      'image/pixels': tensor({ role: 'state', shape: [size, size], description: 'the image so far' }),
      'image/change': scalar({ description: 'the largest pixel change of the step' }),
      'quality/psnr': scalar({ description: 'peak signal-to-noise ratio of the image to the clean scene, in dB' }),
      'quality/psnrStart': scalar({ description: 'the noisy start’s ratio to the clean scene, for comparison' }),
      'image/conduction': derived<DiffusionState>({
        description: 'the conduction g(|∇I|) at every pixel: 1 where the flow is free, 0 across an edge',
        compute: (s): DecisionSurface =>
          picture(conductionOf(s.image, size, kappa, quadratic), size, size, { name: 'conduction' }),
      }),
    }
  },
  defaults: {
    'image/pixels': Log({ detail: 'full' }),
    'image/change': Log(),
    'quality/psnr': Log(),
    'quality/psnrStart': Log(),
  },
})
