/**
 * GrabCut watched, as a method of nothing but its configuration: a rectangle drawn around the largest polygon of a
 * synthetic scene (its own grey level, with background and bits of the other polygons inside the rectangle too),
 * then, by the library's algorithm, each step fitting a Gaussian mixture to the grey levels of each side and cutting
 * the pixels by the minimum cut of the likelihoods and a contrast-sensitive smoothness. The mask is on the tape with
 * the cut's energy, the pixels it changed and its overlap with the polygon, as a board of background and foreground,
 * and the scene dimmed outside the mask with the rectangle over it is a probe.
 */

import { boardSteps, type DecisionSurface } from '../../lib/exposition/index.ts'
import type { Key } from '../../lib/random/index.ts'
import { dict, int, real } from '../../lib/space/index.ts'
import {
  type Algorithm,
  type Config,
  type Contract,
  type Derived,
  defineMethod,
  derived,
  Log,
  type Method,
  metricSeries,
  type Run,
  scalar,
  tensor,
  tensorSeries,
} from '../../lib/trace/index.ts'
import { type GrabCutState, grabCutAlgorithm } from '../../lib/vision/index.ts'
import { picture } from './_picture.ts'
import { grid, type Scene, scene } from './_scene.ts'

const POLYGONS = 3

/** The scene with the object to cut out: its largest polygon (by the pixels showing), and a rectangle around it. */
interface Task {
  readonly scene: Scene
  /** 1 for the object's pixels. */
  readonly truth: number[]
  /** `[x, y, width, height]`. */
  readonly rect: readonly [number, number, number, number]
}

function taskOf(key: Key, c: Config): Task {
  const size = c['size'] as number
  const sc = scene(key, { size, noise: c['noise'] as number, polygons: POLYGONS })
  const showing = new Array<number>(POLYGONS + 1).fill(0)
  for (const r of sc.regions) showing[r] = (showing[r] ?? 0) + 1
  const largest = showing.slice(1).reduce((best, n, i) => (n > (showing[best] ?? 0) ? i + 1 : best), 1)
  const truth = sc.regions.map((r) => (r === largest ? 1 : 0))
  let [x0, y0, x1, y1] = [size, size, -1, -1]
  truth.forEach((v, i) => {
    if (v === 0) return
    const [x, y] = [i % size, Math.floor(i / size)]
    x0 = Math.min(x0, x)
    y0 = Math.min(y0, y)
    x1 = Math.max(x1, x)
    y1 = Math.max(y1, y)
  })
  const pad = 3
  const rx = Math.max(0, x0 - pad)
  const ry = Math.max(0, y0 - pad)
  return { scene: sc, truth, rect: [rx, ry, Math.min(size, x1 + pad + 1) - rx, Math.min(size, y1 + pad + 1) - ry] }
}

/** Intersection over union of the mask with the object. */
function overlap(task: Task, mask: ArrayLike<number>): number {
  let both = 0
  let either = 0
  for (let i = 0; i < task.truth.length; i++) {
    const a = (task.truth[i] ?? 0) === 1
    const b = (mask[i] ?? 0) === 1
    if (a && b) both++
    if (a || b) either++
  }
  return either === 0 ? 1 : both / either
}

/** The scene dimmed outside the mask, with the rectangle's outline over it. */
function pictureOf(task: Task, s: GrabCutState): DecisionSurface {
  const { size } = task.scene
  const shown = task.scene.noisy.map((v, i) => ((s.mask[i] ?? 0) === 1 ? v : 0.25 * v))
  const [rx, ry, rw, rh] = task.rect
  const outline: [number, number][] = []
  for (let x = rx; x < rx + rw; x++) outline.push([x, ry], [x, ry + rh - 1])
  for (let y = ry + 1; y < ry + rh - 1; y++) outline.push([rx, y], [rx + rw - 1, y])
  return picture(shown, size, size, { points: outline, classes: ['rectangle'], labels: outline.map(() => 0) })
}

/**
 * The paths of {@link grabCutSegmentation}: the mask, the energy, the pixels relabelled and the overlap with the true outline of each step, and the derived `cut/picture`.
 *
 * @example
 * ```ts
 * import type { GrabCutPath } from 'omai-engine/methods/vision'
 * const path: GrabCutPath = 'cut/energy'
 * path // => 'cut/energy'
 * ```
 */
export type GrabCutPath = 'cut/mask' | 'cut/energy' | 'cut/changed' | 'cut/overlap' | 'cut/picture'

/**
 * GrabCut of the largest polygon of a scene, one mixture fit and cut per step.
 *
 * Configuration: `components` (Gaussians per side's grey-level mixture), `gamma` (the weight of smoothness against
 * the likelihoods: too high and the cut flattens the rectangle into background), `noise` (the scene's pixel noise) and `size` (the image side). Paths: `cut/mask` (1 for the
 * foreground, full), `cut/energy` (the cut's energy), `cut/changed` (pixels the cut relabelled), `cut/overlap`
 * (intersection over union with the polygon), the projection `board` (the mask as a board of background and
 * foreground) and the probe `cut/picture` (the scene dimmed outside the mask, with the rectangle).
 */
export const grabCutSegmentation: Method<GrabCutState, Config, GrabCutPath> = defineMethod<
  GrabCutState,
  Config,
  GrabCutPath
>({
  name: 'vision/grabcut',
  config: dict({
    components: int(1, 5, { default: 3, label: 'mixture components' }),
    // The paper's 50 weighs smoothness against colour likelihoods in three dimensions; grey levels in [0, 1] carry a
    // third of that, and 10 keeps the cut from flattening everything into background.
    gamma: real(1, 100, { default: 10, log: true, label: 'smoothness γ' }),
    noise: real(0, 0.3, { default: 0.08, label: 'pixel noise σ' }),
    size: int(16, 64, { default: 40, label: 'image side' }),
  }),
  make: (c, key) => {
    const task = taskOf(key, c)
    const { size } = task.scene
    const inner: Algorithm<GrabCutState> = grabCutAlgorithm(grid(task.scene.noisy, size), task.rect, {
      components: c['components'] as number,
      gamma: c['gamma'] as number,
      key,
    })
    return {
      name: 'grabcut',
      config: inner.config,
      init: (k) => inner.init(k),
      step: (s, ctx) => {
        const next = inner.step(s, ctx)
        ctx.log('cut/mask', grid(next.mask, size))
        ctx.log('cut/energy', next.energy)
        ctx.log('cut/changed', next.changed)
        ctx.log('cut/overlap', overlap(task, next.mask))
        return next
      },
      ...(inner.done === undefined ? {} : { done: inner.done }),
      serialise: (s) => ({ mask: Array.from(s.mask), energy: s.energy, iteration: s.iteration, changed: s.changed }),
      deserialise: (j) => {
        const o = j as { mask: number[]; energy: number; iteration: number; changed: number }
        return { mask: Uint8Array.from(o.mask), energy: o.energy, iteration: o.iteration, changed: o.changed }
      },
    }
  },
  projections: {
    board: (run: Run): Derived | undefined => {
      const masks = tensorSeries(run, 'cut/mask').filter((e) => e.values !== undefined && e.indices === undefined)
      const first = masks[0]
      if (first === undefined) return undefined
      const [rows, columns] = [first.shape[0] ?? 0, first.shape[1] ?? 0]
      const changed = metricSeries(run, 'cut/changed').values
      const metrics = changed.length === masks.length ? [{ name: 'changed', values: changed }] : []
      return boardSteps({
        method: 'grabcut',
        layout: { type: 'grid', rows, columns },
        categories: ['background', 'foreground'],
        steps: masks.map((e) => ({ cells: [...(e.values ?? [])] })),
        metrics,
        stepName: 'cut',
      })
    },
  },
  contract: (c, key): Contract<GrabCutState, GrabCutPath> => {
    const task = taskOf(key, c)
    const { size } = task.scene
    return {
      'cut/mask': tensor({ role: 'state', shape: [size, size], description: '1 for the foreground' }),
      'cut/energy': scalar({ description: 'the energy of the cut: likelihoods plus smoothness' }),
      'cut/changed': scalar({ description: 'pixels the cut relabelled' }),
      'cut/overlap': scalar({ description: 'intersection over union of the mask with the polygon' }),
      'cut/picture': derived<GrabCutState>({
        description: 'the scene dimmed outside the mask, with the rectangle over it',
        compute: (s): DecisionSurface => pictureOf(task, s),
      }),
    }
  },
  defaults: { 'cut/mask': Log({ detail: 'full' }), 'cut/energy': Log(), 'cut/changed': Log(), 'cut/overlap': Log() },
})
