/**
 * The slots of the stepped vision algorithms that vary as option tables: how soft non-maximum suppression decays a score
 * (Gaussian or linear), how strongly seam-carving masks bias the energy, when mean shift stops, whether flood fill
 * compares with its neighbour or the seed and how it connects, and how a polygon is simplified. Each table is one `union`
 * space for controls and grids.
 */

import {
  choice,
  defaults,
  dict,
  int,
  type Option,
  type OptionTable,
  optionTable,
  real,
  type SpaceValue,
} from '../space/index.ts'
import type { SoftNmsOptions } from './boxes.ts'
import type { FloodFillOptions } from './components.ts'
import type { SeamCarveOptions } from './seams.ts'
import type { MeanShiftOptions } from './tracking.ts'

function option<T>(
  description: string,
  space: ReturnType<typeof dict>,
  build: (k: (name: string) => number | string | boolean) => T,
): Option<T, Record<string, never>> {
  return {
    description,
    properties: {},
    space,
    make: (config) => {
      const full = { ...(defaults(space) as Readonly<Record<string, SpaceValue>>), ...config }
      return build((name) => full[name] as number | string | boolean)
    },
  }
}

const scoreThreshold = (): ReturnType<typeof real> =>
  real(1e-6, 0.5, { default: 0.001, log: true, label: 'score threshold' })

/**
 * How soft non-maximum suppression decays the score of an overlapping box, as one slot: `gaussian` (s exp(-iou^2 / sigma),
 * every overlap decays a little) and `linear` (s (1 - iou) above the IoU threshold only).
 *
 * @concept non-maximum-suppression
 * @example
 * ```ts
 * import { softNmsMethods } from 'mdml-engine/vision'
 * softNmsMethods.make({ kind: 'linear', iouThreshold: 0.5 }).iouThreshold // => 0.5
 * ```
 */
export const softNmsMethods: OptionTable<SoftNmsOptions, Record<string, never>> = optionTable(
  {
    gaussian: option<SoftNmsOptions>(
      'every overlap decays the score by exp(-iou^2 / sigma)',
      dict({ sigma: real(0.05, 5, { default: 0.5, label: 'sigma' }), scoreThreshold: scoreThreshold() }),
      (k) => ({ method: 'gaussian', sigma: k('sigma') as number, scoreThreshold: k('scoreThreshold') as number }),
    ),
    linear: option<SoftNmsOptions>(
      'overlaps above the IoU threshold decay the score by 1 - iou',
      dict({
        iouThreshold: real(0.05, 0.95, { default: 0.3, label: 'IoU threshold' }),
        scoreThreshold: scoreThreshold(),
      }),
      (k) => ({
        method: 'linear',
        iouThreshold: k('iouThreshold') as number,
        scoreThreshold: k('scoreThreshold') as number,
      }),
    ),
  },
  { default: 'gaussian', label: 'score decay', description: 'How an overlapping box loses score.' },
)

/**
 * How strongly the protect and remove masks of seam carving bias the energy, as one slot: `gentle` (1e3: a protected
 * region is avoided unless the image is very busy), `firm` (1e6, the default) and `absolute` (1e12: never crossed unless
 * unavoidable).
 *
 * @concept seam-carving
 * @example
 * ```ts
 * import { seamMaskStrengths } from 'mdml-engine/vision'
 * seamMaskStrengths.make({ kind: 'gentle' }).bias // => 1000
 * ```
 */
export const seamMaskStrengths: OptionTable<SeamCarveOptions, Record<string, never>> = optionTable(
  {
    gentle: option<SeamCarveOptions>(
      'a mask is a strong hint',
      dict({ bias: real(1, 1e15, { default: 1e3, log: true, label: 'bias' }) }),
      (k) => ({ bias: k('bias') as number }),
    ),
    firm: option<SeamCarveOptions>(
      'a mask is nearly a wall',
      dict({ bias: real(1, 1e15, { default: 1e6, log: true, label: 'bias' }) }),
      (k) => ({ bias: k('bias') as number }),
    ),
    absolute: option<SeamCarveOptions>(
      'a mask is a wall',
      dict({ bias: real(1, 1e15, { default: 1e12, log: true, label: 'bias' }) }),
      (k) => ({ bias: k('bias') as number }),
    ),
  },
  {
    default: 'firm',
    label: 'mask strength',
    description: 'How much the protect and remove masks weigh against the image energy.',
  },
)

/**
 * When mean shift stops, as one slot: `loose` (a window that moves less than 2 pixels, at most 20 updates), `standard`
 * (OpenCV's default: less than 1 pixel, 100 updates) and `tight` (a stationary window, 500 updates).
 *
 * @concept mean-shift
 * @example
 * ```ts
 * import { meanShiftCriteria } from 'mdml-engine/vision'
 * meanShiftCriteria.make({ kind: 'standard' }) // => { maxIter: 100, eps: 1 }
 * ```
 */
export const meanShiftCriteria: OptionTable<MeanShiftOptions, Record<string, never>> = optionTable(
  {
    loose: option<MeanShiftOptions>(
      'stop early',
      dict({
        maxIter: int(1, 1000, { default: 20, label: 'most updates' }),
        eps: real(0.1, 10, { default: 2, label: 'movement below which to stop' }),
      }),
      (k) => ({ maxIter: k('maxIter') as number, eps: k('eps') as number }),
    ),
    standard: option<MeanShiftOptions>(
      "OpenCV's default",
      dict({
        maxIter: int(1, 1000, { default: 100, label: 'most updates' }),
        eps: real(0.1, 10, { default: 1, label: 'movement below which to stop' }),
      }),
      (k) => ({ maxIter: k('maxIter') as number, eps: k('eps') as number }),
    ),
    tight: option<MeanShiftOptions>(
      'stop only when the window sits still',
      dict({
        maxIter: int(1, 1000, { default: 500, label: 'most updates' }),
        eps: real(0.01, 10, { default: 0.5, label: 'movement below which to stop' }),
      }),
      (k) => ({ maxIter: k('maxIter') as number, eps: k('eps') as number }),
    ),
  },
  { default: 'standard', label: 'termination', description: 'When the window update stops.' },
)

/**
 * How flood fill decides a pixel joins, as one slot: `floating` (against the neighbour it grows from, so the region can
 * drift along a gradient) and `fixed` (against the seed, so it cannot), each with the tolerances below and above and the
 * connectivity as knobs.
 *
 * @concept flood-fill
 * @example
 * ```ts
 * import { floodFillModes } from 'mdml-engine/vision'
 * floodFillModes.make({ kind: 'fixed', loDiff: 0.2 }).fixedRange // => true
 * ```
 */
export const floodFillModes: OptionTable<FloodFillOptions, Record<string, never>> = optionTable(
  {
    floating: option<FloodFillOptions>(
      'compare with the neighbour the region grows from',
      dict({
        loDiff: real(0, 1, { default: 0.1, label: 'step down' }),
        upDiff: real(0, 1, { default: 0.1, label: 'step up' }),
        connectivity: choice([4, 8], { default: 4, label: 'connectivity' }),
      }),
      (k) => ({
        loDiff: k('loDiff') as number,
        upDiff: k('upDiff') as number,
        connectivity: k('connectivity') as 4 | 8,
        fixedRange: false,
      }),
    ),
    fixed: option<FloodFillOptions>(
      'compare with the seed',
      dict({
        loDiff: real(0, 1, { default: 0.1, label: 'step down' }),
        upDiff: real(0, 1, { default: 0.1, label: 'step up' }),
        connectivity: choice([4, 8], { default: 4, label: 'connectivity' }),
      }),
      (k) => ({
        loDiff: k('loDiff') as number,
        upDiff: k('upDiff') as number,
        connectivity: k('connectivity') as 4 | 8,
        fixedRange: true,
      }),
    ),
  },
  { default: 'floating', label: 'range', description: 'What a pixel is compared with to join the region.' },
)

/** The settings of a polygon simplification. */
export interface SimplificationOptions {
  readonly epsilon: number
  readonly closed: boolean
}

/**
 * How a curve is simplified, as one slot: `polygon` (a closed curve, the usual case for contours) and `polyline` (an open
 * curve), each with the tolerance epsilon as a knob.
 *
 * @concept polygon-simplification
 * @example
 * ```ts
 * import { simplifications } from 'mdml-engine/vision'
 * simplifications.make({ kind: 'polyline', epsilon: 2 }) // => { epsilon: 2, closed: false }
 * ```
 */
export const simplifications: OptionTable<SimplificationOptions, Record<string, never>> = optionTable(
  {
    polygon: option<SimplificationOptions>(
      'a closed curve',
      dict({ epsilon: real(0, 50, { default: 1, label: 'epsilon' }) }),
      (k) => ({ epsilon: k('epsilon') as number, closed: true }),
    ),
    polyline: option<SimplificationOptions>(
      'an open curve',
      dict({ epsilon: real(0, 50, { default: 1, label: 'epsilon' }) }),
      (k) => ({ epsilon: k('epsilon') as number, closed: false }),
    ),
  },
  { default: 'polygon', label: 'curve', description: 'Whether the curve closes on itself.' },
)
