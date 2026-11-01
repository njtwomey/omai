/**
 * What the greedy vision algorithms log: soft non-maximum suppression and Hough peak picking log the box or peak kept
 * and how much they suppressed; region growing logs the pixel absorbed and its distance to the region mean; flood fill
 * logs the span filled.
 */

import { type Contract, defineContract, scalar, tensor } from '../trace/index.ts'

/** The paths the greedy algorithms log, as a type. */
export type GreedyPath =
  | 'greedy/pick'
  | 'greedy/suppress'
  | 'greedy/score'
  | 'region/pixel'
  | 'region/mean'
  | 'fill/span'
  | 'fill/area'

/** What the greedy algorithms log, by path. */
export const greedyContract: Contract<unknown, GreedyPath> = defineContract({
  'greedy/pick': scalar({ description: 'the index of the box (or the cell of the Hough peak) kept at this step' }),
  'greedy/suppress': scalar({ description: 'the candidates whose score was decayed or removed by the pick' }),
  'greedy/score': scalar({ description: 'the score (or votes) of the pick when it was kept', direction: 'higher' }),
  'region/pixel': scalar({ description: 'the pixel (row-major index) absorbed into a region at this step' }),
  'region/mean': scalar({
    description: 'the colour distance of that pixel to the mean of its region',
    direction: 'lower',
  }),
  'fill/span': scalar({ description: 'the length of the span of pixels filled at this step' }),
  'fill/area': scalar({ description: 'the pixels filled so far' }),
})

/** The paths the geometric constructions log, as a type. */
export type GeomPath = 'geom/circle' | 'geom/split' | 'geom/points'

/**
 * What the minimum enclosing circle and the polygon simplification log: the circle (centre x, y and radius) after each
 * point added, the index of the point a curve was split at, and the number of points processed or kept so far.
 */
export const geomContract: Contract<unknown, GeomPath> = defineContract({
  'geom/circle': tensor({ role: 'state', shape: [3], description: 'the circle: centre x, centre y and radius' }),
  'geom/split': scalar({
    description: 'the index of the curve point the segment was split at (or -1: the segment is kept)',
  }),
  'geom/points': scalar({ description: 'points added to the circle so far, or vertices kept so far' }),
})
