/**
 * Series whose class is a short pattern placed at a random position in noise: class 0 is noise alone, class c has the
 * c-th pattern (a bump, a ramp, a pulse) added once. The patterns are the shapelets a shapelet method should find, and
 * the truth holds them and where each was put, so a discovered or learned shapelet can be compared with what generated
 * the data.
 */

import { type ClassificationData, type DatasetGenerator, xyClasses } from '../data/index.ts'
import { foldIn, normal, uniform } from '../random/index.ts'
import { dict, int, real } from '../space/index.ts'
import { toFlat } from '../tensor/index.ts'
import { labels, matrix, stamp } from './_util.ts'

/** The planted patterns, each of length 7: a bump, a ramp and a pulse. */
export const shapeletPatterns: readonly (readonly number[])[] = [
  [0, 1, 2.5, 3, 2.5, 1, 0],
  [0, 0.75, 1.5, 2.25, 3, 3.75, 4.5],
  [0, 3, 3, 3, 3, 3, 0],
]

/**
 * Series with a planted shapelet: `n` series of noise at standard deviation `noise`, class c (0 to `classes - 1`) with
 * the c-th pattern of `shapeletPatterns` added once at a uniformly random position (class 0 has none). `x` is
 * `[n, length]`, `y` the class. The truth holds the patterns and the position of the pattern in each series (-1 for
 * class 0).
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { shapeletSeries } from 'mdml-engine/datasets'
 * generate(shapeletSeries, { knobs: { n: 20, length: 40 } }).data.x.shape // => [20, 40]
 * ```
 */
export const shapeletSeries: DatasetGenerator<ClassificationData> = {
  name: 'shapeletSeries',
  kind: 'classification',
  description: 'Series of noise in which each class (but the first) carries one planted pattern at a random position.',
  knobs: dict({
    n: int(2, 100_000, { default: 60, label: 'series', description: 'The number of series, classes in turn.' }),
    length: int(20, 10_000, { default: 60, label: 'length', description: 'The number of points in a series.' }),
    classes: int(2, 4, {
      default: 2,
      label: 'classes',
      description: 'The number of classes; the first has no pattern.',
    }),
    noise: real(0, 3, { default: 0.15, label: 'noise', description: 'The standard deviation of the noise.' }),
  }),
  generate: (k, knobs) => {
    const n = knobs['n'] as number
    const length = knobs['length'] as number
    const classes = knobs['classes'] as number
    const noise = knobs['noise'] as number
    const z = Array.from(toFlat(normal(foldIn(k, 0), [n, length])))
    const where = Array.from(toFlat(uniform(foldIn(k, 1), [n])))
    const x = new Float64Array(n * length)
    const y: number[] = []
    const positions: number[] = []
    for (let i = 0; i < n; i++) {
      const c = i % classes
      y.push(c)
      for (let t = 0; t < length; t++) x[i * length + t] = noise * (z[i * length + t] ?? 0)
      if (c === 0) {
        positions.push(-1)
        continue
      }
      const pattern = shapeletPatterns[c - 1] ?? []
      const at = Math.min(length - pattern.length, Math.floor((where[i] ?? 0) * (length - pattern.length + 1)))
      positions.push(at)
      for (const [j, v] of pattern.entries()) x[i * length + at + j] = (x[i * length + at + j] ?? 0) + v
    }
    const data = xyClasses(matrix(x, n, length), labels(y), {
      ...stamp('shapeletSeries', shapeletSeries.description, knobs),
      classes: Array.from({ length: classes }, (_, c) => (c === 0 ? 'noise' : `pattern ${c}`)),
      featureNames: Array.from({ length }, (_, t) => `t${t}`),
    })
    return {
      data,
      truth: {
        clusters: labels(y),
        latents: { positions },
        extra: { patterns: shapeletPatterns.slice(0, classes - 1).map((p) => [...p]) },
      },
    }
  },
}
