/**
 * The slot of the streaming quantile sketches that varies as an option table: Greenwald-Khanna (a deterministic rank
 * error epsilon), KLL (a size parameter k and a key for its coins) and P-squared (five markers for one quantile), each
 * with its own knob. The table is one `union` space for controls and grids, and builds the stepped algorithm for a stream.
 */

import type { Key } from '../random/index.ts'
import { dict, int, knobOption, type Option, type OptionTable, optionTable, real } from '../space/index.ts'
import type { Algorithm } from '../trace/index.ts'
import { gkSummaryAlgorithm, kllSketchAlgorithm, p2Algorithm, type QuantileSketchPath } from './quantiles.ts'

/** The stream a sketch reads, the quantiles it reports, and the key of the randomised sketches. */
export interface SketchInput {
  readonly items: readonly number[]
  readonly quantiles?: readonly number[]
  readonly key: Key
}

/** Builds the stepped algorithm of a sketch for a stream. */
export type QuantileSketchBuilder = (input: SketchInput) => Algorithm<unknown, QuantileSketchPath>

/** What a sketch guarantees, for validity rules and selectors. */
export interface QuantileSketchProperties {
  /** Whether it flips coins (so a run depends on the key). */
  readonly randomised: boolean
  /** Whether the error is a guarantee on every rank (Greenwald-Khanna) or a probabilistic or heuristic one. */
  readonly guaranteed: boolean
  /** Whether it answers any quantile after the stream (as opposed to the one it tracks). */
  readonly anyQuantile: boolean
}

function sketch(
  description: string,
  properties: QuantileSketchProperties,
  space: ReturnType<typeof dict>,
  build: (k: (name: string) => number) => QuantileSketchBuilder,
): Option<QuantileSketchBuilder, QuantileSketchProperties> {
  return knobOption(description, properties, space, build)
}

/**
 * The streaming quantile sketches as one slot: `greenwald-khanna` (deterministic, every rank within epsilon n),
 * `kll` (randomised compactors, rank error about 1.65 / k, mergeable) and `p2` (five markers for one quantile, no
 * guarantee). A table's `make` gives a builder from a stream to a stepped algorithm.
 *
 * @concept quantile-sketch
 * @example
 * ```ts
 * import { quantileSketches } from 'mdml-engine/sketches'
 * import { key } from 'mdml-engine/random'
 * quantileSketches.make({ kind: 'kll', k: 16 })({ items: [1, 2, 3], key: key(0) }).name // => 'sketches/kll'
 * ```
 */
export const quantileSketches: OptionTable<QuantileSketchBuilder, QuantileSketchProperties> = optionTable(
  {
    'greenwald-khanna': sketch(
      'deterministic summary of tuples; every rank within epsilon n',
      { randomised: false, guaranteed: true, anyQuantile: true },
      dict({ epsilon: real(0.001, 0.5, { default: 0.05, log: true, label: 'rank error epsilon' }) }),
      (k) => (input) =>
        gkSummaryAlgorithm({
          epsilon: k('epsilon'),
          items: input.items,
          ...(input.quantiles === undefined ? {} : { quantiles: input.quantiles }),
        }) as unknown as Algorithm<unknown, QuantileSketchPath>,
    ),
    kll: sketch(
      'randomised compactors; rank error about 1.65 / k, mergeable',
      { randomised: true, guaranteed: false, anyQuantile: true },
      dict({ k: int(8, 1000, { default: 200, label: 'size k' }) }),
      (k) => (input) =>
        kllSketchAlgorithm({
          k: k('k'),
          key: input.key,
          items: input.items,
          ...(input.quantiles === undefined ? {} : { quantiles: input.quantiles }),
        }) as unknown as Algorithm<unknown, QuantileSketchPath>,
    ),
    p2: sketch(
      'five markers tracking one quantile by parabolic interpolation',
      { randomised: false, guaranteed: false, anyQuantile: false },
      dict({ p: real(0.01, 0.99, { default: 0.5, label: 'quantile p' }) }),
      (k) => (input) =>
        p2Algorithm({ p: k('p'), items: input.items }) as unknown as Algorithm<unknown, QuantileSketchPath>,
    ),
  },
  {
    default: 'greenwald-khanna',
    label: 'quantile sketch',
    description: 'How a stream of numbers is summarised for quantile queries.',
  },
)
