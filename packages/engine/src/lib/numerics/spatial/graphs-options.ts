/**
 * The slot of the HNSW build that varies as an option table: the profile of the graph (how many links each node keeps
 * and how wide the beam is while building) as fast-build, balanced or high-recall, with the distance as a knob. The
 * table is one `union` space for controls and grids, and builds the options of `hnswIndex` and `hnswAlgorithm` (the key
 * is the caller's).
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
} from '../../space/index.ts'
import type { Metric } from '../geometry/index.ts'
import type { HnswOptions } from './graphs.ts'

/** What an HNSW profile is built from: everything of {@link HnswOptions} but the key. */
export type HnswProfile = Omit<HnswOptions, 'key'>

/** What a profile trades, for validity rules and selectors. */
export interface HnswProfileProperties {
  /** The links per node on upper layers it starts from. */
  readonly m: number
}

const METRICS: readonly Metric[] = ['euclidean', 'sqeuclidean', 'cityblock', 'cosine']

function profile(description: string, m: number, ef: number): Option<HnswProfile, HnswProfileProperties> {
  const space = dict({
    m: int(2, 64, { default: m, label: 'links per node (M)' }),
    efConstruction: int(8, 800, { default: ef, label: 'build beam width' }),
    levelMultiplier: real(0.1, 2, { default: 1 / Math.log(Math.max(2, m)), label: 'level multiplier' }),
    metric: choice(METRICS, { default: 'euclidean', label: 'distance' }),
  })
  return {
    description,
    properties: { m },
    space,
    make: (config) => {
      const full = { ...(defaults(space) as Readonly<Record<string, SpaceValue>>), ...config }
      return {
        m: full['m'] as number,
        efConstruction: full['efConstruction'] as number,
        levelMultiplier: full['levelMultiplier'] as number,
        metric: full['metric'] as Metric,
      }
    },
  }
}

/**
 * HNSW build profiles as one slot: `fast-build` (M 8, a narrow beam), `balanced` (M 16, beam 200, the paper's usual
 * choice) and `high-recall` (M 32, beam 400), each with the links, beam, level multiplier and distance as knobs.
 *
 * @concept approximate-nearest-neighbours
 * @example
 * ```ts
 * import { hnswProfiles } from 'mdml-engine/numerics/spatial'
 * hnswProfiles.make({ kind: 'high-recall' }).m // => 32
 * ```
 */
export const hnswProfiles: OptionTable<HnswProfile, HnswProfileProperties> = optionTable(
  {
    'fast-build': profile('few links and a narrow beam: quick to build, lower recall', 8, 50),
    balanced: profile('M 16 and beam 200', 16, 200),
    'high-recall': profile('many links and a wide beam: slow to build, high recall', 32, 400),
  },
  {
    default: 'balanced',
    label: 'HNSW profile',
    description: 'How densely the graph is linked and how hard it is searched while building.',
  },
)
