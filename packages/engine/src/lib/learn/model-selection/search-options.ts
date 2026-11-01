/**
 * The candidate design of a hyperparameter search as an option table: the full grid of a set of value lists, or a
 * number of random candidates drawn from a space or from the lists. The table is one `union` space for controls, grids
 * and tuners, and builds the function that turns what is searched into a list of candidates.
 */

import type { Key } from '../../random/index.ts'
import { dict, int, type Option, type OptionTable, optionTable, type Space } from '../../space/index.ts'
import { type Candidate, type ParamGrid, parameterGrid, parameterSampler } from './search.ts'

/** What is searched: lists of values by name (or several grids), or a space to sample. */
export type SearchDomain = ParamGrid | Space

/** Turns what is searched into candidates; the key matters only to random designs. */
export type CandidateDesign = (domain: SearchDomain, key: Key) => Candidate[]

/** What a design does, for validity rules and selectors. */
export interface CandidateDesignProperties {
  /** Whether it covers every combination (so no candidate is missed). */
  readonly exhaustive: boolean
  /** Whether it can search a continuous space, not only lists of values. */
  readonly continuous: boolean
}

function design(
  description: string,
  properties: CandidateDesignProperties,
  space: ReturnType<typeof dict>,
  make: (nIter: number) => CandidateDesign,
): Option<CandidateDesign, CandidateDesignProperties> {
  return {
    description,
    properties,
    space,
    make: (config) => make(typeof config['nIter'] === 'number' ? config['nIter'] : 10),
  }
}

/**
 * The candidate designs of a search as one slot: `grid` (every combination of the lists, in sklearn's order) and
 * `random` (`nIter` draws from a space, or distinct grid points drawn without replacement from lists).
 *
 * @concept hyperparameter-search
 * @example
 * ```ts
 * import { searchDesigns } from 'mdml-engine/learn/model-selection'
 * import { key } from 'mdml-engine/random'
 * searchDesigns.make({ kind: 'grid' })({ a: [1, 2], b: [3] }, key(0)) // => [{ a: 1, b: 3 }, { a: 2, b: 3 }]
 * ```
 */
export const searchDesigns: OptionTable<CandidateDesign, CandidateDesignProperties> = optionTable(
  {
    grid: design(
      'every combination of the lists of values',
      { exhaustive: true, continuous: false },
      dict({}),
      () => (domain) => parameterGrid(domain as ParamGrid),
    ),
    random: design(
      'a number of candidates drawn from a space, or distinct grid points without replacement',
      { exhaustive: false, continuous: true },
      dict({ nIter: int(1, 1000, { default: 10, label: 'candidates' }) }),
      (nIter) => (domain, key) => parameterSampler(domain as Parameters<typeof parameterSampler>[0], nIter, key),
    ),
  },
  { default: 'grid', label: 'candidate design', description: 'How the hyperparameter settings to try are chosen.' },
)
