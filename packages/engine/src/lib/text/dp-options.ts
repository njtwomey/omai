/**
 * The slot of the edit-distance tables that varies as an option table: the weighted Levenshtein table (with the cost of
 * each operation as a knob), the longest common subsequence, the unit-cost table behind error-rate counts, and the
 * Sellers table of approximate substring matching (with the edit budget k). The table is one `union` space for controls
 * and grids, and builds the options of `dpAlgorithm`.
 */

import { dict, int, knobOption, type Option, type OptionTable, optionTable, real } from '../space/index.ts'
import type { DpOptions } from './dp-tables.ts'

/** What a table computes, for validity rules and selectors. */
export interface DpKindProperties {
  /** Whether the match may start anywhere in the second sequence. */
  readonly freeStart: boolean
  /** Whether the result is a distance (lower is closer) rather than a length. */
  readonly distance: boolean
}

function kind(
  description: string,
  properties: DpKindProperties,
  space: ReturnType<typeof dict>,
  build: (k: (name: string) => number) => DpOptions,
): Option<DpOptions, DpKindProperties> {
  return knobOption(description, properties, space, build)
}

/**
 * The edit-distance tables as one slot: `levenshtein` (cheapest insertions, deletions and substitutions, costs as
 * knobs), `lcs` (the longest common subsequence), `edit-counts` (hits, substitutions, deletions and insertions of a
 * minimal alignment, for error rates) and `sellers` (approximate substring matching with a free start, within k edits).
 *
 * @concept edit-distance
 * @example
 * ```ts
 * import { dpKinds } from 'mdml-engine/text'
 * dpKinds.make({ kind: 'sellers', k: 2 }) // => { kind: 'sellers', k: 2 }
 * ```
 */
export const dpKinds: OptionTable<DpOptions, DpKindProperties> = optionTable(
  {
    levenshtein: kind(
      'the cheapest insertions, deletions and substitutions turning one sequence into the other',
      { freeStart: false, distance: true },
      dict({
        insertion: real(0.1, 10, { default: 1, label: 'insertion cost' }),
        deletion: real(0.1, 10, { default: 1, label: 'deletion cost' }),
        substitution: real(0.1, 10, { default: 1, label: 'substitution cost' }),
      }),
      (k) => ({
        kind: 'levenshtein',
        weights: { insertion: k('insertion'), deletion: k('deletion'), substitution: k('substitution') },
      }),
    ),
    lcs: kind('the longest subsequence both sequences share', { freeStart: false, distance: false }, dict({}), () => ({
      kind: 'lcs',
    })),
    'edit-counts': kind(
      'hits, substitutions, deletions and insertions of a minimal alignment, for error rates',
      { freeStart: false, distance: true },
      dict({}),
      () => ({ kind: 'edit-counts' }),
    ),
    sellers: kind(
      'the pattern against substrings of the text: each end within k edits',
      { freeStart: true, distance: true },
      dict({ k: int(0, 20, { default: 1, label: 'edits allowed' }) }),
      (k) => ({ kind: 'sellers', k: k('k') }),
    ),
  },
  { default: 'levenshtein', label: 'edit table', description: 'Which dynamic-programming table the alignment fills.' },
)
