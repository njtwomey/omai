/**
 * Hub entry for greedy best-first search, the frontier ordered by the heuristic alone, played on grid maps and mazes
 * where a wall can fool it.
 */

import type { HubEntry } from '../../../entry/entry.ts'
import { searchEntry } from './_search.ts'

/** Greedy best-first search: the frontier ordered by the heuristic alone. */
export const greedy: HubEntry = searchEntry({
  name: 'greedy',
  title: 'Greedy best-first search',
  summary: 'Expand the node that looks closest to the goal: fast in the open, fooled by a wall cupped towards it.',
  algorithm: 'greedy',
  map: 'detour',
  tags: ['informed', 'heuristic'],
  method: 'lib/search/informed.ts#greedyBestFirstSearch',
  references: ['Russell and Norvig (2020), Artificial Intelligence: A Modern Approach, section 3.5.1.'],
})
