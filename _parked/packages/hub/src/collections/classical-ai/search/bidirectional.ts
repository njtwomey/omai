/**
 * Hub entry for bidirectional search, one search forward from the start and one back from the goal until the frontiers
 * meet, played on grid maps and mazes.
 */

import type { HubEntry } from '../../../entry/entry.ts'
import { searchEntry } from './_search.ts'

/** Bidirectional search: one search from the start and one back from the goal. */
export const bidirectional: HubEntry = searchEntry({
  name: 'bidirectional',
  title: 'Bidirectional search',
  summary: 'Search forward from the start and backward from the goal until the two frontiers meet.',
  algorithm: 'bidirectional',
  tags: ['uninformed', 'optimal'],
  method: 'lib/search/uninformed.ts#bidirectionalSearch',
  references: ['Pohl (1971), Bi-directional search.'],
})
