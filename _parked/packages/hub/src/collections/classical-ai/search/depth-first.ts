/**
 * Hub entry for depth-first search, the frontier as a stack diving down one corridor at a time, played on grid maps and
 * mazes.
 */

import type { HubEntry } from '../../../entry/entry.ts'
import { searchEntry } from './_search.ts'

/** Depth-first search: the frontier as a stack. */
export const depthFirst: HubEntry = searchEntry({
  name: 'depth-first',
  title: 'Depth-first search',
  summary: 'Expand the deepest node first: it dives down one corridor at a time and backs up at dead ends.',
  algorithm: 'depth-first',
  tags: ['uninformed', 'stack'],
  method: 'lib/search/uninformed.ts#depthFirstSearch',
  references: ['Russell and Norvig (2020), Artificial Intelligence: A Modern Approach, section 3.4.3.'],
})
