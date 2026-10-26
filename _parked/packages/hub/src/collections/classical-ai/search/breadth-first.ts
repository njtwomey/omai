/**
 * Hub entry for breadth-first search, the frontier as a queue expanding layer by layer from the start, played on grid
 * maps and mazes.
 */

import type { HubEntry } from '../../../entry/entry.ts'
import { searchEntry } from './_search.ts'

/** Breadth-first search: the frontier as a queue, layer by layer. */
export const breadthFirst: HubEntry = searchEntry({
  name: 'breadth-first',
  title: 'Breadth-first search',
  summary: 'Expand the shallowest node first: a wave spreads out from the start and meets the goal by fewest moves.',
  algorithm: 'breadth-first',
  tags: ['uninformed', 'queue', 'complete'],
  method: 'lib/search/uninformed.ts#breadthFirstSearch',
  references: ['Russell and Norvig (2020), Artificial Intelligence: A Modern Approach, section 3.4.1.'],
})
