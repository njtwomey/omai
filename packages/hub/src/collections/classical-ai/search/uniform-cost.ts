import type { HubEntry } from '../../../entry/entry.ts'
import { searchEntry } from './_search.ts'

/** Uniform-cost search (Dijkstra's algorithm): the frontier ordered by path cost. */
export const uniformCost: HubEntry = searchEntry({
  name: 'uniform-cost',
  title: 'Uniform-cost search',
  summary: 'Expand the cheapest path first: across the river it walks to the bridge rather than wading.',
  algorithm: 'uniform-cost',
  map: 'river',
  tags: ['uninformed', 'optimal', 'dijkstra'],
  method: 'lib/search/uninformed.ts#uniformCostSearch',
  references: ['Dijkstra (1959), A note on two problems in connexion with graphs.'],
})
