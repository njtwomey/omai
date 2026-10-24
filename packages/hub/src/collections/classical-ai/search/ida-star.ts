import type { HubEntry } from '../../../entry/entry.ts'
import { searchEntry } from './_search.ts'

/** Iterative-deepening A*: depth-first search under a growing cost bound. */
export const idaStar: HubEntry = searchEntry({
  name: 'ida-star',
  title: 'Iterative-deepening A*',
  summary: 'Depth-first search cut off where g + h exceeds a bound, raised to the smallest excess each round.',
  algorithm: 'ida-star',
  map: 'open',
  tags: ['informed', 'heuristic', 'memory-bounded', 'optimal'],
  method: 'lib/search/informed.ts#idaStarSearch',
  references: ['Korf (1985), Depth-first iterative-deepening: an optimal admissible tree search.'],
})
