import { choice, real } from 'whitebox-engine/space'
import type { HubEntry } from '../../../entry/entry.ts'
import { searchEntry } from './_search.ts'

/** A* search: the frontier ordered by path cost plus heuristic. */
export const aStar: HubEntry = searchEntry({
  name: 'a-star',
  title: 'A* search',
  summary:
    'Expand the node with the least cost so far plus estimated cost to go: optimal with an admissible heuristic.',
  algorithm: 'a-star',
  map: 'detour',
  tags: ['informed', 'heuristic', 'optimal'],
  extra: { metric: choice(['manhattan', 'euclidean', 'chebyshev'], { default: 'manhattan', label: 'heuristic' }) },
  method: 'lib/search/informed.ts#aStarSearch',
  references: [
    'Hart, Nilsson and Raphael (1968), A formal basis for the heuristic determination of minimum cost paths.',
  ],
})

/** Weighted A*: the heuristic counted several times over. */
export const weightedAStar: HubEntry = searchEntry({
  name: 'weighted-a-star',
  title: 'Weighted A*',
  summary: 'A* with the heuristic weighted up: fewer expansions, paths at most that weight times the cheapest.',
  algorithm: 'weighted-a-star',
  map: 'rooms',
  tags: ['informed', 'heuristic', 'bounded suboptimal'],
  extra: { weight: real(1, 5, { default: 2, label: 'heuristic weight' }) },
  method: 'lib/search/informed.ts#weightedAStarSearch',
  references: ['Pohl (1970), Heuristic search viewed as path finding in a graph.'],
})
