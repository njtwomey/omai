/**
 * Hub entries for the n-queens puzzle as a constraint satisfaction problem, solved by backtracking with forward
 * checking and by min-conflicts local search.
 */

import { solveQueens } from 'mdml-engine/methods/gofai'
import { choice, dict, int } from 'mdml-engine/space'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { backtrackingControls, solverOptions } from './_csp.ts'

/** n queens by backtracking: queens placed column by column, attacked squares ruled out. */
export const nQueensEntry: HubEntry = defineEntry({
  id: 'classical-ai/csp/n-queens',
  title: 'n-queens by backtracking',
  summary: 'Place queens column by column; forward checking strikes out attacked squares and backtracks early.',
  collection: 'classical-ai/csp',
  tags: ['constraint satisfaction', 'backtracking', 'forward checking', 'n-queens'],
  datasets: [],
  params: dict({ n: int(4, 14, { default: 8, label: 'queens' }), ...backtrackingControls }),
  run: ({ params, key }) => solveQueens(Number(params['n']), { ...solverOptions(params), key }),
  views: [{ record: 'boardSteps', title: 'The board, value by value' }],
  methods: [
    { file: 'methods/gofai/csp.ts', name: 'solveQueens' },
    { file: 'lib/search/csp.ts', name: 'backtrackingSearch' },
  ],
  references: ['Russell and Norvig (2020), Artificial Intelligence: A Modern Approach, section 6.3.'],
})

/** n queens by min-conflicts: move a queen in conflict to the square with the fewest attacks. */
export const minConflictsEntry: HubEntry = defineEntry({
  id: 'classical-ai/csp/min-conflicts',
  title: 'Min-conflicts local search',
  summary: 'Start with every queen placed, then move a conflicted queen to its least attacked square until none are.',
  collection: 'classical-ai/csp',
  tags: ['constraint satisfaction', 'local search', 'n-queens'],
  datasets: [],
  params: dict({
    n: int(4, 40, { default: 16, label: 'queens' }),
    solver: choice(['min-conflicts'], { default: 'min-conflicts', label: 'solver' }),
  }),
  run: ({ params, key }) => solveQueens(Number(params['n']), { solver: 'min-conflicts', key }),
  views: [{ record: 'boardSteps', title: 'Queens moved one at a time' }],
  methods: [
    { file: 'methods/gofai/csp.ts', name: 'solveQueens' },
    { file: 'lib/search/csp.ts', name: 'minConflicts' },
  ],
  references: ['Minton et al. (1992), Minimizing conflicts: a heuristic repair method.'],
})
