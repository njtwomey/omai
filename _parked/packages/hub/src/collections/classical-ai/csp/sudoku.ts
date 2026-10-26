/**
 * Hub entry for Sudoku as a constraint satisfaction problem, solved by backtracking that fills the cell with the fewest
 * candidates first and strikes out values by forward checking.
 */

import { SUDOKUS, solveSudoku } from 'mdml-engine/methods/gofai'
import { choice, dict } from 'mdml-engine/space'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { backtrackingControls, solverOptions } from './_csp.ts'

/** Sudoku as a CSP. */
export const sudokuEntry: HubEntry = defineEntry({
  id: 'classical-ai/csp/sudoku',
  title: 'Sudoku by constraint propagation',
  summary: 'Backtracking with the fewest-candidates cell first and forward checking: an easy grid needs no backtracks.',
  collection: 'classical-ai/csp',
  tags: ['constraint satisfaction', 'backtracking', 'all-different', 'sudoku'],
  datasets: [],
  params: dict({
    puzzle: choice(['easy', 'hard', 'mini'], { default: 'easy', label: 'puzzle' }),
    ...backtrackingControls,
  }),
  run: ({ params, key }) =>
    solveSudoku(SUDOKUS[params['puzzle'] as 'easy'] ?? SUDOKUS.easy, { ...solverOptions(params), key }),
  views: [{ record: 'boardSteps', title: 'The grid, digit by digit' }],
  methods: [
    { file: 'methods/gofai/csp.ts', name: 'solveSudoku' },
    { file: 'lib/search/csp.ts', name: 'backtrackingSearch' },
  ],
  references: ['Russell and Norvig (2020), Artificial Intelligence: A Modern Approach, section 6.2.6.'],
})
