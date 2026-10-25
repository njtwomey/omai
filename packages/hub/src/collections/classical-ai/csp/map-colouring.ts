/**
 * Hub entry for map colouring as a constraint satisfaction problem: Australia in three colours or a random planar map
 * in four, solved region by region with the colours each choice rules out.
 */

import { AUSTRALIA, randomMap, solveMapColouring } from 'whitebox-engine/methods/gofai'
import { choice, dict, int } from 'whitebox-engine/space'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'
import { backtrackingControls, solverOptions } from './_csp.ts'

const COLOURS = ['red', 'green', 'blue', 'yellow']

/** Map colouring as a CSP. */
export const mapColouringEntry: HubEntry = defineEntry({
  id: 'classical-ai/csp/map-colouring',
  title: 'Map colouring',
  summary: 'Colour regions so no neighbours match: Australia in three colours, or a random planar map in four.',
  collection: 'classical-ai/csp',
  tags: ['constraint satisfaction', 'graph colouring', 'backtracking'],
  datasets: [],
  params: dict({
    map: choice(['Australia', 'random'], { default: 'Australia', label: 'map' }),
    regions: int(6, 30, { default: 16, label: 'regions (random map)' }),
    colours: int(2, 4, { default: 3, label: 'colours' }),
    solver: choice(['backtracking', 'min-conflicts'], { default: 'backtracking', label: 'solver' }),
    ...backtrackingControls,
  }),
  run: ({ params, key }) => {
    const map = params['map'] === 'random' ? randomMap(key, Number(params['regions'])) : AUSTRALIA
    return solveMapColouring(map, COLOURS.slice(0, Number(params['colours'])), { ...solverOptions(params), key })
  },
  views: [{ record: 'boardSteps', title: 'The map, region by region' }],
  methods: [
    { file: 'methods/gofai/csp.ts', name: 'solveMapColouring' },
    { file: 'lib/search/csp.ts', name: 'backtrackingSearch' },
  ],
  references: ['Russell and Norvig (2020), Artificial Intelligence: A Modern Approach, figure 6.1.'],
})
