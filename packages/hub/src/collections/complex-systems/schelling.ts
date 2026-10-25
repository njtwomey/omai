/**
 * Hub entry for Schelling's segregation model, where agents that move when too few of their neighbours are alike sort
 * themselves into segregated blocks.
 */

import { schelling } from 'whitebox-engine/methods/complex-systems'
import { dict, int, real } from 'whitebox-engine/space'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'

/** Schelling's segregation model. */
export const schellingEntry: HubEntry = defineEntry({
  id: 'complex-systems/schelling',
  title: 'Schelling segregation',
  summary: 'Agents happy with a minority of like neighbours still sort themselves into segregated blocks.',
  collection: 'complex-systems',
  tags: ['agent-based model', 'segregation', 'emergence'],
  datasets: [],
  params: dict({
    threshold: real(0, 1, { default: 0.4, label: 'like neighbours wanted' }),
    vacancy: real(0.02, 0.5, { default: 0.1, label: 'empty houses' }),
    size: int(10, 40, { default: 24, label: 'town size' }),
    rounds: int(5, 100, { default: 30, label: 'rounds' }),
  }),
  run: ({ params, key }) =>
    schelling({
      threshold: Number(params['threshold']),
      vacancy: Number(params['vacancy']),
      size: Number(params['size']),
      rounds: Number(params['rounds']),
      key,
    }),
  views: [{ record: 'boardSteps', title: 'The town, round by round (× unhappy)' }],
  methods: [{ file: 'methods/complex-systems/agent-based.ts', name: 'schelling' }],
  references: ['Schelling (1971), Dynamic models of segregation.'],
})
