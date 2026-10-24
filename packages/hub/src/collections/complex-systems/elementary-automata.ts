import { elementaryAutomaton } from 'whitebox-engine/methods/complex-systems'
import { choice, dict, int } from 'whitebox-engine/space'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'

/** Wolfram's elementary cellular automata. */
export const elementaryAutomataEntry: HubEntry = defineEntry({
  id: 'complex-systems/elementary-automata',
  title: 'Elementary cellular automata',
  summary: 'A row of cells updated by a rule of three neighbours: order, fractals, chaos and computation from 8 bits.',
  collection: 'complex-systems',
  tags: ['cellular automata', 'Wolfram', 'emergence'],
  datasets: [],
  params: dict({
    rule: int(0, 255, { default: 30, label: 'rule' }),
    start: choice(['single', 'random'], { default: 'single', label: 'start' }),
    width: int(21, 121, { default: 63, label: 'width' }),
    steps: int(10, 80, { default: 40, label: 'steps' }),
  }),
  run: ({ params, key }) =>
    elementaryAutomaton(Number(params['rule']), {
      start: params['start'] === 'random' ? 'random' : 'single',
      width: Number(params['width']),
      steps: Number(params['steps']),
      key,
    }),
  views: [{ record: 'boardSteps', title: 'Space-time diagram, row by row' }],
  methods: [{ file: 'methods/complex-systems/cellular-automata.ts', name: 'elementaryAutomaton' }],
  references: ['Wolfram (1983), Statistical mechanics of cellular automata.'],
})
