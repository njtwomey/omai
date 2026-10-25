/**
 * Hub entry for an epidemic among people moving about a room: the SEIR agent simulation of 2020's explainers, with
 * distancing, reach, transmission and the incubation and infectious periods to change, and the SEIR equations beside.
 */

import { epidemic as epidemicMethod } from 'mdml-engine/methods/complex-systems'
import { dict, int, real, type SpaceValue } from 'mdml-engine/space'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'

const num = (v: SpaceValue | undefined, fallback: number): number => (typeof v === 'number' ? v : fallback)

/** An SEIR epidemic among people moving about a room, with distancing. */
export const epidemic: HubEntry = defineEntry({
  id: 'complex-systems/epidemic',
  title: 'An epidemic in a room (SEIR)',
  summary:
    'People are susceptible, exposed, infectious or recovered; the infectious pass it on to whoever comes close. Keep a share of people still and watch the curve flatten.',
  collection: 'complex-systems',
  tags: ['epidemics', 'SEIR', 'agent-based models', 'flattening the curve'],
  datasets: [],
  params: dict({
    people: int(50, 600, { default: 300 }),
    distancing: real(0, 0.95, { default: 0, label: 'share staying put' }),
    radius: real(0.5, 8, { default: 4, label: 'reach' }),
    transmission: real(0.01, 0.5, { log: true, default: 0.08, label: 'transmission per step' }),
    incubation: real(0.5, 10, { default: 3, label: 'days exposed (mean)' }),
    infectious: real(1, 20, { default: 7, label: 'days infectious (mean)' }),
    infected: int(1, 20, { default: 3, label: 'infectious at the start' }),
    days: int(20, 300, { default: 120 }),
  }),
  run: ({ params, key }) =>
    epidemicMethod({
      people: num(params['people'], 300),
      distancing: num(params['distancing'], 0),
      radius: num(params['radius'], 4),
      transmission: num(params['transmission'], 0.08),
      incubation: num(params['incubation'], 3),
      infectious: num(params['infectious'], 7),
      infected: num(params['infected'], 3),
      days: num(params['days'], 120),
      key,
    }),
  views: [
    {
      record: 'agentSteps',
      title: 'The room, day by day; the share in each state (solid) beside the well-mixed SEIR equations (dashed)',
    },
  ],
  methods: [
    { file: 'methods/complex-systems/epidemics.ts', name: 'epidemic' },
    { file: 'lib/dynamics/epidemics.ts', name: 'epidemicAlgorithm' },
    { file: 'lib/dynamics/epidemics.ts', name: 'seirCurves' },
  ],
  references: [
    'Kermack and McKendrick (1927), A contribution to the mathematical theory of epidemics',
    'Stevens (2020), Why outbreaks like coronavirus spread exponentially, and how to “flatten the curve”, Washington Post',
  ],
})
