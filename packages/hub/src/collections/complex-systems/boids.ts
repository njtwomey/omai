/**
 * Hub entry for Reynolds' boids: a flock on a torus steering by separation, alignment and cohesion, played frame by
 * frame with its polarisation.
 */

import { flocking } from 'mdml-engine/methods/complex-systems'
import { dict, int, real, type SpaceValue } from 'mdml-engine/space'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'

const num = (v: SpaceValue | undefined, fallback: number): number => (typeof v === 'number' ? v : fallback)

/** Reynolds' boids: order from three local rules. */
export const boids: HubEntry = defineEntry({
  id: 'complex-systems/boids',
  title: 'Boids: flocking from three rules',
  summary:
    'Each bird only keeps its distance, matches its neighbours’ heading and drifts towards their centre; a flock emerges. Turn one rule off and watch what breaks.',
  collection: 'complex-systems',
  tags: ['agent-based models', 'flocking', 'emergence', 'self-organisation'],
  datasets: [],
  params: dict({
    count: int(10, 400, { default: 200, label: 'boids' }),
    separation: real(0, 4, { default: 1.5 }),
    alignment: real(0, 4, { default: 1 }),
    cohesion: real(0, 4, { default: 1 }),
    radius: real(2, 30, { default: 10, label: 'sight radius' }),
    maxSpeed: real(0.5, 4, { default: 2, label: 'top speed' }),
    noise: real(0, 0.5, { default: 0.05 }),
    steps: int(100, 2000, { log: true, default: 800, label: 'time steps' }),
  }),
  run: ({ params, key }) =>
    flocking({
      count: num(params['count'], 200),
      separation: num(params['separation'], 1.5),
      alignment: num(params['alignment'], 1),
      cohesion: num(params['cohesion'], 1),
      radius: num(params['radius'], 10),
      maxSpeed: num(params['maxSpeed'], 2),
      minSpeed: Math.min(0.5, num(params['maxSpeed'], 2)),
      noise: num(params['noise'], 0.05),
      steps: num(params['steps'], 800),
      every: 2,
      key,
    }),
  views: [{ record: 'agentSteps', title: 'The flock, coloured by heading, and its polarisation' }],
  methods: [
    { file: 'methods/complex-systems/flocking.ts', name: 'flocking' },
    { file: 'lib/dynamics/flocking.ts', name: 'boidsAlgorithm' },
  ],
  references: ['Reynolds (1987), Flocks, herds and schools: a distributed behavioral model'],
})
