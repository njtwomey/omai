/**
 * Hub entry for a real-coded genetic algorithm with simulated binary crossover, on test functions of two variables.
 */

import { evolveOnPlane, type PlaneFunctionName } from 'mdml-engine/methods/evolutionary'
import { choice, dict, int } from 'mdml-engine/space'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'
import { objectiveControl } from './_plane.ts'

/** A real-coded genetic algorithm on a function of two variables. */
export const realVectorGa: HubEntry = defineEntry({
  id: 'evolutionary/real-vectors',
  title: 'Real-coded genetic algorithm',
  summary: 'A population of points bred by simulated binary crossover and mutation, settling into the minima.',
  collection: 'evolutionary',
  tags: ['genetic algorithm', 'continuous optimisation', 'SBX'],
  datasets: [],
  params: dict({
    objective: objectiveControl('himmelblau'),
    populationSize: int(6, 100, { default: 40, label: 'population' }),
    generations: int(5, 100, { default: 40, label: 'generations' }),
    crossover: choice(['sbx', 'blend', 'arithmetic'], { default: 'sbx', label: 'crossover' }),
    mutation: choice(['gaussian', 'polynomial'], { default: 'gaussian', label: 'mutation' }),
  }),
  run: ({ params, key }) =>
    evolveOnPlane(params['objective'] as PlaneFunctionName, {
      populationSize: Number(params['populationSize']),
      generations: Number(params['generations']),
      crossover: params['crossover'] as 'sbx',
      mutation: params['mutation'] as 'gaussian',
      key,
    }),
  views: [{ record: 'populationSteps', title: 'The population over the objective' }],
  methods: [
    { file: 'methods/evolutionary/genetic-algorithms.ts', name: 'evolveOnPlane' },
    { file: 'lib/search/evolutionary/genetic.ts', name: 'realVectors' },
  ],
  references: ['Deb and Agrawal (1995), Simulated binary crossover for continuous search space.'],
})
