/**
 * Hub entry for CMA-ES, an evolution strategy whose Gaussian search distribution adapts its covariance to the
 * objective's shape, on test functions of two variables.
 */

import { cmaesOnPlane, type PlaneFunctionName } from 'mdml-engine/methods/evolutionary'
import { dict, int, real } from 'mdml-engine/space'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'
import { objectiveControl } from './_plane.ts'

/** CMA-ES: a Gaussian search distribution whose covariance learns the objective's shape. */
export const cmaEsEntry: HubEntry = defineEntry({
  id: 'evolutionary/cma-es',
  title: 'CMA-ES',
  summary: 'Sample, rank, and move the mean and reshape the covariance towards the best samples: the ellipse learns.',
  collection: 'evolutionary',
  tags: ['evolution strategy', 'covariance matrix adaptation', 'continuous optimisation'],
  datasets: [],
  params: dict({
    objective: objectiveControl('rosenbrock'),
    popSize: int(4, 40, { default: 8, label: 'offspring per generation' }),
    sigma0: real(0.05, 3, { default: 0.8, label: 'initial step size' }),
    generations: int(5, 150, { default: 60, label: 'generations' }),
  }),
  run: ({ params, key }) =>
    cmaesOnPlane(params['objective'] as PlaneFunctionName, {
      popSize: Number(params['popSize']),
      sigma0: Number(params['sigma0']),
      generations: Number(params['generations']),
      key,
    }),
  views: [{ record: 'populationSteps', title: 'Samples and the search distribution' }],
  methods: [
    { file: 'methods/evolutionary/evolution-strategies.ts', name: 'cmaesOnPlane' },
    { file: 'lib/optim/minimize/cmaes.ts', name: 'cmaes' },
  ],
  references: ['Hansen and Ostermeier (2001), Completely derandomized self-adaptation in evolution strategies.'],
})
