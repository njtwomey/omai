/**
 * Hub entry for a self-adaptive evolution strategy, parents breeding Gaussian offspring with their own step sizes under
 * comma or plus selection, on test functions of two variables.
 */

import { evolutionStrategyOnPlane, type PlaneFunctionName } from 'whitebox-engine/methods/evolutionary'
import { bool, dict, int, real } from 'whitebox-engine/space'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'
import { objectiveControl } from './_plane.ts'

/** A self-adaptive evolution strategy. */
export const evolutionStrategyEntry: HubEntry = defineEntry({
  id: 'evolutionary/evolution-strategy',
  title: 'Evolution strategy',
  summary: 'Parents breed Gaussian offspring with their own step sizes; the best mu survive, comma or plus.',
  collection: 'evolutionary',
  tags: ['evolution strategy', 'self-adaptation', 'continuous optimisation'],
  datasets: [],
  params: dict({
    objective: objectiveControl('rastrigin'),
    mu: int(1, 20, { default: 5, label: 'parents (mu)' }),
    lambda: int(2, 80, { default: 30, label: 'offspring (lambda)' }),
    plus: bool({ default: false, label: 'plus selection (keep parents)' }),
    sigma: real(0.05, 3, { default: 1, label: 'initial step size' }),
    generations: int(5, 100, { default: 40, label: 'generations' }),
  }),
  run: ({ params, key }) =>
    evolutionStrategyOnPlane(params['objective'] as PlaneFunctionName, {
      mu: Number(params['mu']),
      lambda: Math.max(Number(params['lambda']), Number(params['mu'])),
      plus: params['plus'] === true,
      sigma: Number(params['sigma']),
      generations: Number(params['generations']),
      key,
    }),
  views: [{ record: 'populationSteps', title: 'Offspring and parents over the objective' }],
  methods: [
    { file: 'methods/evolutionary/evolution-strategies.ts', name: 'evolutionStrategyOnPlane' },
    { file: 'lib/search/evolutionary/genetic.ts', name: 'evolutionStrategy' },
  ],
  references: ['Beyer and Schwefel (2002), Evolution strategies: a comprehensive introduction.'],
})
