/**
 * Hub entry for symbolic regression by genetic programming, evolving expression trees by subtree crossover and mutation
 * until a formula fits the points.
 */

import { type SymbolicTarget, symbolicRegression } from 'mdml-engine/methods/evolutionary'
import { bool, choice, dict, int, real } from 'mdml-engine/space'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'

/** Symbolic regression by genetic programming. */
export const geneticProgrammingEntry: HubEntry = defineEntry({
  id: 'evolutionary/genetic-programming',
  title: 'Genetic programming',
  summary: 'Evolve expression trees by subtree crossover and mutation until a formula fits the points.',
  collection: 'evolutionary',
  tags: ['genetic programming', 'symbolic regression', 'expression trees'],
  datasets: [],
  params: dict({
    target: choice(['quartic', 'cubic', 'sine', 'rational'], { default: 'quartic', label: 'target' }),
    trig: bool({ default: false, label: 'sine and cosine' }),
    populationSize: int(20, 400, { default: 200, label: 'population' }),
    generations: int(5, 100, { default: 40, label: 'generations' }),
    parsimony: real(0, 0.02, { default: 0.001, label: 'size penalty' }),
  }),
  run: ({ params, key }) =>
    symbolicRegression({
      target: params['target'] as SymbolicTarget,
      trig: params['trig'] === true,
      populationSize: Number(params['populationSize']),
      generations: Number(params['generations']),
      parsimony: Number(params['parsimony']),
      key,
    }),
  views: [{ record: 'populationSteps', title: 'The best program of each generation' }],
  methods: [
    { file: 'methods/evolutionary/genetic-programming.ts', name: 'symbolicRegression' },
    { file: 'lib/search/evolutionary/programming.ts', name: 'geneticProgramming' },
  ],
  references: ['Koza (1992), Genetic Programming: On the Programming of Computers by Means of Natural Selection.'],
})
