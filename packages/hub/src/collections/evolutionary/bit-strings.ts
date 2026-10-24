import { evolveBitStrings, knapsack, oneMax } from 'whitebox-engine/methods/evolutionary'
import { choice, dict, int, real } from 'whitebox-engine/space'
import { defineEntry, type HubEntry } from '../../entry/entry.ts'

/** A genetic algorithm on bit strings: OneMax and the knapsack. */
export const bitStringGa: HubEntry = defineEntry({
  id: 'evolutionary/bit-strings',
  title: 'Genetic algorithm on bit strings',
  summary: 'Selection, crossover and bit-flip mutation on OneMax or a knapsack: watch the population fill with ones.',
  collection: 'evolutionary',
  tags: ['genetic algorithm', 'selection', 'crossover', 'mutation', 'OneMax', 'knapsack'],
  datasets: [],
  params: dict({
    problem: choice(['OneMax', 'knapsack'], { default: 'OneMax', label: 'problem' }),
    length: int(8, 64, { default: 32, label: 'bits (items)' }),
    populationSize: int(4, 80, { default: 30, label: 'population' }),
    generations: int(5, 200, { default: 60, label: 'generations' }),
    selection: choice(['tournament', 'roulette', 'rank', 'universal'], { default: 'tournament', label: 'selection' }),
    crossover: choice(['one-point', 'two-point', 'uniform'], { default: 'one-point', label: 'crossover' }),
    mutation: real(0, 0.2, { default: 0.03, label: 'bit-flip probability' }),
  }),
  run: ({ params, key }) => {
    const n = Number(params['length'])
    const problem = params['problem'] === 'knapsack' ? knapsack(key, n) : oneMax(n)
    return evolveBitStrings(problem, {
      populationSize: Number(params['populationSize']),
      generations: Number(params['generations']),
      selection: params['selection'] as 'tournament',
      crossover: params['crossover'] as 'one-point',
      bitFlipRate: Number(params['mutation']),
      key,
    })
  },
  views: [
    { record: 'boardSteps', title: 'The population, fittest first, generation by generation' },
    { record: 'populationSteps', title: 'Fitness of every genome', area: 'below' },
  ],
  methods: [
    { file: 'methods/evolutionary/genetic-algorithms.ts', name: 'evolveBitStrings' },
    { file: 'lib/search/evolutionary/genetic.ts', name: 'geneticAlgorithm' },
  ],
  references: ['Holland (1975), Adaptation in Natural and Artificial Systems.', 'Goldberg (1989), Genetic Algorithms.'],
})
