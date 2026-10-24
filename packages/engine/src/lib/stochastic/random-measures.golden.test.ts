import { goldenSuite } from '../../testing/suite.ts'
import { expectedDishes, expectedTables, partitionLogProbability } from './random-measures.ts'

goldenSuite('stochastic/random-measures against scipy.special', 'stochastic/random-measures', {
  tables: (_i, p) => expectedTables(p['alpha'] as number, p['n'] as number, { discount: p['d'] as number }),
  eppf: (_i, p) =>
    partitionLogProbability(p['sizes'] as number[], p['alpha'] as number, { discount: p['d'] as number }),
  dishes: (_i, p) => expectedDishes(p['alpha'] as number, p['n'] as number),
})
