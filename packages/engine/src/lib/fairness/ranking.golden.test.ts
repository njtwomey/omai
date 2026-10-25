import { goldenSuite } from '../../testing/suite.ts'
import { tensor } from '../tensor/index.ts'
import { fairMinimumTable } from './ranking.ts'

goldenSuite("FA*IR's minimum table against scipy", 'fairness/ranking', {
  mtable: (_, o) => tensor(fairMinimumTable(o['k'] as number, o['p'] as number, o['alpha'] as number)),
})
