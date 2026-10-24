import { goldenSuite } from '../../testing/suite.ts'
import { tensor, type Value } from '../tensor/index.ts'
import {
  absorption,
  classifyStates,
  distributionAfter,
  expectedHittingTimes,
  hittingProbabilities,
  mixingTime,
  spectralGap,
  stationaryDistribution,
  timeReversal,
} from './markov-chains.ts'

goldenSuite('stochastic/markov-chains against numpy, scipy and networkx', 'stochastic/markov-chains', {
  stationary: ([p]) => stationaryDistribution(p),
  after: ([p, p0], c) => distributionAfter(p, p0, c['steps'] as number),
  moduli: ([p]) => tensor(spectralGap(p).moduli as number[]),
  reversal: ([p]) => timeReversal(p),
  mixing: ([p], c) => mixingTime(p, { epsilon: c['epsilon'] as number }).time,
  gap: ([p]) => spectralGap(p).gap,
  'hitting-times': ([p], c) => expectedHittingTimes(p, c['target'] as number[]),
  'hitting-probabilities': ([p], c) => hittingProbabilities(p, c['target'] as number[]),
  absorption: ([p], c) => absorption(p)[c['part'] as 'fundamental' | 'probabilities' | 'expectedSteps'] as Value,
  'class-of': ([p]) => tensor(Array.from(classifyStates(p).classOf)),
  recurrent: ([p]) => tensor(classifyStates(p).recurrent.map((r) => (r ? 1 : 0))),
})
