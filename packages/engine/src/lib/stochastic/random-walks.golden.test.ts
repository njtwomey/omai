import { goldenSuite } from '../../testing/suite.ts'
import { graph } from '../graph/index.ts'
import { tensor } from '../tensor/index.ts'
import { countSelfAvoidingWalks, gamblersRuin, graphWalkStationary, simpleWalkPmf } from './random-walks.ts'

goldenSuite('stochastic/random-walks against scipy, numpy, OEIS and networkx', 'stochastic/random-walks', {
  'walk-pmf': ([k], p) => simpleWalkPmf(k, p['steps'] as number, p['p'] as number),
  'ruin-win': (_i, p) => gamblersRuin(p['i'] as number, p['n'] as number, p['p'] as number).winProbability,
  'ruin-duration': (_i, p) => gamblersRuin(p['i'] as number, p['n'] as number, p['p'] as number).expectedDuration,
  'saw-count': (_i, p) =>
    tensor(Array.from({ length: (p['n'] as number) + 1 }, (_, n) => countSelfAvoidingWalks(n, p['dim'] as number))),
  'graph-stationary': (_i, p) => graphWalkStationary(graph(p['edges'] as [number, number, number][])),
})
