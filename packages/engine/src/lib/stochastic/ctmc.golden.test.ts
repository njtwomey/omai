import { goldenSuite } from '../../testing/suite.ts'
import { ctmcHittingTimes, ctmcStationary, ctmcTransition, embeddedChain, uniformise } from './ctmc.ts'

goldenSuite('stochastic/ctmc against scipy.linalg', 'stochastic/ctmc', {
  transition: ([q], p) => ctmcTransition(q, p['t'] as number),
  stationary: ([q]) => ctmcStationary(q),
  hitting: ([q], p) => ctmcHittingTimes(q, p['target'] as number[]),
  embedded: ([q]) => embeddedChain(q).transition,
  uniformise: ([q], p) => uniformise(q, p['rate'] as number).transition,
})
