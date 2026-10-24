import { goldenSuite } from '../../testing/suite.ts'
import { blackScholes } from './calculus.ts'

goldenSuite('stochastic/calculus against the closed-form Black-Scholes price', 'stochastic/calculus', {
  'black-scholes': (_inputs, p) =>
    blackScholes(
      p['spot'] as number,
      p['strike'] as number,
      p['maturity'] as number,
      p['rate'] as number,
      p['sigma'] as number,
      { kind: p['kind'] as 'call' | 'put', dividend: p['dividend'] as number },
    ),
})
