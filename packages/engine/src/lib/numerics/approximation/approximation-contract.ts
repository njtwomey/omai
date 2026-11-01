/**
 * What the approximation algorithms log: the Remez exchange (the reference points, the levelled error, the number of
 * alternations found) and the adaptive Chebyshev approximation (the sample size and the coefficients' tail).
 */

import { type Contract, defineContract, scalar, tensor } from '../../trace/index.ts'

/** The paths the approximation algorithms log, as a type. */
export type ApproximationPath =
  | 'remez/reference'
  | 'remez/error'
  | 'remez/maxError'
  | 'remez/alternation'
  | 'chebfun/degree'
  | 'chebfun/tail'
  | 'chebfun/coefficients'

/** What the Remez exchange and the adaptive Chebyshev approximation log, by path. */
export const approximationContract: Contract<unknown, ApproximationPath> = defineContract({
  'remez/reference': tensor({ role: 'state', shape: ['reference'], description: 'the reference points on [a, b]' }),
  'remez/error': scalar({ description: 'the levelled error E of the solve on the reference' }),
  'remez/maxError': scalar({ description: 'the largest error of the approximation on the grid', direction: 'lower' }),
  'remez/alternation': scalar({
    description: 'the alternating extrema of the error found (the reference needs m + n + 2)',
  }),
  'chebfun/degree': scalar({ description: 'the degree of the interpolant sampled at this step' }),
  'chebfun/tail': scalar({
    description: 'the largest of the last coefficients relative to the largest',
    direction: 'lower',
  }),
  'chebfun/coefficients': tensor({ role: 'state', shape: ['degree'], description: 'the Chebyshev coefficients' }),
})
