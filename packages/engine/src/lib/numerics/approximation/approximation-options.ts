/**
 * The slots of the approximation algorithms as option tables: the type of the minimax approximation the Remez
 * exchange computes (a polynomial, or a rational function with a denominator degree) and how finely the adaptive
 * Chebyshev approximation insists the coefficients decay. Each table is one `union` space for controls and grids.
 */

import { dict, int, type OptionTable, optionTable, plainOption, real } from '../../space/index.ts'
import type { RemezOptions } from './best.ts'
import type { ChebfunOptions } from './chebyshev.ts'

/**
 * The types of minimax approximation as one slot: `polynomial` (denominator degree 0) and `rational` (a denominator of
 * degree n, found by iterating the linearised exchange), with the tolerance and exchange cap. A table's `make` gives
 * the `RemezOptions`.
 *
 * @concept minimax-approximation
 * @example
 * ```ts
 * import { minimaxTypes } from 'omai-engine/numerics/approximation'
 * minimaxTypes.make({ kind: 'rational', n: 2 }).n // => 2
 * ```
 */
export const minimaxTypes: OptionTable<RemezOptions, Record<string, never>> = optionTable(
  {
    polynomial: plainOption(
      'the best polynomial of the numerator degree',
      dict({ tol: real(1e-14, 1e-3, { default: 1e-10, log: true, label: 'tolerance' }) }),
      (k) => ({ n: 0, tol: k('tol') }),
    ),
    rational: plainOption(
      'the best rational function: numerator degree m over a denominator of degree n',
      dict({
        n: int(1, 10, { default: 1, label: 'denominator degree' }),
        tol: real(1e-14, 1e-3, { default: 1e-10, log: true, label: 'tolerance' }),
      }),
      (k) => ({ n: k('n'), tol: k('tol') }),
    ),
  },
  { default: 'polynomial', label: 'minimax type', description: 'What the Remez exchange approximates by.' },
)

/**
 * How finely the adaptive Chebyshev approximation resolves a function, as one slot: `standard` (ten machine epsilons,
 * as chebfun), `loose` (1e-8, fewer points) and `strict` (one machine epsilon, may fail on functions with noise).
 *
 * @concept chebyshev-approximation
 * @example
 * ```ts
 * import { chebyshevResolutions } from 'omai-engine/numerics/approximation'
 * chebyshevResolutions.make({ kind: 'loose' }).tol // => 1e-8
 * ```
 */
export const chebyshevResolutions: OptionTable<ChebfunOptions, Record<string, never>> = optionTable(
  {
    standard: plainOption(
      'ten machine epsilons',
      dict({ maxPoints: int(17, 1048577, { default: 65537, label: 'largest sample' }) }),
      (k) => ({
        tol: 10 * 2.220446049250313e-16,
        maxPoints: k('maxPoints'),
      }),
    ),
    loose: plainOption(
      '1e-8, a few terms for a plot',
      dict({ maxPoints: int(17, 1048577, { default: 65537, label: 'largest sample' }) }),
      (k) => ({
        tol: 1e-8,
        maxPoints: k('maxPoints'),
      }),
    ),
    strict: plainOption(
      'one machine epsilon',
      dict({ maxPoints: int(17, 1048577, { default: 65537, label: 'largest sample' }) }),
      (k) => ({
        tol: 2.220446049250313e-16,
        maxPoints: k('maxPoints'),
      }),
    ),
  },
  { default: 'standard', label: 'resolution', description: 'How small the Chebyshev coefficients must get.' },
)
