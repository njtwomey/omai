/**
 * The slot of the shooting methods that varies as an option table: the integrator and its tolerances (precise, standard,
 * fast), with the Newton iteration cap and residual tolerance as knobs. The table is one `union` space for controls and
 * grids, and builds the options of `shooting`, `multipleShooting` and their algorithms.
 */

import { dict, fillKnobs, int, type Option, type OptionTable, optionTable, real } from '../../space/index.ts'
import type { ShootingOptions } from './boundary.ts'

function accuracy(
  description: string,
  method: 'DOP853' | 'RK45',
  rtol: number,
  atol: number,
): Option<ShootingOptions, Record<string, never>> {
  const space = dict({
    maxIterations: int(1, 200, { default: 50, label: 'Newton iterations' }),
    ftol: real(1e-14, 1e-2, { default: 1e-10, log: true, label: 'residual tolerance' }),
  })
  return {
    description,
    properties: {},
    space,
    make: (config) => {
      const full = fillKnobs(space, config)
      return { method, rtol, atol, maxIterations: full['maxIterations'] as number, ftol: full['ftol'] as number }
    },
  }
}

/**
 * The accuracy of the integration inside a shooting method, as one slot: `precise` (DOP853 at 1e-10, the default),
 * `standard` (RK45 at 1e-6) and `fast` (RK45 at 1e-3, for a first look), each with the Newton iteration cap and the
 * residual tolerance as knobs.
 *
 * @concept shooting-method
 * @example
 * ```ts
 * import { shootingAccuracies } from 'mdml-engine/dynamics/ode'
 * shootingAccuracies.make({ kind: 'standard' }).method // => 'RK45'
 * ```
 */
export const shootingAccuracies: OptionTable<ShootingOptions, Record<string, never>> = optionTable(
  {
    precise: accuracy('DOP853 at 1e-10', 'DOP853', 1e-10, 1e-12),
    standard: accuracy('RK45 at 1e-6', 'RK45', 1e-6, 1e-9),
    fast: accuracy('RK45 at 1e-3', 'RK45', 1e-3, 1e-6),
  },
  { default: 'precise', label: 'integration accuracy', description: 'How precisely each shot is integrated.' },
)
