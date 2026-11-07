/**
 * The slot of the restarted Krylov eigensolvers that varies as an option table: which end of the spectrum is wanted
 * (largest magnitude, largest or smallest algebraic), with the knobs of the solve (tolerance, Krylov block size, restart
 * cap). The table is one `union` space for controls and grids, and builds the options of `eigsh` and `eigshAlgorithm`.
 */

import { dict, fillKnobs, int, type Option, type OptionTable, optionTable, real } from '../../space/index.ts'
import type { EigshOptions, Which } from './eigen.ts'

/** What a target of the spectrum means, for validity rules and selectors. */
export interface SpectrumTargetProperties {
  /** The scipy `which` code. */
  readonly which: Which
}

function target(description: string, which: Which): Option<EigshOptions, SpectrumTargetProperties> {
  const space = dict({
    tol: real(1e-14, 1e-2, { default: 1e-10, log: true, label: 'tolerance' }),
    ncv: int(2, 200, { default: 20, label: 'Krylov block size' }),
    maxRestarts: int(1, 1000, { default: 300, label: 'restart cap' }),
  })
  return {
    description,
    properties: { which },
    space,
    make: (config) => {
      const full = fillKnobs(space, config)
      return {
        which,
        tol: full['tol'] as number,
        ncv: full['ncv'] as number,
        maxRestarts: full['maxRestarts'] as number,
      }
    },
  }
}

/**
 * The ends of the spectrum as one slot: `largest-magnitude` (scipy's LM, the default), `largest` (LA) and `smallest`
 * (SA), each with the tolerance, Krylov block size and restart cap. A table's `make` gives the options of `eigsh`.
 *
 * @concept lanczos
 * @example
 * ```ts
 * import { spectrumTargets } from 'mdml-engine/numerics/sparse'
 * spectrumTargets.make({ kind: 'smallest' }).which // => 'SA'
 * ```
 */
export const spectrumTargets: OptionTable<EigshOptions, SpectrumTargetProperties> = optionTable(
  {
    'largest-magnitude': target('the eigenvalues of largest absolute value', 'LM'),
    largest: target('the algebraically largest eigenvalues', 'LA'),
    smallest: target('the algebraically smallest eigenvalues', 'SA'),
  },
  { default: 'largest-magnitude', label: 'spectrum end', description: 'Which eigenvalues the Krylov solver finds.' },
)
