/**
 * The slots of warm-up as option tables: the form of the mass matrix Stan-style window adaptation learns (a diagonal or
 * a dense one) with its knobs (target acceptance, the buffer and window sizes, the shrinkage of the estimate, dual
 * averaging), and the knobs of random-walk adaptation (target acceptance, whether to learn the proposal's shape).
 * Each table is one `union` space for controls and grids, and builds the options of the warm-up algorithm.
 */

import {
  bool,
  defaults,
  dict,
  int,
  type Option,
  type OptionTable,
  optionTable,
  real,
  type SpaceValue,
} from '../space/index.ts'
import type { WindowAdaptationOptions } from './hamiltonian/adaptation.ts'
import type { RandomWalkAdaptationOptions } from './metropolis.ts'

type Knobs = (name: string) => number | boolean

function option<T>(
  description: string,
  space: ReturnType<typeof dict>,
  build: (k: Knobs) => T,
): Option<T, Record<string, never>> {
  return {
    description,
    properties: {},
    space,
    make: (config) => {
      const full = { ...(defaults(space) as Readonly<Record<string, SpaceValue>>), ...config }
      return build((name) => full[name] as number | boolean)
    },
  }
}

const windowKnobs = (): ReturnType<typeof dict> =>
  dict({
    targetAcceptance: real(0.5, 0.99, { default: 0.8, label: 'target acceptance' }),
    initialBuffer: int(0, 500, { default: 75, label: 'first fast buffer' }),
    firstWindow: int(1, 500, { default: 25, label: 'first slow window' }),
    finalBuffer: int(0, 500, { default: 50, label: 'last fast buffer' }),
    regularisation: real(1e-6, 1, { default: 1e-3, log: true, label: 'shrinkage towards the identity' }),
    findStepSize: bool({ default: true, label: 'search for a first step size' }),
    t0: real(1, 100, { default: 10, label: 'dual averaging offset t0' }),
    gamma: real(0.01, 1, { default: 0.05, log: true, label: 'dual averaging shrinkage' }),
    kappa: real(0.51, 1, { default: 0.75, label: 'dual averaging exponent' }),
  })

const windowOptions = (k: Knobs, dense: boolean): WindowAdaptationOptions => ({
  dense,
  targetAcceptance: k('targetAcceptance') as number,
  initialBuffer: k('initialBuffer') as number,
  firstWindow: k('firstWindow') as number,
  finalBuffer: k('finalBuffer') as number,
  regularisation: k('regularisation') as number,
  findStepSize: k('findStepSize') as boolean,
  t0: k('t0') as number,
  gamma: k('gamma') as number,
  kappa: k('kappa') as number,
})

/**
 * The forms of window adaptation as one slot: `diagonal` (a per-coordinate scale, the Stan default) and `dense` (the
 * full covariance), each with the schedule, the shrinkage and dual-averaging knobs. A table's `make` gives the options
 * of `windowAdaptation` and `windowAdaptationAlgorithm`.
 *
 * @concept window-adaptation
 * @example
 * ```ts
 * import { windowAdaptationForms } from 'mdml-engine/sampling'
 * windowAdaptationForms.make({ kind: 'dense', targetAcceptance: 0.9 }).dense // => true
 * ```
 */
export const windowAdaptationForms: OptionTable<WindowAdaptationOptions, Record<string, never>> = optionTable(
  {
    diagonal: option('estimate one variance per coordinate', windowKnobs(), (k) => windowOptions(k, false)),
    dense: option('estimate the full covariance', windowKnobs(), (k) => windowOptions(k, true)),
  },
  {
    default: 'diagonal',
    label: 'mass matrix',
    description: 'What window adaptation learns about the posterior scale.',
  },
)

/**
 * The knobs of random-walk adaptation as one slot: `scale-only` (tune the proposal scale towards the target acceptance)
 * and `scale-and-shape` (also learn the proposal's shape from the covariance of the second half of warm-up).
 *
 * @concept adaptive-metropolis
 * @example
 * ```ts
 * import { randomWalkAdaptations } from 'mdml-engine/sampling'
 * randomWalkAdaptations.make({ kind: 'scale-only' }).adaptCovariance // => false
 * ```
 */
export const randomWalkAdaptations: OptionTable<RandomWalkAdaptationOptions, Record<string, never>> = optionTable(
  {
    'scale-only': option<RandomWalkAdaptationOptions>(
      'Robbins-Monro on the log proposal scale only',
      dict({ targetAcceptance: real(0.05, 0.9, { default: 0.234, label: 'target acceptance' }) }),
      (k) => ({ targetAcceptance: k('targetAcceptance') as number, adaptCovariance: false }),
    ),
    'scale-and-shape': option<RandomWalkAdaptationOptions>(
      'also learn the proposal covariance from the second half of warm-up',
      dict({ targetAcceptance: real(0.05, 0.9, { default: 0.234, label: 'target acceptance' }) }),
      (k) => ({ targetAcceptance: k('targetAcceptance') as number, adaptCovariance: true }),
    ),
  },
  { default: 'scale-and-shape', label: 'proposal adaptation', description: 'What random-walk adaptation tunes.' },
)
