/**
 * The temperature ladder of annealed importance sampling as one slot: an option table over a power ladder (the default,
 * dense near the prior), a linear ladder and a log-spaced ladder, each with the knobs of its own. Each entry is one
 * `union` space for controls, grids and tuners and builds the increasing list of temperatures from 0 to 1 that
 * `aisAlgorithm` takes.
 */

import { dict, int, type OptionTable, optionTable, real } from '../space/index.ts'

/** What a ladder is like, for validity rules and selectors. */
export interface LadderProperties {
  /** Whether the rungs are denser near the prior (temperature 0) than near the target. */
  readonly denseNearPrior: boolean
}

/**
 * The temperature ladders as one slot: `power` (lambda_t = (t / T)^p: the default is p = 4), `linear` (evenly spaced)
 * and `log-spaced` (0, then a geometric ladder from a small first temperature up to 1). More rungs make the weights
 * steadier; the spacing decides where along the path the effort goes.
 *
 * @concept annealed-importance-sampling
 * @example
 * ```ts
 * import { annealingLadders } from 'omai-engine/sampling'
 * annealingLadders.make({ kind: 'linear', rungs: 4 }) // => [0, 0.25, 0.5, 0.75, 1]
 * ```
 */
export const annealingLadders: OptionTable<readonly number[], LadderProperties> = optionTable(
  {
    power: {
      description: 'lambda_t = (t / T)^p: small steps at first, where the tempered target changes fastest',
      properties: { denseNearPrior: true },
      space: dict({
        rungs: int(1, 10_000, { log: true, default: 50, label: 'rungs T' }),
        power: real(1, 10, { default: 4, label: 'exponent p' }),
      }),
      make: (c) => {
        const rungs = (c['rungs'] as number | undefined) ?? 50
        const power = (c['power'] as number | undefined) ?? 4
        return Array.from({ length: rungs + 1 }, (_, i) => (i / rungs) ** power)
      },
    },
    linear: {
      description: 'evenly spaced temperatures',
      properties: { denseNearPrior: false },
      space: dict({ rungs: int(1, 10_000, { log: true, default: 50, label: 'rungs T' }) }),
      make: (c) => {
        const rungs = (c['rungs'] as number | undefined) ?? 50
        return Array.from({ length: rungs + 1 }, (_, i) => i / rungs)
      },
    },
    'log-spaced': {
      description: '0, then a geometric ladder from a small first temperature to 1',
      properties: { denseNearPrior: true },
      space: dict({
        rungs: int(2, 10_000, { log: true, default: 50, label: 'rungs T' }),
        first: real(1e-8, 0.5, { log: true, default: 1e-4, label: 'first nonzero temperature' }),
      }),
      make: (c) => {
        const rungs = (c['rungs'] as number | undefined) ?? 50
        const first = (c['first'] as number | undefined) ?? 1e-4
        return [0, ...Array.from({ length: rungs }, (_, i) => first ** (1 - i / (rungs - 1)))]
      },
    },
  },
  {
    default: 'power',
    label: 'temperature ladder',
    description: 'How the temperature rises from the prior (0) to the posterior (1).',
  },
)
