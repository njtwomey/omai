/**
 * Convolutional dictionary learning as a method a session can drive: `convolutionalDictionaryMethod` composes a
 * synthetic pulse train (the `pulseTrain` generator, whose truth is the planted pulses) with
 * `convolutionalDictionaryLearningAlgorithm` of `optim`, and adds to its contract the planted filters and how well the
 * learnt ones match them (up to shift and sign). Everything a viewer may choose, from the number and length of the
 * pulses to the penalty and the filters to learn, is a knob of the method's configuration.
 */

import { generate } from '../data/index.ts'
import { pulseTrain } from '../datasets/index.ts'
import { curveSet } from '../exposition/index.ts'
import {
  type ConvolutionalDictionaryOptions,
  type ConvolutionalDictionaryState,
  type ConvolutionalPath,
  convolutionalDictionaryContract,
  convolutionalDictionaryLearningAlgorithm,
  matchFilters,
} from '../optim/index.ts'
import { foldIn, type Key } from '../random/index.ts'
import { choice, dict, int, real, type SpaceValue } from '../space/index.ts'
import { type Contract, defineContract, defineMethod, derived, Log, type Method } from '../trace/index.ts'

/** The configuration of {@link convolutionalDictionaryMethod}. */
export interface ConvolutionalDictionaryConfig {
  readonly [knob: string]: SpaceValue
  /** The length of the signal. */
  readonly n: number
  /** The number of different pulses planted (1 to 4). */
  readonly pulses: number
  /** The length of a planted pulse. */
  readonly pulseLength: number
  /** How many times each pulse occurs. */
  readonly occurrences: number
  /** The standard deviation of the noise. */
  readonly noise: number
  /** The number of filters to learn. */
  readonly filters: number
  /** The length of a learnt filter (a few samples longer than the pulses holds them at any shift). */
  readonly length: number
  /** The penalty on the activations. */
  readonly lambda: number
  /** Proximal gradient steps of coding per round. */
  readonly codeSteps: number
  /** Projected FISTA steps of the filter update per round. */
  readonly filterSteps: number
  /** Where the initial filters come from: `energy` (the most energetic stretches) or `random` (from the key). */
  readonly initialisation: 'energy' | 'random'
}

/** The paths of {@link convolutionalDictionaryMethod}: the learner's, the planted filters and the recovery. */
export type ConvolutionalMethodPath = ConvolutionalPath | 'cdl/planted' | 'cdl/recovery'

/** The signal, its truth and the learner's options a configuration and a key make. */
export interface ConvolutionalDictionaryProblem {
  /** The noisy pulse train, `[n]`. */
  readonly signal: Float64Array
  /** The planted filters, one row of `pulseLength` values per pulse. */
  readonly planted: readonly (readonly number[])[]
  /** The planted activation maps, one row of `n - pulseLength + 1` values per pulse. */
  readonly plantedCodes: readonly (readonly number[])[]
  readonly options: ConvolutionalDictionaryOptions & { readonly initialisation: 'energy' | 'random' }
}

/**
 * The problem a configuration describes: the pulse train the generator makes from the key, its truth, and the options
 * of the learner. The same key makes the same problem.
 *
 * @param config - The method's configuration.
 * @param key - The key.
 * @returns The signal, the planted filters and maps, and the learner's options.
 * @example
 * ```ts
 * import { convolutionalDictionaryProblem } from 'omai-engine/decomposition'
 * import { key } from 'omai-engine/random'
 * const p = convolutionalDictionaryProblem(
 *   { n: 120, pulses: 2, pulseLength: 8, occurrences: 3, noise: 0.01, filters: 2, length: 10, lambda: 0.1, codeSteps: 10, filterSteps: 10, initialisation: 'energy' },
 *   key(0),
 * )
 * [p.signal.length, p.planted.length, p.planted[0]?.length] // => [120, 2, 8]
 * ```
 */
export function convolutionalDictionaryProblem(
  config: ConvolutionalDictionaryConfig,
  key: Key,
): ConvolutionalDictionaryProblem {
  const g = generate(pulseTrain, {
    key: foldIn(key, 0),
    knobs: {
      n: config.n,
      pulses: config.pulses,
      length: config.pulseLength,
      occurrences: config.occurrences,
      noise: config.noise,
    },
  })
  const extra = g.truth.extra as { filters: number[][]; codes: number[][] }
  return {
    signal: Float64Array.from(g.data.values.data),
    planted: extra.filters,
    plantedCodes: extra.codes,
    options: {
      filters: config.filters,
      length: config.length,
      lambda: config.lambda,
      codeSteps: config.codeSteps,
      filterSteps: config.filterSteps,
      initialisation: config.initialisation,
    },
  }
}

/** The learner's contract with the planted filters and the recovery of them added. */
function methodContract(
  problem: ConvolutionalDictionaryProblem,
): Contract<ConvolutionalDictionaryState, ConvolutionalMethodPath> {
  const { signal, planted, options } = problem
  const L = options.length
  return defineContract({
    ...convolutionalDictionaryContract(signal, options),
    'cdl/planted': derived<ConvolutionalDictionaryState>({
      description: 'the planted pulses against the tap, one curve each',
      compute: () =>
        curveSet({
          x: Array.from({ length: planted[0]?.length ?? 0 }, (_, l) => l),
          xLabel: 'tap',
          series: planted.map((p, k) => ({ name: `pulse ${k + 1}`, y: [...p] })),
        }),
    }),
    'cdl/recovery': derived<ConvolutionalDictionaryState>({
      description: 'the mean over the planted pulses of the best match among the learnt filters, 1 when found exactly',
      compute: (s) => {
        const scores = matchFilters(
          Array.from({ length: options.filters }, (_, k) => Array.from(s.filters.subarray(k * L, (k + 1) * L))),
          planted,
        ).map((m) => m.score)
        return scores.reduce((a, b) => a + b, 0) / Math.max(scores.length, 1)
      },
    }),
  })
}

/**
 * Learning filters from a pulse train as a method: choose how many pulses are planted, their length, the noise, and the
 * number, length and penalty of the filters to learn, and a session runs the rounds, logging the objective, its fit
 * and the number of activations, and offering the filters, the maps, the reconstruction and the planted pulses at any
 * step, with `cdl/recovery`, the match of the learnt filters with the planted ones.
 *
 * @concept convolutional-sparse-coding
 * @index estimator Learn filters from a planted pulse train as a session, with the recovery.
 * @example
 * ```ts
 * import { convolutionalDictionaryMethod } from 'omai-engine/decomposition'
 * import { Log, metricSeries, session } from 'omai-engine/trace'
 * const s = session(convolutionalDictionaryMethod, {
 *   config: { pulses: 1, filters: 1, length: 8, codeSteps: 10, filterSteps: 20 },
 *   logs: { 'cdl/objective': Log() },
 *   mode: 'eager',
 *   steps: 5,
 * })
 * metricSeries(s.run(), 'cdl/objective').values.length // => 5
 * ```
 */
export const convolutionalDictionaryMethod: Method<
  ConvolutionalDictionaryState,
  ConvolutionalDictionaryConfig,
  ConvolutionalMethodPath
> = defineMethod<ConvolutionalDictionaryState, ConvolutionalDictionaryConfig, ConvolutionalMethodPath>({
  name: 'convolutional-dictionary-learning',
  config: dict({
    n: int(60, 4000, { default: 240, label: 'samples' }),
    pulses: int(1, 4, { default: 2, label: 'planted pulses' }),
    pulseLength: int(4, 60, { default: 12, label: 'pulse length' }),
    occurrences: int(1, 40, { default: 5, label: 'copies of each pulse' }),
    noise: real(0, 1, { default: 0.02, label: 'noise' }),
    filters: int(1, 8, { default: 2, label: 'filters to learn' }),
    length: int(4, 80, { default: 16, label: 'filter length' }),
    lambda: real(0.001, 5, { log: true, default: 0.1, label: 'penalty λ' }),
    codeSteps: int(1, 200, { default: 25, label: 'coding steps per round' }),
    filterSteps: int(1, 400, { default: 60, label: 'filter steps per round' }),
    initialisation: choice(['energy', 'random'], { default: 'energy', label: 'initial filters' }),
  }),
  make: (config, key) => {
    const p = convolutionalDictionaryProblem(config, key)
    return convolutionalDictionaryLearningAlgorithm(p.signal, p.options)
  },
  contract: (config, key) => methodContract(convolutionalDictionaryProblem(config, key)),
  defaults: { 'cdl/objective': Log(), 'cdl/fit': Log(), 'cdl/nonzeros': Log() },
})
