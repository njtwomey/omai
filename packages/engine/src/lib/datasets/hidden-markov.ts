/**
 * Sequences read off a hidden Markov chain, with the hidden states as the labels: the occasionally dishonest casino of
 * Durbin, Eddy, Krogh and Mitchison (a fair die and a loaded one, swapped in secret) and a series that switches between
 * a calm and an active Gaussian regime. Each is one long sequence, rows are time steps, `x` is the observation and `y`
 * the true state, and the generating model is in the truth, for checking what a hidden-state method recovers.
 */

import { type ClassificationData, type DatasetGenerator, xyClasses } from '../data/index.ts'
import { type HiddenMarkovModel, hiddenMarkovModel, hmmSample } from '../pgm/index.ts'
import { dict, int, real } from '../space/index.ts'
import { labels, matrix, stamp } from './_util.ts'

const lengthKnob = int(10, 100_000, { default: 300, label: 'length', description: 'The number of time steps.' })

/**
 * The occasionally dishonest casino: a fair die and a loaded one that shows a six more often than a sixth of the time;
 * at each roll the casino may swap the dice. `x` holds the face rolled as an index 0 to 5 (the faces 1 to 6) and `y` the
 * die used (0 fair, 1 loaded). The truth holds the generating `HiddenMarkovModel` and the true states.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { dishonestCasino } from 'mdml-engine/datasets'
 * generate(dishonestCasino, { knobs: { length: 50 } }).data.y.shape // => [50]
 * ```
 */
export const dishonestCasino: DatasetGenerator<ClassificationData> = {
  name: 'dishonestCasino',
  kind: 'classification',
  description: 'Rolls of a die the casino secretly swaps between a fair one and one loaded towards six.',
  knobs: dict({
    length: lengthKnob,
    toLoaded: real(0.001, 0.5, {
      default: 0.05,
      log: true,
      label: 'chance of loading the die',
      description: 'The probability of swapping the fair die for the loaded one before a roll.',
    }),
    toFair: real(0.001, 0.5, {
      default: 0.1,
      log: true,
      label: 'chance of restoring the die',
      description: 'The probability of swapping the loaded die for the fair one before a roll.',
    }),
    loadedSix: real(0.17, 0.95, {
      default: 0.5,
      label: 'chance of a six on the loaded die',
      description: 'The probability of a six on the loaded die; the other faces share the rest.',
    }),
  }),
  generate: (k, knobs) => {
    const toLoaded = knobs['toLoaded'] as number
    const toFair = knobs['toFair'] as number
    const six = knobs['loadedSix'] as number
    const model: HiddenMarkovModel = hiddenMarkovModel({
      states: ['fair', 'loaded'],
      initial: [0.5, 0.5],
      transition: [
        [1 - toLoaded, toLoaded],
        [toFair, 1 - toFair],
      ],
      emission: {
        kind: 'categorical',
        symbols: ['1', '2', '3', '4', '5', '6'],
        probabilities: [
          [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6],
          [...Array.from({ length: 5 }, () => (1 - six) / 5), six],
        ],
      },
    })
    const n = knobs['length'] as number
    const { states, observations } = hmmSample(model, k, n)
    const data = xyClasses(matrix(observations as number[], n, 1), labels(states), {
      ...stamp('dishonestCasino', dishonestCasino.description, knobs),
      classes: ['fair', 'loaded'],
      featureNames: ['face'],
    })
    return { data, truth: { clusters: labels(states), latents: { states }, extra: { model } } }
  },
}

/**
 * A series that switches between two Gaussian regimes, a calm one around zero and an active one that sits higher and
 * wanders more, with sticky switching. `x` is the series `[n, 1]` and `y` the regime (0 calm, 1 active). The truth
 * holds the generating `HiddenMarkovModel` and the true regimes.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { regimeSwitching } from 'mdml-engine/datasets'
 * generate(regimeSwitching, { knobs: { length: 50 } }).data.x.shape // => [50, 1]
 * ```
 */
export const regimeSwitching: DatasetGenerator<ClassificationData> = {
  name: 'regimeSwitching',
  kind: 'classification',
  description: 'A series that switches between a calm and an active Gaussian regime.',
  knobs: dict({
    length: lengthKnob,
    gap: real(0.5, 10, {
      default: 3,
      label: 'gap between the regimes',
      description: 'How far the active regime’s mean sits above the calm one’s, in units of the noise.',
    }),
    noise: real(0.1, 5, {
      default: 1,
      label: 'noise',
      description: 'The standard deviation within the calm regime; the active regime’s is one and a half times it.',
    }),
    stay: real(0.5, 0.999, {
      default: 0.95,
      label: 'chance of staying',
      description: 'The probability of remaining in a regime from one step to the next.',
    }),
  }),
  generate: (k, knobs) => {
    const noise = knobs['noise'] as number
    const stay = knobs['stay'] as number
    const model: HiddenMarkovModel = hiddenMarkovModel({
      states: ['calm', 'active'],
      initial: [0.5, 0.5],
      transition: [
        [stay, 1 - stay],
        [1 - stay, stay],
      ],
      emission: {
        kind: 'gaussian',
        means: [[0], [(knobs['gap'] as number) * noise]],
        covariances: [[[noise * noise]], [[(1.5 * noise) ** 2]]],
      },
    })
    const n = knobs['length'] as number
    const { states, observations } = hmmSample(model, k, n)
    const data = xyClasses(matrix((observations as number[][]).flat(), n, 1), labels(states), {
      ...stamp('regimeSwitching', regimeSwitching.description, knobs),
      classes: ['calm', 'active'],
      featureNames: ['value'],
    })
    return { data, truth: { clusters: labels(states), latents: { states }, extra: { model } } }
  },
}
