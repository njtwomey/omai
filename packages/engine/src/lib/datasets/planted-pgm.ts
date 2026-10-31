/**
 * Samples from probabilistic graphical models with their truth: item-response data from a known test (the true abilities
 * and item parameters) and configurations of a Potts lattice at a chosen coupling, drawn by the Swendsen-Wang cluster
 * update (the critical coupling, the exact Onsager magnetisation for the Ising case, and each sample's energy and
 * magnetisation).
 */

import { type DatasetGenerator, type RegressionData, xy } from '../data/index.ts'
import {
  type IrtItem,
  irtSimulate,
  ISING_CRITICAL_ENERGY_PER_SITE,
  onsagerMagnetisation,
  pottsCriticalCoupling,
  pottsSamplerAlgorithm,
} from '../pgm/index.ts'
import { foldIn } from '../random/index.ts'
import { choice, dict, int, real } from '../space/index.ts'
import { iterate } from '../trace/index.ts'
import { matrix, stamp, vector } from './_util.ts'

/**
 * Right and wrong answers of respondents to a test whose items are known: abilities from the standard normal, items with
 * difficulties spread evenly over [-2, 2], and (per `model`) the same discrimination (`1pl`), discriminations cycling
 * through 0.6 to 1.8 (`2pl`), or those with a guessing floor of 0.2 (`3pl`). `x` is the 0/1 response matrix
 * `[respondents, items]` and `y` the true ability of each respondent. The truth holds the abilities and the true items.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { irtResponses } from 'mdml-engine/datasets'
 * generate(irtResponses, { knobs: { respondents: 30, items: 6 } }).data.x.shape // => [30, 6]
 * ```
 */
export const irtResponses: DatasetGenerator<RegressionData> = {
  name: 'irtResponses',
  kind: 'regression',
  description: 'Scored test responses from known abilities and item parameters (1PL, 2PL or 3PL).',
  knobs: dict({
    respondents: int(2, 1_000_000, { default: 500, label: 'respondents', description: 'The number of test takers.' }),
    items: int(2, 1000, { default: 12, label: 'items', description: 'The number of test items.' }),
    model: choice(['1pl', '2pl', '3pl'], {
      default: '2pl',
      label: 'model',
      description: 'Which item parameters vary: difficulty only, plus discrimination, plus a guessing floor.',
    }),
  }),
  generate: (k, knobs) => {
    const n = knobs['respondents'] as number
    const m = knobs['items'] as number
    const model = knobs['model'] as string
    const items: IrtItem[] = Array.from({ length: m }, (_, j) => ({
      a: model === '1pl' ? 1 : 0.6 + 0.4 * (j % 4),
      b: m === 1 ? 0 : -2 + (4 * j) / (m - 1),
      c: model === '3pl' ? 0.2 : 0,
    }))
    const { responses, theta } = irtSimulate(foldIn(k, 0), n, items)
    const data = xy(matrix(responses.flat(), n, m), vector(theta), {
      ...stamp('irtResponses', irtResponses.description, knobs),
      featureNames: Array.from({ length: m }, (_, j) => `item${j}`),
      targetNames: ['ability'],
    })
    return { data, truth: { latents: { theta }, extra: { items, model } } }
  },
}

/**
 * Configurations of the q-state Potts model on an L by L periodic lattice at the coupling `relativeCoupling` times the
 * critical one (log(1 + sqrt q)), drawn by Swendsen-Wang updates from a random start, one configuration every `thin`
 * updates after a burn-in. `x` is the spins `[samples, L * L]` and `y` the magnetisation per site of each sample (about
 * 0 in the disordered phase, up to 1 in the ordered). The truth holds each sample's energy per site, the critical
 * coupling and, for the Ising case, Onsager's magnetisation of the infinite lattice at that coupling.
 *
 * @example
 * ```ts
 * import { generate } from 'mdml-engine/data'
 * import { pottsSamples } from 'mdml-engine/datasets'
 * generate(pottsSamples, { knobs: { samples: 5, side: 6, burnIn: 10 } }).data.x.shape // => [5, 36]
 * ```
 */
export const pottsSamples: DatasetGenerator<RegressionData> = {
  name: 'pottsSamples',
  kind: 'regression',
  description: 'Spin configurations of a Potts lattice at a chosen coupling, with magnetisation and energy.',
  knobs: dict({
    samples: int(1, 100_000, { default: 100, label: 'samples', description: 'The number of configurations.' }),
    side: int(2, 128, { default: 12, label: 'side', description: 'The lattice side L (L * L spins, periodic).' }),
    states: int(2, 8, {
      default: 2,
      label: 'states',
      description: 'The number of Potts states q (2 is the Ising model).',
    }),
    relativeCoupling: real(0.05, 4, {
      default: 1.2,
      label: 'coupling over the critical one',
      description: 'The coupling in units of the critical coupling log(1 + sqrt q): below 1 disordered, above ordered.',
    }),
    burnIn: int(0, 100_000, {
      default: 100,
      label: 'burn-in',
      description: 'Updates discarded before the first sample.',
    }),
    thin: int(1, 1000, { default: 5, label: 'thinning', description: 'Updates between samples.' }),
  }),
  generate: (k, knobs) => {
    const n = knobs['samples'] as number
    const L = knobs['side'] as number
    const q = knobs['states'] as number
    const beta = (knobs['relativeCoupling'] as number) * pottsCriticalCoupling(q)
    const burn = knobs['burnIn'] as number
    const thin = knobs['thin'] as number
    const alg = pottsSamplerAlgorithm('swendsen-wang', { L, q, beta, start: 'random' })
    const spins: number[] = []
    const magnetisation: number[] = []
    const energy: number[] = []
    for (const { state, t } of iterate(alg, { key: foldIn(k, 0), maxSteps: burn + n * thin })) {
      if (t > burn && (t - burn) % thin === 0 && magnetisation.length < n) {
        spins.push(...state.spins)
        magnetisation.push(state.magnetisation)
        energy.push(state.energy)
      }
    }
    const data = xy(matrix(spins, n, L * L), vector(magnetisation), {
      ...stamp('pottsSamples', pottsSamples.description, knobs),
      featureNames: Array.from({ length: L * L }, (_, i) => `spin${i}`),
      targetNames: ['magnetisation'],
    })
    return {
      data,
      truth: {
        latents: { energy },
        extra: {
          beta,
          criticalCoupling: pottsCriticalCoupling(q),
          ...(q === 2
            ? {
                onsagerMagnetisation: onsagerMagnetisation(beta),
                criticalEnergyPerSite: ISING_CRITICAL_ENERGY_PER_SITE,
              }
            : {}),
        },
      },
    }
  },
}
