/**
 * The setups of expectation propagation where a discrete class chooses which regression explains a row: a mixture of two linear
 * regressions on a shared design, written with the model layer's `pick` (the class picks the prediction of its branch) and compiled
 * to one gate of Gaussian likelihoods per row. The exact answer enumerates the assignments of rows to classes, each class's
 * coefficients Gaussian given its rows, so every coefficient's posterior is a mixture of Gaussians.
 */

import { dict, real } from '../../space/index.ts'
import type { Config } from '../../trace/index.ts'
import { configNumber } from '../inference/method.ts'
import * as d from '../model/dists/index.ts'
import { defineModel, ops } from '../model/index.ts'
import { mixtureMarginal, mixtureOfRegressions } from '../validation/index.ts'
import { type AtomParts, defineAtom, type EpAtom } from './ep-atoms.ts'
import { tolerance } from './ep-tolerance.ts'

// Two lines, y = 1 + 2 x and y = -1 - x, eight rows whose classes are not told.
const XS = [-1.5, -0.5, 0.2, 1.1, -0.9, 0.7, 1.6, -1.2]
const YS = [-2.1, 0.2, 1.4, 3.3, -0.1, -1.6, -2.4, -0.3]
// The prior means (an intercept and a slope of each class) tell the classes apart softly, which fixes the labels.
const PRIOR_MEAN = [
  [1, 1.5],
  [-1, -1.5],
]

function mixtureRegressionProblem(k: Config): AtomParts {
  const noise = configNumber(k, 'noise')
  const priorSd = configNumber(k, 'priorSd')
  const design = XS.map((x) => [1, x])
  // region mixture-regression
  const definition = defineModel('mixture-regression', (m) => {
    const row = m.range('row')
    const feature = m.range('feature', { size: 2 })
    const component = m.range('component', { size: 2 })
    const x = m.data('X', { dims: [row, feature] })
    const y = m.data('y', { dims: [row], role: 'response' })
    // an intercept and a slope for each class, with priors that tell the classes apart
    const prior = m.const(
      [
        [1, -1],
        [1.5, -1.5],
      ],
      [feature, component],
    )
    const b = m.latent('B', d.normal(prior, priorSd), { dims: [feature, component] })
    // the class of each row, and the prediction of that class for the row
    const z = m.latent('z', d.categorical({ probs: [0.5, 0.5] }), { dims: [row] })
    const mean = ops.pick(ops.matmul(x, b), z)
    m.observe('y', d.normal(mean, noise), y)
  })
  // endregion
  const post = mixtureOfRegressions({ X: design, y: YS, priorMean: PRIOR_MEAN, priorSd, noiseSd: noise })
  const names = ['intercept of class 1', 'intercept of class 2', 'slope of class 1', 'slope of class 2']
  return {
    build: {
      definition,
      data: { X: design, y: YS },
      quantities: [0, 1, 2, 3].map((i) => ({
        name: `B[${i}]`,
        label: names[i] ?? `B_${i}`,
        support: 'real' as const,
      })),
    },
    reference: {
      exact: Object.fromEntries(
        [0, 1, 2, 3].map((i) => {
          const feature = Math.floor(i / 2)
          const klass = i % 2
          return [`B[${i}]`, mixtureMarginal(((post.coefficients[klass] ?? [])[feature]?.components ?? []).slice())]
        }),
      ),
      referenceSource: 'the 256 assignments of the eight rows to the two classes, each class conjugate given its rows',
    },
    present: {
      equations: [
        `B_class,0 and B_class,1 (an intercept and a slope) ~ N(prior mean, ${priorSd}²), prior means (1, 1.5) and (-1, -1.5)`,
        `z_i ~ Categorical(1/2, 1/2),  y_i ~ N(B_z[i],0 + B_z[i],1 x_i, ${noise}²)`,
        `observe the eight points drawn in the figure`,
      ],
      figure: {
        caption: 'The eight points and the two lines at the means of the beliefs.',
        xLabel: 'x',
        yLabel: 'y',
        xLimits: [-2, 2],
        yLimits: [-3.5, 4],
        points: [{ name: 'rows', x: XS, y: YS }],
        answer: (mean) =>
          [0, 1].map((c) => ({
            name: `class ${c + 1}`,
            x: [-2, 2],
            y: [-2, 2].map((x) => (mean[`B[${c}]`] ?? Number.NaN) + (mean[`B[${2 + c}]`] ?? Number.NaN) * x),
          })),
      },
    },
  }
}

/**
 * A mixture of two linear regressions on a shared design: eight points on two lines, a class per row chosen by a categorical
 * latent, the row's prediction picked from its class's line. The exact posterior enumerates the class assignments.
 *
 * @index estimator EP atom: a mixture of two regressions, a class per row picking the line (a gate of Gaussian likelihoods).
 * @example
 * ```ts
 * import { mixtureRegressionAtom } from 'omai-engine/probprog/ep'
 * mixtureRegressionAtom.name // => 'mixture-regression'
 * ```
 */
export const mixtureRegressionAtom: EpAtom = defineAtom({
  name: 'mixture-regression',
  title: 'Mixture of regressions (a class picks the line)',
  what: 'Eight points on two lines with a categorical class per row and a coefficient vector per class: the class picks the prediction of its branch, one gate of Gaussian likelihoods per row, against the enumeration of the assignments.',
  factors: [
    'prior (gaussian) x4',
    'dot-constant (the prediction of each class)',
    'gate (the class picks the branch of Gaussian likelihoods)',
  ],
  family: 'gates',
  reference: 'exact',
  tolerance: tolerance(
    'measured',
    'the coefficients are shared by every row, so the classes of the rows are coupled through them and EP matches one Gaussian per coefficient to a mixture of Gaussians over the assignments: measured within 0.14 reference sds, spreads down to 0.72 of the reference',
    { meanZ: 0.2, sdRatio: [0.65, 1.2], ks: 0.1 },
  ),
  knobs: dict({
    noise: real(0.3, 1.5, { default: 0.6, step: 0.05, label: 'noise sd of each row' }),
    priorSd: real(0.5, 4, { default: 1.5, step: 0.1, label: 'prior sd of each coefficient' }),
  }),
  parts: mixtureRegressionProblem,
})
