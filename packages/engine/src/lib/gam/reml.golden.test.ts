import { learnGolden } from '../../testing/learn.ts'
import type { Tensor } from '../tensor/index.ts'
import { toFlat } from '../tensor/index.ts'
import { fitGam, GAM_DEFAULTS, gamProblem, gamSolution } from './pirls.ts'
import { factor } from './terms.ts'

// A random effect is a penalty of the identity with lambda = sigma^2 / sigma_b^2: the REML optimum of the additive model
// is the REML fit of the random-intercept mixed model.
learnGolden('REML smoothing selection against statsmodels MixedLM', 'gam/reml', {
  mixedModelReml: ([groups, y]) => {
    const g = toFlat(groups as Tensor)
    const hyper = { ...GAM_DEFAULTS, terms: [factor(0, { randomEffect: true })], method: 'reml' as const, tol: 1e-10 }
    const problem = gamProblem(
      g.map((v) => [v]),
      y as Tensor,
      hyper,
    )
    const { state } = fitGam(problem, hyper)
    const { solution } = gamSolution(problem, state)
    const n = g.length
    return {
      lambda: [Math.exp(state.logLambda[0] ?? 0)],
      sigma2: [(solution.rss + solution.penalty) / (n - 1)],
      intercept: [state.beta[0] ?? 0],
    }
  },
})
