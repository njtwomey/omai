import { learnGolden, num, str } from '../../testing/learn.ts'
import { binomial, gamma, gaussian, log, poisson } from '../dists/families/index.ts'
import type { Tensor } from '../tensor/index.ts'
import { toFlat } from '../tensor/index.ts'
import { fitGam, GAM_DEFAULTS, type GamHyper, type GamProblem, influenceDiagonal } from './pirls.ts'

const FAMILIES = {
  gaussian: gaussian(),
  poisson: poisson(),
  binomial: binomial(),
  'gamma-log': gamma(log()),
} as const

// The design and penalty come from statsmodels; the engine's penalised IRLS runs on them at the given smoothing parameter.
learnGolden('penalised IRLS against statsmodels GLMGam', 'gam/glmgam', {
  glmgam: ([design, penalty, y], p) => {
    const X = Float64Array.from(toFlat(design as Tensor))
    const [n = 0, cols = 0] = (design as Tensor).shape
    const k = (penalty as Tensor).shape[0] ?? 0
    const family = FAMILIES[str(p, 'family') as keyof typeof FAMILIES]
    const hyper: GamHyper = { ...GAM_DEFAULTS, family, method: 'fixed', sp: [num(p, 'lam')], tol: 1e-13, maxIter: 300 }
    const problem: GamProblem = {
      design: { n, p: cols, nFeatures: 1, X, terms: [], nPenalties: 1 },
      structure: {
        p: cols,
        blocks: [{ offset: cols - k, size: k, term: 0, matrix: Float64Array.from(toFlat(penalty as Tensor)) }],
        terms: [],
      },
      y: Float64Array.from(toFlat(y as Tensor)),
      prior: new Float64Array(n).fill(1),
      family,
      link: family.defaultLink,
      knownScale: undefined,
    }
    const { state, converged } = fitGam(problem, hyper)
    if (!converged) throw new Error('penalised IRLS did not converge')
    const hat = Array.from(influenceDiagonal(problem, state))
    return { coef: state.beta, hat, edf: [hat.reduce((a, b) => a + b, 0)], deviance: [state.deviance] }
  },
})
