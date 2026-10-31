import { learnGolden, num, str } from '../../testing/learn.ts'
import { binomial, gamma, gaussian, log, poisson } from '../dists/families/index.ts'
import type { Tensor } from '../tensor/index.ts'
import { toFlat } from '../tensor/index.ts'
import { fitGamOuter } from './outer.ts'
import { GAM_DEFAULTS, type GamHyper, type GamProblem } from './pirls.ts'

const FAMILIES = {
  gaussian: gaussian(),
  poisson: poisson(),
  binomial: binomial(),
  'gamma-log': gamma(log()),
} as const

// The design and penalty are statsmodels' B-spline smooth. The reference selects the smoothing parameter by a brute-force
// profile of the Laplace-approximate REML over a grid of log lambda, refined by scipy; the engine's outer iteration
// (Newton on the analytic gradient) must land on the same optimum, criterion, effective degrees of freedom and fit.
learnGolden('outer-iteration REML against a grid-and-scipy reference', 'gam/outer', {
  outerReml: ([design, penalty, y], p) => {
    const X = Float64Array.from(toFlat(design as Tensor))
    const [n = 0, cols = 0] = (design as Tensor).shape
    const family = FAMILIES[str(p, 'family') as keyof typeof FAMILIES]
    const hyper: GamHyper = { ...GAM_DEFAULTS, family, iteration: 'outer', method: 'reml', maxIter: 200 }
    const problem: GamProblem = {
      design: { n, p: cols, nFeatures: 1, X, terms: [], nPenalties: 1 },
      structure: {
        p: cols,
        blocks: [{ offset: 0, size: cols, term: 0, matrix: Float64Array.from(toFlat(penalty as Tensor)) }],
        terms: [{ offset: 0, size: cols, blocks: [0], rank: num(p, 'rank') }],
      },
      y: Float64Array.from(toFlat(y as Tensor)),
      prior: new Float64Array(n).fill(1),
      family,
      link: family.defaultLink,
      knownScale: num(p, 'knownScale') === 1 ? 1 : undefined,
    }
    const { state, converged } = fitGamOuter(problem, hyper)
    if (!converged) throw new Error('the outer iteration did not converge')
    return { logLambda: state.logLambda, criterion: [state.criterion], edf: [state.edf], coef: state.beta }
  },
})
