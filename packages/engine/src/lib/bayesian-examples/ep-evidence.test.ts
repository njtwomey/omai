// EP's log evidence against the grid evidence of the same model, on the atoms whose grid is fast: every rule that carries an evidence
// is checked at model level, not only per operator (operators/evidence.test.ts). The tolerances are in nats and say why they are not tight.
import { describe, expect, it } from 'vitest'
import { toProgram } from '../probprog/model/index.ts'
import { gridPosterior } from '../probprog/validation/index.ts'
import { epAtom } from './ep-atom-list.ts'
import { atomBound, atomRun } from './ep-atoms.ts'

/** [atom, tolerance in nats, why]. */
const CASES: readonly (readonly [string, number, string])[] = [
  ['gaussian', 1e-5, 'conjugate: exact'],
  ['gaussian-learned', 1e-5, 'a normal-gamma pair: exact'],
  ['beta-bernoulli', 1e-5, 'conjugate: exact'],
  ['gamma-poisson', 1e-5, 'conjugate: exact'],
  ['prior-normal-inverse-gamma', 1e-5, 'normal-gamma: exact'],
  ['prior-gamma-exponential', 1e-5, 'conjugate: exact'],
  ['prior-beta-geometric', 1e-5, 'conjugate: exact'],
  ['prior-dirichlet-categorical', 1e-5, 'conjugate: exact'],
  ['prior-inverse-gamma-hyperprior', 1e-3, 'the gamma precision rule on a grid in log lambda'],
  ['prior-gamma-poisson-hyperprior', 1e-3, 'a gamma rate with a gamma hyperprior'],
  ['prior-normal-hyperprior', 1e-3, 'a Gaussian centre with a Gaussian hyperprior'],
  ['gate-random', 1e-3, 'the gate: a mixture of branch evidences'],
  ['mixture-components', 1e-3, 'the gate again, written with components'],
  ['spike-and-slab', 1e-3, 'a gate between a narrow and a wide Gaussian'],
  ['spike-and-slab-marginal', 1e-3, 'the mixture site, switch summed out: exact'],
  ['clutter', 1e-3, 'a switch per observation: exact moment matching'],
  ['gate-weight', 1e-3, 'a gate with a beta weight'],
  ['gate-weight-hyperprior', 1e-3, 'a gate with a beta weight and a hyperprior'],
  ['ordered-means', 1e-3, 'an ordering constraint on two Gaussians'],
  ['exp', 1e-3, 'the apply rule (exp) by quadrature'],
  ['tanh', 1e-3, 'the apply rule (tanh) by quadrature'],
  ['log-scale', 1e-3, 'the Gaussian with a log scale by quadrature'],
  ['poisson-log', 1e-3, 'a log-link Poisson by quadrature'],
  ['product', 0.02, 'the product of two Gaussians: loopy, moment matched'],
  ['max', 0.01, 'the max of two Gaussians: moment matched'],
  ['gamma-precision', 0.05, 'a Gaussian with a gamma precision: the gamma is projected from a Student-t shaped tilt'],
  ['gamma-precision-learned', 0.05, 'the same with a learned mean'],
  ['student-t', 1e-3, 'the Student-t site by quadrature'],
  ['laplace', 0.05, 'the Laplace site: a kinked density, moment matched'],
  ['probit', 0.03, 'a probit link: moment matched truncated Gaussians'],
  ['logistic', 0.03, 'a logistic link by quadrature, moment matched'],
  ['skill-difference', 1e-2, 'two skills compared through a probit'],
  ['prior-lognormal-poisson', 0.01, 'a log-normal rate through exp'],
  ['prior-normal-weibull-scale', 0.02, 'a Weibull with a random scale'],
  ['prior-logit-normal-bernoulli', 0.01, 'a logistic-normal probability'],
  ['prior-beta-strength-gamma', 1e-3, 'a beta with a random concentration'],
  ['prior-beta-two-shapes', 1e-3, 'a beta with two random shapes: the grid in both log shapes (2e-5 measured)'],
  ['ordinal-probit', 0.01, 'the ordinal numeric site by quadrature (0.002 measured)'],
  ['ordinal-probit-learned', 0.01, 'the same on simulated outcomes (0.002 measured)'],
  ['quantile-regression', 0.15, 'the asymmetric Laplace: a kinked density, moment matched (0.07 measured)'],
  ['quantile-regression-learned', 0.08, 'the same on simulated points (0.02 measured)'],
  ['student-t-df-learned', 0.02, 'a Student-t with a gamma df: the pair site (0.006 measured)'],
  ['weibull-shape-learned', 0.08, 'a Weibull with a log-normal shape: the pair site (0.036 measured)'],
  ['skill-draws', 0.25, 'draws as an ordinal probit on two skills: loopy, moment matched (0.12 measured)'],
  ['softmax-regression', 0.5, 'a softmax of three scores shared by every point: loopy, moment matched (0.3 measured)'],
  ['half-cauchy', 0.3, 'the half-Cauchy prior is projected once on a log scale: the model is not the original one'],
  ['half-normal', 0.5, 'the half-normal prior is projected once on a log scale: the model is not the original one'],
  ['prior-uniform-mean', 0.15, 'the uniform prior is projected once on the value: the model is not the original one'],
]

describe('the EP log evidence of an atom against its grid evidence', () => {
  it.each(CASES)('%s (within %d nats: %s)', { timeout: 60000 }, (name, tol) => {
    const atom = epAtom(name)
    const problem = atom.problem(atom.defaults)
    const run = atomRun(problem, atom.ep ?? {}, { light: true })
    expect(run.status).toBe('ran')
    if (run.status !== 'ran') return
    const grid = gridPosterior(toProgram(atomBound(problem)), undefined, { maxRefinements: 0, ...problem.grid })
    expect(Number.isFinite(run.evidence)).toBe(true)
    expect(Math.abs((run.evidence ?? Number.NaN) - grid.logEvidence)).toBeLessThan(tol)
  })
})
