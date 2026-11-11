// EP's log evidence against the grid evidence of the same model, on the atoms whose grid is fast: every rule that carries an evidence
// is checked at model level, not only per operator (operators/evidence.test.ts). The tolerances are in nats and say why they are not tight.
import { describe, expect, it } from 'vitest'
import { toProgram } from '../model/index.ts'
import { gridPosterior } from '../validation/index.ts'
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
  [
    'clutter-mixture-learned',
    0.05,
    'the summed-out outlier mixture with a learned scale and share: a hidden switch per observation under a gate (0.02 measured)',
  ],
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

describe('the EP log evidence of the summed-out spike and slab with a learned share and slab precision', () => {
  it('is within 0.05 nats (0.014 measured) of the grid over the share and the slab precision, the coefficients summed out', {
    timeout: 60000,
  }, () => {
    const atom = epAtom('spike-and-slab-marginal-learned')
    const problem = atom.problem(atom.defaults)
    const run = atomRun(problem, atom.ep ?? {}, { light: true })
    expect(run.status).toBe('ran')
    if (run.status !== 'ran') return
    const y = (problem.data as { y: number[] }).y
    const noise = atom.defaults['noise'] as number
    const normal = (v: number, sd: number): number => -0.5 * Math.log(2 * Math.PI * sd * sd) - (v * v) / (2 * sd * sd)
    const logBeta = (p: number, a: number, b: number): number =>
      lgamma(a + b) - lgamma(a) - lgamma(b) + (a - 1) * Math.log(p) + (b - 1) * Math.log(1 - p)
    const lgamma = (z: number): number => {
      const c = [
        676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905,
        -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
      ]
      const x = z - 1
      let a = 0.99999999999980993
      for (let i = 0; i < 8; i++) a += (c[i] as number) / (x + i + 1)
      const t = x + 7.5
      return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a)
    }
    // log p(share) p(tau) prod_j [share N(y_j; 0, slab^2 + noise^2) + (1 - share) N(y_j; 0, 0.1^2 + noise^2)], integrated on a grid.
    const nP = 400
    const nT = 400
    const [lo, hi] = [Math.log(1e-4), Math.log(60)]
    const terms: number[] = []
    for (let a = 0; a < nP; a++) {
      const p = (a + 0.5) / nP
      for (let b = 0; b < nT; b++) {
        const lt = lo + ((hi - lo) * (b + 0.5)) / nT
        const tau = Math.exp(lt)
        let lp = logBeta(p, 1.5, 3.5) + 2 * Math.log(8) - lgamma(2) + 1 * lt - 8 * tau + lt
        for (const v of y) {
          const s1 = Math.log(p) + normal(v, Math.hypot(1 / Math.sqrt(tau), noise))
          const s0 = Math.log(1 - p) + normal(v, Math.hypot(0.1, noise))
          lp += Math.max(s0, s1) + Math.log1p(Math.exp(-Math.abs(s0 - s1)))
        }
        terms.push(lp)
      }
    }
    const top = terms.reduce((a, b) => Math.max(a, b), Number.NEGATIVE_INFINITY)
    const grid =
      top + Math.log(terms.reduce((a, l) => a + Math.exp(l - top), 0)) + Math.log(1 / nP) + Math.log((hi - lo) / nT)
    expect(Math.abs((run.evidence ?? Number.NaN) - grid)).toBeLessThan(0.05)
  })
})
