// EP's log evidence against the grid evidence of the same model, on the atoms whose grid is fast: every rule that carries an evidence
// is checked at model level, not only per operator (operators/evidence.test.ts). The tolerances are in nats and say why they are not tight.
import { describe, expect, it } from 'vitest'
import { type Value, toFlat } from '../../tensor/index.ts'
import { session } from '../../trace/index.ts'
import { epSpecMethod } from '../inference/index.ts'
import { toProgram } from '../model/index.ts'
import { gridPosterior } from '../validation/index.ts'
import { epAtom, epAtoms } from './ep-atom-list.ts'
import { atomBound, atomRun } from './ep-atoms.ts'

/** The atoms that declare an evidence tolerance (`evidence: { nats, reason }`): the ones whose grid is fast. */
const CASES = epAtoms.flatMap((a) =>
  a.evidence === undefined ? [] : [[a.name, a.evidence.nats, a.evidence.reason] as const],
)

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
    // the simulated values are a tensor (the atom builds them with tensor ops)
    const y = toFlat((problem.data as { y: Value }).y)
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

describe('the EP evidence of the learned logistic regression (the weights share an unknown precision)', () => {
  const atom = epAtom('logistic-learned')
  const problem = atom.problem(atom.defaults)
  const bound = atomBound(problem)
  const posteriorOf = (plates: 'vectorise' | 'expand') => {
    const method = epSpecMethod(bound, { plates })
    const s = session(method, { seed: 1, steps: 400, mode: 'eager', config: { tol: 1e-9, ...(atom.ep ?? {}) } })
    return method.posterior?.(s.state, s.config as never)
  }

  it('has the evidence of the plated factor itself, though its cavity has no density: the rows-expanded model has the same', () => {
    // The scale mixture on the weights sends a site with a negative precision, so the cavity of the plate (all rows, one factor)
    // is indefinite at the fixed point and the integral of the plate against it diverges. The plate reads its cavity only through the
    // posterior it makes with its rows (proper), so it runs there (`improperTarget`) and its energy is taken in natural parameters:
    // the same fixed point and the same evidence as the model with a factor per row, to rounding.
    const plated = posteriorOf('vectorise')
    const rows = posteriorOf('expand')
    expect(plated?.logEvidence?.method).toBe('ep-energy')
    expect(plated?.diagnostics['evidenceUndefinedAt']).toBeUndefined()
    expect(plated?.diagnostics['unresolved']).toBe(0)
    expect(
      Math.abs((plated?.logEvidence?.value ?? Number.NaN) - (rows?.logEvidence?.value ?? Number.NaN)),
    ).toBeLessThan(1e-8)
    const grid = gridPosterior(toProgram(bound), undefined, { maxRefinements: 0, ...problem.grid })
    expect(Math.abs((plated?.logEvidence?.value ?? Number.NaN) - grid.logEvidence)).toBeLessThan(0.1)
  })

  it('has the evidence with a factor per row (a proper cavity for each), within 0.1 nats of the grid (0.046 measured)', () => {
    const post = posteriorOf('expand')
    expect(post?.diagnostics['evidenceUndefinedAt']).toBeUndefined()
    const grid = gridPosterior(toProgram(bound), undefined, { maxRefinements: 0, ...problem.grid })
    expect(Math.abs((post?.logEvidence?.value ?? Number.NaN) - grid.logEvidence)).toBeLessThan(0.1)
  })
})
