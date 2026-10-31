import { describe, expect, it } from 'vitest'
import { matrixView } from '../learn/index.ts'
import { smoothSolverConfig } from '../optim/index.ts'
import { key } from '../random/index.ts'
import type { SpaceValue } from '../space/index.ts'
import { type Algorithm, runAlgorithm } from '../trace/index.ts'
import { mixedAlgorithm, mixedModel, mixedProblemOf } from './mixed.ts'
import { multinomial, multinomialAlgorithm, multinomialProblemOf } from './multinomial.ts'
import { ordinal, ordinalAlgorithm, ordinalProblemOf } from './ordinal.ts'
import { countModelAlgorithm, countProblemOf, zeroInflated } from './zero-inflated.ts'

// The solvers each model is fitted with: Newton, two quasi-Newton methods, and the stochastic mode with Adam.
const adam = (learningRate: number): SpaceValue =>
  smoothSolverConfig({ kind: 'gradient', optimiser: { kind: 'adam' }, learningRate, gradTol: 1e-7 })
const SOLVERS = {
  newton: smoothSolverConfig({ kind: 'newton', gradTol: 1e-10 }),
  lbfgs: smoothSolverConfig({ kind: 'lbfgs', gradTol: 1e-9, ftol: 1e-15 }),
  bfgs: smoothSolverConfig({ kind: 'bfgs', gradTol: 1e-9 }),
}

const near = (a: ArrayLike<number>, b: ArrayLike<number>, digits: number, what: string): void => {
  expect(a.length, what).toBe(b.length)
  for (let j = 0; j < a.length; j++) expect(a[j], `${what} ${j}`).toBeCloseTo(b[j] ?? 0, digits)
}

/** Run an algorithm straight through, and in two halves through its serialised state: the iterates must agree. */
function resumes<S>(algorithm: Algorithm<S>, read: (state: S) => ArrayLike<number>, steps = 8): void {
  const straight = runAlgorithm(algorithm, { key: key(0), maxSteps: steps })
  const head = runAlgorithm(algorithm, { key: key(0), maxSteps: Math.floor(steps / 2) })
  const json = algorithm.serialise?.(head.state)
  expect(json).toBeDefined()
  const resumed = runAlgorithm(algorithm, {
    key: key(0),
    maxSteps: steps - head.steps,
    start: { state: json, step: head.steps, serialised: true },
  })
  expect(Array.from(read(resumed.state))).toEqual(Array.from(read(straight.state)))
}

const n = 72
const x2 = Array.from({ length: n }, (_, i) => [Math.sin(i * 1.7) * 2, Math.cos(i * 0.9)])
const latent = x2.map((r, i) => (r[0] ?? 0) - 0.5 * (r[1] ?? 0) + 0.9 * Math.sin(i * 12.9898))

describe('multinomial over the solver slot', () => {
  const y = latent.map((v) => (v < -0.6 ? 0 : v < 0.8 ? 1 : 2))
  const reference = multinomial({ alpha: 0.05 }).fit({ x: x2, y }).params

  it('reaches one optimum with Newton, BFGS, L-BFGS and Adam', () => {
    for (const [name, solver] of Object.entries({ ...SOLVERS, adam: adam(0.1) })) {
      const fit = multinomial({ alpha: 0.05, solver, maxIter: 6000 }).fit({ x: x2, y }).params
      for (let c = 0; c < reference.coef.length; c++) {
        near(fit.coef[c] ?? [], reference.coef[c] ?? [], 3, `${name} class ${c}`)
        // The intercepts are identified up to a common shift (the softmax ignores it, the penalty does not touch them).
        const shift = (p: typeof fit): number => (p.intercept[c] ?? 0) - (p.intercept[0] ?? 0)
        expect(shift(fit), `${name} intercept ${c}`).toBeCloseTo(shift(reference), 3)
      }
    }
  })

  it('serialises the state of any solver and resumes from it', () => {
    const { problem } = multinomialProblemOf({ x: x2, y }, 'test')
    for (const solver of [SOLVERS.lbfgs, adam(0.1)])
      resumes(multinomialAlgorithm(problem, { alpha: 0.05, solver }), (s) => s.theta)
  })
})

describe('ordinal over the solver slot', () => {
  const y = latent.map((v) => (v < -1 ? 0 : v < 0 ? 1 : v < 1.2 ? 2 : 3))
  const reference = ordinal().fit({ x: x2, y }).params

  it('reaches one optimum with Newton, BFGS, L-BFGS and Adam', () => {
    for (const [name, solver] of Object.entries({ ...SOLVERS, adam: adam(0.05) })) {
      const fit = ordinal({ solver, maxIter: 8000 }).fit({ x: x2, y }).params
      near(fit.coef, reference.coef, 3, `${name} slopes`)
      near(fit.thresholds, reference.thresholds, 3, `${name} thresholds`)
    }
  })

  it('serialises the state of any solver and resumes from it', () => {
    const { problem } = ordinalProblemOf({ x: x2, y }, 'test')
    for (const solver of [SOLVERS.bfgs, adam(0.05)]) resumes(ordinalAlgorithm(problem, { solver }), (s) => s.u)
  })
})

describe('zero-inflated counts over the solver slot', () => {
  const xs = Array.from({ length: 90 }, (_, i) => [Math.sin(i * 0.77) + 0.5 * Math.cos(i * 0.31)])
  const counts = xs.map((r, i) => {
    const mu = Math.exp(0.6 + 0.5 * (r[0] ?? 0))
    const u = (Math.sin(i * 91.7) + 1) / 2
    return u < 0.3 ? 0 : Math.max(0, Math.round(mu + 1.5 * Math.sin(i * 7.1)))
  })
  const reference = zeroInflated().fit({ x: xs, y: counts }).params

  it('reaches one optimum with Newton, BFGS, L-BFGS and Adam', () => {
    for (const [name, solver] of Object.entries({ ...SOLVERS, adam: adam(0.05) })) {
      const fit = zeroInflated({ solver, maxIter: 8000 }).fit({ x: xs, y: counts }).params
      near(fit.coef, reference.coef, 3, `${name} counts`)
      expect(fit.intercept, `${name} intercept`).toBeCloseTo(reference.intercept, 3)
      expect(fit.zeroIntercept, `${name} zero part`).toBeCloseTo(reference.zeroIntercept, 3)
    }
  })

  it('serialises the state of any solver and resumes from it', () => {
    const problem = countProblemOf({ x: xs, y: counts }, 'constant', 'test')
    for (const solver of [SOLVERS.lbfgs, adam(0.05)]) resumes(countModelAlgorithm(problem, { solver }), (s) => s.theta)
  })
})

describe('mixed models over the solver slot', () => {
  const groups = 6
  const per = 8
  const rows = Array.from({ length: groups * per }, (_, i) => [Math.floor(i / per) + 1, (i % per) - 3.5])
  const effects = [0.8, -0.6, 1.2, -1, 0.3, -0.4]
  const y = rows.map((r, i) => 1 + 0.5 * (r[1] ?? 0) + (effects[(r[0] ?? 1) - 1] ?? 0) + 0.4 * Math.sin(i * 12.9898))
  const reference = mixedModel({ covariance: 'diagonal' }).fit({ x: rows, y }).params

  it('reaches one optimum with Newton, BFGS and L-BFGS', () => {
    for (const [name, solver] of Object.entries(SOLVERS)) {
      const fit = mixedModel({ covariance: 'diagonal', solver, maxIter: 500 }).fit({ x: rows, y }).params
      near(fit.coef, reference.coef, 3, `${name} slopes`)
      expect(fit.intercept, `${name} intercept`).toBeCloseTo(reference.intercept, 3)
      expect(fit.sigma2, `${name} sigma2`).toBeCloseTo(reference.sigma2, 3)
    }
  })

  it('serialises the state of any solver and resumes from it', () => {
    const layout = { groupColumn: 0, fitIntercept: true, randomIntercept: true, randomSlopes: [] }
    const problem = mixedProblemOf({ x: rows, y }, layout, 'test')
    for (const solver of [SOLVERS.bfgs, SOLVERS.lbfgs])
      resumes(mixedAlgorithm(problem, { covariance: 'diagonal', solver }), (s) => s.theta)
    expect(matrixView(rows, 'test').n).toBe(groups * per)
  })
})
