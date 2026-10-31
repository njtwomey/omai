import { describe, expect, it } from 'vitest'
import { families } from '../dists/index.ts'
import { key } from '../random/index.ts'
import { defaults } from '../space/index.ts'
import { runAlgorithm, type StepContext } from '../trace/index.ts'
import { fitGlm, type GlmProblem, glmGradient, glmSolvers, glmValue } from './glm.ts'
import { loggedSolverOf } from './solvers.ts'

// A small deterministic design: n rows, d = 3 standardised-ish columns.
const n = 80
const d = 3
const x = Array.from({ length: n * d }, (_, k) => {
  const i = Math.floor(k / d)
  const j = k % d
  return Math.sin(1.7 * i * (j + 1) + j) + 0.3 * Math.cos(0.37 * i)
})
const eta = Array.from(
  { length: n },
  (_, i) => 0.4 + 0.9 * (x[i * d] ?? 0) - 0.7 * (x[i * d + 1] ?? 0) + 0.2 * (x[i * d + 2] ?? 0),
)
const binary = eta.map((e, i) => (1 / (1 + Math.exp(-e)) > 0.5 + 0.35 * Math.sin(5.3 * i) ? 1 : 0))
const counts = eta.map((e, i) => Math.round(Math.exp(e) * (1 + 0.4 * Math.sin(3.1 * i))))

const logistic = (extra: Partial<GlmProblem> = {}): GlmProblem => ({
  x,
  n,
  d,
  y: binary,
  family: families.binomial(),
  link: families.logit(),
  fitIntercept: true,
  penalty: { l2: 0.05 },
  ...extra,
})

const SOLVERS = [
  { kind: 'irls' },
  { kind: 'coordinateDescent', sweeps: 5 },
  { kind: 'newton', gradTol: 1e-10 },
  { kind: 'bfgs', gradTol: 1e-10 },
  { kind: 'lbfgs', gradTol: 1e-10 },
  { kind: 'conjugateGradient', gradTol: 1e-8 },
] as const

describe('glmSolvers', () => {
  it('is one slot over eight solvers, IRLS the default', () => {
    expect(glmSolvers.names).toEqual([
      'irls',
      'coordinateDescent',
      'newton',
      'bfgs',
      'lbfgs',
      'conjugateGradient',
      'gradientDescent',
      'gradient',
    ])
    expect(defaults(glmSolvers.space)).toMatchObject({ kind: 'irls' })
  })

  it('reaches one optimum with IRLS, Newton, BFGS, L-BFGS, conjugate gradient and coordinate descent', () => {
    const problem = logistic()
    const reference = fitGlm(problem, { kind: 'irls' })
    expect(reference.converged).toBe(true)
    expect(Math.max(...Array.from(glmGradient(problem, reference.x), Math.abs))).toBeLessThan(1e-8)
    for (const config of SOLVERS) {
      const fit = fitGlm(problem, config, { maxSteps: 500 })
      expect(fit.converged, config.kind).toBe(true)
      for (let j = 0; j <= d; j++) expect(fit.x[j], `${config.kind} ${j}`).toBeCloseTo(reference.x[j] ?? 0, 5)
      expect(fit.objective, config.kind).toBeCloseTo(reference.objective, 10)
    }
  })

  it('reaches the same optimum with the stochastic gradient mode (Adam, whole set and a batch)', () => {
    const problem = logistic()
    const reference = fitGlm(problem, { kind: 'irls' })
    const whole = fitGlm(
      problem,
      { kind: 'gradient', optimiser: { kind: 'adam' }, learningRate: 0.05, gradTol: 1e-6 },
      { maxSteps: 4000 },
    )
    for (let j = 0; j <= d; j++) expect(whole.x[j]).toBeCloseTo(reference.x[j] ?? 0, 3)
    const batched = fitGlm(
      problem,
      { kind: 'gradient', optimiser: { kind: 'adam' }, learningRate: 0.02, batch: 20, schedule: { kind: 'cosine' } },
      { maxSteps: 3000 },
    )
    // A batch is a noisy estimate of the gradient: near the optimum, not on it.
    for (let j = 0; j <= d; j++) expect(batched.x[j]).toBeCloseTo(reference.x[j] ?? 0, 0)
    expect(batched.objective).toBeLessThan(reference.objective + 1e-2)
  })

  it('handles a Poisson response with an offset and prior weights on every table solver', () => {
    const offset = Array.from({ length: n }, (_, i) => 0.2 * Math.cos(i))
    const weights = Array.from({ length: n }, (_, i) => 1 + (i % 3))
    const problem: GlmProblem = {
      x,
      n,
      d,
      y: counts,
      weights,
      offset,
      family: families.poisson(),
      link: families.log(),
      fitIntercept: true,
      penalty: { l2: 0.01 },
    }
    const reference = fitGlm(problem, { kind: 'irls' })
    for (const config of [
      { kind: 'newton', gradTol: 1e-10 },
      { kind: 'lbfgs', gradTol: 1e-10 },
    ] as const) {
      const fit = fitGlm(problem, config, { maxSteps: 500 })
      for (let j = 0; j <= d; j++) expect(fit.x[j], config.kind).toBeCloseTo(reference.x[j] ?? 0, 5)
    }
  })

  it('uses a non-canonical link: Newton on the observed information agrees with Fisher scoring', () => {
    const problem: GlmProblem = {
      x,
      n,
      d,
      y: counts.map((c) => c + 1),
      family: families.gamma(families.log()),
      link: families.log(),
      fitIntercept: true,
      penalty: { l2: 0.01 },
    }
    const fisher = fitGlm(problem, { kind: 'irls' })
    const observed = fitGlm(problem, { kind: 'newton', gradTol: 1e-10 }, { maxSteps: 200 })
    for (let j = 0; j <= d; j++) expect(observed.x[j]).toBeCloseTo(fisher.x[j] ?? 0, 6)
  })

  it('zeroes coordinates under an l1 weight: IRLS, coordinate descent and OWL-QN agree', () => {
    const problem = logistic({ penalty: { l1: 0.08, l2: 0.02 } })
    const reference = fitGlm(problem, { kind: 'irls' })
    expect(reference.converged).toBe(true)
    expect(reference.coef.some((v) => v === 0)).toBe(true)
    for (const config of [
      { kind: 'coordinateDescent', sweeps: 10 },
      { kind: 'lbfgs', gradTol: 1e-9, memory: 10 },
    ] as const) {
      const fit = fitGlm(problem, config, { maxSteps: 1000 })
      for (let j = 0; j <= d; j++) expect(fit.x[j], config.kind).toBeCloseTo(reference.x[j] ?? 0, 4)
    }
    expect(() => fitGlm(problem, { kind: 'newton' })).toThrow(/smooth/)
  })

  it('takes a general penalty matrix, as an additive model needs: IRLS and Newton agree', () => {
    const p = d + 1
    const S = new Float64Array(p * p)
    for (let j = 0; j < d; j++) S[j * p + j] = 0.3 + 0.1 * j
    S[0 * p + 1] = 0.05
    S[1 * p + 0] = 0.05
    const problem = logistic({ penalty: { matrix: S }, normalise: false })
    const a = fitGlm(problem, { kind: 'irls' })
    const b = fitGlm(problem, { kind: 'newton', gradTol: 1e-10 }, { maxSteps: 200 })
    for (let j = 0; j < p; j++) expect(b.x[j]).toBeCloseTo(a.x[j] ?? 0, 6)
    expect(glmValue(problem, a.x)).toBeCloseTo(a.objective, 12)
  })

  it('serialises the state of every mode and resumes from it', () => {
    const problem = logistic()
    const stochastic = { kind: 'gradient', optimiser: { kind: 'adam' }, learningRate: 0.05, batch: 16 }
    for (const config of [...SOLVERS, stochastic]) {
      const algorithm = loggedSolverOf(glmSolvers, problem, config, key(3))
      const straight = runAlgorithm(algorithm, { key: key(3), maxSteps: 6 })
      const head = runAlgorithm(algorithm, { key: key(3), maxSteps: 3 })
      const json = algorithm.serialise?.(head.state)
      expect(json, config.kind).toBeDefined()
      const resumed = runAlgorithm(algorithm, {
        key: key(3),
        maxSteps: 6 - head.steps,
        start: { state: json, step: head.steps, serialised: true },
      })
      const x = (r: { readonly state: unknown }): number[] => Array.from((r.state as { x: Float64Array }).x)
      expect(x(resumed), config.kind).toEqual(x(straight))
    }
  })

  it('logs the objective, the gradient norm and the step by the solver convention', () => {
    const algorithm = loggedSolverOf(glmSolvers, logistic(), { kind: 'irls' }, key(0))
    const logged: string[] = []
    const ctx = { t: 0, key: key(0), log: (path: string) => logged.push(path), emit: () => undefined }
    algorithm.step(algorithm.init(key(0)), ctx as unknown as StepContext)
    expect(logged).toEqual(['solver/objective', 'solver/gradNorm', 'solver/step'])
  })
})
