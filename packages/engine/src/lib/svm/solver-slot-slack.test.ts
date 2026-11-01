/**
 * The solver of the support vector regressor and of the support vector data description is a slot: SMO (the dual) is the
 * default and reaches either loss; with the squared loss the primal is smooth and every solver of `smoothSolvers`
 * reaches the optimum SMO does. The one-class machine has no slot.
 */

import { describe, expect, it } from 'vitest'
import { solverConfigOf } from '../optim/index.ts'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { metricSeries, runAlgorithm } from '../trace/index.ts'
import { svddDual, svrDual } from './_dual.ts'
import { svddSolvers, svrSolvers } from './_squared-slack.ts'
import { oneClassSvm, svdd } from './one-class.ts'
import { svr } from './regressor.ts'

const solvers = [
  ['newton', { gradTol: 1e-9 }],
  ['bfgs', { gradTol: 1e-9 }],
  ['lbfgs', { gradTol: 1e-9, ftol: 1e-15 }],
  ['conjugateGradient', { gradTol: 1e-9 }],
] as const

describe('the solver slot of the support vector regressor', () => {
  const x = Array.from({ length: 16 }, (_, i) => [-3 + (6 * i) / 15])
  const data = { x, y: x.map(([v]) => Math.sin(v as number) + 0.1 * Math.cos(7 * (v as number))) }
  const base = { kernel: 'rbf' as const, gamma: 1, C: 5, epsilon: 0.05, tol: 1e-8 }
  const squared = { ...base, loss: 'squaredEpsilonInsensitive' as const }
  const slot = (kind: string, more: Record<string, number> = {}): ReturnType<typeof solverConfigOf> =>
    solverConfigOf(svrSolvers, { kind, ...more })

  it('lists SMO first, the default, beside the smooth solvers', () => {
    expect(svrSolvers.names[0]).toBe('smo')
    expect(svrSolvers.names).toEqual(expect.arrayContaining(['bfgs', 'lbfgs', 'newton', 'conjugateGradient']))
  })

  it('the default is still the epsilon-insensitive SMO fit: the library dual run directly gives the same model', () => {
    const lin = { x: [[0], [1], [2], [3], [4]], y: [0.2, 1.1, 1.9, 3.2, 3.9] }
    const fitted = svr.with({ kernel: 'linear', C: 10, epsilon: 0.1, tol: 1e-8 }).fit(lin)
    const K = Float64Array.from({ length: 25 }, (_, k) => Math.floor(k / 5) * (k % 5))
    const dual = svrDual(K, Float64Array.from(lin.y), new Float64Array(5).fill(10), 0.1, {
      tol: 1e-8,
      selection: 'second-order',
    })
    const r = runAlgorithm(dual.algorithm, { maxSteps: 200_000 })
    const e = dual.expansion(r.state)
    expect(fitted.params.intercept).toBe(-e.rho)
    expect(fitted.params.coef).toEqual(Array.from(e.coef).filter((c) => c !== 0))
    const named = svr
      .with({ kernel: 'linear', C: 10, epsilon: 0.1, tol: 1e-8, loss: 'epsilonInsensitive', solver: slot('smo') })
      .fit(lin)
    expect(named.params).toEqual(fitted.params)
  })

  it('every solver reaches the decision function SMO does on the squared epsilon-insensitive loss', () => {
    const reference = svr.with(squared).fit(data)
    const f0 = toFlat(reference.predict(x))
    expect(reference.score(data)).toBeGreaterThan(0.9)
    for (const [kind, more] of solvers) {
      const fitted = svr.with({ ...squared, solver: slot(kind, more) }).fit(data)
      toFlat(fitted.predict(x)).forEach((v, i) => {
        expect(v, `${kind} ${i}`).toBeCloseTo(f0[i] ?? 0, 3)
      })
    }
  }, 60_000)

  it('refuses a smooth solver for the epsilon-insensitive loss and the nu form with the squared loss', () => {
    expect(() => svr.with({ ...base, solver: slot('bfgs') }).fit(data)).toThrow(/not smooth/)
    expect(() => svr.with({ ...squared, form: 'nu' }).fit(data)).toThrow(/epsilon form/)
  })

  it('weights act on the squared loss: a zero-weight sample is ignored by SMO and by a smooth solver alike', () => {
    const weights = data.x.map((_, i) => (i === 3 ? 0 : 1))
    const a = svr.with(squared).fit({ ...data, weights })
    const b = svr.with({ ...squared, solver: slot('lbfgs', { gradTol: 1e-9 }) }).fit({ ...data, weights })
    const fa = toFlat(a.predict(x))
    toFlat(b.predict(x)).forEach((v, i) => {
      expect(v).toBeCloseTo(fa[i] ?? 0, 3)
    })
  }, 60_000)

  it('a smooth solver is stepped on the tape: the early model exists and the gap closes', () => {
    const est = svr.with({ ...squared, solver: slot('lbfgs', { gradTol: 1e-8 }) })
    const s = est.session(data, { key: key(0), logs: 'all' })
    s.advance({ steps: 3 })
    expect(toFlat(s.fittedAt(3).predict(x)).length).toBe(x.length)
    s.advance()
    expect(s.status).toBe('converged')
    const gap = metricSeries(s.run(), 'solver/gap').values
    expect(gap.every((g) => g > -1e-9)).toBe(true)
    expect(gap[gap.length - 1] as number).toBeLessThan(1e-3)
    expect(gap[0] as number).toBeGreaterThan(gap[gap.length - 1] as number)
  }, 60_000)
})

describe('the solver slot of the support vector data description', () => {
  const x = Array.from({ length: 18 }, (_, i) => [Math.cos(i * 1.7) * (1 + 0.3 * Math.sin(i)), Math.sin(i * 1.7) * 0.6])
  const base = { kernel: 'rbf' as const, gamma: 3, C: 0.3, tol: 1e-8 }
  const squared = { ...base, loss: 'squaredSlack' as const }
  const slot = (kind: string, more: Record<string, number> = {}): ReturnType<typeof solverConfigOf> =>
    solverConfigOf(svddSolvers, { kind, ...more })
  const probe = [
    [0, 0],
    [0.5, 0.2],
    [1.2, 0.1],
    [3, 3],
    [-2, 0.4],
  ]

  it('lists SMO first, the default, beside the smooth solvers', () => {
    expect(svddSolvers.names[0]).toBe('smo')
    expect(svddSolvers.names).toEqual(expect.arrayContaining(['bfgs', 'lbfgs', 'newton', 'conjugateGradient']))
  })

  it('the default is still the linear-slack SMO fit: the library dual run directly gives the same ball', () => {
    const fitted = svdd.with(base).fit({ x })
    const named = svdd.with({ ...base, loss: 'slack', solver: slot('smo') }).fit({ x })
    expect(named.params).toEqual(fitted.params)
    const K = fitted.params.kernel
    expect(K).toBeDefined()
    const dual = svddDual(
      Float64Array.from({ length: 18 * 18 }, (_, k) => {
        const a = x[Math.floor(k / 18)] as number[]
        const b = x[k % 18] as number[]
        return Math.exp(
          -3 * ((a[0] as number) - (b[0] as number)) ** 2 - 3 * ((a[1] as number) - (b[1] as number)) ** 2,
        )
      }),
      0.3,
      { tol: 1e-8, selection: 'second-order' },
    )
    const r = runAlgorithm(dual.algorithm, { maxSteps: 200_000 })
    expect(fitted.params.rho).toBeCloseTo(dual.expansion(r.state).rho, 6)
  })

  it('every solver reaches the decision function SMO does on the squared slack', () => {
    const reference = svdd.with(squared).fit({ x })
    const f0 = toFlat(reference.decisionFunction(probe))
    expect(reference.params.radiusSquared).toBeGreaterThan(0)
    for (const [kind, more] of solvers) {
      const fitted = svdd.with({ ...squared, solver: slot(kind, more) }).fit({ x })
      toFlat(fitted.decisionFunction(probe)).forEach((v, i) => {
        expect(v, `${kind} ${i}`).toBeCloseTo(f0[i] ?? 0, 3)
      })
      expect(fitted.params.radiusSquared as number).toBeCloseTo(reference.params.radiusSquared as number, 3)
    }
  }, 60_000)

  it('accepts any positive C with the squared slack, refuses a smooth solver for the linear slack', () => {
    expect(() => svdd.with({ ...base, C: 0.01 }).fit({ x })).toThrow(/at least/)
    expect(() => svdd.with({ ...squared, C: 0.01 }).fit({ x })).not.toThrow()
    expect(() => svdd.with({ ...base, solver: slot('bfgs') }).fit({ x })).toThrow(/not smooth/)
  })

  it('a smooth solver is stepped on the tape: the early model exists and the gap closes', () => {
    const est = svdd.with({ ...squared, solver: slot('lbfgs', { gradTol: 1e-8 }) })
    const s = est.session({ x }, { key: key(0), logs: 'all' })
    s.advance({ steps: 3 })
    expect(toFlat(s.fittedAt(3).decisionFunction(probe)).length).toBe(probe.length)
    s.advance()
    expect(s.status).toBe('converged')
    const gap = metricSeries(s.run(), 'solver/gap').values
    expect(gap.every((g) => g > -1e-9)).toBe(true)
    expect(gap[gap.length - 1] as number).toBeLessThan(1e-3)
    expect(gap[0] as number).toBeGreaterThan(gap[gap.length - 1] as number)
  }, 60_000)

  it('the one-class machine is SMO-only and still fits', () => {
    const fitted = oneClassSvm.with({ nu: 0.3, gamma: 3 }).fit({ x })
    expect(fitted.params.supportIndex.length).toBeGreaterThan(0)
  })
})
