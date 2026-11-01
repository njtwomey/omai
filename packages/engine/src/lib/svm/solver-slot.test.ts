/**
 * The solver of the support vector classifier is a slot: SMO (the dual, libsvm's) is the default and reaches either
 * loss; with the squared hinge the primal is smooth, and every solver of `smoothSolvers` reaches the optimum SMO does.
 */

import { describe, expect, it } from 'vitest'
import { solverConfigOf } from '../optim/index.ts'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { metricSeries } from '../trace/index.ts'
import { svcSolvers } from './_squared.ts'
import { svc } from './classifier.ts'

const blobs = (centres: readonly (readonly [number, number])[], per: number): { x: number[][]; y: number[] } => {
  const x: number[][] = []
  const y: number[] = []
  let s = 7
  const next = (): number => {
    s = (s * 1103515245 + 12345) % 2147483648
    return s / 2147483648 - 0.5
  }
  centres.forEach(([cx, cy], c) => {
    for (let i = 0; i < per; i++) {
      x.push([cx + 1.6 * next(), cy + 1.6 * next()])
      y.push(c)
    }
  })
  return { x, y }
}
const data = blobs(
  [
    [-1, 0],
    [1, 0],
  ],
  12,
)
const base = { kernel: 'rbf' as const, gamma: 0.5, C: 2, tol: 1e-8, loss: 'squaredHinge' as const }
const slot = (kind: string, more: Record<string, number> = {}): ReturnType<typeof solverConfigOf> =>
  solverConfigOf(svcSolvers, { kind, ...more })

describe('the solver slot of the support vector classifier', () => {
  it('lists SMO first, the default, beside the smooth solvers', () => {
    expect(svcSolvers.names[0]).toBe('smo')
    expect(svcSolvers.names).toEqual(expect.arrayContaining(['bfgs', 'lbfgs', 'newton', 'conjugateGradient']))
  })

  it('SMO on the squared hinge is the optimum of the smooth primal: every solver reaches the same decision function', () => {
    const reference = svc.with(base).fit(data)
    const f0 = toFlat(reference.decisionFunction(data.x))
    for (const [kind, more] of [
      ['newton', { gradTol: 1e-9 }],
      ['bfgs', { gradTol: 1e-9 }],
      ['lbfgs', { gradTol: 1e-9 }],
      ['conjugateGradient', { gradTol: 1e-9 }],
    ] as const) {
      const fitted = svc.with({ ...base, solver: slot(kind, more) }).fit(data)
      const f = toFlat(fitted.decisionFunction(data.x))
      f.forEach((v, i) => {
        expect(v, `${kind} ${i}`).toBeCloseTo(f0[i] ?? 0, 3)
      })
      expect(fitted.score(data)).toBe(reference.score(data))
    }
  }, 60_000)

  it('keeps the default hinge fit on SMO and refuses a smooth solver for the hinge', () => {
    const hinge = svc.with({ kernel: 'rbf', gamma: 0.5, C: 2 }).fit(data)
    const named = svc.with({ kernel: 'rbf', gamma: 0.5, C: 2, solver: slot('smo') }).fit(data)
    expect(toFlat(named.decisionFunction(data.x))).toEqual(toFlat(hinge.decisionFunction(data.x)))
    expect(() => svc.with({ kernel: 'rbf', solver: slot('bfgs') }).fit(data)).toThrow(/not smooth/)
    expect(() => svc.with({ ...base, form: 'nu' }).fit(data)).toThrow(/C form/)
  })

  it('a smooth solver is stepped on the tape: the early model exists and the gap closes', () => {
    const est = svc.with({ ...base, solver: slot('lbfgs', { gradTol: 1e-8 }) })
    const s = est.session(data, { key: key(0), logs: 'all' })
    s.advance({ steps: 3 })
    expect(toFlat(s.fittedAt(3).predict(data.x)).length).toBe(data.x.length)
    s.advance()
    expect(s.status).toBe('converged')
    const gap = metricSeries(s.run(), 'solver/gap').values
    expect(gap.every((g) => g > -1e-9)).toBe(true)
    expect(gap[gap.length - 1] as number).toBeLessThan(1e-3)
    expect(gap[0] as number).toBeGreaterThan(gap[gap.length - 1] as number)
  }, 60_000)

  it('fits Platt probabilities with the squared hinge', () => {
    const fitted = svc.with({ ...base, probability: true, tol: 1e-4 }).fit(data)
    const p = fitted.predictProba?.(data.x)
    expect(p).toBeDefined()
    const flat = toFlat(p as never)
    expect(flat.length).toBe(data.x.length * 2)
    expect(Math.abs((flat[0] ?? 0) + (flat[1] ?? 0) - 1)).toBeLessThan(1e-9)
  }, 60_000)
})
