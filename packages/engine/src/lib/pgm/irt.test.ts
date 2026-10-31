import { describe, expect, it } from 'vitest'
import { flatObjective } from '../optim/index.ts'
import { key } from '../random/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import * as P from './index.ts'

const truth2: P.IrtItem[] = [
  { a: 0.8, b: -1.5, c: 0 },
  { a: 1.2, b: -0.8, c: 0 },
  { a: 1.0, b: -0.2, c: 0 },
  { a: 1.5, b: 0.3, c: 0 },
  { a: 0.9, b: 0.8, c: 0 },
  { a: 1.3, b: 1.4, c: 0 },
  { a: 1.1, b: 0.0, c: 0 },
  { a: 0.7, b: 0.5, c: 0 },
]
const data = P.irtSimulate(key(21), 1500, truth2)

describe('item functions', () => {
  it('has probability one half at the difficulty and information a^2 / 4 there', () => {
    expect(P.irtProbability({ a: 2, b: 1, c: 0 }, 1)).toBe(0.5)
    expect(P.irtItemInformation({ a: 2, b: 1, c: 0 }, 1)).toBeCloseTo(1, 12)
    // The guessing floor lowers the information.
    expect(P.irtItemInformation({ a: 1, b: 0, c: 0.25 }, 0)).toBeLessThan(0.25)
  })
  it('peaks at the difficulty and adds across items', () => {
    const it: P.IrtItem = { a: 1.3, b: 0.4, c: 0 }
    const at = (t: number) => P.irtItemInformation(it, t)
    expect(at(0.4)).toBeGreaterThan(at(0.1))
    expect(at(0.4)).toBeGreaterThan(at(0.9))
    expect(P.irtTestInformation([it, it], 0.4)).toBeCloseTo(2 * at(0.4), 12)
  })
  it('matches the numerical derivative of the response curve (information = slope^2 / (P Q))', () => {
    const it: P.IrtItem = { a: 1.1, b: -0.3, c: 0.15 }
    const h = 1e-6
    const slope = (P.irtProbability(it, 0.2 + h) - P.irtProbability(it, 0.2 - h)) / (2 * h)
    const p = P.irtProbability(it, 0.2)
    expect(P.irtItemInformation(it, 0.2)).toBeCloseTo((slope * slope) / (p * (1 - p)), 6)
  })
})

describe('marginal maximum likelihood', () => {
  it('has an analytic gradient equal to the numerical one', () => {
    for (const model of ['1pl', '2pl', '3pl'] as const) {
      const { objective, x0 } = P.irtSmoothProblem({ responses: data.responses.slice(0, 150), model })
      const flat = flatObjective(objective, x0 as number[])
      const x = (x0 as number[]).map((v, i) => v + 0.1 * Math.sin(i + 1))
      const g = Array.from(flat.valueAndGrad(x).grad)
      x.forEach((_, i) => {
        const h = 1e-5
        const up = x.map((v, k) => (k === i ? v + h : v))
        const dn = x.map((v, k) => (k === i ? v - h : v))
        expect(g[i]).toBeCloseTo((flat.value(up) - flat.value(dn)) / (2 * h), 6)
      })
    }
  })
  it('recovers the planted 2PL items', () => {
    const fit = P.fitIrt({ responses: data.responses, model: '2pl' })
    expect(fit.converged).toBe(true)
    let sb = 0
    let sa = 0
    fit.items.forEach((it, j) => {
      sb += (it.b - (truth2[j]?.b ?? 0)) ** 2
      sa += (it.a - (truth2[j]?.a ?? 0)) ** 2
    })
    expect(Math.sqrt(sb / fit.items.length)).toBeLessThan(0.15)
    expect(Math.sqrt(sa / fit.items.length)).toBeLessThan(0.2)
  })
  it('increases the marginal likelihood at every EM step', () => {
    const algorithm = P.irtAlgorithm({ responses: data.responses.slice(0, 400), model: '2pl' })
    let state = algorithm.init(key(0))
    let last = state.objective
    for (let i = 0; i < 12; i++) {
      state = algorithm.step(state, undefined as never)
      expect(state.objective).toBeLessThanOrEqual(last + 1e-10)
      last = state.objective
    }
  })
  it('reaches the same items and likelihood from EM, Newton and L-BFGS', () => {
    const problem: P.IrtProblem = { responses: data.responses.slice(0, 600), model: '2pl' }
    const base = P.fitIrt(problem, { tol: 1e-7 })
    for (const solver of [
      { kind: 'bfgs', gradTol: 1e-7 },
      { kind: 'lbfgs', gradTol: 1e-7 },
    ]) {
      const fit = P.fitIrt(problem, { solver, tol: 1e-6, maxSteps: 400 })
      expect(fit.solver).toBe(solver.kind)
      expect(fit.logLikelihood).toBeCloseTo(base.logLikelihood, 2)
      for (const [j, it] of fit.items.entries()) expect(it.b).toBeCloseTo(base.items[j]?.b ?? 0, 1)
    }
  })
  it('fits a Rasch model with unit discrimination and a 3PL with a floor', () => {
    const rasch = P.fitIrt({ responses: data.responses, model: '1pl' })
    expect(rasch.items.every((it) => it.a === 1 && it.c === 0)).toBe(true)
    const guessing: P.IrtItem[] = truth2.map((it, j) => ({ ...it, c: j % 2 === 0 ? 0.2 : 0 }))
    const g = P.irtSimulate(key(8), 3000, guessing)
    const f3 = P.fitIrt({ responses: g.responses, model: '3pl' }, { maxSteps: 300 })
    const f2 = P.fitIrt({ responses: g.responses, model: '2pl' })
    expect(f3.logLikelihood).toBeGreaterThan(f2.logLikelihood)
    expect(f3.items[0]?.c ?? 0).toBeGreaterThan(0.05)
  })
  it('handles missing responses', () => {
    const holes = data.responses.map((row, i) => row.map((x, j) => ((i + j) % 7 === 0 ? null : x)))
    const fit = P.fitIrt({ responses: holes, model: '2pl' })
    expect(fit.converged).toBe(true)
    expect(Math.abs((fit.items[3]?.b ?? 0) - 0.3)).toBeLessThan(0.3)
  })
})

describe('ability estimation', () => {
  const fit = P.fitIrt({ responses: data.responses, model: '2pl' })
  const corr = (a: readonly number[], b: readonly number[]): number => {
    const ma = a.reduce((x, y) => x + y, 0) / a.length
    const mb = b.reduce((x, y) => x + y, 0) / b.length
    let sab = 0
    let saa = 0
    let sbb = 0
    a.forEach((x, i) => {
      sab += (x - ma) * ((b[i] ?? 0) - mb)
      saa += (x - ma) ** 2
      sbb += ((b[i] ?? 0) - mb) ** 2
    })
    return sab / Math.sqrt(saa * sbb)
  }
  it('EAP tracks the true abilities and its error falls with more items', () => {
    const est = data.responses.slice(0, 400).map((r) => P.irtAbility(r, fit.items, 'eap'))
    // Corr(EAP, theta) = sqrt(1 - E[posterior variance]) for a calibrated prior.
    const expected = Math.sqrt(1 - est.reduce((a, e) => a + e.se * e.se, 0) / est.length)
    expect(
      Math.abs(
        corr(
          est.map((e) => e.theta),
          data.theta.slice(0, 400),
        ) - expected,
      ),
    ).toBeLessThan(0.06)
    const short = P.irtAbility(
      (data.responses[0] ?? []).map((x, j) => (j < 3 ? x : null)),
      fit.items,
      'eap',
    )
    const full = P.irtAbility(data.responses[0] ?? [], fit.items, 'eap')
    expect(full.se).toBeLessThan(short.se)
  })
  it('MLE is infinite for perfect scores and MAP shrinks it towards zero', () => {
    const items = fit.items
    expect(
      P.irtAbility(
        items.map(() => 1),
        items,
        'mle',
      ).theta,
    ).toBe(Number.POSITIVE_INFINITY)
    const mixed = items.map((_, j): 0 | 1 => (j % 2 === 0 ? 1 : 0))
    const mle = P.irtAbility(mixed, items, 'mle')
    const map = P.irtAbility(mixed, items, 'map')
    const eap = P.irtAbility(mixed, items, 'eap')
    expect(Math.abs(map.theta)).toBeLessThanOrEqual(Math.abs(mle.theta) + 1e-6)
    expect(eap.theta).toBeCloseTo(map.theta, 0)
    // The MLE's standard error is the reciprocal root of the test information at the estimate.
    expect(mle.se).toBeCloseTo(1 / Math.sqrt(P.irtTestInformation(items, mle.theta)), 8)
  })
  it('under the Rasch model equal total scores give equal MLE abilities (the sum score is sufficient)', () => {
    const items: P.IrtItem[] = [-1, -0.3, 0.4, 1.1].map((b) => ({ a: 1, b, c: 0 }))
    const a = P.irtAbility([1, 1, 0, 0], items, 'mle').theta
    const b = P.irtAbility([0, 1, 1, 0], items, 'mle').theta
    expect(a).toBeCloseTo(b, 6)
  })
  it('serialises the traced fit', () => {
    const algorithm = P.irtAlgorithm(
      { responses: data.responses.slice(0, 200), model: '2pl' },
      { solver: { kind: 'bfgs', gradTol: 1e-6 } },
    )
    const run = runAlgorithm(algorithm, { maxSteps: 3 })
    expect(algorithm.deserialise?.(algorithm.serialise?.(run.state) as never)?.u).toEqual(run.state.u)
  })
})
