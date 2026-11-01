import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { policies, recorder, runAlgorithm, stateToJson } from '../../trace/index.ts'
import * as R from './index.ts'

const f1 = (x: number): number => x ** 3 - x - 1
const d1 = (x: number): number => 3 * x * x - 1
const f3 = (x: number): number => Math.exp(x) - 5

// What the functions returned before they became runs of the algorithms (taken from the previous implementation).
const BEFORE = {
  bisect: { root: 1.324717957244502, it: 39, calls: 41, flag: 'converged' },
  brentq: { root: 1.5707963267948966, it: 7, calls: 8, flag: 'converged' },
  ridder: { root: 1.6094379124330997, it: 5, calls: 12, flag: 'converged' },
  toms: { root: 1.324717957244746, it: 3, calls: 10, flag: 'converged' },
  newton: { root: 1.324717957244746, it: 5, calls: 5, flag: 'converged' },
  secant: { root: 1.6094379124341003, it: 9, calls: 10, flag: 'converged' },
  newtonSec: { root: 1.3247179572447458, it: 6, calls: 7, flag: 'converged' },
  halley: { root: 1.324717957244746, it: 3, calls: 3, flag: 'converged' },
  bisectMax: { root: 1.34375, it: 5, calls: 7, flag: 'maximum iterations' },
  brentMax: { root: 1.4980102695802378, it: 3, calls: 5, flag: 'maximum iterations' },
  toms2: { root: 1.6094379124341003, it: 3, calls: 11, flag: 'converged' },
  ridMax: { root: 1.609817804105578, it: 2, calls: 6, flag: 'maximum iterations' },
  newtonMax: { root: 1.325200398950907, it: 2, calls: 2, flag: 'maximum iterations' },
  newtonZero: { root: 0, it: 1, calls: 1, flag: 'derivative was zero' },
  bisectEnd: { root: 0, it: 1, calls: 3, flag: 'converged' },
} as const

const brief = (r: R.RootResult): { root: number; it: number; calls: number; flag: string } => ({
  root: r.root,
  it: r.iterations,
  calls: r.functionCalls,
  flag: r.flag,
})

describe('the root finders are runs of their algorithms', () => {
  it('give the numbers they gave before, exactly', () => {
    expect(brief(R.bisect(f1, 1, 2))).toEqual(BEFORE.bisect)
    expect(brief(R.brentq(Math.cos, 0, 2))).toEqual(BEFORE.brentq)
    expect(brief(R.ridder(f3, 0, 3))).toEqual(BEFORE.ridder)
    expect(brief(R.toms748(f1, 1, 2))).toEqual(BEFORE.toms)
    expect(brief(R.newton(f1, 1.5, { fprime: d1 }))).toEqual(BEFORE.newton)
    expect(brief(R.secant(f3, 0, 1))).toEqual(BEFORE.secant)
    expect(brief(R.newton(f1, 1.5))).toEqual(BEFORE.newtonSec)
    expect(brief(R.halley(f1, 1.5, d1, (x) => 6 * x))).toEqual(BEFORE.halley)
    expect(brief(R.bisect(f1, 1, 2, { maxIterations: 5 }))).toEqual(BEFORE.bisectMax)
    expect(brief(R.brentq(f3, 0, 3, { maxIterations: 3 }))).toEqual(BEFORE.brentMax)
    expect(brief(R.toms748(f3, 0, 3))).toEqual(BEFORE.toms2)
    expect(brief(R.ridder(f3, 0, 3, { maxIterations: 2 }))).toEqual(BEFORE.ridMax)
    expect(brief(R.newton(f1, 1.5, { fprime: d1, maxIterations: 2 }))).toEqual(BEFORE.newtonMax)
    expect(brief(R.newton((x) => x * x + 1, 0, { fprime: (x) => 2 * x }))).toEqual(BEFORE.newtonZero)
    expect(brief(R.bisect((x) => x, -1, 1))).toEqual(BEFORE.bisectEnd)
  })

  it('keeps the history of evaluations as the function returned it', () => {
    const r = R.bisect(f1, 1, 2)
    expect(r.history).toHaveLength(41)
    expect(r.history[0]).toEqual({ x: 1, fx: -1 })
    expect(r.history[40]?.fx).toBe(-1.0405010186786967e-12)
  })

  it('rejects a bracket without a sign change', () => {
    for (const make of [R.bisectAlgorithm, R.brentqAlgorithm, R.ridderAlgorithm, R.toms748Algorithm])
      expect(() =>
        runAlgorithm(
          make((x) => x * x + 1, -1, 1),
          { maxSteps: 3 },
        ),
      ).toThrow()
  })
})

type Case = { name: string; make: () => ReturnType<typeof R.bisectAlgorithm>; bracketed: boolean }
const cases: Case[] = [
  { name: 'bisect', make: () => R.bisectAlgorithm(f1, 1, 2), bracketed: true },
  { name: 'brentq', make: () => R.brentqAlgorithm(f3, 0, 3), bracketed: true },
  { name: 'ridder', make: () => R.ridderAlgorithm(f3, 0, 3), bracketed: true },
  { name: 'toms748', make: () => R.toms748Algorithm(f1, 1, 2), bracketed: true },
  { name: 'newton', make: () => R.newtonAlgorithm(f1, 1.5, { fprime: d1 }), bracketed: false },
  { name: 'halley', make: () => R.halleyAlgorithm(f1, 1.5, d1, (x) => 6 * x), bracketed: false },
  { name: 'secant', make: () => R.secantAlgorithm(f3, 0, 1), bracketed: false },
]

describe('the root algorithms: state and paths', () => {
  for (const c of cases) {
    it(`${c.name}: a state that is JSON resumes to the same end`, () => {
      const alg = c.make()
      const whole = runAlgorithm(alg, { maxSteps: 100 })
      const half = runAlgorithm(alg, { maxSteps: 2 }).state
      const stored = JSON.parse(JSON.stringify(alg.serialise?.(half)))
      const rest = runAlgorithm(alg, { maxSteps: 98, start: { state: stored, step: 2, serialised: true } })
      expect(stateToJson(rest.state)).toEqual(stateToJson(whole.state))
      expect(rest.status).toBe(whole.status)
    })

    it(`${c.name}: logs the estimate, f, the step and the rule, and the bracket when there is one`, () => {
      const rec = recorder()
      runAlgorithm(c.make(), {
        maxSteps: 3,
        tracer: rec,
        policy: { ...policies.minimal, metrics: true, tensors: true, messages: true, roles: ['state'], detail: 'full' },
      })
      const metric = (name: string): number[] =>
        rec.events.flatMap((e) => (e.type === 'metric' && e.name === name ? [e.value] : []))
      expect(metric('root/x')).toHaveLength(3)
      expect(metric('root/step')).toHaveLength(3)
      expect(metric('root/fx').length).toBeGreaterThan(0)
      expect(metric('root/bracket-width')).toHaveLength(c.bracketed ? 3 : 0)
      const rules = rec.events.filter((e) => e.type === 'message' && e.kind === 'root/method')
      expect(rules).toHaveLength(3)
    })
  }

  it('a bracket always contains the root, and its width never grows', () => {
    fc.assert(
      fc.property(fc.double({ min: 0.1, max: 3, noNaN: true }), fc.double({ min: 4, max: 9, noNaN: true }), (r, b) => {
        for (const make of [R.bisectAlgorithm, R.brentqAlgorithm, R.toms748Algorithm]) {
          let width = Number.POSITIVE_INFINITY
          const alg = make((x) => x - r, 0, b)
          let s = alg.init(undefined as never)
          for (let i = 0; i < 60 && s.flag === null; i++) {
            s = alg.step(s, ctxOf())
            expect(s.lo).not.toBeNull()
            expect(s.lo as number).toBeLessThanOrEqual(r + 1e-9)
            expect(s.hi as number).toBeGreaterThanOrEqual(r - 1e-9)
            const w = (s.hi as number) - (s.lo as number)
            expect(w).toBeLessThanOrEqual(width + 1e-12)
            width = w
          }
        }
      }),
      { numRuns: 30 },
    )
  })

  it('bisection halves its bracket exactly at every step', () => {
    const alg = R.bisectAlgorithm(f1, 1, 2)
    let s = alg.init(undefined as never)
    for (let i = 1; i <= 10; i++) {
      s = alg.step(s, ctxOf())
      expect((s.hi as number) - (s.lo as number)).toBeCloseTo(2 ** -i, 12)
    }
  })
})

function ctxOf(): Parameters<ReturnType<typeof R.bisectAlgorithm>['step']>[1] {
  return { t: 0, wants: () => false, log: () => undefined, emit: () => undefined } as never
}

describe('rootFinders: the slot', () => {
  const problem: R.RootProblem = { f: f1, bracket: [1, 2], x0: 1.5, fprime: d1, fprime2: (x) => 6 * x }
  for (const kind of ['bisect', 'brentq', 'ridder', 'toms748', 'newton', 'halley', 'secant']) {
    it(`${kind} runs and finds the root`, () => {
      const alg = R.rootAlgorithm({ kind }, problem)
      const r = R.solveRoot(alg, 200)
      expect(r.converged).toBe(true)
      expect(Math.abs(r.root - 1.324717957244746)).toBeLessThan(1e-7)
      expect(alg.name).toBe(kind)
    })
  }

  it('takes the knobs of its option, and names them', () => {
    const loose = R.solveRoot(R.rootAlgorithm(R.rootFinderValue('bisect', { xtol: 1e-3 }), problem), 100)
    const tight = R.solveRoot(R.rootAlgorithm(R.rootFinderValue('bisect'), problem), 100)
    expect(loose.iterations).toBeLessThan(tight.iterations)
    expect(() => R.rootFinderValue('bisect', { tol: 1 })).toThrow()
    expect(() => R.rootFinderValue('nope')).toThrow()
    expect(R.rootFinders.where((p) => p.bracketed)).toEqual(['bisect', 'brentq', 'ridder', 'toms748'])
    expect(R.rootFinders.where((p) => p.derivatives === 0 && !p.bracketed)).toEqual(['secant'])
  })

  it('says what a method needs when the problem lacks it', () => {
    expect(() => R.rootAlgorithm({ kind: 'brentq' }, { f: f1, x0: 1 })).toThrow(/bracket/)
    expect(() => R.rootAlgorithm({ kind: 'newton' }, { f: f1, x0: 1 })).toThrow(/fprime/)
    expect(() => R.rootAlgorithm({ kind: 'halley' }, { f: f1, x0: 1, fprime: d1 })).toThrow(/fprime2/)
  })
})
