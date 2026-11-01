import { describe, expect, it } from 'vitest'
import { toFlat } from '../../tensor/index.ts'
import { policies, recorder, runAlgorithm, stateToJson } from '../../trace/index.ts'
import * as L from './eig.ts'
import { eigenIteration, eigenIterations } from './eig-options.ts'
import { eigh } from './eigh.ts'

const A = [
  [4, 1, 0.5],
  [1, 3, 0.2],
  [0.5, 0.2, 1],
]

// What the functions returned before they became runs of the algorithms (taken from the previous implementation).
const BEFORE = {
  power: { value: 4.6942825492605476, it: 20, conv: true },
  powerKey: { value: 4.694282549260463, it: 22, conv: true },
  powerMax: { value: 4.683865993384312, it: 3, conv: false },
  inverse: { value: 0.9180705192718636, it: 10, conv: true },
  inverseMax: { value: 0.9191818827218753, it: 2, conv: false },
  rqi: { value: 2.3876469314665574, it: 5, conv: true },
  rqiMax: { value: 2.5073736827972875, it: 1, conv: false },
  rqiExact: { value: 2.9999999999999996, it: 1, conv: true },
} as const

const brief = (r: L.EigenIterationResult): { value: number; it: number; conv: boolean } => ({
  value: r.value,
  it: r.iterations,
  conv: r.converged,
})

describe('the eigenvalue iterations are runs of their algorithms', () => {
  it('give the numbers they gave before, exactly', () => {
    expect(brief(L.powerIteration(A, { x0: [1, 0.3, 0.2] }))).toEqual(BEFORE.power)
    expect(brief(L.powerIteration(A))).toEqual(BEFORE.powerKey)
    expect(brief(L.powerIteration(A, { x0: [1, 0.3, 0.2], maxIterations: 3 }))).toEqual(BEFORE.powerMax)
    expect(brief(L.inverseIteration(A, 1.2, { x0: [1, 1, 1] }))).toEqual(BEFORE.inverse)
    expect(brief(L.inverseIteration(A, 1.2, { x0: [1, 1, 1], maxIterations: 2 }))).toEqual(BEFORE.inverseMax)
    expect(brief(L.rayleighQuotientIteration(A, { x0: [0, 1, 0.1] }))).toEqual(BEFORE.rqi)
    expect(brief(L.rayleighQuotientIteration(A, { x0: [0, 1, 0.1], maxIterations: 1 }))).toEqual(BEFORE.rqiMax)
    expect(
      brief(
        L.rayleighQuotientIteration(
          [
            [2, 1],
            [1, 2],
          ],
          { x0: [1, 1] },
        ),
      ),
    ).toEqual(BEFORE.rqiExact)
    expect(L.powerIteration(A, { x0: [1, 0.3, 0.2] }).history).toHaveLength(20)
  })
})

const algorithms = {
  power: () => L.powerIterationAlgorithm(A, { x0: [1, 0.3, 0.2] }),
  inverse: () => L.inverseIterationAlgorithm(A, 1.2, { x0: [1, 1, 1] }),
  rayleigh: () => L.rayleighQuotientIterationAlgorithm(A, { x0: [0, 1, 0.1] }),
} as const

describe('the eigenvalue algorithms: state, paths, invariants', () => {
  for (const [name, make] of Object.entries(algorithms)) {
    it(`${name}: a state that is JSON resumes to the same end`, () => {
      const alg = make()
      const whole = runAlgorithm(alg, { maxSteps: 100 })
      const half = runAlgorithm(alg, { maxSteps: 3 }).state
      const stored = JSON.parse(JSON.stringify(alg.serialise?.(half)))
      const rest = runAlgorithm(alg, { maxSteps: 97, start: { state: stored, step: 3, serialised: true } })
      expect(stateToJson(rest.state)).toEqual(stateToJson(whole.state))
      expect(whole.status).toBe('converged')
    })

    it(`${name}: logs the vector, the estimate, the residual and the angle each step`, () => {
      const rec = recorder()
      runAlgorithm(make(), {
        maxSteps: 4,
        tracer: rec,
        policy: { ...policies.minimal, metrics: true, tensors: true, roles: ['output'], detail: 'full' },
      })
      const metric = (n: string): number[] =>
        rec.events.flatMap((e) => (e.type === 'metric' && e.name === n ? [e.value] : []))
      expect(metric('eig/value')).toHaveLength(4)
      expect(metric('eig/residual')).toHaveLength(4)
      expect(metric('eig/angle')).toHaveLength(4)
      expect(rec.events.filter((e) => e.type === 'tensor' && e.name === 'eig/vector')).toHaveLength(4)
    })
  }

  it('power iteration: the residual shrinks and the estimate approaches the largest eigenvalue', () => {
    const rec = recorder()
    runAlgorithm(algorithms.power(), {
      maxSteps: 30,
      tracer: rec,
      policy: { ...policies.minimal, metrics: true, detail: 'full' },
    })
    const residual = rec.events.flatMap((e) => (e.type === 'metric' && e.name === 'eig/residual' ? [e.value] : []))
    expect(residual.at(-1) as number).toBeLessThan(1e-5)
    expect((residual.at(-1) as number) < (residual[0] as number)).toBe(true)
    const largest = Math.max(...toFlat(eigh(A).eigenvalues as never))
    expect(
      Math.abs((runAlgorithm(algorithms.power(), { maxSteps: 100 }).state.value as number) - largest),
    ).toBeLessThan(1e-9)
  })

  it('every vector in the state is a unit vector', () => {
    const s = runAlgorithm(algorithms.rayleigh(), { maxSteps: 3 }).state
    expect(Math.hypot(...s.x)).toBeCloseTo(1, 12)
  })

  it('rejects a matrix that is not square', () => {
    expect(() => L.powerIterationAlgorithm([[1, 2, 3]])).toThrow()
  })
})

describe('eigenIterations: the slot', () => {
  it('each option runs and finds an eigenvalue of the matrix', () => {
    const values = toFlat(eigh(A).eigenvalues as never)
    for (const kind of eigenIterations.names) {
      const s = runAlgorithm(eigenIteration({ kind, shift: 1.2 }, { a: A, x0: [1, 0.3, 0.2] }), {
        maxSteps: 1000,
      }).state
      expect(Math.min(...values.map((v) => Math.abs(v - (s.value as number))))).toBeLessThan(1e-8)
    }
  })

  it('names what each finds', () => {
    expect(eigenIterations.where((p) => p.solves)).toEqual(['inverse', 'rayleigh'])
    expect(eigenIterations.properties({ kind: 'power' }).finds).toBe('largest magnitude')
  })
})
