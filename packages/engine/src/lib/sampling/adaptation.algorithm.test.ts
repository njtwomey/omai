import { describe, expect, it } from 'vitest'
import { standardNormal } from '../../testing/sampling.ts'
import { key } from '../random/index.ts'
import { policies, recorder, runAlgorithm } from '../trace/index.ts'
import { adaptContract } from './adaptation-contract.ts'
import { randomWalkAdaptations, windowAdaptationForms } from './adaptation-options.ts'
import { sampleChains } from './chains.ts'
import {
  findReasonableStepSize,
  findReasonableStepSizeAlgorithm,
  windowAdaptation,
  windowAdaptationAlgorithm,
  windowAdaptationResult,
} from './hamiltonian/adaptation.ts'
import { hmcKernel } from './hamiltonian/hmc.ts'
import { randomWalkAdaptation, randomWalkAdaptationAlgorithm, randomWalkAdaptationKernel } from './metropolis.ts'
import { numericTarget } from './target.ts'

const viaJson = <T>(x: T): T => JSON.parse(JSON.stringify(x)) as T
const narrow = numericTarget((x) => (-0.5 * (x[0] ?? 0) ** 2) / 0.01, 1, { grad: (x) => [-(x[0] ?? 0) / 0.01] })
const kernelFor = (stepSize: number) => hmcKernel(narrow, { stepSize, numIntegrationSteps: 1 })
const wide = numericTarget((x) => (-0.5 * (x[0] ?? 0) ** 2) / 4, 1, { grad: (x) => [-(x[0] ?? 0) / 4] })
const hmcFor = (p: { stepSize: number; inverseMassMatrix: Float64Array }) =>
  hmcKernel(wide, { ...p, numIntegrationSteps: 5 })

describe('findReasonableStepSizeAlgorithm', () => {
  it('tries one step size per step, flips direction once and agrees with the function', () => {
    const alg = findReasonableStepSizeAlgorithm(kernelFor, kernelFor(1).init([0.05]), { key: key(0) })
    const run = runAlgorithm(alg, { maxSteps: 100 })
    expect(run.status).toBe('converged')
    expect(run.steps).toBe(run.state.tries)
    expect(run.state.stepSize).toBe(findReasonableStepSize(kernelFor, kernelFor(1).init([0.05]), { key: key(0) }))
    expect(run.state.stepSize).toBeLessThan(0.6)
  })

  it('serialises, resumes, and logs the size, acceptance and direction', () => {
    const alg = findReasonableStepSizeAlgorithm(kernelFor, kernelFor(1).init([0.05]), { key: key(0) })
    const whole = runAlgorithm(alg, { maxSteps: 100 })
    const part = runAlgorithm(alg, { maxSteps: 2 }).state
    const rest = runAlgorithm(alg, {
      maxSteps: 100,
      start: { state: viaJson(alg.serialise?.(part) ?? null), step: 2, serialised: true },
    })
    expect(rest.state).toEqual(whole.state)
    const rec = recorder()
    runAlgorithm(alg, { maxSteps: 100, tracer: rec, policy: { ...policies.minimal, metrics: true, detail: 'full' } })
    const sizes = rec.events.flatMap((e) => (e.type === 'metric' && e.name === 'adapt/stepSize' ? [e.value] : []))
    expect(sizes).toHaveLength(whole.steps)
    expect(sizes[0]).toBe(1)
  })
})

describe('windowAdaptationAlgorithm', () => {
  const alg = windowAdaptationAlgorithm(1, hmcFor, Float64Array.of(0), { numWarmup: 120 })

  it('adapts the step size each iteration and ends slow windows with a mass matrix', () => {
    const run = runAlgorithm(alg, { key: key(1), maxSteps: 120 })
    expect(run.steps).toBe(120)
    expect(run.state.windows).toBeGreaterThan(0)
    const tuned = windowAdaptationResult(run.state)
    expect(tuned.stepSize).toBeGreaterThan(0)
    // The target has variance 4: the learned inverse mass matrix moves from the identity towards it.
    expect(tuned.inverseMassMatrix[0]).toBeGreaterThan(1)
  })

  it('gives sampleChains the same parameters as the Adaptation wrapper', () => {
    const c = sampleChains(windowAdaptation(1, hmcFor), { key: key(1), numChains: 1, numWarmup: 120, numDraws: 5 })
    const run = runAlgorithm(alg, { key: key(0), maxSteps: 120 })
    expect(Object.keys(c.parameters[0] as object).sort()).toEqual(['inverseMassMatrix', 'stepSize'])
    expect(windowAdaptationResult(run.state).inverseMassMatrix).toHaveLength(1)
  })

  it('serialises and resumes to the same end', () => {
    const whole = runAlgorithm(alg, { key: key(2), maxSteps: 120 })
    const part = runAlgorithm(alg, { key: key(2), maxSteps: 60 }).state
    const stored = viaJson(alg.serialise?.(part) ?? null)
    const rest = runAlgorithm(alg, { key: key(2), maxSteps: 120, start: { state: stored, step: 60, serialised: true } })
    expect(rest.state.windows).toBe(whole.state.windows)
    expect(rest.state.stepSize).toBeCloseTo(whole.state.stepSize, 9)
    expect(rest.state.chain.logDensity).toBeCloseTo(whole.state.chain.logDensity, 9)
    expect(Array.from(rest.state.mm.inverseMassMatrix)).toEqual(Array.from(whole.state.mm.inverseMassMatrix))
  })

  it('logs the step size, acceptance, window and a message per closed window', () => {
    const rec = recorder()
    const run = runAlgorithm(alg, {
      key: key(1),
      maxSteps: 120,
      tracer: rec,
      policy: { ...policies.minimal, metrics: true, detail: 'full' },
    })
    const metric = (name: string): number[] =>
      rec.events.flatMap((e) => (e.type === 'metric' && e.name === name ? [e.value] : []))
    expect(metric('adapt/stepSize')).toHaveLength(120)
    expect(metric('adapt/accept').every((a) => a >= 0 && a <= 1)).toBe(true)
    expect(Math.max(...metric('adapt/window'))).toBe(run.state.windows)
    expect(Object.keys(adaptContract)).toHaveLength(9)
  })
})

describe('randomWalkAdaptationAlgorithm', () => {
  const alg = randomWalkAdaptationAlgorithm(standardNormal(2), Float64Array.of(0, 0), { numWarmup: 200 })

  it('tunes the scale then learns a shape in the second half', () => {
    const run = runAlgorithm(alg, { key: key(3), maxSteps: 200 })
    expect(run.steps).toBe(200)
    expect(run.state.shape).toBeDefined()
    expect(run.state.history.count).toBe(100)
    expect(typeof randomWalkAdaptationKernel(standardNormal(2), run.state).step).toBe('function')
  })

  it('serialises and resumes to the same end', () => {
    const whole = runAlgorithm(alg, { key: key(3), maxSteps: 200 })
    const part = runAlgorithm(alg, { key: key(3), maxSteps: 90 }).state
    const stored = viaJson(alg.serialise?.(part) ?? null)
    const rest = runAlgorithm(alg, { key: key(3), maxSteps: 200, start: { state: stored, step: 90, serialised: true } })
    expect(rest.state.logScale).toBeCloseTo(whole.state.logScale, 9)
    expect(Array.from(rest.state.shape ?? [])).toEqual(Array.from(whole.state.shape ?? []))
  })

  it('leaves the Adaptation wrapper working and rejects a bad target acceptance', () => {
    const c = sampleChains(randomWalkAdaptation(standardNormal(2)), {
      key: key(4),
      numChains: 1,
      numWarmup: 100,
      numDraws: 5,
    })
    expect(typeof (c.parameters[0] as { scale: number }).scale).toBe('number')
    expect(() =>
      randomWalkAdaptationAlgorithm(standardNormal(2), Float64Array.of(0, 0), { numWarmup: 5, targetAcceptance: 0 }),
    ).toThrow()
  })
})

describe('adaptation option tables', () => {
  it('names the forms and builds options', () => {
    expect(windowAdaptationForms.names).toEqual(['diagonal', 'dense'])
    expect(windowAdaptationForms.make({ kind: 'dense' }).dense).toBe(true)
    expect(windowAdaptationForms.make({ kind: 'diagonal', targetAcceptance: 0.9 }).targetAcceptance).toBe(0.9)
    expect(randomWalkAdaptations.make({ kind: 'scale-only' }).adaptCovariance).toBe(false)
  })
})
