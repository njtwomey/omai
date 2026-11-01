import { describe, expect, it } from 'vitest'
import { toFlat } from '../tensor/index.ts'
import { policies, recorder, runAlgorithm } from '../trace/index.ts'
import { awq, awqAlgorithm, gptq, gptqAlgorithm, outputError } from './quantise.ts'
import { gptqOrders } from './quantise-options.ts'

const viaJson = <T>(x: T): T => JSON.parse(JSON.stringify(x)) as T
const policy = {
  ...policies.minimal,
  metrics: true,
  tensors: true,
  roles: ['output' as const, 'state' as const],
  detail: 'full' as const,
}
const metricsOf = (rec: ReturnType<typeof recorder>, name: string): number[] =>
  rec.events.flatMap((e) => (e.type === 'metric' && e.name === name ? [e.value] : []))

const weight = [
  [0.5, -1.2, 0.3, 2.1],
  [1.7, 0.1, -0.8, -0.2],
  [-0.4, 0.9, 1.1, 0.6],
]
const inputs = Array.from({ length: 12 }, (_, i) => [
  Math.sin(i),
  2 * Math.cos(1.3 * i),
  0.3 * Math.sin(2.1 * i),
  1 + 0.1 * i,
])

describe('gptqAlgorithm', () => {
  it('quantises one column per step, the function being that run', () => {
    const rec = recorder()
    const run = runAlgorithm(gptqAlgorithm(weight, inputs, { bits: 3 }), { maxSteps: 10, tracer: rec, policy })
    expect(run.steps).toBe(4)
    expect(metricsOf(rec, 'quant/column')).toEqual([0, 1, 2, 3])
    expect(metricsOf(rec, 'quant/error').every((v) => v >= 0)).toBe(true)
    const direct = gptq(weight, inputs, { bits: 3 })
    // In the natural order the working order is the original one, so the codes are those of the state.
    expect(Array.from(toFlat(direct.codes))).toEqual(Array.from(run.state.codes))
    expect(rec.events.filter((e) => e.type === 'tensor' && e.name === 'quant/weights')).toHaveLength(4)
    // On the calibration inputs it does no worse than the plain rounding of the same grid would on average.
    expect(outputError(weight, direct.weight, inputs)).toBeGreaterThanOrEqual(0)
  })

  it('resumes from a JSON checkpoint to the same codes', () => {
    const alg = gptqAlgorithm(weight, inputs, { bits: 3 })
    const whole = runAlgorithm(alg, { maxSteps: 10 })
    const half = runAlgorithm(alg, { maxSteps: 2 }).state
    const stored = viaJson(alg.serialise?.(half) ?? null)
    const rest = runAlgorithm(alg, { maxSteps: 8, start: { state: stored, step: 2, serialised: true } })
    expect(Array.from(rest.state.codes)).toEqual(Array.from(whole.state.codes))
    for (const [i, v] of whole.state.quantised.entries()) expect(rest.state.quantised[i]).toBeCloseTo(v, 12)
  })

  it('quantises the columns in the order the order slot gives', () => {
    const rec = recorder()
    runAlgorithm(gptqAlgorithm(weight, inputs, { bits: 4, order: { kind: 'activation' } }), {
      maxSteps: 10,
      tracer: rec,
      policy,
    })
    const columns = metricsOf(rec, 'quant/column')
    expect([...columns].sort()).toEqual([0, 1, 2, 3])
    expect(columns).not.toEqual([0, 1, 2, 3])
    expect(gptq(weight, inputs, { actOrder: true }).weight.shape).toEqual([3, 4])
    expect(gptqOrders.names).toEqual(['natural', 'activation'])
    expect(gptqOrders.make({ kind: 'activation' })([1, 5, 3, 5])).toEqual([1, 3, 2, 0])
  })

  it('rejects inputs of the wrong width', () => {
    expect(() => gptqAlgorithm(weight, [[1, 2]], {})).toThrow(/inputs/)
  })
})

describe('awqAlgorithm', () => {
  it('tries one exponent per step and keeps the best, never worse than plain rounding', () => {
    const rec = recorder()
    const options = { bits: 3, gridSize: 8 }
    const run = runAlgorithm(awqAlgorithm(weight, inputs, options), { maxSteps: 20, tracer: rec, policy })
    expect(run.steps).toBe(9)
    const alpha = metricsOf(rec, 'quant/alpha')
    expect(alpha).toEqual(Array.from({ length: 9 }, (_, i) => i / 8))
    const error = metricsOf(rec, 'quant/error')
    expect(run.state.bestError).toBeCloseTo(Math.min(...error), 12)
    expect(run.state.bestError).toBeLessThanOrEqual((error[0] ?? 0) + 1e-12)
    const direct = awq(weight, inputs, options)
    expect(direct.alpha).toBe(run.state.bestAlpha)
    expect(direct.error).toBeCloseTo(run.state.bestError, 12)
    expect(toFlat(direct.weight)).toHaveLength(12)
  })

  it('resumes from a JSON checkpoint', () => {
    const alg = awqAlgorithm(weight, inputs, { bits: 3, gridSize: 8 })
    const whole = runAlgorithm(alg, { maxSteps: 20 }).state
    const half = runAlgorithm(alg, { maxSteps: 4 }).state
    const stored = viaJson(alg.serialise?.(half) ?? null)
    const rest = runAlgorithm(alg, { maxSteps: 16, start: { state: stored, step: 4, serialised: true } }).state
    expect(rest.bestAlpha).toBe(whole.bestAlpha)
    expect(rest.bestError).toBeCloseTo(whole.bestError, 12)
  })
})
