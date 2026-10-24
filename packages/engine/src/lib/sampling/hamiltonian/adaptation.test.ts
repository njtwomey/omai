import { describe, expect, it } from 'vitest'
import { gaussian2, standardNormal } from '../../../testing/sampling.ts'
import { key } from '../../random/index.ts'
import { messages, recorder, tracePolicy } from '../../trace/index.ts'
import { sampleChains } from '../chains.ts'
import { numericTarget } from '../target.ts'
import {
  dualAveragingFinal,
  dualAveragingInit,
  dualAveragingUpdate,
  findReasonableStepSize,
  massMatrixFinal,
  massMatrixInit,
  massMatrixUpdate,
  windowAdaptation,
  windowSchedule,
} from './adaptation.ts'
import { hmcKernel } from './hmc.ts'
import { nutsKernel } from './nuts.ts'

describe('dual averaging', () => {
  it('converges to the step size whose acceptance rate is the target', () => {
    // A deterministic acceptance curve alpha(eps) = exp(-eps): the target 0.8 is reached at eps = -log 0.8.
    let s = dualAveragingInit(1)
    for (let t = 0; t < 3000; t++)
      s = dualAveragingUpdate(s, Math.exp(-Math.exp(s.logStepSize)), { targetAcceptance: 0.8 })
    expect(dualAveragingFinal(s)).toBeCloseTo(-Math.log(0.8), 2)
  })

  it('treats a NaN acceptance rate as 0 and rejects bad starting step sizes', () => {
    const s = dualAveragingUpdate(dualAveragingInit(1), Number.NaN)
    expect(s).toEqual(dualAveragingUpdate(dualAveragingInit(1), 0))
    expect(() => dualAveragingInit(0)).toThrow()
    expect(() => dualAveragingInit(Number.POSITIVE_INFINITY)).toThrow()
  })
})

describe('findReasonableStepSize', () => {
  it('finds a step size on the scale of a narrow target', () => {
    const narrow = numericTarget((x) => (-0.5 * (x[0] ?? 0) ** 2) / 0.01, 1, { grad: (x) => [-(x[0] ?? 0) / 0.01] })
    const kernelFor = (stepSize: number) => hmcKernel(narrow, { stepSize, numIntegrationSteps: 1 })
    const eps = findReasonableStepSize(kernelFor, kernelFor(1).init([0.05]), { key: key(0) })
    expect(eps).toBeGreaterThan(0.01)
    expect(eps).toBeLessThan(0.6)
  })

  it('stops on degenerate targets: always accepting (flat) or never (NaN)', () => {
    const flat = numericTarget(() => 0, 1, { grad: () => [0] })
    const f = (stepSize: number) => hmcKernel(flat, { stepSize, numIntegrationSteps: 1 })
    expect(findReasonableStepSize(f, f(1).init([0]), { key: key(1), maxIter: 40 })).toBe(2 ** 39)
    const broken = numericTarget(() => Number.NaN, 1, { grad: () => [Number.NaN] })
    const g = (stepSize: number) => hmcKernel(broken, { stepSize, numIntegrationSteps: 1 })
    const eps = findReasonableStepSize(g, g(1).init([0]), { key: key(2) })
    expect(eps).toBeGreaterThan(0)
    expect(eps).toBeLessThan(1e-20)
  })
})

describe('mass-matrix adaptation', () => {
  it("regularises a window's covariance towards 1e-3 I (Stan's formula), densely and symmetrically", () => {
    let s = massMatrixInit(2, { dense: true })
    const pts = [
      [0, 0],
      [1, 2],
      [2, 1],
      [3, 4],
    ]
    for (const p of pts) s = massMatrixUpdate(s, p)
    const m = massMatrixFinal(s).inverseMassMatrix
    // Covariance [[5/3, 5.5/3], [5.5/3, 35/12]] of the four points, weighted 4/9, plus 1e-3 5/9 on the diagonal.
    expect(m[0]).toBeCloseTo((4 / 9) * (5 / 3) + (5 / 9) * 1e-3, 12)
    expect(m[1]).toBeCloseTo((4 / 9) * (5.5 / 3), 12)
    expect(m[1]).toBe(m[2])
    expect(massMatrixFinal(s).count).toBe(0)
  })

  it('starts from a given matrix and checks its shape', () => {
    expect(Array.from(massMatrixInit(2, { initial: [2, 3] }).inverseMassMatrix)).toEqual([2, 3])
    expect(() => massMatrixInit(2, { initial: [1, 2, 3] })).toThrow()
  })
})

describe('windowSchedule', () => {
  it('is all fast below 20 iterations and shrinks the buffers when they do not fit', () => {
    expect(windowSchedule(19).every((s) => !s.slow)).toBe(true)
    const s = windowSchedule(100)
    // Buffers of 15 and 10 around one slow window of 75.
    expect(s.filter((x) => x.slow).length).toBe(75)
    expect(s.flatMap((x, i) => (x.windowEnd ? [i] : []))).toEqual([89])
    expect(() => windowSchedule(-1)).toThrow()
  })
})

describe('windowAdaptation', () => {
  it('tunes NUTS so the sampling acceptance rate is near the target', () => {
    const target = gaussian2([0, 0], [1, 3], 0.5)
    const adapt = windowAdaptation(2, (p) => nutsKernel(target, p), { targetAcceptance: 0.8 })
    const c = sampleChains(adapt, { key: key(3), numChains: 2, numWarmup: 300, numDraws: 300 })
    const a = c.stats.acceptanceRate?.data ?? new Float64Array()
    const mean = a.reduce((x, y) => x + y, 0) / a.length
    expect(mean).toBeGreaterThan(0.65)
    expect(mean).toBeLessThan(0.95)
  })

  it('returns the starting kernel without warm-up, and traces window ends', () => {
    const t = standardNormal(2)
    const adapt = windowAdaptation(2, (p) => hmcKernel(t, { ...p, numIntegrationSteps: 3 }), { initialStepSize: 0.3 })
    const w0 = adapt.run(Float64Array.from([0, 0]), { key: key(4), numWarmup: 0 })
    expect((w0.parameters as { stepSize: number }).stepSize).toBe(0.3)
    const rec = recorder()
    sampleChains(adapt, {
      key: key(5),
      numChains: 1,
      numWarmup: 150,
      numDraws: 5,
      tracer: rec,
      policy: tracePolicy({ messages: true }),
    })
    // 150 iterations: buffers of 22 and 15 around one slow window, so one window end.
    expect(messages(rec.run(), 'windowEnd').length).toBe(1)
  })

  it('keeps warm-up draws when asked', () => {
    const t = standardNormal(1)
    const adapt = windowAdaptation(1, (p) => hmcKernel(t, { ...p, numIntegrationSteps: 3 }))
    const c = sampleChains(adapt, { key: key(6), numChains: 2, numWarmup: 30, numDraws: 10, keepWarmup: true })
    expect(c.warmup?.draws.shape).toEqual([2, 30, 1])
  })
})
