import { describe, expect, it } from 'vitest'
import { defineEstimator, rowsOf } from '../learn/index.ts'
import { key } from '../random/index.ts'
import { full, item, matmul, mean, sigmoid, tensor, type Value } from '../tensor/index.ts'
import { type Algorithm, policies, recorder, runAlgorithm } from '../trace/index.ts'
import {
  anchors,
  anchorsAlgorithm,
  dataShapley,
  dataShapleyAlgorithm,
  diverseCounterfactuals,
  diverseCounterfactualsAlgorithm,
  faceCounterfactual,
  faceCounterfactualAlgorithm,
  growingSpheres,
  growingSpheresAlgorithm,
  permutationImportance,
  permutationImportanceAlgorithm,
  prototypeCounterfactual,
  prototypeCounterfactualAlgorithm,
  sage,
  sageAlgorithm,
  samplingShapley,
  samplingShapleyAlgorithm,
  shapleyEstimates,
} from './index.ts'

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

/** Run to the end, and again from a JSON checkpoint taken after `cut` steps. */
function resumed<S>(alg: Algorithm<S>, cut: number, maxSteps = 5000): { whole: S; rest: S; steps: [number, number] } {
  const whole = runAlgorithm(alg, { maxSteps })
  const half = runAlgorithm(alg, { maxSteps: cut }).state
  const stored = viaJson(alg.serialise?.(half) ?? null)
  const rest = runAlgorithm(alg, { maxSteps: maxSteps - cut, start: { state: stored, step: cut, serialised: true } })
  return { whole: whole.state, rest: rest.state, steps: [whole.steps, rest.steps] }
}

describe('diverseCounterfactualsAlgorithm', () => {
  const proba = (v: Value): Value => sigmoid(matmul(v, [[2], [2]]))

  it('is what diverseCounterfactuals runs, with the objective and validity on the tape', () => {
    const rec = recorder()
    const alg = diverseCounterfactualsAlgorithm(proba, [-1, -1], { count: 3 })
    const run = runAlgorithm(alg, { maxSteps: 60, tracer: rec, policy })
    expect(run.steps).toBe(60)
    const r = diverseCounterfactuals(proba, [-1, -1], { count: 3, maxIterations: 60 })
    expect(Array.from(run.state.c)).toEqual(r.points.flat())
    const loss = metricsOf(rec, 'dice/loss')
    expect(loss).toHaveLength(60)
    expect(loss.at(-1)).toBeLessThan(loss[0] ?? 0)
    expect(metricsOf(rec, 'dice/validity').at(-1)).toBe(r.valid.filter(Boolean).length)
    expect(metricsOf(rec, 'dice/iteration').at(-1)).toBe(60)
  })

  it('resumes from a JSON checkpoint', () => {
    const { whole, rest } = resumed(diverseCounterfactualsAlgorithm(proba, [-1, -1], { count: 2 }), 10, 40)
    expect(Array.from(rest.c)).toEqual(Array.from(whole.c))
    expect(rest.t).toBe(whole.t)
  })
})

describe('growingSpheresAlgorithm', () => {
  const f = (x: { shape: number[]; data: ArrayLike<number> }) =>
    tensor(Array.from({ length: x.shape[0] ?? 0 }, (_, i) => x.data[2 * i] ?? 0))
  const options = { accept: (p: number) => p > 1, key: key(4) }

  it('shrinks, grows and sparsifies, and ends with the counterfactual growingSpheres returns', () => {
    const rec = recorder()
    const run = runAlgorithm(growingSpheresAlgorithm(f as never, [0, 0], options), {
      maxSteps: 300,
      tracer: rec,
      policy,
    })
    expect(run.status).toBe('converged')
    const cf = growingSpheres(f as never, [0, 0], options)
    expect(run.state.phase).toBe('done')
    expect(Array.from(run.state.point)).toEqual(cf?.x)
    expect(cf?.valid).toBe(true)
    expect(cf?.x[1]).toBe(0)
    const radius = metricsOf(rec, 'gs/radius')
    expect(radius.length).toBeGreaterThan(1)
    expect(metricsOf(rec, 'gs/enemies').at(-1)).toBeGreaterThan(0)
  })

  it('resumes from a JSON checkpoint in each phase and ends undefined when nothing is an enemy', () => {
    for (const cut of [1, 4, 10]) {
      const alg = growingSpheresAlgorithm(f as never, [0, 0], options)
      const { whole, rest, steps } = resumed(alg, cut, 300)
      expect(rest.point).toEqual(whole.point)
      expect(steps[1]).toBe(steps[0])
    }
    expect(growingSpheres(f as never, [0, 0], { accept: () => false, maxLayers: 3, nSamples: 20 })).toBeUndefined()
  })
})

describe('faceCounterfactualAlgorithm', () => {
  const sign = (x: { shape: number[]; data: ArrayLike<number> }) =>
    tensor(Array.from({ length: x.shape[0] ?? 0 }, (_, i) => ((x.data[i] ?? 0) > 2 ? 1 : 0)))
  const data = [[0], [0.5], [1], [1.5], [2], [2.5], [3]]
  const options = { accept: (p: number) => p === 1, kNeighbors: 2 }

  it('settles nodes in order of cost and stops once the cheapest qualifying point is known', () => {
    const rec = recorder()
    const run = runAlgorithm(faceCounterfactualAlgorithm(sign as never, [0.1], data, options), {
      maxSteps: 20,
      tracer: rec,
      policy,
    })
    expect(run.status).toBe('converged')
    const cost = metricsOf(rec, 'face/cost')
    expect(cost.every((v, i) => i === 0 || v >= (cost[i - 1] ?? 0))).toBe(true)
    expect(run.state.best).toBe(6)
    expect(metricsOf(rec, 'face/best').at(-1)).toBe(6)
    expect(faceCounterfactual(sign as never, [0.1], data, options)?.counterfactual).toEqual([2.5])
  })

  it('resumes from a JSON checkpoint', () => {
    const alg = faceCounterfactualAlgorithm(sign as never, [0.1], data, options)
    const whole = runAlgorithm(alg, { maxSteps: 20 })
    const half = runAlgorithm(alg, { maxSteps: 2 }).state
    const stored = viaJson(alg.serialise?.(half) ?? null)
    const rest = runAlgorithm(alg, { maxSteps: 20, start: { state: stored, step: 2, serialised: true } })
    expect(rest.state.best).toBe(whole.state.best)
    expect(rest.steps).toBe(whole.steps)
  })

  it('finds nothing when no point is acceptable', () => {
    expect(faceCounterfactual(sign as never, [0.1], data, { accept: () => false, kNeighbors: 2 })).toBeUndefined()
  })
})

describe('prototypeCounterfactualAlgorithm', () => {
  const scores = (v: Value): Value =>
    matmul(v, [
      [1, -1],
      [-1, 1],
    ])
  const train = [
    [2, 0],
    [3, 1],
    [0, 2],
    [1, 3],
  ]
  const options = { kPrototype: 2, rounds: 2, iterations: 60 }

  it('runs one FISTA iteration per step and agrees with prototypeCounterfactual', () => {
    const rec = recorder()
    const alg = prototypeCounterfactualAlgorithm(scores, [2, 1], train, [0, 0, 1, 1], options)
    const run = runAlgorithm(alg, { maxSteps: 500, tracer: rec, policy })
    expect(run.steps).toBe(120)
    expect(run.state.best).not.toBeNull()
    const r = prototypeCounterfactual(scores, [2, 1], train, [0, 0, 1, 1], options)
    expect(r?.delta.length).toBe(2)
    for (const [j, v] of (run.state.best ?? []).entries()) expect(v).toBeCloseTo(r?.delta[j] ?? Number.NaN, 12)
    expect(metricsOf(rec, 'search/round').at(-1)).toBe(1)
  })

  it('resumes from a JSON checkpoint at a round boundary and inside a round', () => {
    for (const cut of [30, 60]) {
      const alg = prototypeCounterfactualAlgorithm(scores, [2, 1], train, [0, 0, 1, 1], options)
      const { whole, rest } = resumed(alg, cut, 500)
      expect(rest.best).toEqual(whole.best)
      expect(rest.c).toBe(whole.c)
    }
  })
})

describe('anchorsAlgorithm', () => {
  const firstPositive = (x: { shape: number[]; data: ArrayLike<number> }) =>
    tensor(Array.from({ length: x.shape[0] ?? 0 }, (_, i) => ((x.data[2 * i] ?? 0) > 0 ? 1 : 0)))
  const train = Array.from({ length: 40 }, (_, i) => [((i * 7) % 20) - 10, (i * 3) % 5])

  it('lengthens the rule by one predicate per step and ends with the anchor anchors returns', () => {
    const rec = recorder()
    const run = runAlgorithm(anchorsAlgorithm(firstPositive as never, [8, 1], train, { key: key(2) }), {
      maxSteps: 5,
      tracer: rec,
      policy,
    })
    expect(run.state.best?.rule).toEqual([0])
    expect(metricsOf(rec, 'anchor/length')).toEqual([1])
    expect(metricsOf(rec, 'anchor/precision')[0]).toBeGreaterThan(0.9)
    const a = anchors(firstPositive as never, [8, 1], train, { key: key(2) })
    expect(a.predicates.map((p) => p.feature)).toEqual([0])
    expect(a.samples).toBe(run.state.samples)
  })

  it('resumes from a JSON checkpoint when the first round is not enough', () => {
    const alg = anchorsAlgorithm(firstPositive as never, [8, 1], train, { key: key(2), threshold: 1.01, maxLength: 2 })
    const { whole, rest, steps } = resumed(alg, 1, 5)
    expect(whole.length).toBe(2)
    expect(rest.beam).toEqual(whole.beam)
    expect(steps[1]).toBe(steps[0])
  })
})

describe('permutation-sampling estimators', () => {
  const linear = (x: { shape: number[]; data: ArrayLike<number> }) =>
    tensor(Array.from({ length: x.shape[0] ?? 0 }, (_, i) => 2 * (x.data[2 * i] ?? 0) + (x.data[2 * i + 1] ?? 0)))

  it('sampling Shapley accumulates one order per step, logs the estimate, and resumes', () => {
    const rec = recorder()
    const alg = samplingShapleyAlgorithm(linear as never, [1, 1], [[0, 0]], { nPermutations: 12 })
    const run = runAlgorithm(alg, { maxSteps: 12, tracer: rec, policy })
    expect(run.steps).toBe(12)
    const { values, standardErrors } = shapleyEstimates(run.state)
    expect(values).toEqual([2, 1])
    expect(standardErrors[0]).toBeCloseTo(0, 12)
    expect(metricsOf(rec, 'shapley/permutation')).toEqual(Array.from({ length: 12 }, (_, i) => i + 1))
    expect(rec.events.filter((e) => e.type === 'tensor' && e.name === 'shapley/phi')).toHaveLength(12)
    expect(samplingShapley(linear as never, [1, 1], [[0, 0]], { nPermutations: 12 }).values).toEqual(values)
    const { whole, rest } = resumed(alg, 5, 12)
    expect(Array.from(rest.sum)).toEqual(Array.from(whole.sum))
    expect(rest.count).toBe(whole.count)
  })

  it('SAGE gives the ignored feature nothing and resumes', () => {
    const first = (x: { shape: number[]; data: ArrayLike<number> }) =>
      tensor(Array.from({ length: x.shape[0] ?? 0 }, (_, i) => x.data[2 * i] ?? 0))
    const x = [
      [0, 5],
      [2, -5],
      [1, 0],
      [3, 2],
    ]
    const data = { x, y: [0, 2, 1, 3] }
    const alg = sageAlgorithm(first as never, data, x, { nPermutations: 40, key: key(1) })
    const run = runAlgorithm(alg, { maxSteps: 40 })
    const est = shapleyEstimates(run.state)
    expect(est.values[1]).toBeCloseTo(0, 12)
    expect(est.values[0]).toBeGreaterThan(0)
    const viaFunction = sage(first as never, data, x, { nPermutations: 40, key: key(1) })
    expect(viaFunction.values[0]).toBeCloseTo(est.values[0] ?? Number.NaN, 12)
    const { whole, rest } = resumed(alg, 15, 40)
    expect(Array.from(rest.sum)).toEqual(Array.from(whole.sum))
    expect(() => sageAlgorithm(first as never, data, x, { exact: true })).toThrow(/exact/)
  })

  it('data Shapley values the bad label lowest, one order per step, and resumes', () => {
    const meanModel = defineEstimator(
      {
        name: 'mean',
        kind: 'regressor',
        fit: (d) => ({ m: item(mean(d.y as never)) }),
        methods: (p) => ({ predict: (x) => full([rowsOf(x)], p.m) }),
      },
      {},
    )
    const options = { scoring: 'neg_mean_absolute_error' as const, nPermutations: 20, emptyUtility: -5, key: key(0) }
    const train = { x: [[0], [0], [0]], y: [1, 1, 9] }
    const test = { x: [[0]], y: [1] }
    const alg = dataShapleyAlgorithm(meanModel, train, test, options)
    const rec = recorder()
    const run = runAlgorithm(alg, { maxSteps: 20, tracer: rec, policy })
    const v = shapleyEstimates(run.state).values
    expect(v[2]).toBeLessThan(v[0] ?? 0)
    expect(metricsOf(rec, 'shapley/permutation').at(-1)).toBe(20)
    const r = dataShapley(meanModel, train, test, options)
    for (const [i, value] of r.values.entries()) expect(value).toBeCloseTo(v[i] ?? Number.NaN, 12)
    const { whole, rest } = resumed(alg, 7, 20)
    expect(Array.from(rest.sum)).toEqual(Array.from(whole.sum))
  })

  it('permutation importance permutes one feature per step and resumes', () => {
    const firstColumn = {
      estimator: 'f',
      kind: 'regressor' as const,
      params: {},
      predict: (x: { shape: number[]; data: ArrayLike<number> }) =>
        tensor(Array.from({ length: x.shape[0] ?? 0 }, (_, i) => x.data[i * 2] ?? 0)),
    }
    const x = [
      [1, 9],
      [2, 8],
      [3, 7],
      [4, 6],
    ]
    const options = { scoring: 'neg_mean_squared_error' as const, permutations: [[3, 2, 1, 0]] }
    const alg = permutationImportanceAlgorithm(firstColumn as never, { x, y: [1, 2, 3, 4] }, options)
    const rec = recorder()
    const run = runAlgorithm(alg, { maxSteps: 5, tracer: rec, policy })
    expect(run.state.drops).toEqual([[5], [0]])
    expect(metricsOf(rec, 'importance/mean')).toEqual([5, 0])
    expect(permutationImportance(firstColumn as never, { x, y: [1, 2, 3, 4] }, options).mean).toEqual([5, 0])
    const { whole, rest } = resumed(alg, 1, 5)
    expect(rest.drops).toEqual(whole.drops)
  })
})
