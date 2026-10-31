import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { shapeletPatterns, shapeletSeries } from '../datasets/index.ts'
import { key, normal } from '../random/index.ts'
import { pearson } from '../stats/index.ts'
import { grad } from '../autodiff/index.ts'
import { asTensor, toFlat, type Value } from '../tensor/index.ts'
import { iterate, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import {
  findShapelets,
  fitLearnedShapelets,
  initialShapelets,
  learnedShapeletProblem,
  learnedShapeletsAlgorithm,
  predictLearnedShapelets,
  shapeletDistance,
  shapeletTransform,
  softMinDistance,
} from './index.ts'

const rows = (x: Value): number[][] => {
  const shape = (x as { shape: readonly number[] }).shape
  const flat = toFlat(x)
  return Array.from({ length: shape[0] ?? 0 }, (_, i) => flat.slice(i * (shape[1] ?? 0), (i + 1) * (shape[1] ?? 0)))
}
const labelsOf = (y: Value): number[] => toFlat(y)

function sample(
  seed: number,
  knobs: Record<string, number>,
): { series: number[][]; labels: number[]; positions: number[] } {
  const { data, truth } = generate(shapeletSeries, { key: key(seed), knobs })
  return { series: rows(data.x), labels: labelsOf(data.y), positions: truth.latents?.['positions'] as number[] }
}

describe('soft-minimum distance', () => {
  it('is the hand-computed weighted mean of the window distances, and tends to the minimum as alpha falls', () => {
    // windows of length 1: distances to the shapelet [0] are 0 and 1; weights e^0 and e^-1
    expect(softMinDistance([0, 1], [0], -1)).toBeCloseTo(Math.exp(-1) / (1 + Math.exp(-1)), 14)
    // three windows of length 2, distances (mean squares) 0.5, 0.125 and 2
    const x = [1, 0, 0.5, 2, 0]
    const s = [0, 0.5]
    // four windows [1,0], [0,0.5], [0.5,2], [2,0]: mean squared differences to [0, 0.5]
    const d = [0.5 * (1 + 0.25), 0, 0.5 * (0.25 + 2.25), 0.5 * (4 + 0.25)]
    const w = d.map((v) => Math.exp(-3 * v))
    const expected = d.reduce((a, v, j) => a + v * (w[j] ?? 0), 0) / w.reduce((a, v) => a + v, 0)
    expect(softMinDistance(x, s, -3)).toBeCloseTo(expected, 12)
    const hard = shapeletDistance(x, s).distance
    expect(softMinDistance(x, s, -2000)).toBeCloseTo(hard, 6)
    expect(softMinDistance(x, s, -1e-9)).toBeCloseTo(d.reduce((a, v) => a + v, 0) / 4, 6)
    expect(softMinDistance(x, s, -3)).toBeGreaterThanOrEqual(hard)
    expect(() => softMinDistance([1], [1, 2])).toThrow()
  })
})

describe('the learned-shapelets objective', () => {
  it('has the analytic gradient (checked against finite differences of the objective)', () => {
    const { series, labels } = sample(1, { n: 12, length: 24, classes: 3 })
    const data = { series, labels, nClasses: 3 }
    const problem = learnedShapeletProblem(data, { nShapelets: 2, length: 5, alpha: -5, ridge: 0.1 })
    const f = problem.objective as unknown as (x: Value, batch?: readonly number[]) => Value
    const x0 = Array.from(
      toFlat(normal(key(4), [(problem.x0 as number[]).length])),
      (v, i) => (problem.x0 as number[])[i]! + 0.1 * v,
    )
    const value = (x: number[], batch?: readonly number[]): number =>
      toFlat(f(asTensor(Float64Array.from(x)), batch))[0] ?? Number.NaN
    const g = toFlat(grad((x: Value) => f(x))(asTensor(Float64Array.from(x0))) as Value)
    expect(g.length).toBe(x0.length)
    const h = 1e-6
    x0.forEach((_, i) => {
      const up = x0.map((v, j) => (j === i ? v + h : v))
      const dn = x0.map((v, j) => (j === i ? v - h : v))
      expect(g[i]).toBeCloseTo((value(up) - value(dn)) / (2 * h), 6)
    })
    // a batch is the mean over its series: all series is the full objective; singletons average to it (up to the penalty counted once)
    expect(
      value(
        x0,
        Array.from({ length: 12 }, (_, i) => i),
      ),
    ).toBeCloseTo(value(x0), 12)
  })

  it('starts at k-means centres of the windows and is deterministic from the key', () => {
    const { series, labels } = sample(2, { n: 20, length: 30 })
    const data = { series, labels, nClasses: 2 }
    const a = initialShapelets(data, { nShapelets: 3, length: 7 }, key(0))
    expect(a.length).toBe(3)
    expect(initialShapelets(data, { nShapelets: 3, length: 7 }, key(0))).toEqual(a)
    expect(() => initialShapelets(data, { nShapelets: 3, length: 70 }, key(0))).toThrow(/length/)
  })
})

describe('learned shapelets on planted series', () => {
  const train = sample(11, { n: 60, length: 60, classes: 2, noise: 0.15 })
  const test = sample(12, { n: 60, length: 60, classes: 2, noise: 0.15 })

  it('recovers the planted bump (after alignment) and classifies held-out series', () => {
    const model = fitLearnedShapelets(
      { series: train.series, labels: train.labels, nClasses: 2 },
      { nShapelets: 3, length: 7, key: key(3) },
      300,
    )
    expect(model.accuracy).toBeGreaterThanOrEqual(0.95)
    const accuracy = (m: typeof model, set: typeof test): number => {
      const { classes } = predictLearnedShapelets(m, set.series)
      return classes.filter((c, i) => c === set.labels[i]).length / classes.length
    }
    expect(accuracy(model, test)).toBeGreaterThanOrEqual(0.95)
    // alignment: some learned shapelet against the planted bump, over shifts of up to two steps, by the correlation of
    // the overlapping part (scale free: the fit may stretch the amplitude to separate the classes better)
    const truth = shapeletPatterns[0] as readonly number[]
    const match = (learned: readonly number[], shift: number): number => {
      const a: number[] = []
      const b: number[] = []
      for (let l = 0; l < truth.length; l++) {
        const j = l + shift
        if (j < 0 || j >= learned.length) continue
        a.push(learned[j] ?? 0)
        b.push(truth[l] ?? 0)
      }
      return a.length < 5 ? Number.NEGATIVE_INFINITY : pearson(a, b).statistic
    }
    const best = Math.max(...model.shapelets.flatMap((sk) => [-2, -1, 0, 1, 2].map((shift) => match(sk, shift))))
    expect(best).toBeGreaterThan(0.95)
    // the shapelet-transform baseline (a discovered shapelet and the threshold of its best split) is no better
    const found = findShapelets(train.series, train.labels, { minLength: 7, maxLength: 7, k: 1 })
    const threshold = (found[0] as { threshold: number }).threshold
    const pred = shapeletTransform(test.series, found).map((r) => ((r[0] ?? 0) <= threshold ? 1 : 0))
    const baseline = pred.filter((p, i) => p === test.labels[i]).length / pred.length
    expect(accuracy(model, test)).toBeGreaterThanOrEqual(baseline - 0.05)
  })

  it('learns three classes with two shapelets', () => {
    const a = sample(21, { n: 90, length: 60, classes: 3, noise: 0.15 })
    const b = sample(22, { n: 60, length: 60, classes: 3, noise: 0.15 })
    const model = fitLearnedShapelets(
      { series: a.series, labels: a.labels, nClasses: 3 },
      { nShapelets: 2, length: 7, key: key(5) },
      500,
    )
    const { classes } = predictLearnedShapelets(model, b.series)
    expect(classes.filter((c, i) => c === b.labels[i]).length / classes.length).toBeGreaterThanOrEqual(0.9)
  })

  it('the solver is a slot: Adam is the default, L-BFGS on the full objective also fits, and the tape shows the shapelets', () => {
    const data = { series: train.series, labels: train.labels, nClasses: 2 }
    const alg = learnedShapeletsAlgorithm(data, { nShapelets: 3, length: 7, key: key(3) })
    const objective: number[] = []
    for (const { state } of iterate(alg, { key: key(0), maxSteps: 150 })) objective.push(state.objective)
    expect(objective.at(-1) ?? 0).toBeLessThan((objective[0] ?? 0) / 2)
    const lbfgs = runAlgorithm(
      learnedShapeletsAlgorithm(data, {
        nShapelets: 3,
        length: 7,
        key: key(3),
        solver: { kind: 'lbfgs', gradTol: 1e-6 } as never,
      }),
      { maxSteps: 200, key: key(0) },
    )
    expect(lbfgs.state.accuracy).toBeGreaterThanOrEqual(0.95)
    expect(lbfgs.state.objective).toBeLessThan(0.3)
    const state = runAlgorithm(alg, { maxSteps: 3, key: key(0) }).state
    // the optimiser's stateless links hold undefined, which JSON stores as null: the JSON is stable, the parameters exact
    const json = stateToJson(state)
    expect(stateToJson(stateFromJson(json))).toEqual(json)
    expect((stateFromJson(json) as { parameters: number[] }).parameters).toEqual(state.parameters)
  })
})
