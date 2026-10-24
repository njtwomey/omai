import { describe, expect, it } from 'vitest'
import {
  bayesPredict,
  type ClassificationData,
  type DatasetGenerator,
  generate,
  rowCount,
} from '../../lib/data/index.ts'
import { matrixRank } from '../../lib/numerics/linalg/index.ts'
import { key } from '../../lib/random/index.ts'
import { sum, type Tensor, take, tensor, toFlat } from '../../lib/tensor/index.ts'
import {
  blobs,
  checkerboard,
  circles,
  gaussianQuantiles,
  hastie10_2,
  makeClassification,
  makeMultilabelClassification,
  moons,
  spirals,
  xor,
} from './index.ts'

const mean = (xs: readonly number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length

/** The Bayes rule's error on the sample agrees with the stated Bayes error within a few standard errors. */
function checkBayes(
  g: DatasetGenerator<ClassificationData>,
  knobs: Record<string, number | boolean | string>,
  seed = 1,
): void {
  const { data, truth } = generate(g, { key: key(seed), knobs })
  const pred = toFlat(bayesPredict(truth, data.x))
  const y = toFlat(data.y)
  const err = mean(pred.map((p, i) => (p === y[i] ? 0 : 1)))
  const b = truth.bayesError ?? Number.NaN
  const se = Math.sqrt(Math.max(b * (1 - b), 1e-4) / y.length)
  expect(Math.abs(err - b), g.name).toBeLessThan(5 * se + 0.005)
  const p = truth.posterior?.(take(data.x, tensor([0, 1, 2], { dtype: 'int32' }), 0)) as Tensor
  for (const s of toFlat(sum(p, 1))) expect(s).toBeCloseTo(1, 10)
}

describe('classification generators', () => {
  it('blobs: even split, the asked spread and an exact posterior', () => {
    const { data, truth } = generate(blobs, { key: key(0), knobs: { n: 3000, centers: 3, clusterStd: 2 } })
    const y = toFlat(data.y)
    expect([0, 1, 2].map((c) => y.filter((v) => v === c).length)).toEqual([1000, 1000, 1000])
    const centres = truth.latents?.['centers'] as number[][]
    const x = toFlat(data.x)
    const dev = y.flatMap((c, i) => [
      (x[2 * i] ?? 0) - (centres[c]?.[0] ?? 0),
      (x[2 * i + 1] ?? 0) - (centres[c]?.[1] ?? 0),
    ])
    expect(Math.sqrt(mean(dev.map((v) => v * v)))).toBeCloseTo(2, 1)
    expect(toFlat(truth.clusters as Tensor)).toEqual(y)
    checkBayes(blobs, { n: 4000, clusterStd: 3, boxSize: 4 })
  })

  it('moons and circles: noise around the arcs and a calibrated Bayes error', () => {
    const { data, truth } = generate(moons, { key: key(2), knobs: { n: 2000, noise: 0.2 } })
    const angle = toFlat(truth.latents?.['angle'] as Tensor)
    const x = toFlat(data.x)
    const y = toFlat(data.y)
    const dev = y.flatMap((c, i) => {
      const t = angle[i] ?? 0
      const [cx, cy] = c === 0 ? [Math.cos(t), Math.sin(t)] : [1 - Math.cos(t), 1 - Math.sin(t) - 0.5]
      return [(x[2 * i] ?? 0) - cx, (x[2 * i + 1] ?? 0) - cy]
    })
    expect(Math.sqrt(mean(dev.map((v) => v * v)))).toBeCloseTo(0.2, 1)
    checkBayes(moons, { n: 4000, noise: 0.3 })
    checkBayes(circles, { n: 4000, noise: 0.15, factor: 0.6 })
    expect(generate(moons, { knobs: { noise: 0 } }).truth.bayesError).toBe(0)
  })

  it('xor, spirals and checkerboards', () => {
    checkBayes(xor, { n: 4000, noise: 0.6 })
    checkBayes(spirals, { n: 3000, noise: 0.08 })
    checkBayes(checkerboard, { n: 4000, labelNoise: 0.2 })
    expect(generate(checkerboard, { knobs: { labelNoise: 0.8 } }).truth.bayesError).toBeCloseTo(0.2, 12)
    const s = generate(spirals, { key: key(4), knobs: { n: 90, arms: 3 } })
    expect([0, 1, 2].map((c) => toFlat(s.data.y).filter((v) => v === c).length)).toEqual([30, 30, 30])
  })

  it('Gaussian quantiles: equal classes and population thresholds', () => {
    const { data, truth } = generate(gaussianQuantiles, { key: key(5), knobs: { n: 3000, nClasses: 3, nFeatures: 2 } })
    const y = toFlat(data.y)
    expect([0, 1, 2].map((c) => y.filter((v) => v === c).length)).toEqual([1000, 1000, 1000])
    // chi2(2) quantiles at 1/3 and 2/3: 2 log(3/2) and 2 log 3.
    expect(truth.extra?.['thresholds']).toEqual([
      expect.closeTo(2 * Math.log(1.5), 10),
      expect.closeTo(2 * Math.log(3), 10),
    ])
    const agree = mean(toFlat(bayesPredict(truth, data.x)).map((p, i) => (p === y[i] ? 1 : 0)))
    expect(agree).toBeGreaterThan(0.97)
  })

  it('Hastie 10.2 is balanced', () => {
    const { data } = generate(hastie10_2, { key: key(6), knobs: { n: 4000 } })
    expect(Math.abs(mean(toFlat(data.y)) - 0.5)).toBeLessThan(4 * Math.sqrt(0.25 / 4000))
    expect(data.classes).toEqual(['-1', '1'])
  })
})

describe('makeClassification', () => {
  it('builds informative, redundant, repeated and noise features where the truth says', () => {
    const knobs = { n: 400, nFeatures: 10, nInformative: 3, nRedundant: 2, nRepeated: 1, flipY: 0 }
    const { data, truth } = generate(makeClassification, { key: key(7), knobs })
    expect(data.x.shape).toEqual([400, 10])
    const relevant = truth.relevantFeatures as number[]
    expect(relevant.length).toBe(6)
    const cols = (idx: readonly number[]) => take(data.x, tensor(idx as number[], { dtype: 'int32' }), 1)
    expect(matrixRank(cols(relevant))).toBe(3)
    const repeated = truth.extra?.['repeated'] as number[]
    expect(repeated.length).toBe(1)
    const noise = truth.extra?.['noise'] as number[]
    const z = toFlat(cols(noise))
    expect(Math.abs(mean(z))).toBeLessThan(0.1)
    expect(mean(z.map((v) => v * v))).toBeCloseTo(1, 0)
    expect(toFlat((truth.latents?.['cleanLabels'] as Tensor) ?? tensor([]))).toEqual(toFlat(data.y))
  })

  it('flips labels, imbalances classes and calibrates its Bayes error', () => {
    const { data, truth } = generate(makeClassification, {
      key: key(8),
      knobs: { n: 4000, nFeatures: 4, nInformative: 2, nRedundant: 0, flipY: 0.2, classSep: 1.5 },
    })
    const clean = toFlat(truth.latents?.['cleanLabels'] as Tensor)
    const changed = mean(toFlat(data.y).map((c, i) => (c === clean[i] ? 0 : 1)))
    // A flipped label is redrawn uniformly, so it changes half the time with two classes.
    expect(Math.abs(changed - 0.1)).toBeLessThan(4 * Math.sqrt(0.09 / 4000))
    checkBayes(makeClassification, { n: 4000, nFeatures: 5, nInformative: 2, nRedundant: 1, classSep: 0.8 }, 3)
    const imb = generate(makeClassification, { key: key(9), knobs: { n: 1000, imbalance: 9, flipY: 0 } })
    expect(mean(toFlat(imb.data.y))).toBeCloseTo(0.1, 2)
    const ordered = generate(makeClassification, { key: key(1), knobs: { shuffle: false, nFeatures: 6 } })
    expect(ordered.truth.extra?.['informative']).toEqual([0, 1])
    expect(rowCount(ordered.data)).toBe(100)
  })
})

describe('makeMultilabelClassification', () => {
  it('draws documents from the label mixture', () => {
    const { data, truth } = generate(makeMultilabelClassification, {
      key: key(10),
      knobs: { n: 300, nFeatures: 15, nClasses: 4, length: 40, allowUnlabeled: false },
    })
    expect(data.x.shape).toEqual([300, 15])
    const x = toFlat(data.x)
    const lengths = Array.from({ length: 300 }, (_, i) => x.slice(i * 15, (i + 1) * 15).reduce((a, b) => a + b, 0))
    expect(Math.abs(mean(lengths) - 40)).toBeLessThan(4 * Math.sqrt(40 / 300))
    const y = toFlat(data.y)
    const perRow = Array.from({ length: 300 }, (_, i) => y.slice(i * 4, (i + 1) * 4).reduce((a, b) => a + b, 0))
    expect(Math.min(...perRow.slice(0, 50).concat([1]))).toBeGreaterThanOrEqual(1)
    expect(perRow.every((v) => v >= 1 && v <= 4)).toBe(true)
    const probs = truth.latents?.['classProbabilities'] as number[]
    expect(probs.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
  })
})
