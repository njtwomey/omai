import { describe, expect, it } from 'vitest'
import { datasetCollections, loadEntry } from '../../../lib/datasets/index.ts'
import { recordsOf } from '../../../lib/exposition/index.ts'
import type { LearnData } from '../../../lib/learn/index.ts'
import { key } from '../../../lib/random/index.ts'
import { toFlat } from '../../../lib/tensor/index.ts'
import { backpropByHand } from './backprop.ts'
import { MLPClassifier, mlpClassifier, mlpRegressor } from './mlp.ts'
import { rosenblattPerceptron } from './perceptron.ts'

const load = (collection: string, name: string, knobs = {}): LearnData => {
  const entry = datasetCollections[collection]?.entries.find((e) => e.name === name)
  if (entry === undefined) throw new Error(`no ${collection}/${name}`)
  return loadEntry(entry, key(0), knobs).data as unknown as LearnData
}

describe('mlpClassifier', () => {
  it('learns the moons within a short budget and records how', () => {
    const data = load('classification2d', 'moons', { n: 120 })
    const fitted = mlpClassifier({
      hiddenLayerSizes: [8, 8],
      activation: 'tanh',
      learningRateInit: 0.05,
      maxIter: 80,
    }).fit(data, { key: key(0) })
    expect(fitted.score?.(data)).toBeGreaterThan(0.9)
    const [surface] = recordsOf(fitted.exposition, 'decisionSurface')
    expect(surface?.labels?.length).toBe(30)
    expect(surface?.probabilities?.length).toBe(2)
    // The surface at geometrically spaced epochs, from the initial network to the fitted one.
    const steps = surface?.steps ?? []
    expect(steps.length).toBeGreaterThan(10)
    expect(steps.length).toBeLessThanOrEqual(40)
    expect(steps[0]?.iteration).toBe(0)
    expect(steps.at(-1)?.iteration).toBe(fitted.params.epochs)
    expect(steps[0]?.labels?.length).toBe(30)
    expect(surface?.stepName).toBe('epoch')
    expect(surface?.rows?.points.length).toBe(120)
    const curves = recordsOf(fitted.exposition, 'learningCurve')
    expect(curves.length).toBe(2)
    expect(curves[0]?.sizes.length).toBe(fitted.params.epochs)
    const [hidden] = recordsOf(fitted.exposition, 'embeddingSteps')
    expect(hidden?.iterations[0]).toBe(0)
    expect(hidden?.positions.length).toBeLessThanOrEqual(40)
    expect(hidden?.positions[0]?.length).toBe(120)
    // The first hidden layer's eight units over the plane: three across, three down, 15 by 15 each with gaps.
    const units = recordsOf(fitted.exposition, 'decisionSurface')[1]
    expect([units?.values?.length, units?.values?.[0]?.length]).toEqual([47, 47])
    expect(units?.steps?.length).toBe(steps.length)
    expect(fitted.describe().nodes.length).toBeGreaterThan(3)
    expect(fitted.hidden([[0, 0]]).map((h) => h.shape)).toEqual([
      [1, 8],
      [1, 8],
    ])
  })

  it('handles several classes with a softmax, early stopping and the class form', () => {
    const data = load('classification2d', 'blobs', { n: 90 })
    const m = new MLPClassifier({ hiddenLayerSizes: [8], maxIter: 40, learningRateInit: 0.05, earlyStopping: true })
    m.fit(data.x, data.y as never)
    expect(m.model.params.output).toBe('softmax')
    expect(toFlat(m.model.predictProba?.([[0, 0]]) as never).length).toBe(3)
    expect(recordsOf(m.exposition, 'learningCurve')[1]?.labels?.validation).toBe('validation accuracy')
  })

  it('reads hidden layer sizes from a string control and revives', () => {
    const est = mlpClassifier().with({ hiddenLayerSizes: '4,3', maxIter: 3 })
    expect(est.hyper.hiddenLayerSizes).toEqual([4, 3])
    const data = load('classification2d', 'xor', { n: 40 })
    const fitted = est.fit(data, { key: key(1) })
    const revived = est.revive(fitted.params)
    expect(toFlat(revived.predict([[0.5, -0.5]]))).toEqual(toFlat(fitted.predict([[0.5, -0.5]])))
  })

  it('reports progress at snapshots', () => {
    const data = load('classification2d', 'moons', { n: 60 })
    const seen: number[] = []
    mlpClassifier({ hiddenLayerSizes: [4], maxIter: 10, onProgress: (p) => seen.push(p.fraction) }).fit(data)
    expect(seen.at(-1)).toBe(1)
    expect(seen.length).toBeGreaterThan(5)
  })
})

describe('mlpRegressor', () => {
  it('fits a sine curve and records its curve', () => {
    const data = load('regression1d', 'sineCurve')
    const fitted = mlpRegressor({
      hiddenLayerSizes: [16, 16],
      activation: 'tanh',
      learningRateInit: 0.02,
      batchSize: 25,
      maxIter: 80,
      nIterNoChange: 80,
    }).fit(data, { key: key(0) })
    expect(fitted.score?.(data)).toBeGreaterThan(0.7)
    const [curve] = recordsOf(fitted.exposition, 'decisionSurface')
    expect(curve?.values?.length).toBe(1)
    expect(curve?.values?.[0]?.length).toBe(30)
  })

  it('trains with SGD and the adaptive and invscaling schedules', () => {
    const data = load('regression1d', 'sineCurve')
    for (const learningRate of ['adaptive', 'invscaling'] as const) {
      const fitted = mlpRegressor({ hiddenLayerSizes: [8], solver: 'sgd', learningRate, maxIter: 20 }).fit(data)
      expect(Number.isFinite(fitted.params.loss)).toBe(true)
    }
  })
})

describe('rosenblattPerceptron', () => {
  it('converges on separable data within the Novikoff bound, recording every update', () => {
    const data = load('classification2d', 'blobs', { n: 60, centers: 2, clusterStd: 0.5 })
    const fitted = rosenblattPerceptron().fit(data, { key: key(0) })
    expect(fitted.params.converged).toBe(true)
    expect(fitted.score?.(data)).toBe(1)
    const table = recordsOf(fitted.exposition, 'coefficientTable')[0]
    const bound = table?.estimates[table.names.indexOf('bound (R/γ)²')] ?? 0
    expect(fitted.params.updates).toBeLessThanOrEqual(bound)
    expect(fitted.params.history.length).toBe(fitted.params.updates)
    expect(recordsOf(fitted.exposition, 'learningCurve')[0]?.train.at(-1)).toBe(0)
    const surface = recordsOf(fitted.exposition, 'decisionSurface')[0]
    const steps = surface?.steps ?? []
    expect(steps[0]?.iteration).toBe(0)
    expect(steps.at(-1)?.iteration).toBe(fitted.params.updates)
    expect(steps.at(-1)?.objective).toBe(0)
    expect(steps[1]?.highlight).toEqual([fitted.params.history[0]?.row])
  })
})

describe('backpropByHand', () => {
  it('agrees with autodiff at every step and lowers the loss', () => {
    const r = backpropByHand({ steps: 12 })
    expect(r.maxError).toBeLessThan(1e-12)
    expect(r.steps.length).toBe(12)
    expect(r.steps[0]?.gradients.length).toBe(9)
    // Per step: three layers forward, the loss, two layers back.
    expect(r.frames.length).toBe(12 * 6)
    expect(r.captions.length).toBe(r.frames.length)
    expect(r.structure.nodes.length).toBe(6)
    expect(recordsOf(r.exposition, 'coefficientTable')[0]?.names.length).toBe(9)
    const wide = backpropByHand({
      sizes: [2, 3, 2],
      activation: 'tanh',
      y: [
        [0, 1],
        [1, 0],
        [1, 0],
        [0, 1],
      ],
    })
    expect(wide.maxError).toBeLessThan(1e-12)
  })
})
