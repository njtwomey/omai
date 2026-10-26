import { describe, expect, it } from 'vitest'
import { digits } from '../../lib/datasets/index.ts'
import { recordsOf } from '../../lib/exposition/index.ts'
import { takeData } from '../../lib/learn/index.ts'
import { key } from '../../lib/random/index.ts'
import { lenet } from '../neural/vision/index.ts'
import { explainLayer, explainNetwork, type NetworkMethod } from './gradients.ts'

const data = takeData(
  digits(),
  Array.from({ length: 120 }, (_, i) => i),
)
const net = lenet({ epochs: 1, channels: [3, 4], hidden: 8, snapshots: 2 }).fit(data, { key: key(1), explain: false })

describe('network gradient methods', () => {
  it('explains an image by every gradient method, as image-laid records', () => {
    const methods: NetworkMethod[] = [
      'saliency',
      'inputXGradient',
      'integratedGradients',
      'smoothGrad',
      'varGrad',
      'gradientShap',
      'guidedBackprop',
      'deconvolution',
      'gradCam',
      'guidedGradCam',
    ]
    for (const method of methods) {
      const r = explainNetwork(net, data, 2, { method, nSamples: 3, nSteps: 4, background: 4, key: key(2) })
      expect(r.values.length).toBe(64)
      expect(r.values.every(Number.isFinite)).toBe(true)
      expect(recordsOf(r.exposition, 'featureAttributions')[0]?.layout).toEqual({ kind: 'image', shape: [8, 8] })
    }
    expect(explainNetwork(net, data, 2, { method: 'gradCam' }).values.every((v) => v >= 0)).toBe(true)
  })

  it('reads the last convolution channel by channel', () => {
    const r = explainLayer(net, data, 2, { nSteps: 4 })
    expect(r.conductance.length).toBe(4)
    expect(recordsOf(r.exposition, 'valueGrids')[0]?.grids.length).toBe(4)
  })
})
