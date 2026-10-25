// The layers of fitted networks as functions, for the gradient and concept explainers (private to methods/explain):
// a LeNet of methods/neural rebuilt from its saved state, scaling first, its ReLUs marked.
import { WhiteboxError } from '../../lib/base/index.ts'
import type * as explain from '../../lib/explain/index.ts'
import type { Fitted } from '../../lib/learn/index.ts'
import { loadState, type Module, type Params, type SavedState } from '../../lib/nn/index.ts'
import { key as makeKey } from '../../lib/random/index.ts'
import { div, type Value } from '../../lib/tensor/index.ts'
import { type LeNetHyper, type LeNetParams, lenetNetwork } from '../neural/vision/index.ts'

/** A network read off a fitted model: its layers as functions, which are ReLUs, and its image shape. */
export interface NetworkLayers {
  readonly layers: readonly explain.DeepLiftLayer[]
  /** The index (in `layers`) of the last convolution's activation, where Grad-CAM reads. */
  readonly camLayer: number
  /** The index of the hidden fully connected layer's activation (before the output layer), where concepts are read. */
  readonly hiddenLayer: number
  readonly shape: readonly [number, number, number]
  readonly classes: readonly number[]
}

function isLeNet(p: unknown): p is LeNetParams {
  const q = p as Partial<LeNetParams>
  return (
    Array.isArray(q.imageShape) && Array.isArray(q.channels) && typeof q.scale === 'number' && q.state !== undefined
  )
}

/** The layers of a fitted LeNet, scaling first, ReLUs marked, in eval mode. */
export function networkOf(fitted: Fitted): NetworkLayers {
  const p = fitted.params
  if (!isLeNet(p))
    throw new WhiteboxError('value', `explain: ${fitted.estimator} is not a LeNet (methods/neural/vision)`)
  const shape = p.imageShape as [number, number, number]
  const net = lenetNetwork(
    shape,
    p.classes.length,
    {
      channels: p.channels as [number, number],
      kernelSize: p.kernelSize,
      hidden: p.hidden,
      activation: p.activation as LeNetHyper['activation'],
      pool: p.pool as LeNetHyper['pool'],
    },
    makeKey(0),
  )
  loadState(net, p.state as unknown as SavedState)
  const params = net.params
  const children = net.children()
  const layers: explain.DeepLiftLayer[] = [{ fn: (v: Value) => div(v, p.scale) }]
  let camLayer = 0
  children.forEach(([name, m], i) => {
    const mod = m as Module
    layers.push({
      fn: (v: Value) => mod.apply((params[name] ?? {}) as Params, v, { train: false }),
      ...(mod.kind === 'relu' ? { nonlinear: true } : {}),
    })
    if (children[i - 1]?.[1].kind === 'Conv2d') camLayer = layers.length - 1
  })
  return { layers, camLayer, hiddenLayer: layers.length - 2, shape, classes: p.classes }
}

export const compose =
  (layers: readonly explain.DeepLiftLayer[]) =>
  (v: Value): Value =>
    layers.reduce((h, l) => l.fn(h), v)
