// The layers of fitted networks as functions, for the gradient and concept explainers (private to methods/explain):
// a LeNet of methods/neural rebuilt from its saved state, scaling first, its ReLUs marked.
import { MdmlError } from '../../lib/base/index.ts'
import type * as explain from '../../lib/explain/index.ts'
import type { Fitted } from '../../lib/learn/index.ts'
import { loadState, type Module, type Params, type SavedState } from '../../lib/nn/index.ts'
import { foldIn, type Key, key as makeKey, normal } from '../../lib/random/index.ts'
import { div, mul, type Tensor, toFlat, type Value } from '../../lib/tensor/index.ts'
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

/**
 * The layers of a fitted LeNet, scaling first, ReLUs marked, in eval mode; with `randomise`, the weights of that many
 * of its parametrised layers, counting from the output, replaced by Gaussian noise of the same scale (Adebayo et al.'s
 * cascading randomisation).
 */
export function networkOf(fitted: Fitted, randomise?: { readonly layers: number; readonly key: Key }): NetworkLayers {
  const p = fitted.params
  if (!isLeNet(p)) throw new MdmlError('value', `explain: ${fitted.estimator} is not a LeNet (methods/neural/vision)`)
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
  const children = net.children()
  const params: Record<string, unknown> = { ...net.params }
  if (randomise !== undefined && randomise.layers > 0) {
    const named = children.filter(([name]) => {
      const q = params[name] as Readonly<Record<string, unknown>> | undefined
      return q !== undefined && Object.keys(q).length > 0
    })
    named.slice(-randomise.layers).forEach(([name], i) => {
      const q = params[name] as Readonly<Record<string, Tensor>>
      params[name] = Object.fromEntries(
        Object.entries(q).map(([k, t]) => {
          const v = toFlat(t)
          const mean = v.reduce((a, b) => a + b, 0) / Math.max(1, v.length)
          const sd = Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(1, v.length)) || 0.1
          return [k, mul(normal(foldIn(foldIn(randomise.key, i), k.length), t.shape), sd) as Tensor]
        }),
      )
    })
  }
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
