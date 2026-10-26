/**
 * Gradient explanations of fitted neural networks (the LeNet of `methods/neural`): saliency, input times gradient,
 * integrated gradients, SmoothGrad, VarGrad, GradientSHAP, guided backpropagation, deconvolution, Grad-CAM and guided
 * Grad-CAM over an image, as image-laid feature-attribution records; and a convolutional layer's conductance per
 * channel as value grids.
 */
import { argmaxOf, MdmlError } from '../../lib/base/index.ts'
import * as explain from '../../lib/explain/index.ts'
import { type Explained, featureAttributions, valueGrids } from '../../lib/exposition/index.ts'
import { type Fitted, featureNamesOf, type LearnData, matrixView } from '../../lib/learn/index.ts'
import { type Key, key as makeKey } from '../../lib/random/index.ts'
import { fromStorage, type Tensor, toFlat } from '../../lib/tensor/index.ts'
import { compose, networkOf } from './_networks.ts'

/** The gradient explainers {@link explainNetwork} offers. */
export type NetworkMethod =
  | 'saliency'
  | 'inputXGradient'
  | 'integratedGradients'
  | 'smoothGrad'
  | 'varGrad'
  | 'gradientShap'
  | 'guidedBackprop'
  | 'deconvolution'
  | 'gradCam'
  | 'guidedGradCam'

/** Options of {@link explainNetwork}. */
export interface ExplainNetworkOptions {
  /** The explainer (default saliency). */
  readonly method?: NetworkMethod
  /** The class explained (default the class predicted). */
  readonly classIndex?: number
  /** Path steps of integrated gradients (default 32). */
  readonly nSteps?: number
  /** Noisy copies or samples of the noise tunnels and GradientSHAP (default 30). */
  readonly nSamples?: number
  /** The noise standard deviation as a fraction of the image's range (default 0.15). */
  readonly noise?: number
  /** Background images for GradientSHAP (default 20 rows of the data). */
  readonly background?: number
  readonly key?: Key
}

/** A network's attribution map for one image. */
export interface NetworkExplanation {
  readonly method: NetworkMethod
  /** One value per pixel. */
  readonly values: readonly number[]
  /** The explained class's logit. */
  readonly logit: number
  readonly classIndex: number
}

/**
 * Explain a fitted LeNet's logit for one image by its gradients: saliency (absolute), input times gradient,
 * integrated gradients from a black image, SmoothGrad and VarGrad (noise tunnels over the gradient), GradientSHAP over
 * background images, guided backpropagation and deconvolution (ReLUs pass only positive signals back), Grad-CAM at
 * the last convolution (upsampled to the image), or guided Grad-CAM (the two multiplied). The record is laid out as an
 * image.
 *
 * @param fitted - A fitted LeNet (`methods/neural` `lenet`).
 * @param data - The images (flattened rows).
 * @param row - The image explained.
 * @param options - method, classIndex, nSteps, nSamples, noise, background and key.
 * @returns The map with a `featureAttributions` record.
 */
export function explainNetwork(
  fitted: Fitted,
  data: LearnData,
  row: number,
  options: ExplainNetworkOptions = {},
): Explained<NetworkExplanation> {
  const x = matrixView(data.x, 'explainNetwork')
  if (!(Number.isInteger(row) && row >= 0 && row < x.n)) throw new MdmlError('value', `explainNetwork: no row ${row}`)
  const net = networkOf(fitted)
  const [, h, w] = net.shape
  const pixels = Array.from(x.data.subarray(row * x.d, (row + 1) * x.d))
  const image = fromStorage(Float64Array.from(pixels), [1, x.d], 'float64')
  const model = compose(net.layers)
  const logits = toFlat(model(image) as Tensor)
  const target = options.classIndex ?? argmaxOf(logits)
  const method = options.method ?? 'saliency'
  const key = options.key ?? makeKey(0)
  const range = Math.max(...pixels) - Math.min(...pixels) || 1
  const stdev = (options.noise ?? 0.15) * range
  const nSamples = options.nSamples ?? 30
  const cam = (): number[] => {
    const head = compose(net.layers.slice(0, net.camLayer + 1))
    const tail = compose(net.layers.slice(net.camLayer + 1))
    const map = explain.gradCam(head, tail, image, { target })[0] ?? []
    return explain.interpolateMap(map, [h, w]).flat()
  }
  let values: number[]
  if (method === 'saliency') values = toFlat(explain.saliency(model, image, { target }))
  else if (method === 'inputXGradient') values = toFlat(explain.inputXGradient(model, image, { target }))
  else if (method === 'integratedGradients')
    values = toFlat(explain.integratedGradients(model, image, { target, nSteps: options.nSteps ?? 32 }).attributions)
  else if (method === 'smoothGrad' || method === 'varGrad')
    values = toFlat(
      explain.noiseTunnel((v) => explain.saliency(model, v, { target, abs: false }), image, {
        type: method === 'smoothGrad' ? 'smoothgrad' : 'vargrad',
        nSamples,
        stdev,
        key,
      }),
    )
  else if (method === 'gradientShap') {
    const m = Math.min(x.n, options.background ?? 20)
    const bg = Array.from({ length: m }, (_, b) => {
      const i = Math.floor((b * x.n) / m)
      return Array.from(x.data.subarray(i * x.d, (i + 1) * x.d))
    })
    values = toFlat(explain.gradientShap(model, image, bg, { target, nSamples, stdev: stdev / 3, key }))
  } else if (method === 'guidedBackprop') values = toFlat(explain.guidedBackprop(net.layers, image, { target }))
  else if (method === 'deconvolution') values = toFlat(explain.deconvolution(net.layers, image, { target }))
  else if (method === 'gradCam') values = cam()
  else {
    const guided = toFlat(explain.guidedBackprop(net.layers, image, { target }))
    const c = cam()
    values = guided.map((g, i) => g * (c[i] ?? 0))
  }
  const logit = logits[target] ?? 0
  const name = String(net.classes[target] ?? target)
  return {
    method,
    values,
    logit,
    classIndex: target,
    exposition: [
      featureAttributions({
        method,
        features: featureNamesOf(data, x.d),
        output: `logit of ${name}`,
        baseValue: 0,
        values: [values],
        data: [pixels],
        predictions: [logit],
        indices: [row],
        layout: { kind: 'image', shape: [h, w] },
      }),
    ],
  }
}

/**
 * How a fitted LeNet's last convolution carries its decision for one image: each channel's layer conductance (the
 * flow of integrated gradients through it, from a black image) as a value grid, so the channels the logit relies on
 * stand out.
 *
 * @param fitted - A fitted LeNet.
 * @param data - The images.
 * @param row - The image explained.
 * @param options - classIndex and nSteps (default 24).
 * @returns The conductance per channel and the logit, with a `valueGrids` record.
 */
export function explainLayer(
  fitted: Fitted,
  data: LearnData,
  row: number,
  options: { readonly classIndex?: number; readonly nSteps?: number } = {},
): Explained<{ readonly conductance: readonly (readonly (readonly number[])[])[]; readonly classIndex: number }> {
  const x = matrixView(data.x, 'explainLayer')
  if (!(Number.isInteger(row) && row >= 0 && row < x.n)) throw new MdmlError('value', `explainLayer: no row ${row}`)
  const net = networkOf(fitted)
  const image = fromStorage(Float64Array.from(x.data.subarray(row * x.d, (row + 1) * x.d)), [1, x.d], 'float64')
  const model = compose(net.layers)
  const target = options.classIndex ?? argmaxOf(toFlat(model(image) as Tensor))
  const head = compose(net.layers.slice(0, net.camLayer + 1))
  const tail = compose(net.layers.slice(net.camLayer + 1))
  const c = explain.layerConductance(head, tail, image, { target, nSteps: options.nSteps ?? 24 })
  const [, ch = 1, gh = 1, gw = 1] = c.shape
  const flat = toFlat(c)
  const conductance = Array.from({ length: ch }, (_, k) =>
    Array.from({ length: gh }, (_, r) => flat.slice((k * gh + r) * gw, (k * gh + r + 1) * gw)),
  )
  const name = String(net.classes[target] ?? target)
  return {
    conductance,
    classIndex: target,
    exposition: [
      valueGrids({
        grids: conductance.map((g, k) => ({
          name: `channel ${k + 1} · ${g
            .flat()
            .reduce((a, b) => a + b, 0)
            .toFixed(2)}`,
          values: g,
        })),
        scale: 'shared',
        diverging: true,
        valueName: `conductance to the logit of ${name}`,
      }),
    ],
  }
}
