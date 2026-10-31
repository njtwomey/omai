/**
 * Feature visualisation by activation maximisation (Erhan et al. 2009, Mordvintsev et al. 2015, Olah et al. 2017):
 * gradient ascent on an input image to maximise a unit of a network (a class logit, a channel of a feature map). The
 * raw ascent produces adversarial noise, so the standard remedies are options: a parametrisation in the Fourier
 * domain with the spectrum scaled by 1/f (frequency preconditioning, so low frequencies move first), random jitter,
 * scale and rotation of the image before each forward pass (transformation robustness), a total-variation penalty and
 * an optional blur. The ascent is an `Algorithm` over a serialisable state, so a session can scrub through the image
 * emerging from noise.
 */
import { valueAndGrad } from '../autodiff/index.ts'
import type { Json } from '../base/index.ts'
import { MdmlError } from '../base/index.ts'
import { affineGrid, gridSample } from '../nn/index.ts'
import { ifft2 } from '../numerics/fourier/index.ts'
import { adam, applyUpdates, type LearningRate, sgd } from '../optim/index.ts'
import { foldIn, type Key, normal, uniform } from '../random/index.ts'
import { gaussianFilter } from '../signal/image/index.ts'
import {
  abs,
  add,
  asTensor,
  complex,
  fromStorage,
  mean,
  mul,
  primalOf,
  real,
  reshape,
  roll,
  sigmoid,
  sliceAxis,
  sub,
  sum,
  type Tensor,
  tensor,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  scalar,
  stateFromJson,
  stateToJson,
  tensor as tensorSource,
} from '../trace/index.ts'

/** A network seen through an image: `[1, C, H, W]` in, anything out (logits `[1, K]`, a feature map `[1, C', h, w]`). */
export type ImageModel = (image: Value) => Value

/** The scalar of the network's output that the ascent raises. */
export type UnitObjective = (output: Value) => Value

/**
 * Raise one class logit (or any output unit of a `[1, K]` output).
 *
 * @param unit - The index.
 * @returns The objective.
 * @example
 * ```ts
 * import { unitObjective } from 'mdml-engine/explain'
 * import { tensor, toFlat } from 'mdml-engine/tensor'
 * toFlat(unitObjective(1)(tensor([[3, 5, 9]]))) // => [5]
 * ```
 */
export function unitObjective(unit: number): UnitObjective {
  return (output) => sum(sliceAxis(reshape(output, [output.shape[0] ?? 1, -1]), 1, unit, unit + 1))
}

/**
 * Raise the mean activation of one channel of a `[1, C, h, w]` feature map.
 *
 * @param channel - The channel.
 * @returns The objective.
 * @example
 * ```ts
 * import { channelObjective } from 'mdml-engine/explain'
 * import { tensor, toFlat } from 'mdml-engine/tensor'
 * toFlat(channelObjective(1)(tensor([[[[1, 1]], [[2, 4]]]]))) // => [3]
 * ```
 */
export function channelObjective(channel: number): UnitObjective {
  return (output) => mean(sliceAxis(output, 1, channel, channel + 1))
}

/** Options of {@link featureVisualisation}. */
export interface FeatureVisualisationOptions {
  /** The image `[channels, height, width]`. */
  readonly shape: readonly [number, number, number]
  /** What to raise (default the first output unit). */
  readonly objective?: UnitObjective
  /** `fourier` (default) optimises a spectrum scaled by 1/f; `pixel` optimises the pixels directly. */
  readonly parametrisation?: 'fourier' | 'pixel'
  /** The exponent of the 1/f scaling (default 1). */
  readonly decay?: number
  /** Optimiser (default `adam`; `sgd` is plain gradient ascent). */
  readonly optimiser?: 'adam' | 'sgd'
  readonly learningRate?: LearningRate
  /** Maximum jitter in pixels, a random roll before each pass (default 0). */
  readonly jitter?: number
  /** Scale factors sampled uniformly per pass, `[low, high]` (default none). */
  readonly scale?: readonly [number, number]
  /** Maximum rotation in degrees per pass (default 0). */
  readonly rotate?: number
  /** Weight of the total-variation penalty of the image (default 0). */
  readonly totalVariation?: number
  /** Gaussian blur of the pixels every `every` steps (default none; with the `pixel` parametrisation only). */
  readonly blur?: { readonly sigma: number; readonly every: number }
  /** The number of steps before completion (default 100). */
  readonly steps?: number
}

/** The state of {@link featureVisualisation}. */
export interface FeatureVisualisationState {
  /** The spectrum (`real`, `imag`) or the pixels (`pixels`) being optimised. */
  readonly params: { readonly real?: Value; readonly imag?: Value; readonly pixels?: Value }
  readonly optimiser: unknown
  readonly step: number
  /** The objective on the current image without any transformation. */
  readonly activation: number
  /** The image `[C, H, W]` in [0, 1]. */
  readonly image: Tensor
}

/** The paths of {@link featureVisualisation}. */
export type FeatureVisualisationPath = 'vis/activation' | 'vis/loss' | 'vis/image'

/** What a feature-visualisation run logs. */
export const featureVisualisationContract: Contract<FeatureVisualisationState, FeatureVisualisationPath> =
  defineContract({
    'vis/activation': scalar({ description: 'the objective on the current image, untransformed', direction: 'higher' }),
    'vis/loss': scalar({
      description: 'minus the objective on the transformed image plus the penalties',
      direction: 'lower',
    }),
    'vis/image': tensorSource({
      role: 'state',
      shape: ['channels', 'height', 'width'],
      description: 'the image so far, in [0, 1]',
    }),
  })

/**
 * The total variation of an image `[1, C, H, W]`: the sum of absolute differences between neighbouring pixels (the
 * anisotropic form), which penalises high-frequency noise.
 *
 * @param image - The image.
 * @returns The sum.
 * @example
 * ```ts
 * import { totalVariation } from 'mdml-engine/explain'
 * import { tensor, toFlat } from 'mdml-engine/tensor'
 * toFlat(totalVariation(tensor([[[[0, 1], [1, 0]]]]))) // => [4]
 * ```
 */
export function totalVariation(image: Value): Value {
  const [, , h = 1, w = 1] = image.shape
  const dy = abs(sub(sliceAxis(image, 2, 1, h), sliceAxis(image, 2, 0, h - 1)))
  const dx = abs(sub(sliceAxis(image, 3, 1, w), sliceAxis(image, 3, 0, w - 1)))
  return add(sum(dy), sum(dx))
}

/** The 1/f^decay scaling of a spectrum `[C, H, W]` (frequencies in cycles per pixel, floored at one cycle per image). */
function spectrumScale(c: number, h: number, w: number, decay: number): Tensor {
  const out = new Float64Array(c * h * w)
  const freq = (i: number, n: number): number => (i <= n / 2 ? i : i - n) / n
  const floor = 1 / Math.max(h, w)
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const f = Math.max(Math.hypot(freq(y, h), freq(x, w)), floor)
      const s = (1 / f) ** decay * Math.sqrt(h * w)
      for (let q = 0; q < c; q++) out[(q * h + y) * w + x] = s
    }
  return fromStorage(out, [c, h, w], 'float64')
}

/**
 * Activation maximisation as an algorithm: each step is one gradient step on minus the objective of the transformed
 * image plus the penalties. The image is a sigmoid of either the pixels or the real part of the inverse Fourier
 * transform of a scaled spectrum. The state's `activation` is the objective on the current untransformed image, which
 * is the quantity to watch rise.
 *
 * @param model - The network, from an image `[1, C, H, W]`.
 * @param options - The image shape, objective, parametrisation, transformations and penalties.
 * @returns The algorithm (its key draws the initial image and the transformations).
 * @concept feature-visualisation
 * @example
 * ```ts
 * import { featureVisualisation, unitObjective } from 'mdml-engine/explain'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * import { mean } from 'mdml-engine/tensor'
 * // A "network" whose logit is the mean brightness: ascent makes the image brighter.
 * const alg = featureVisualisation((x) => mean(x, [1, 2, 3], true), { shape: [1, 4, 4], steps: 20, objective: unitObjective(0), parametrisation: 'pixel', optimiser: 'sgd', learningRate: 5 })
 * runAlgorithm(alg, { maxSteps: 20, seed: 0 }).state.activation > 0.6 // => true
 * ```
 */
export function featureVisualisation(
  model: ImageModel,
  options: FeatureVisualisationOptions,
): Algorithm<FeatureVisualisationState, FeatureVisualisationPath> {
  const [c, h, w] = options.shape
  if (![c, h, w].every((n) => Number.isInteger(n) && n >= 1))
    throw new MdmlError('shape', 'featureVisualisation: shape must be [channels, height, width]')
  const objective = options.objective ?? unitObjective(0)
  const fourier = (options.parametrisation ?? 'fourier') === 'fourier'
  if (fourier && options.blur !== undefined)
    throw new MdmlError(
      'value',
      'featureVisualisation: blur needs the pixel parametrisation (the Fourier one is already low-pass)',
    )
  const scale = spectrumScale(c, h, w, options.decay ?? 1)
  const steps = options.steps ?? 100
  const jitter = options.jitter ?? 0
  const rotate = ((options.rotate ?? 0) * Math.PI) / 180
  const tv = options.totalVariation ?? 0
  const optimiser =
    (options.optimiser ?? 'adam') === 'adam'
      ? adam({ learningRate: options.learningRate ?? 0.05 })
      : sgd({ learningRate: options.learningRate ?? 0.1 })
  const imageOf = (p: FeatureVisualisationState['params']): Value => {
    if (fourier) {
      const spec = complex(mul(p.real as Value, scale), mul(p.imag as Value, scale))
      return sigmoid(real(ifft2(spec)))
    }
    return sigmoid(p.pixels as Value)
  }
  const batch = (img: Value): Value => reshape(img, [1, c, h, w])
  const activationOf = (p: FeatureVisualisationState['params']): number => {
    const out = model(batch(primalOf(imageOf(p))))
    return toFlat(asTensor(primalOf(objective(out))))[0] as number
  }
  const view = (p: FeatureVisualisationState['params']): Tensor => primalOf(imageOf(p))
  return {
    name: 'featureVisualisation',
    config: {
      shape: [c, h, w],
      parametrisation: fourier ? 'fourier' : 'pixel',
      jitter,
      rotate: options.rotate ?? 0,
      scale: options.scale === undefined ? null : [...options.scale],
      totalVariation: tv,
      steps,
    } as Json,
    init: (key: Key) => {
      const noise = (k: Key): Value => mul(normal(k, [c, h, w]), fourier ? 0.01 : 0.05)
      const params = fourier ? { real: noise(foldIn(key, 0)), imag: noise(foldIn(key, 1)) } : { pixels: noise(key) }
      return {
        params,
        optimiser: optimiser.init(params),
        step: 0,
        activation: activationOf(params),
        image: view(params),
      }
    },
    step: (s, ctx) => {
      const u = uniform(ctx.key, [5]).data
      const dx = jitter > 0 ? Math.round((u[0] as number) * 2 * jitter - jitter) : 0
      const dy = jitter > 0 ? Math.round((u[1] as number) * 2 * jitter - jitter) : 0
      const angle = rotate > 0 ? ((u[2] as number) * 2 - 1) * rotate : 0
      const zoom =
        options.scale === undefined ? 1 : options.scale[0] + (u[3] as number) * (options.scale[1] - options.scale[0])
      const transformed = (img: Value): Value => {
        let x = batch(img)
        if (dx !== 0) x = roll(x, dx, 3)
        if (dy !== 0) x = roll(x, dy, 2)
        if (angle !== 0 || zoom !== 1) {
          const cos = Math.cos(angle) / zoom
          const sin = Math.sin(angle) / zoom
          const theta = tensor([
            [
              [cos, -sin, 0],
              [sin, cos, 0],
            ],
          ])
          x = gridSample(x, affineGrid(theta, [1, c, h, w]), { paddingMode: 'reflection' })
        }
        return x
      }
      const loss = (p: FeatureVisualisationState['params']): Value => {
        const img = imageOf(p)
        const raised = objective(model(transformed(img)))
        const penalty = tv > 0 ? mul(totalVariation(batch(img)), tv / (c * h * w)) : 0
        return add(mul(sum(raised), -1), penalty)
      }
      const { value, grad } = valueAndGrad<FeatureVisualisationState['params']>(loss)(s.params)
      const { updates, state } = optimiser.update(grad, s.optimiser as unknown[], s.params)
      let params = applyUpdates(s.params, updates) as FeatureVisualisationState['params']
      const t = s.step + 1
      if (options.blur !== undefined && t % options.blur.every === 0) {
        const blurred = (v: Value): Value =>
          gaussianFilter(asTensor(primalOf(v)), [0, options.blur?.sigma ?? 0, options.blur?.sigma ?? 0])
        params = { pixels: blurred(params.pixels as Value) }
      }
      const activation = activationOf(params)
      const image = view(params)
      ctx.log('vis/activation', activation)
      ctx.log('vis/loss', toFlat(asTensor(primalOf(value)))[0] as number)
      if (ctx.wants('vis/image')) ctx.log('vis/image', image)
      return { params, optimiser: state, step: t, activation, image }
    },
    done: (s) => (s.step >= steps ? 'completed' : undefined),
    observe: (s) => ({
      metrics: { activation: s.activation },
      tensors: { image: { role: 'state', value: s.image } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as FeatureVisualisationState,
  }
}
