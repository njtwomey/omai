/**
 * Variational autoencoders (Kingma and Welling, 2014): an encoder maps each row to a Gaussian over latent codes, a
 * decoder maps codes back to a likelihood of the data (Gaussian or Bernoulli pixels), and both are trained together by
 * the reparameterisation trick on the evidence lower bound. Records the latent space, reconstructions and samples as
 * training goes on.
 */

import { valueAndGrad } from '../../lib/autodiff/index.ts'
import { WhiteboxError } from '../../lib/base/index.ts'
import { type Explained, type Exposition, embeddingSteps, imageSteps } from '../../lib/exposition/index.ts'
import type { FitOptions, LearnData } from '../../lib/learn/index.ts'
import { bernoulliLogLikelihood, gaussianKl, gaussianLogLikelihood } from '../../lib/losses/index.ts'
import { type ActivationName, type Params, zerosInit } from '../../lib/nn/index.ts'
import { adam } from '../../lib/optim/index.ts'
import { foldIn, type Key, normal } from '../../lib/random/index.ts'
import {
  add,
  exp,
  item,
  mean,
  mul,
  neg,
  primalOf,
  sigmoid,
  sliceAxis,
  type Tensor,
  toFlat,
  type Value,
} from '../../lib/tensor/index.ts'
import { DATA_LIMIT, samplesSteps, stepCurve } from './_records.ts'
import {
  descend,
  type Frame,
  type GenerativeProgress,
  logScale,
  minibatch,
  mlp,
  noise,
  type Optimised,
  optimising,
  type Prepared,
  prepare,
  runKey,
  thin,
  toData,
  toModel,
  trainLoop,
} from './_shared.ts'

/** How a VAE's decoder scores the data: a Gaussian with a learned variance per feature, or Bernoulli pixels. */
export type VaeLikelihood = 'gaussian' | 'bernoulli'

/** Options of {@link vae}. */
export interface VaeOptions {
  /** Latent dimensions (default 2, so the latent space can be drawn). */
  readonly latent?: number
  /** Hidden width of the encoder's and decoder's two hidden layers (default 32). */
  readonly hidden?: number
  readonly activation?: ActivationName
  /**
   * The decoder's likelihood (default 'gaussian'; 'bernoulli' takes data in [0, 1], such as pixel intensities, and
   * leaves it unstandardised).
   */
  readonly likelihood?: VaeLikelihood
  /**
   * A Gaussian decoder's noise standard deviation per feature, on standardised data (default 0.2), or 'learned' (one
   * per feature, trained with the rest; the decoder may then explain the data as noise and its means collapse to a
   * line).
   */
  readonly noiseScale?: number | 'learned'
  /** The weight of the KL term (beta-VAE; default 1, the ELBO). */
  readonly beta?: number
  /** Adam steps (default 600). */
  readonly steps?: number
  /** Minibatch size (default 64). */
  readonly batchSize?: number
  /** Adam's learning rate (default 0.005). */
  readonly learningRate?: number
  /** Samples decoded from fixed prior draws at each recorded step (default 300). */
  readonly samples?: number
  /** The most recorded steps (default 40). */
  readonly frames?: number
  /**
   * The rows are images of this height and width (row-major pixels): record the data, reconstructions and samples as
   * images at each step.
   */
  readonly imageShape?: readonly [number, number]
  /** Images drawn per group (default 10). */
  readonly images?: number
  /** Called at every recorded step with the records so far. */
  readonly onProgress?: (progress: GenerativeProgress) => void
}

/** A VAE's state at a recorded step, for drawing. */
export interface VaeFrame {
  /** Encoder means of the shown rows (their first two latent coordinates). */
  readonly latent: number[][]
  /** Decoder means of the shown rows' encoder means, in the data's coordinates. */
  readonly reconstructions: number[][]
  /** Decoder means of fixed draws from the prior, in the data's coordinates. */
  readonly samples: number[][]
  /** The negative ELBO per row on the shown rows (one fixed draw each), and its two terms. */
  readonly negativeElbo: number
  readonly reconstruction: number
  readonly kl: number
}

/** What {@link vae} returns beside its records. */
export interface VaeResult {
  /** The negative ELBO per row at the start and at the end (in the data's coordinates for a Gaussian decoder). */
  readonly initialNegativeElbo: number
  readonly negativeElbo: number
  /** The final reconstruction term -E log p(x | z) and KL term per row. */
  readonly reconstruction: number
  readonly kl: number
  readonly steps: number
  readonly status: string
  /** Every recorded frame (latent codes, reconstructions and samples, per step), for views of images. */
  readonly frames: readonly Frame<VaeFrame>[]
  /** The rows shown in the frames, in the data's coordinates, and their class indices when labelled. */
  readonly shown: readonly (readonly number[])[]
}

interface Halves {
  readonly mean: Value
  readonly logVariance: Value
}

/**
 * A variational autoencoder (Kingma and Welling, 2014): an encoder maps each row to a Gaussian q(z | x), a decoder maps
 * latent codes back to a likelihood p(x | z), and both are trained by the reparameterised ELBO, E_q log p(x | z) -
 * beta KL(q(z | x) || N(0, I)) (beta = 1 is the ELBO, more is a beta-VAE). A Gaussian decoder has a fixed noise scale (or learns one
 * per feature) on standardised data; a Bernoulli decoder scores pixel intensities in [0, 1].
 *
 * Exposition, in order: the encoder's means of the data (the latent space) at each recorded step, coloured by class
 * when the data is labelled; for two-dimensional data, the reconstructions (decoder means of the encoder means) and
 * samples (decoder means of fixed prior draws) beside the data at each step; the reconstruction and KL terms per step;
 * and the negative ELBO per step.
 *
 * @param data - Rows `[n, d]` (labels, when given, only colour the latent space).
 * @param options - Latent size, widths, likelihood, beta, steps, batch size, learning rate and recording.
 * @param fit - The key, tracer and policy.
 * @returns The final bound and its terms, the frames and the records.
 */
export function vae(data: LearnData, options: VaeOptions = {}, fit: FitOptions = {}): Explained<VaeResult> {
  const likelihood = options.likelihood ?? 'gaussian'
  const p = prepare(data, 'vae', { standardise: likelihood === 'gaussian' })
  if (likelihood === 'bernoulli' && p.rows.some((r) => r.some((v) => v < 0 || v > 1)))
    throw new WhiteboxError('value', 'vae: a Bernoulli decoder needs data in [0, 1]')
  const key = runKey(fit)
  const d = p.d
  const latent = options.latent ?? 2
  const hidden = options.hidden ?? 32
  const activation = options.activation ?? 'tanh'
  const beta = options.beta ?? 1
  const noiseScale = options.noiseScale ?? 0.2
  if (noiseScale !== 'learned' && !(noiseScale > 0))
    throw new WhiteboxError('value', 'vae: noiseScale must be positive')
  const encoder = mlp([d, hidden, hidden, 2 * latent], activation, foldIn(key, 1))
  const decoder = mlp([latent, hidden, hidden, d], activation, foldIn(key, 2))
  const initial: Params = {
    encoder: encoder.params,
    decoder: decoder.params,
    ...(likelihood === 'gaussian' && noiseScale === 'learned' ? { logVariance: zerosInit([d]) } : {}),
  }
  const optimizer = adam({ learningRate: options.learningRate ?? 0.005 })
  const batchSize = options.batchSize ?? 64
  const encode = (params: Params, x: Value): Halves => {
    const h = encoder.apply(params['encoder'] as Params, x)
    return { mean: sliceAxis(h, -1, 0, latent), logVariance: sliceAxis(h, -1, latent, 2 * latent) }
  }
  const decode = (params: Params, z: Value): Value => decoder.apply(params['decoder'] as Params, z)
  // log p(x | z) per row: Gaussian with the learned variances, or Bernoulli from logits.
  const logLik = (params: Params, out: Value, x: Value): Value =>
    likelihood === 'gaussian'
      ? gaussianLogLikelihood(out, x, {
          logVariance: noiseScale === 'learned' ? (params['logVariance'] as Value) : 2 * Math.log(noiseScale),
        })
      : bernoulliLogLikelihood(out, x)
  const terms = (params: Params, x: Value, eps: Value): { reconstruction: Value; kl: Value } => {
    const q = encode(params, x)
    const z = add(q.mean, mul(exp(mul(q.logVariance, 0.5)), eps))
    return {
      reconstruction: neg(mean(logLik(params, decode(params, z), x))),
      kl: gaussianKl(q.mean, q.logVariance),
    }
  }
  // A Gaussian decoder's likelihood is of standardised rows: the data's own is lower by the log scale.
  const shift = likelihood === 'gaussian' ? logScale(p) : 0
  const shown = thin(p.rows, DATA_LIMIT)
  const shownX = toModel(p, shown)
  const shownEps = normal(foldIn(key, 3), [shown.length, latent]) as Tensor
  const prior = noise(foldIn(key, 4), options.samples ?? 300, latent)
  const decoded = (params: Params, z: Value): number[][] => {
    const out = decode(params, z)
    return likelihood === 'gaussian' ? toData(p, out) : rowsOf(sigmoid(out) as Tensor, d)
  }
  const capture = (s: Optimised): VaeFrame => {
    const q = encode(s.params, shownX)
    const t = terms(s.params, shownX, shownEps)
    const reconstruction = item(t.reconstruction) + shift
    const kl = item(t.kl)
    return {
      latent: rowsOf(q.mean as Tensor, latent).map((r) => [r[0] ?? 0, r[1] ?? 0]),
      reconstructions: decoded(s.params, q.mean),
      samples: decoded(s.params, prior),
      negativeElbo: reconstruction + kl,
      reconstruction,
      kl,
    }
  }
  const shape = options.imageShape
  if (shape !== undefined && shape[0] * shape[1] !== d)
    throw new WhiteboxError('value', `vae: images of ${shape[0]} by ${shape[1]} do not have ${d} pixels`)
  const pictured = shape === undefined ? [] : pictureRows(p, shown.length, options.images ?? 10)
  const records = (frames: readonly Frame<VaeFrame>[], losses: Readonly<Record<string, readonly number[]>>) =>
    vaeRecords(p, shown, frames, losses, likelihood, shape === undefined ? undefined : { shape, rows: pictured })
  const run = trainLoop<Optimised, VaeFrame>({
    name: 'vae',
    steps: options.steps ?? 600,
    frames: options.frames ?? 40,
    init: () => optimising(initial, optimizer),
    step: (s, _t, k: Key) => {
      const batch = minibatch(p.x, foldIn(k, 0), batchSize)
      const eps = normal(foldIn(k, 1), [batch.shape[0] ?? 0, latent])
      const { grad, aux } = valueAndGrad(
        (params: Params) => {
          const t = terms(params, batch, eps)
          return [add(t.reconstruction, mul(t.kl, beta)), [t.reconstruction, t.kl]] as const
        },
        { hasAux: true },
      )(s.params)
      const [r, q] = aux as readonly [Value, Value]
      const reconstruction = item(primalOf(r)) + shift
      const kl = item(primalOf(q))
      return {
        state: descend(optimizer, s, grad as Params),
        losses: { reconstruction, kl, negativeElbo: reconstruction + kl },
      }
    },
    capture,
    records,
    fit,
    ...(options.onProgress === undefined ? {} : { onProgress: options.onProgress }),
  })
  const last = run.frames.at(-1)?.frame
  return {
    initialNegativeElbo: run.frames[0]?.frame.negativeElbo ?? Number.NaN,
    negativeElbo: last?.negativeElbo ?? Number.NaN,
    reconstruction: last?.reconstruction ?? Number.NaN,
    kl: last?.kl ?? Number.NaN,
    steps: run.steps,
    status: run.status,
    frames: run.frames,
    shown,
    exposition: run.exposition,
  }
}

function rowsOf(t: Tensor, width: number): number[][] {
  const flat = toFlat(t)
  return Array.from({ length: Math.floor(flat.length / width) }, (_, i) => flat.slice(i * width, (i + 1) * width))
}

// The shown rows to picture: one per class in turn when labelled (so every digit appears), else the first ones.
function pictureRows(p: Prepared, shown: number, count: number): number[] {
  const labels = p.labels === undefined ? undefined : thin(p.labels, DATA_LIMIT)
  if (labels === undefined) return Array.from({ length: Math.min(count, shown) }, (_, i) => i)
  const byClass = new Map<number, number[]>()
  for (const [i, l] of labels.entries()) byClass.set(l, [...(byClass.get(l) ?? []), i])
  const classes = [...byClass.keys()].sort((a, b) => a - b)
  const out: number[] = []
  for (let round = 0; out.length < Math.min(count, shown); round++)
    for (const c of classes) {
      const i = byClass.get(c)?.[round]
      if (i !== undefined && out.length < count) out.push(i)
    }
  return out
}

function vaeRecords(
  p: Prepared,
  shown: readonly (readonly number[])[],
  frames: readonly Frame<VaeFrame>[],
  losses: Readonly<Record<string, readonly number[]>>,
  likelihood: VaeLikelihood,
  pictures?: { readonly shape: readonly [number, number]; readonly rows: readonly number[] },
): Exposition[] {
  const steps = frames.map((f) => f.step)
  const elbo = frames.map((f) => f.frame.negativeElbo)
  const labels = p.labels === undefined ? undefined : thin(p.labels, DATA_LIMIT)
  const out: Exposition[] = [
    embeddingSteps({
      method: 'VAE latent space (encoder means)',
      positions: frames.map((f) => f.frame.latent),
      iterations: steps,
      objective: elbo,
      ...(labels === undefined
        ? {
            colourBy: {
              kind: 'values' as const,
              values: shown.map((r) => r[0] ?? 0),
              name: p.featureNames[0] ?? 'x0',
            },
          }
        : {
            colourBy: { kind: 'classes' as const, labels, ...(p.classes === undefined ? {} : { classes: p.classes }) },
          }),
    }),
  ]
  if (p.d === 2 && likelihood === 'gaussian')
    out.push(
      samplesSteps(
        'VAE reconstructions',
        p,
        steps,
        frames.map((f) => f.frame.reconstructions),
        { name: 'reconstruction', objective: elbo },
      ),
      samplesSteps(
        'VAE samples from the prior',
        p,
        steps,
        frames.map((f) => f.frame.samples),
        { objective: elbo },
      ),
    )
  if (pictures !== undefined) {
    const k = pictures.rows.length
    out.push(
      imageSteps({
        method: 'VAE',
        height: pictures.shape[0],
        width: pictures.shape[1],
        groups: ['data', 'reconstruction', 'samples'],
        steps: frames.map((f) => ({
          iteration: f.step,
          images: [
            pictures.rows.map((i) => [...(shown[i] ?? [])]),
            pictures.rows.map((i) => f.frame.reconstructions[i] ?? []),
            f.frame.samples.slice(0, k),
          ],
          objective: f.frame.negativeElbo,
        })),
        ...(likelihood === 'gaussian' ? { range: imageRange(shown) } : {}),
        objectiveName: 'negative ELBO',
      }),
    )
  }
  out.push(
    stepCurve(
      { values: losses['reconstruction'] ?? [], label: 'reconstruction −log p(x | z)' },
      { values: losses['kl'] ?? [], label: 'KL(q(z | x) ‖ p(z))' },
    ),
    stepCurve({ values: losses['negativeElbo'] ?? [], label: 'negative ELBO' }),
  )
  return out
}

// The grey range of rows pictured as images.
function imageRange(rows: readonly (readonly number[])[]): [number, number] {
  let lo = Number.POSITIVE_INFINITY
  let hi = Number.NEGATIVE_INFINITY
  for (const r of rows)
    for (const v of r) {
      lo = Math.min(lo, v)
      hi = Math.max(hi, v)
    }
  return hi > lo ? [lo, hi] : [0, 1]
}
