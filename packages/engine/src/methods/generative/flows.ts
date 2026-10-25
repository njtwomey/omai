/**
 * Normalising flows on tabular, mostly two-dimensional, data: RealNVP (Dinh, Sohl-Dickstein and Bengio, 2017), a stack
 * of affine coupling layers that each transform half the features using the other half, so the map stays invertible
 * with an exact log density. Trained by maximum likelihood, recording how samples and the density change.
 */

import { valueAndGrad } from '../../lib/autodiff/index.ts'
import { WhiteboxError } from '../../lib/base/index.ts'
import { type EmbeddingSteps, type Explained, type Exposition, embeddingSteps } from '../../lib/exposition/index.ts'
import type { FitOptions, LearnData } from '../../lib/learn/index.ts'
import { type ActivationName, blocks, type Params } from '../../lib/nn/index.ts'
import { adam } from '../../lib/optim/index.ts'
import { foldIn } from '../../lib/random/index.ts'
import {
  add,
  asTensor,
  fromStorage,
  item,
  mean,
  mul,
  neg,
  square,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
} from '../../lib/tensor/index.ts'
import { boxOf, DATA_LIMIT, densitySteps, gridEdges, gridOf, samplesSteps, stepCurve } from './_records.ts'
import {
  descend,
  type GenerativeProgress,
  logScale,
  minibatch,
  mlp,
  noise,
  type Optimised,
  optimising,
  prepare,
  runKey,
  thin,
  toData,
  toModel,
  trainLoop,
} from './_shared.ts'

/** Options of {@link realNvp}. */
export interface RealNvpOptions {
  /** Affine coupling layers, alternating which half of the features passes through (default 6). */
  readonly layers?: number
  /** Hidden width of each coupling's conditioner, two hidden layers (default 24). */
  readonly hidden?: number
  readonly activation?: ActivationName
  /** Adam steps (default 400). */
  readonly steps?: number
  /** Minibatch size (default 64). */
  readonly batchSize?: number
  /** Adam's learning rate (default 0.01). */
  readonly learningRate?: number
  /** Grid points per axis of the recorded density (default 32). */
  readonly resolution?: number
  /** Samples drawn from fixed base noise at each recorded step (default 300). */
  readonly samples?: number
  /** The most recorded steps (default 40). */
  readonly frames?: number
  /** Called at every recorded step with the records so far. */
  readonly onProgress?: (progress: GenerativeProgress) => void
}

/** What {@link realNvp} returns beside its records. */
export interface RealNvpResult {
  /** The negative log likelihood per row of the data, in its own coordinates, at the end. */
  readonly negativeLogLikelihood: number
  /** The same for the initial (identity) flow: a standard normal fitted by standardising. */
  readonly initialNegativeLogLikelihood: number
  readonly steps: number
  readonly status: string
  /** The log density of rows in the data's coordinates under the trained flow. */
  readonly logProb: (x: TensorLike) => number[]
}

type Coupling = blocks.AffineCoupling

// The flow's normalising direction, data to base: every coupling in order, summing log |det J|.
function toBase(couplings: readonly Coupling[], params: Params, x: Value): { z: Value; logDet: Value } {
  let h = x
  let logDet: Value | number = 0
  for (const [l, c] of couplings.entries()) {
    const r = c.apply((params[String(l)] ?? {}) as Params, h)
    h = r.output
    logDet = add(logDet, r.logDet)
  }
  return { z: h, logDet: logDet as Value }
}

// The generative direction, base to data: the inverses in reverse order.
function fromBase(couplings: readonly Coupling[], params: Params, z: Value): Value {
  let h = z
  for (let l = couplings.length - 1; l >= 0; l--)
    h = (couplings[l] as Coupling).inverse((params[String(l)] ?? {}) as Params, h).output
  return h
}

// Every coupling's output on the way from the data to the base, the data first.
function layerOutputs(couplings: readonly Coupling[], params: Params, x: Value): Value[] {
  const out = [x]
  let h = x
  for (const [l, c] of couplings.entries()) {
    h = c.apply((params[String(l)] ?? {}) as Params, h).output
    out.push(h)
  }
  return out
}

// log N(z; 0, I) per row.
function standardLogProb(z: Value, d: number): Value {
  return sub(mul(sum(square(z), -1), -0.5), (d / 2) * Math.log(2 * Math.PI))
}

interface FlowFrame {
  readonly density: Float64Array
  readonly samples: number[][]
  readonly warped: number[][]
  readonly nll: number
}

/**
 * RealNVP (Dinh, Sohl-Dickstein and Bengio, 2017): a stack of affine coupling layers (`nn.blocks.AffineCoupling`), each
 * passing half the features through unchanged and using them to scale and shift the other half, trained by maximum
 * likelihood under a standard normal base. The data are standardised first and every conditioner's last layer starts
 * at zero, so training starts from the identity map (a Gaussian fitted to the data's means and variances).
 *
 * Exposition, in order: the density on a grid over the first two features at each recorded step (a decision surface
 * whose steps carry values, with the negative log likelihood); samples pushed from fixed base noise beside the data at
 * each step; a regular grid of the base space pushed into the data space at each step (embedding steps with the grid's
 * edges as the graph, coloured by the radius in the base); the trained flow's layers, the data carried layer by layer
 * to the Gaussian; and the minibatch negative log likelihood per step.
 *
 * @param data - Rows `[n, d]`, d at least 2 (labels, when given, only colour the layer view).
 * @param options - Layers, conditioner width, steps, batch size, learning rate and recording.
 * @param fit - The key, tracer and policy.
 * @returns The final likelihoods, the log density and the records.
 */
export function realNvp(data: LearnData, options: RealNvpOptions = {}, fit: FitOptions = {}): Explained<RealNvpResult> {
  const p = prepare(data, 'realNvp')
  if (p.d < 2) throw new WhiteboxError('value', 'realNvp: needs at least two features')
  const key = runKey(fit)
  const d = p.d
  const hidden = options.hidden ?? 24
  const nLayers = options.layers ?? 6
  const couplings: Coupling[] = Array.from(
    { length: nLayers },
    (_, l) =>
      new blocks.AffineCoupling(
        mlp([d, hidden, hidden, 2 * d], options.activation ?? 'tanh', foldIn(key, 100 + l), { zeroLast: true }),
        d,
        {
          parity: l % 2,
        },
      ),
  )
  const initial: Params = Object.fromEntries(couplings.map((c, l) => [String(l), c.params]))
  const optimizer = adam({ learningRate: options.learningRate ?? 0.01 })
  const batchSize = options.batchSize ?? 64
  const grid = gridOf(p, options.resolution ?? 32)
  const base = noise(foldIn(key, 1), options.samples ?? 300, d)
  // A regular grid of the base space (other coordinates at zero), to push into the data space.
  const nodes = 17
  const span = 2.5
  const baseGrid: number[][] = []
  for (let i = 0; i < nodes; i++)
    for (let j = 0; j < nodes; j++)
      baseGrid.push([
        -span + (2 * span * j) / (nodes - 1),
        -span + (2 * span * i) / (nodes - 1),
        ...new Array<number>(d - 2).fill(0),
      ])
  const baseGridTensor = fromStorage(Float64Array.from(baseGrid.flat()), [baseGrid.length, d], 'float64') as Tensor
  const radius = baseGrid.map((r) => Math.hypot(r[0] ?? 0, r[1] ?? 0))
  const evalRows = toModel(p, thin(p.rows, 300))
  const shift = logScale(p)
  const logProbModel = (params: Params, x: Value): Value => {
    const { z, logDet } = toBase(couplings, params, x)
    return add(standardLogProb(z, d), logDet)
  }
  const loss =
    (x: Value) =>
    (params: Params): Value =>
      neg(mean(logProbModel(params, x)))
  // The other coordinates of a grid point sit at the data's mean (zero once standardised).
  const capture = (s: Optimised): FlowFrame => {
    const lp = toFlat(logProbModel(s.params, grid.points) as Tensor)
    return {
      density: Float64Array.from(lp, (v) => Math.exp(v - shift)),
      samples: toData(p, fromBase(couplings, s.params, base)),
      warped: toData(p, fromBase(couplings, s.params, baseGridTensor)),
      nll: -item(mean(logProbModel(s.params, evalRows))) + shift,
    }
  }
  const box = boxOf(p.rows, 0.6)
  const records = (
    frames: readonly { step: number; frame: FlowFrame }[],
    losses: Readonly<Record<string, readonly number[]>>,
  ): Exposition[] => {
    const steps = frames.map((f) => f.step)
    const nll = frames.map((f) => f.frame.nll)
    const warp: EmbeddingSteps = embeddingSteps({
      method: 'RealNVP: the base grid in data space',
      positions: frames.map((f) =>
        f.frame.warped.map((q) => [
          Math.min(Math.max(q[0] ?? 0, box[0]), box[1]),
          Math.min(Math.max(q[1] ?? 0, box[2]), box[3]),
        ]),
      ),
      iterations: steps,
      graph: gridEdges(nodes, nodes),
      colourBy: { kind: 'values', values: radius, name: 'radius in the base' },
    })
    return [
      densitySteps(
        p,
        grid,
        steps,
        frames.map((f) => f.frame.density),
        { name: 'negative log likelihood', values: nll },
      ),
      samplesSteps(
        'RealNVP samples',
        p,
        steps,
        frames.map((f) => f.frame.samples),
        { objective: nll },
      ),
      warp,
      stepCurve({ values: losses['nll'] ?? [], label: 'minibatch negative log likelihood' }),
    ]
  }
  const run = trainLoop<Optimised, FlowFrame>({
    name: 'realNvp',
    steps: options.steps ?? 400,
    frames: options.frames ?? 40,
    init: () => optimising(initial, optimizer),
    step: (s, _t, k) => {
      const batch = minibatch(p.x, k, batchSize)
      const { value, grad } = valueAndGrad(loss(batch))(s.params)
      return { state: descend(optimizer, s, grad as Params), losses: { nll: item(value) + shift } }
    },
    capture,
    records,
    fit,
    ...(options.onProgress === undefined ? {} : { onProgress: options.onProgress }),
  })
  const final = run.state.params
  // The data carried through the trained flow layer by layer (standardised coordinates).
  const shown = thin(p.rows, DATA_LIMIT)
  const outputs = layerOutputs(couplings, final, toModel(p, shown))
  const shownLabels = p.labels === undefined ? undefined : thin(p.labels, DATA_LIMIT)
  const layersRecord = embeddingSteps({
    method: 'RealNVP layer by layer',
    positions: outputs.map((h) => {
      const flat = toFlat(h as Tensor)
      return shown.map((_, i) => [flat[i * d] ?? 0, flat[i * d + 1] ?? 0])
    }),
    iterations: outputs.map((_, l) => l),
    ...(shownLabels === undefined
      ? {}
      : {
          colourBy: {
            kind: 'classes' as const,
            labels: shownLabels,
            ...(p.classes === undefined ? {} : { classes: p.classes }),
          },
        }),
  })
  const [density, samples, warp, curve] = run.exposition
  const nlls = run.frames.map((f) => f.frame.nll)
  return {
    negativeLogLikelihood: nlls.at(-1) ?? Number.NaN,
    initialNegativeLogLikelihood: nlls[0] ?? Number.NaN,
    steps: run.steps,
    status: run.status,
    logProb: (x) => toFlat(logProbModel(final, toModel(p, toRows(x, d))) as Tensor).map((v) => v - shift),
    exposition: [density, samples, warp, layersRecord, curve].filter((r): r is Exposition => r !== undefined),
  }
}

function toRows(x: TensorLike, d: number): number[][] {
  const flat = toFlat(asTensor(x))
  return Array.from({ length: Math.floor(flat.length / d) }, (_, i) => flat.slice(i * d, (i + 1) * d))
}
