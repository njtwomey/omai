import { type Json, WhiteboxError } from '../../../lib/base/index.ts'
import { coefficientTable, type Exposition } from '../../../lib/exposition/index.ts'
import {
  defineExplainedEstimator,
  type ExplainedEstimator,
  ExplainedEstimatorClass,
  type FitOptions,
  type Fitted,
  type LearnData,
  matrixView,
} from '../../../lib/learn/index.ts'
import {
  AvgPool2d,
  activationModule,
  Conv2d,
  describe,
  Flatten,
  Linear,
  loadState,
  MaxPool2d,
  type Module,
  type Params,
  type SavedState,
  Sequential,
  saveState,
  Unflatten,
} from '../../../lib/nn/index.ts'
import { foldIn, type Key, key as makeKey } from '../../../lib/random/index.ts'
import type { Structure } from '../../../lib/scene/index.ts'
import { choice, dict, int, real, type Space } from '../../../lib/space/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat, type Value } from '../../../lib/tensor/index.ts'
import { lossOf, optimiserOf, predictorOf, scoreOf, type Task, taskOf } from '../_nets.ts'
import {
  lossAtSnapshots,
  lossCurve,
  mapSteps,
  progressReporter,
  scoreCurve,
  type TrainingProgress,
} from '../_records.ts'
import { type Snapshot, train } from '../_train.ts'

/** Hyperparameters of {@link lenet}. */
export interface LeNetHyper {
  /** The image shape [channels, height, width] of each flattened row (default inferred: square, one channel). */
  readonly imageShape?: readonly [number, number, number]
  /** Channels of the two convolutions (default [6, 12]). */
  readonly channels: readonly [number, number]
  /** The convolutions' kernel size (default 3, padded to keep the size). */
  readonly kernelSize: number
  /** Units of the hidden fully connected layer (default 32). */
  readonly hidden: number
  readonly activation: 'relu' | 'tanh'
  /** Subsampling after each convolution (LeNet-5 averaged; max pooling is today's default). */
  readonly pool: 'max' | 'avg'
  readonly epochs: number
  readonly batchSize: number
  /** Adam's learning rate (default 0.02). */
  readonly learningRate: number
  /** Rows held out for the validation curves (default 0.2). */
  readonly validationFraction: number
  /** Divide the inputs by this (default: their largest absolute value, so pixels lie in [0, 1]). */
  readonly scale?: number
  /** The most snapshots of the feature maps (spaced geometrically over minibatch steps, default 30). */
  readonly snapshots: number
  /** Which row's feature maps are tapped (default 0, the first row). */
  readonly tapRow: number
  readonly onProgress?: (progress: TrainingProgress) => void
}

/** The learnt parameters of a LeNet. */
export interface LeNetParams {
  readonly [key: string]: Json
  readonly imageShape: readonly number[]
  readonly channels: readonly number[]
  readonly kernelSize: number
  readonly hidden: number
  readonly activation: string
  readonly pool: string
  readonly classes: readonly number[]
  readonly scale: number
  readonly state: Json
}

/** The verbs of a fitted LeNet. */
export interface LeNetVerbs {
  readonly predict: (x: TensorLike) => Tensor
  readonly predictProba?: (x: TensorLike) => Tensor
  readonly classes: readonly number[]
  readonly nFeaturesIn: number
  readonly describe: () => Structure
  /** The feature maps of each convolution (after the activation) for some rows: maps[layer][row][channel][i][j]. */
  readonly featureMaps: (x: TensorLike) => number[][][][][]
}

/**
 * A LeNet-style network for small images given as flattened rows: unflatten, two convolution-activation-pooling
 * stages, a hidden fully connected layer and a linear output over the classes.
 *
 * @param shape - [channels, height, width] (height and width divisible by 4).
 * @param classes - The number of classes.
 * @param h - Channels, kernel size, hidden units, activation and pooling.
 * @param key - The initialisation key.
 * @returns The module.
 */
export function lenetNetwork(
  shape: readonly [number, number, number],
  classes: number,
  h: Pick<LeNetHyper, 'channels' | 'kernelSize' | 'hidden' | 'activation' | 'pool'>,
  key: Key,
): Sequential {
  const [c, hh, ww] = shape
  const [c1, c2] = h.channels
  const pad = Math.floor(h.kernelSize / 2)
  const pool = (): Module => (h.pool === 'avg' ? new AvgPool2d(2) : new MaxPool2d(2))
  const flat = c2 * Math.floor(hh / 4) * Math.floor(ww / 4)
  return new Sequential(
    new Unflatten(1, [c, hh, ww]),
    new Conv2d(c, c1, h.kernelSize, { padding: pad, key: foldIn(key, 0) }),
    activationModule(h.activation),
    pool(),
    new Conv2d(c1, c2, h.kernelSize, { padding: pad, key: foldIn(key, 1) }),
    activationModule(h.activation),
    pool(),
    new Flatten(),
    new Linear(flat, h.hidden, { key: foldIn(key, 2) }),
    activationModule(h.activation),
    new Linear(h.hidden, classes, { key: foldIn(key, 3) }),
  )
}

// The children after which a convolution's activated maps are read (the activation following each Conv2d).
function convTaps(net: Sequential): number[] {
  const children = net.children()
  return children.flatMap(([, m], i) => (children[i - 1]?.[1].kind === 'Conv2d' && m.kind !== 'Conv2d' ? [i] : []))
}

/** The activated maps of each convolution for some rows: maps[layer][row][channel][i][j]. */
function featureMapsOf(net: Sequential, params: Params, x: Value): number[][][][][] {
  const taps = new Set(convTaps(net))
  const out: number[][][][][] = []
  let h = x
  for (const [i, [name, m]] of net.children().entries()) {
    h = (m as Module).apply((params[name] ?? {}) as Params, h, { train: false })
    if (!taps.has(i)) continue
    const t = h as Tensor
    const [n = 0, c = 0, hh = 0, ww = 0] = t.shape
    const flat = toFlat(t)
    out.push(
      Array.from({ length: n }, (_, r) =>
        Array.from({ length: c }, (_, k) =>
          Array.from({ length: hh }, (_, a) =>
            flat.slice(((r * c + k) * hh + a) * ww, ((r * c + k) * hh + a + 1) * ww),
          ),
        ),
      ),
    )
  }
  return out
}

/** The first convolution's filters, one map per output channel (averaged over the input channels). */
function firstFilters(net: Sequential, params: Params): number[][][] {
  const name = net.children().find(([, m]) => m.kind === 'Conv2d')?.[0] ?? '1'
  const w = (params[name] as { readonly weight: Tensor }).weight
  const [o = 0, c = 0, kh = 0, kw = 0] = w.shape
  const flat = toFlat(w)
  return Array.from({ length: o }, (_, k) =>
    Array.from({ length: kh }, (_, a) =>
      Array.from({ length: kw }, (_, b) => {
        let s = 0
        for (let ch = 0; ch < c; ch++) s += flat[((k * c + ch) * kh + a) * kw + b] ?? 0
        return s / Math.max(1, c)
      }),
    ),
  )
}

function inferShape(d: number, given?: readonly [number, number, number]): [number, number, number] {
  if (given !== undefined) {
    if (given[0] * given[1] * given[2] !== d)
      throw new WhiteboxError('shape', `lenet: imageShape [${given.join(', ')}] does not hold ${d} features`)
    return [...given]
  }
  const side = Math.round(Math.sqrt(d))
  if (side * side !== d)
    throw new WhiteboxError('shape', `lenet: ${d} features are not a square image; give imageShape`)
  return [1, side, side]
}

function scaled(x: TensorLike, scale: number): Tensor {
  const m = matrixView(x, 'lenet')
  return fromStorage(
    Float64Array.from(m.data, (v) => v / scale),
    [m.n, m.d],
    'float64',
  ) as Tensor
}

function fitLeNet(
  data: LearnData,
  h: LeNetHyper,
  options: FitOptions,
): { params: LeNetParams; exposition: Exposition[] } {
  const key = options.key ?? makeKey(0)
  const raw = matrixView(data.x, 'lenet')
  const shape = inferShape(raw.d, h.imageShape)
  if (shape[1] % 4 !== 0 || shape[2] % 4 !== 0)
    throw new WhiteboxError('shape', 'lenet: the image height and width must be divisible by 4 (two poolings by 2)')
  let peak = 0
  for (const v of raw.data) peak = Math.max(peak, Math.abs(v))
  const scale = h.scale ?? (peak > 0 ? peak : 1)
  const task: Task = taskOf({ ...data, x: scaled(data.x, scale) }, true, {
    who: 'lenet',
    key,
    holdOut: h.validationFraction,
  })
  // The LeNet always has a softmax over its classes (two classes too), so its loss and score are the softmax's.
  const output = 'softmax' as const
  const net = lenetNetwork(shape, task.classes.length, h, foldIn(key, 1))
  const tapped = task.validation?.x ?? task.rows.x
  const tapIndex = Math.min(Math.max(0, h.tapRow), (tapped.shape[0] ?? 1) - 1)
  const tapInput = fromStorage(
    Float64Array.from(toFlat(tapped).slice(tapIndex * raw.d, (tapIndex + 1) * raw.d)),
    [1, raw.d],
    'float64',
  ) as Tensor
  const mapsAt = (s: Pick<Snapshot, 'params'>): number[][][][] =>
    featureMapsOf(net, s.params, tapInput).map((l) => l[0] ?? [])
  const training = train({
    name: 'lenet',
    model: net,
    train: task.rows,
    ...(task.validation === undefined ? {} : { validation: task.validation }),
    loss: lossOf(output),
    score: scoreOf(output),
    optimizer: optimiserOf('adam', h.learningRate),
    learningRate: h.learningRate,
    epochs: h.epochs,
    batchSize: h.batchSize,
    snapshots: h.snapshots,
    snapshotUnit: 'step',
    fit: { ...options, key: foldIn(key, 2) },
    ...(h.onProgress === undefined
      ? {}
      : {
          onEpoch: progressReporter(h.onProgress, (s) =>
            mapsAt(s).map((maps) => mapSteps([maps], [s.step], Math.ceil(Math.sqrt(maps.length)))),
          ),
        }),
  })
  net.setParams(training.params)
  const params: LeNetParams = {
    imageShape: shape,
    channels: [...h.channels],
    kernelSize: h.kernelSize,
    hidden: h.hidden,
    activation: h.activation,
    pool: h.pool,
    classes: [...task.classes],
    scale,
    state: saveState(net) as unknown as Json,
  }
  const iterations = training.snapshots.map((s) => s.step)
  const objective = lossAtSnapshots(training)
  const perLayer = training.snapshots.map(mapsAt)
  const layers = perLayer[0]?.length ?? 0
  const mapRecords = Array.from({ length: layers }, (_, l) => {
    const count = perLayer[0]?.[l]?.length ?? 1
    return {
      ...mapSteps(
        perLayer.map((p) => p[l] ?? []),
        iterations,
        Math.ceil(Math.sqrt(count)),
        objective,
      ),
      stepName: 'step',
    }
  })
  const filters = {
    ...mapSteps(
      training.snapshots.map((s) => firstFilters(net, s.params)),
      iterations,
      Math.ceil(Math.sqrt(h.channels[0])),
      objective,
      false,
    ),
    stepName: 'step',
  }
  const image = mapSteps([[tapInput2d(tapInput, shape)]], [0], 1)
  // Accuracy per class on the validation rows (or the training rows).
  const rows = task.validation ?? task.rows
  const predicted = toFlat(
    predictorOf(
      net,
      training.params,
      training.buffers,
      output,
      task.classes.map((_, i) => i),
    ).predict(rows.x),
  )
  const truth = toFlat(rows.y)
  const perClass = task.classes.map((_, c) => {
    const idx = truth.flatMap((t, i) => (t === c ? [i] : []))
    return idx.length === 0 ? Number.NaN : idx.filter((i) => predicted[i] === c).length / idx.length
  })
  const exposition: Exposition[] = [
    lossCurve(training),
    ...(scoreCurve(training, 'accuracy') === undefined ? [] : [scoreCurve(training, 'accuracy') as Exposition]),
    image,
    ...mapRecords,
    filters,
    coefficientTable({
      names: (task.classNames ?? task.classes.map(String)).map((c) => `accuracy on ${c}`),
      estimates: perClass,
    }),
  ]
  return { params, exposition }
}

function tapInput2d(x: Tensor, shape: readonly [number, number, number]): number[][] {
  const flat = toFlat(x)
  const [, hh, ww] = shape
  return Array.from({ length: hh }, (_, a) => flat.slice(a * ww, (a + 1) * ww))
}

function lenetVerbs(p: LeNetParams): LeNetVerbs {
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
  const d = shape[0] * shape[1] * shape[2]
  return {
    ...predictorOf(net, net.params, net.buffers, 'softmax', p.classes, (x) => scaled(x, p.scale)),
    classes: p.classes,
    nFeaturesIn: d,
    describe: () => describe(net, [1, d]),
    featureMaps: (x) => featureMapsOf(net, net.params, scaled(x, p.scale)),
  }
}

/**
 * A LeNet (LeCun et al., 1998) for small images given as flattened rows (the 8 by 8 digits by default): two
 * convolution, activation and pooling stages, a hidden fully connected layer and a softmax over the classes, trained
 * by Adam on minibatches. Exposition, in order: the loss and accuracy per epoch (training against held-out rows), the
 * tapped input image, the activated feature maps of each convolution for that image at about 30 geometrically spaced
 * steps (playable: the filters' responses sharpen as training proceeds), the first convolution's filters at the same
 * steps, and the accuracy per class.
 *
 * @param hyper - Image shape, channels, kernel, hidden units, activation, pooling, epochs, batching, rate and taps.
 * @returns The estimator.
 */
export function lenet(
  hyper: Partial<LeNetHyper> = {},
): ExplainedEstimator<LeNetHyper, Fitted<LeNetParams> & LeNetVerbs> {
  return defineExplainedEstimator(
    {
      name: 'lenet',
      kind: 'classifier',
      hyperSpace: (h: LeNetHyper): Space =>
        dict({
          epochs: int(1, 50, { default: h.epochs }),
          batchSize: int(8, 256, { log: true, default: h.batchSize }),
          learningRate: real(1e-4, 0.1, { log: true, default: h.learningRate }),
          activation: choice(['relu', 'tanh'] as const, { default: h.activation }),
          pool: choice(['max', 'avg'] as const, { default: h.pool }),
          hidden: int(8, 128, { log: true, default: h.hidden }),
        }),
      fit: fitLeNet,
      methods: lenetVerbs,
    },
    {
      channels: [6, 12],
      kernelSize: 3,
      hidden: 32,
      activation: 'relu',
      pool: 'max',
      epochs: 5,
      batchSize: 32,
      learningRate: 0.02,
      validationFraction: 0.2,
      snapshots: 30,
      tapRow: 0,
      ...hyper,
    },
  )
}

/** The class form of {@link lenet}. */
export class LeNet extends ExplainedEstimatorClass<LeNetHyper, Fitted<LeNetParams> & LeNetVerbs> {
  /** @param hyper - As {@link lenet}. */
  constructor(hyper: Partial<LeNetHyper> = {}) {
    super(lenet(hyper))
  }
}
