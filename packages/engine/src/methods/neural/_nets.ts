// Private network plumbing of the neural methods (methods/neural): the task a network learns from a dataset (inputs,
// encoded targets, the output layer's kind, held-out rows), its loss and score, MLP builders with a choice of
// initialisation, normalisation and dropout, the predictions of a network under any parameters (for snapshots), and
// the decision surface they draw.
import { WhiteboxError } from '../../lib/base/index.ts'
import type { DecisionSurface } from '../../lib/exposition/index.ts'
import {
  classNamesOf,
  decisionSurfaceOf,
  encodeLabels,
  type Fitted,
  featureNamesOf,
  type LearnData,
  matrixView,
} from '../../lib/learn/index.ts'
import { binaryCrossEntropyWithLogits, crossEntropy, mse } from '../../lib/losses/index.ts'
import { accuracyScore, r2Score } from '../../lib/metrics/index.ts'
import {
  type ActivationName,
  activationModule,
  BatchNorm1d,
  Dropout,
  kaimingNormal,
  LayerNorm,
  Linear,
  type Module,
  normalInit,
  type Params,
  Sequential,
  uniformInit,
  xavierNormal,
  zerosInit,
} from '../../lib/nn/index.ts'
import {
  adagrad,
  adam,
  adamw,
  addDecayedWeights,
  chain,
  type GradientTransformation,
  type LearningRate,
  masked,
  rmsprop,
  sgd,
} from '../../lib/optim/index.ts'
import { foldIn, type Key } from '../../lib/random/index.ts'
import {
  argmax,
  asTensor,
  fromStorage,
  mul,
  sigmoid,
  softmax,
  type Tensor,
  type TensorLike,
  tensor,
  toFlat,
  type Value,
} from '../../lib/tensor/index.ts'
import { lossAtSnapshots } from './_records.ts'
import { holdOut, type Rows, type Training } from './_train.ts'

/** How a network's output maps to predictions: a linear output, a logistic for two classes, a softmax for more. */
export type NetOutput = 'identity' | 'logistic' | 'softmax'

/** What a network learns from a dataset. */
export interface Task {
  readonly classifier: boolean
  readonly output: NetOutput
  /** The class labels (classifiers). */
  readonly classes: readonly number[]
  readonly classNames?: readonly string[]
  readonly featureNames: readonly string[]
  /** Output units: 1 for two classes or one target, k for k classes or targets. */
  readonly nOut: number
  /** Every row's inputs `[n, d]` (the surface spans them). */
  readonly x: Tensor
  /** Every row's targets as given (class labels for classifiers), to label the rows a surface carries. */
  readonly y: Tensor
  /** The rows trained on: inputs and encoded targets (class indices, or a `[n, k]` matrix). */
  readonly rows: Rows
  readonly validation?: Rows
}

/** Features as a float matrix tensor `[n, d]`. */
export function inputs(x: TensorLike, who = 'neural'): Tensor {
  const m = matrixView(x, who)
  return fromStorage(m.data, [m.n, m.d], 'float64') as Tensor
}

function column(t: Tensor): Tensor {
  return fromStorage(Float64Array.from(toFlat(t)), [t.shape[0] ?? 0, 1], 'float64') as Tensor
}

/**
 * The task of a dataset: inputs, encoded targets, the output kind and, given a fraction, held-out validation rows
 * (stratified for classifiers).
 */
export function taskOf(
  data: LearnData,
  classifier: boolean,
  options: { readonly who: string; readonly key: Key; readonly holdOut?: number },
): Task {
  const x = inputs(data.x, options.who)
  const n = x.shape[0] ?? 0
  const d = x.shape[1] ?? 0
  if (data.y === undefined) throw new WhiteboxError('value', `${options.who}: needs targets y`)
  const featureNames = featureNamesOf(data, d)
  let rows: Rows
  let classes: number[] = []
  let output: NetOutput = 'identity'
  let nOut = 1
  if (classifier) {
    const enc = encodeLabels(data.y, options.who, n)
    classes = [...enc.classes]
    if (classes.length < 2) throw new WhiteboxError('value', `${options.who}: needs at least two classes`)
    output = classes.length === 2 ? 'logistic' : 'softmax'
    nOut = classes.length === 2 ? 1 : classes.length
    rows = { x, y: tensor(Array.from(enc.codes)) }
  } else {
    const y = toFlat(asTensor(data.y))
    nOut = Math.max(1, Math.round(y.length / Math.max(1, n)))
    rows = { x, y: fromStorage(Float64Array.from(y), [n, nOut], 'float64') as Tensor }
  }
  const base = {
    classifier,
    output,
    classes,
    ...(classifier ? { classNames: classNamesOf(data, classes) } : {}),
    featureNames,
    nOut,
    x,
    y: asTensor(data.y) as Tensor,
  }
  const fraction = options.holdOut ?? 0
  if (!(fraction > 0)) return { ...base, rows }
  const parts = holdOut(rows, fraction, foldIn(options.key, 7), classifier)
  return { ...base, rows: parts.train, validation: parts.validation }
}

/** The training objective of an output kind: log loss, or scikit-learn's squared loss (half the mean squared error). */
export function lossOf(output: NetOutput): (prediction: Value, target: Value) => Value {
  if (output === 'softmax') return (p, t) => crossEntropy(p, t)
  if (output === 'logistic')
    return (p, t) => binaryCrossEntropyWithLogits(p, (t as Tensor).shape.length === 1 ? column(t as Tensor) : t)
  return (p, t) => mul(mse(p, t), 0.5)
}

/** The score of an output kind on raw outputs: accuracy for classifiers, R² for regressors. */
export function scoreOf(output: NetOutput): (prediction: Tensor, target: Tensor) => number {
  if (output === 'identity') return (p, t) => r2Score(t, p).estimate.value
  return (p, t) => {
    const pred = output === 'softmax' ? toFlat(argmax(p, -1)) : toFlat(p).map((z) => (z > 0 ? 1 : 0))
    return accuracyScore(toFlat(t), pred).estimate.value
  }
}

/** Predictions (and probabilities) of a network under some parameters and buffers, in eval mode. */
export function predictorOf(
  net: Module,
  params: Params,
  buffers: Params,
  output: NetOutput,
  classes: readonly number[],
  prepare: (x: TensorLike) => Tensor = (x) => inputs(x),
): { readonly predict: (x: TensorLike) => Tensor; readonly predictProba?: (x: TensorLike) => Tensor } {
  const raw = (x: TensorLike): Value => net.apply(params, prepare(x), { train: false, buffers })
  if (output === 'identity')
    return {
      predict: (x) => {
        const z = raw(x) as Tensor
        return (z.shape[1] ?? 1) === 1
          ? (fromStorage(Float64Array.from(toFlat(z)), [z.shape[0] ?? 0], 'float64') as Tensor)
          : z
      },
    }
  const proba = (x: TensorLike): Tensor => {
    const z = raw(x)
    if (output === 'softmax') return softmax(z, -1) as Tensor
    return tensor(toFlat(sigmoid(z)).map((v) => [1 - v, v]))
  }
  return {
    predictProba: proba,
    predict: (x) => tensor(toFlat(argmax(proba(x), -1)).map((i) => classes[i] ?? 0)),
  }
}

/** A fitted-like value of a network under some parameters (a snapshot), for {@link decisionSurfaceOf}. */
export function fittedOf(net: Module, params: Params, buffers: Params, task: Task): Fitted {
  return {
    estimator: 'network',
    kind: task.classifier ? 'classifier' : 'regressor',
    params: {},
    nFeaturesIn: task.x.shape[1] ?? 0,
    ...(task.classifier ? { classes: task.classes } : {}),
    ...predictorOf(net, params, buffers, task.output, task.classes),
  }
}

/**
 * The decision surface (or regression curve) of a network under some parameters, over the task's rows, with the
 * rows; given the training run, the surface at every snapshot (the rows each misclassifies highlighted) and the
 * training loss per epoch, so a page can play the training.
 */
export function surfaceOf(
  net: Module,
  params: Params,
  buffers: Params,
  task: Task,
  resolution = 30,
  training?: Training,
): DecisionSurface {
  const steps =
    training === undefined || training.snapshots.length < 2
      ? undefined
      : {
          models: training.snapshots.map((s) => fittedOf(net, s.params, s.buffers, task)),
          iterations: training.snapshots.map((s) => s.epoch),
          objective: lossAtSnapshots(training),
          ...(task.classifier ? { highlight: 'misclassified' as const } : {}),
        }
  return decisionSurfaceOf(fittedOf(net, params, buffers, task), task.x, {
    resolution,
    featureNames: task.featureNames,
    ...(task.classNames === undefined ? {} : { classNames: task.classNames }),
    y: task.y,
    ...(steps === undefined
      ? {}
      : {
          steps,
          stepName: 'epoch',
          objective: {
            name: 'training loss',
            iterations: [0, ...(training?.epochs ?? [])],
            values: [training?.initial.loss ?? Number.NaN, ...(training?.evalLoss ?? [])],
          },
        }),
  })
}

/** How a network's linear layers are initialised. */
export type InitScheme = 'sklearn' | 'pytorch' | 'xavier' | 'kaiming' | 'zeros' | 'small' | 'large'

/** Options of {@link buildMlp}. */
export interface MlpBuild {
  /** Widths, input first and output last. */
  readonly sizes: readonly number[]
  /** The hidden activation (undefined for none, a linear network). */
  readonly activation?: ActivationName
  /** Initialisation (default 'sklearn': weights and biases uniform on ±√(6 / (fan in + fan out)), ±√2/... for the logistic). */
  readonly init?: InitScheme
  /** Normalisation after each hidden linear layer (default none). */
  readonly norm?: 'none' | 'batch' | 'layer'
  /** Dropout after each hidden activation (default 0). */
  readonly dropout?: number
  readonly key: Key
}

function initLinear(lin: Linear, scheme: InitScheme, logistic: boolean, gain: number, key: Key): void {
  const out = lin.outFeatures
  const inn = lin.inFeatures
  if (scheme === 'pytorch') return
  if (scheme === 'sklearn') {
    const bound = Math.sqrt((logistic ? 2 : 6) / (inn + out))
    lin.setParams({
      weight: uniformInit(foldIn(key, 10), [out, inn], -bound, bound),
      bias: uniformInit(foldIn(key, 11), [out], -bound, bound),
    })
    return
  }
  const weight =
    scheme === 'zeros'
      ? zerosInit([out, inn])
      : scheme === 'xavier'
        ? xavierNormal(foldIn(key, 10), [out, inn], gain)
        : scheme === 'kaiming'
          ? kaimingNormal(foldIn(key, 10), [out, inn])
          : normalInit(foldIn(key, 10), [out, inn], 0, scheme === 'small' ? 0.01 : 1)
  lin.setParams({ weight, bias: zerosInit([out]) })
}

const GAINS: Partial<Record<ActivationName, number>> = { tanh: 5 / 3, relu: Math.SQRT2 }

/**
 * An MLP as a `Sequential`: linear layers with, after each hidden one, the normalisation, the activation and dropout.
 *
 * @param build - Widths, activation, initialisation, normalisation, dropout and key.
 * @returns The module.
 */
export function buildMlp(build: MlpBuild): Sequential {
  const { sizes, activation } = build
  const layers: Module[] = []
  for (let l = 0; l + 1 < sizes.length; l++) {
    const fanIn = sizes[l] ?? 1
    const fanOut = sizes[l + 1] ?? 1
    const k = foldIn(build.key, l)
    const lin = new Linear(fanIn, fanOut, { key: k })
    initLinear(
      lin,
      build.init ?? 'sklearn',
      activation === 'sigmoid',
      activation === undefined ? 1 : (GAINS[activation] ?? 1),
      k,
    )
    layers.push(lin)
    if (l + 2 >= sizes.length) continue
    if (build.norm === 'batch') layers.push(new BatchNorm1d(fanOut))
    if (build.norm === 'layer') layers.push(new LayerNorm(fanOut))
    if (activation !== undefined) layers.push(activationModule(activation))
    if ((build.dropout ?? 0) > 0) layers.push(new Dropout(build.dropout))
  }
  return new Sequential(...layers)
}

/** The child names of a network's linear layers, in order. */
export function linearNames(net: Sequential): string[] {
  return net
    .children()
    .filter(([, m]) => m.kind === 'Linear')
    .map(([name]) => name)
}

/** The weights-only mask (no biases, no normalisation parameters) of an L2 penalty on a {@link buildMlp} network. */
export function weightMask(net: Sequential): Params {
  return Object.fromEntries(
    net.children().flatMap(([name, m]) => {
      const leaves = Object.keys(net.params[name] ?? {})
      if (leaves.length === 0) return []
      return [[name, Object.fromEntries(leaves.map((leaf) => [leaf, m.kind === 'Linear' && leaf === 'weight']))]]
    }),
  )
}

/**
 * The activations at the end of every hidden layer of a `Sequential` MLP (after its activation, before the next
 * linear layer), in eval mode.
 */
export function hiddenActivations(net: Sequential, params: Params, x: Value, buffers: Params = net.buffers): Tensor[] {
  const out: Tensor[] = []
  let h = x
  const children = net.children()
  const linears = children.filter(([, m]) => m.kind === 'Linear').length
  let seen = 0
  for (const [i, [name, m]] of children.entries()) {
    h = (m as Module).apply((params[name] ?? {}) as Params, h, {
      train: false,
      ...(buffers[name] === undefined ? {} : { buffers: buffers[name] as Params }),
    })
    if (m.kind === 'Linear') seen++
    const next = children[i + 1]?.[1]
    if (seen < linears && seen > 0 && (next === undefined || next.kind === 'Linear')) out.push(h as Tensor)
  }
  return out
}

/** The optimisers the studies compare. */
export type OptimiserName = 'sgd' | 'momentum' | 'nesterov' | 'rmsprop' | 'adagrad' | 'adam' | 'adamw'

/** Learning rates that suit each optimiser on the small demo networks. */
export const DEMO_LEARNING_RATES: Readonly<Record<OptimiserName, number>> = {
  sgd: 0.1,
  momentum: 0.05,
  nesterov: 0.05,
  rmsprop: 0.005,
  adagrad: 0.1,
  adam: 0.01,
  adamw: 0.01,
}

/**
 * An optimiser by name: plain SGD, SGD with heavy-ball or Nesterov momentum (0.9), RMSprop, Adagrad, Adam or AdamW
 * (decoupled decay), with an optional L2 penalty (coupled weight decay) on the masked leaves.
 */
export function optimiserOf(
  name: OptimiserName,
  learningRate: LearningRate,
  options: { readonly weightDecay?: number; readonly mask?: Params } = {},
): GradientTransformation {
  const base =
    name === 'sgd'
      ? sgd({ learningRate })
      : name === 'momentum'
        ? sgd({ learningRate, momentum: 0.9 })
        : name === 'nesterov'
          ? sgd({ learningRate, momentum: 0.9, nesterov: true })
          : name === 'rmsprop'
            ? rmsprop({ learningRate })
            : name === 'adagrad'
              ? adagrad({ learningRate })
              : name === 'adam'
                ? adam({ learningRate })
                : adamw({ learningRate, weightDecay: options.weightDecay ?? 0.01 })
  const decay = options.weightDecay ?? 0
  if (decay <= 0 || name === 'adamw') return base
  const penalty = addDecayedWeights(decay)
  return chain(options.mask === undefined ? penalty : masked(penalty, options.mask), base)
}

/** Whether learn data is a regression problem (a dataset of kind 'regression'); otherwise classification. */
export function isRegression(data: LearnData): boolean {
  return (data as { readonly kind?: unknown }).kind === 'regression'
}
