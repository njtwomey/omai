import { type Json, WhiteboxError } from '../../../lib/base/index.ts'
import type { DecisionSurface, Exposition } from '../../../lib/exposition/index.ts'
import {
  classNamesOf,
  decisionSurfaceOf,
  defineExplainedEstimator,
  type ExplainedEstimator,
  ExplainedEstimatorClass,
  type ExplainedFitted,
  encodeLabels,
  type FitOptions,
  type Fitted,
  featureNamesOf,
  type LearnData,
  matrixView,
} from '../../../lib/learn/index.ts'
import { binaryCrossEntropyWithLogits, crossEntropy, mse } from '../../../lib/losses/index.ts'
import { accuracyScore, r2Score } from '../../../lib/metrics/index.ts'
import {
  type ActivationName,
  activationModule,
  describe,
  Linear,
  type Module,
  type Params,
  Sequential,
  uniformInit,
} from '../../../lib/nn/index.ts'
import {
  adam,
  addDecayedWeights,
  chain,
  type GradientTransformation,
  type LearningRate,
  masked,
  sgd,
} from '../../../lib/optim/index.ts'
import { foldIn, type Key, key as makeKey } from '../../../lib/random/index.ts'
import type { Structure } from '../../../lib/scene/index.ts'
import { bool, choice, dict, int, real, type Space } from '../../../lib/space/index.ts'
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
} from '../../../lib/tensor/index.ts'
import {
  lossAtSnapshots,
  lossCurve,
  progressReporter,
  representationSteps,
  scoreCurve,
  type TrainingProgress,
} from '../_records.ts'
import { holdOut, type Rows, type Training, train } from '../_train.ts'

/** scikit-learn's hidden-layer activations. */
export type MlpActivation = 'identity' | 'logistic' | 'tanh' | 'relu'

/** scikit-learn's learning-rate schedules (for the 'sgd' solver; Adam keeps its rate constant). */
export type MlpLearningRate = 'constant' | 'invscaling' | 'adaptive'

/** Hyperparameters of {@link mlpClassifier} and {@link mlpRegressor} (sklearn.neural_network names, camel-cased). */
export interface MlpHyper {
  /** The width of each hidden layer (default [100]). */
  readonly hiddenLayerSizes: readonly number[]
  /** The hidden activation (default 'relu'). */
  readonly activation: MlpActivation
  /** 'adam' (default) or 'sgd' (with momentum). TODO: 'lbfgs'. */
  readonly solver: 'adam' | 'sgd'
  /** The L2 penalty on the weights (not the biases), as scikit-learn scales it: alpha/2 ‖W‖² per minibatch row. */
  readonly alpha: number
  /** Minibatch size; 0 means scikit-learn's 'auto', min(200, n). */
  readonly batchSize: number
  readonly learningRate: MlpLearningRate
  readonly learningRateInit: number
  /** The exponent of 'invscaling': rate = init / (t + 1)^powerT, t the rows seen. */
  readonly powerT: number
  /** Epochs (default 200). */
  readonly maxIter: number
  /** Stop (or, with 'adaptive', divide the rate by five) after `nIterNoChange` epochs not improving by `tol`. */
  readonly tol: number
  readonly nIterNoChange: number
  readonly momentum: number
  readonly nesterovsMomentum: boolean
  /** Hold out `validationFraction` of the rows and stop on their score, keeping the best epoch's weights. */
  readonly earlyStopping: boolean
  readonly validationFraction: number
  /** Hold out `validationFraction` of the rows to record validation curves, without stopping on them. */
  readonly monitorValidation: boolean
  readonly beta1: number
  readonly beta2: number
  readonly epsilon: number
  /** The most training snapshots recorded (geometrically spaced epochs, default 40). */
  readonly snapshots: number
  /** Grid points per axis of the recorded decision surfaces (default 30). */
  readonly resolution: number
  /** Called at every snapshot with the decision surface and loss curve so far (to draw training live). */
  readonly onProgress?: (progress: TrainingProgress) => void
}

const DEFAULTS: Omit<MlpHyper, 'onProgress'> = {
  hiddenLayerSizes: [100],
  activation: 'relu',
  solver: 'adam',
  alpha: 1e-4,
  batchSize: 0,
  learningRate: 'constant',
  learningRateInit: 1e-3,
  powerT: 0.5,
  maxIter: 200,
  tol: 1e-4,
  nIterNoChange: 10,
  momentum: 0.9,
  nesterovsMomentum: true,
  earlyStopping: false,
  validationFraction: 0.1,
  monitorValidation: false,
  beta1: 0.9,
  beta2: 0.999,
  epsilon: 1e-8,
  snapshots: 40,
  resolution: 30,
}

/** How the output layer maps to predictions. */
export type MlpOutput = 'identity' | 'logistic' | 'softmax'

/** The learnt parameters of an MLP: its layer widths, activations and weights (PyTorch's layout, by layer index). */
export interface MlpParams {
  readonly [key: string]: Json
  readonly sizes: readonly number[]
  readonly activation: MlpActivation
  readonly output: MlpOutput
  /** Class labels, for classifiers. */
  readonly classes: readonly number[]
  /** layers[l]: the weight `[out, in]` (rows) and bias of linear layer l. */
  readonly layers: readonly { readonly weight: readonly (readonly number[])[]; readonly bias: readonly number[] }[]
  readonly epochs: number
  readonly loss: number
}

/** The verbs of a fitted MLP. */
export interface MlpVerbs {
  readonly predict: (x: TensorLike) => Tensor
  readonly predictProba?: (x: TensorLike) => Tensor
  readonly classes?: readonly number[]
  readonly nFeaturesIn: number
  /** The network's structure (layers, shapes, parameter counts), for an architecture diagram. */
  readonly describe: () => Structure
  /** The hidden activations of rows at every hidden layer (after the nonlinearity). */
  readonly hidden: (x: TensorLike) => Tensor[]
}

/** A fitted MLP classifier or regressor. */
export type FittedMlp = ExplainedFitted<MlpParams, MlpVerbs>

const ACTIVATIONS: Readonly<Record<MlpActivation, ActivationName | undefined>> = {
  identity: undefined,
  logistic: 'sigmoid',
  tanh: 'tanh',
  relu: 'relu',
}

/**
 * The network of an MLP: linear layers of the given widths with the activation between them (none for 'identity')
 * and a linear output; the parameters are initialised as scikit-learn does (weights and biases uniform on ±√(6 / (fan
 * in + fan out)), ±√(2 / ...) for the logistic).
 *
 * @param sizes - The widths, input first and output last.
 * @param activation - The hidden activation.
 * @param key - The initialisation key.
 * @returns The module.
 */
export function mlpNetwork(sizes: readonly number[], activation: MlpActivation, key: Key): Sequential {
  const layers: Module[] = []
  const name = ACTIVATIONS[activation]
  for (let l = 0; l + 1 < sizes.length; l++) {
    const fanIn = sizes[l] ?? 1
    const fanOut = sizes[l + 1] ?? 1
    const lin = new Linear(fanIn, fanOut, { key: foldIn(key, l) })
    const bound = Math.sqrt((activation === 'logistic' ? 2 : 6) / (fanIn + fanOut))
    lin.setParams({
      weight: uniformInit(foldIn(foldIn(key, l), 10), [fanOut, fanIn], -bound, bound),
      bias: uniformInit(foldIn(foldIn(key, l), 11), [fanOut], -bound, bound),
    })
    layers.push(lin)
    if (l + 2 < sizes.length && name !== undefined) layers.push(activationModule(name))
  }
  return new Sequential(...layers)
}

/** The linear layers' parameter names in a {@link mlpNetwork} (by child index). */
function linearIndices(net: Sequential): string[] {
  return net
    .children()
    .filter(([, m]) => m.kind === 'Linear')
    .map(([name]) => name)
}

/** The weights-only mask of scikit-learn's L2 penalty. */
function weightMask(net: Sequential): Params {
  return Object.fromEntries(linearIndices(net).map((name) => [name, { weight: true, bias: false }]))
}

/** The activations after each hidden layer of a {@link mlpNetwork} under some parameters. */
export function hiddenActivations(net: Sequential, params: Params, x: Value): Tensor[] {
  const out: Tensor[] = []
  let h = x
  const children = net.children()
  const linears = children.filter(([, m]) => m.kind === 'Linear').length
  let seen = 0
  for (const [i, [name, m]] of children.entries()) {
    h = (m as Module).apply((params[name] ?? {}) as Params, h, { train: false })
    if (m.kind === 'Linear') seen++
    const next = children[i + 1]?.[1]
    const endOfHidden = seen < linears && (next === undefined || next.kind === 'Linear')
    if (endOfHidden) out.push(h as Tensor)
  }
  return out
}

function toLayers(net: Sequential, params: Params): MlpParams['layers'] {
  return linearIndices(net).map((name) => {
    const p = params[name] as { readonly weight: Tensor; readonly bias: Tensor }
    const [out = 0, inn = 0] = p.weight.shape
    const w = toFlat(p.weight)
    return {
      weight: Array.from({ length: out }, (_, r) => w.slice(r * inn, (r + 1) * inn)),
      bias: toFlat(p.bias),
    }
  })
}

function fromLayers(net: Sequential, layers: MlpParams['layers']): Params {
  const names = linearIndices(net)
  return Object.fromEntries(
    layers.map((l, i) => [
      names[i] ?? String(i),
      { weight: tensor(l.weight.map((r) => [...r])), bias: tensor([...l.bias]) },
    ]),
  )
}

/** The output of a network under some parameters mapped to probabilities or predictions. */
function verbsFor(
  net: Sequential,
  params: Params,
  output: MlpOutput,
  classes: readonly number[],
): Pick<MlpVerbs, 'predict' | 'predictProba'> {
  const raw = (x: TensorLike): Value => net.apply(params, inputs(x), { train: false })
  if (output === 'identity') {
    return {
      predict: (x) => {
        const z = raw(x) as Tensor
        return (z.shape[1] ?? 1) === 1
          ? (fromStorage(Float64Array.from(toFlat(z)), [z.shape[0] ?? 0], 'float64') as Tensor)
          : z
      },
    }
  }
  const proba = (x: TensorLike): Tensor => {
    const z = raw(x)
    if (output === 'softmax') return softmax(z, -1) as Tensor
    const p = toFlat(sigmoid(z))
    return tensor(p.map((v) => [1 - v, v]))
  }
  return {
    predictProba: proba,
    predict: (x) => {
      const idx = toFlat(argmax(proba(x), -1))
      return tensor(idx.map((i) => classes[i] ?? 0))
    },
  }
}

/** The fitted-like value of a snapshot, for drawing its decision surface. */
function snapshotFitted(
  net: Sequential,
  params: Params,
  output: MlpOutput,
  classes: readonly number[],
  nFeatures: number,
): Fitted {
  return {
    estimator: 'mlp',
    kind: output === 'identity' ? 'regressor' : 'classifier',
    params: {},
    nFeaturesIn: nFeatures,
    ...(output === 'identity' ? {} : { classes }),
    ...verbsFor(net, params, output, classes),
  }
}

/** The optimiser of a solver, with scikit-learn's L2 penalty on the weights as a coupled decay. */
function optimizerFor(
  h: MlpHyper,
  net: Sequential,
  batch: number,
  n: number,
): { opt: GradientTransformation; lr: LearningRate } {
  const perEpoch = Math.ceil(n / batch)
  const lr: LearningRate =
    h.solver === 'sgd' && h.learningRate === 'invscaling'
      ? (u: number) => h.learningRateInit / (Math.floor(u / perEpoch) * n + 1) ** h.powerT
      : h.learningRateInit
  const base =
    h.solver === 'adam'
      ? adam({ learningRate: lr, b1: h.beta1, b2: h.beta2, eps: h.epsilon })
      : sgd({ learningRate: lr, momentum: h.momentum, nesterov: h.nesterovsMomentum && h.momentum > 0 })
  const opt = h.alpha > 0 ? chain(masked(addDecayedWeights(h.alpha / batch), weightMask(net)), base) : base
  return { opt, lr }
}

function parseSizes(v: unknown): readonly number[] | undefined {
  if (Array.isArray(v)) return v as number[]
  if (typeof v === 'number') return [v]
  if (typeof v !== 'string') return undefined
  const sizes = v
    .split(/[,\s]+/)
    .filter((s) => s !== '')
    .map(Number)
  if (sizes.some((s) => !(Number.isInteger(s) && s > 0)))
    throw new WhiteboxError('value', `mlp: hiddenLayerSizes '${v}' is not a list of positive widths`)
  return sizes
}

/** Merge hyperparameter changes, reading `hiddenLayerSizes` from a space's string choice ('16,16'). */
function updateHyper(h: MlpHyper, changes: Readonly<Record<string, unknown>>): MlpHyper {
  const sizes = parseSizes(changes['hiddenLayerSizes'])
  return { ...h, ...changes, ...(sizes === undefined ? {} : { hiddenLayerSizes: sizes }) } as MlpHyper
}

const SIZE_CHOICES = ['4', '8', '16', '32', '64', '100', '8,8', '16,16', '32,32', '16,16,16', '32,16,8']

function hyperSpace(h: MlpHyper): Space {
  const current = h.hiddenLayerSizes.join(',')
  return dict({
    hiddenLayerSizes: choice(SIZE_CHOICES.includes(current) ? SIZE_CHOICES : [current, ...SIZE_CHOICES], {
      default: current,
    }),
    activation: choice(['relu', 'tanh', 'logistic', 'identity'] as const, { default: h.activation }),
    solver: choice(['adam', 'sgd'] as const, { default: h.solver }),
    learningRateInit: real(1e-4, 1, { log: true, default: h.learningRateInit }),
    alpha: real(1e-6, 1, { log: true, default: Math.max(h.alpha, 1e-6) }),
    maxIter: int(1, 2000, { log: true, default: h.maxIter }),
    batchSize: int(0, 512, { default: h.batchSize }),
    learningRate: choice(['constant', 'invscaling', 'adaptive'] as const, { default: h.learningRate }),
    earlyStopping: bool({ default: h.earlyStopping }),
  })
}

interface Prepared {
  readonly rows: Rows
  readonly validation?: Rows
  readonly classes: readonly number[]
  readonly output: MlpOutput
  readonly nOut: number
}

function prepare(data: LearnData, h: MlpHyper, classifier: boolean, key: Key): Prepared {
  const x = matrixView(data.x, 'mlp')
  const xt = inputs(data.x)
  if (data.y === undefined) throw new WhiteboxError('value', 'mlp: needs targets y')
  let rows: Rows
  let classes: number[] = []
  let output: MlpOutput = 'identity'
  let nOut = 1
  if (classifier) {
    const enc = encodeLabels(data.y, 'mlpClassifier', x.n)
    classes = [...enc.classes]
    if (classes.length < 2) throw new WhiteboxError('value', 'mlpClassifier: needs at least two classes')
    output = classes.length === 2 ? 'logistic' : 'softmax'
    nOut = classes.length === 2 ? 1 : classes.length
    rows = { x: xt, y: tensor(Array.from(enc.codes)) }
  } else {
    const y = toFlat(asTensor(data.y))
    nOut = Math.max(1, Math.round(y.length / x.n))
    rows = { x: xt, y: fromStorage(Float64Array.from(y), [x.n, nOut], 'float64') as Tensor }
  }
  const split = h.earlyStopping || h.monitorValidation
  if (!split) return { rows, classes, output, nOut }
  const parts = holdOut(rows, h.validationFraction, foldIn(key, 7), classifier)
  return { rows: parts.train, validation: parts.validation, classes, output, nOut }
}

function lossFor(output: MlpOutput): (p: Value, t: Value) => Value {
  if (output === 'softmax') return (p, t) => crossEntropy(p, t)
  if (output === 'logistic')
    return (p, t) => binaryCrossEntropyWithLogits(p, (t as Tensor).shape.length === 1 ? reshapeColumn(t as Tensor) : t)
  // scikit-learn's squared loss is half the mean squared error.
  return (p, t) => mul(mse(p, t), 0.5)
}

/** Features as a float matrix tensor `[n, d]`. */
function inputs(x: TensorLike): Tensor {
  const m = matrixView(x, 'mlp')
  return fromStorage(m.data, [m.n, m.d], 'float64') as Tensor
}

function reshapeColumn(t: Tensor): Tensor {
  return fromStorage(Float64Array.from(toFlat(t)), [t.shape[0] ?? 0, 1], 'float64') as Tensor
}

function scoreFor(output: MlpOutput): (p: Tensor, t: Tensor) => number {
  if (output === 'identity') return (p, t) => r2Score(t, p).estimate.value
  return (p, t) => {
    const pred = output === 'softmax' ? toFlat(argmax(p, -1)) : toFlat(p).map((z) => (z > 0 ? 1 : 0))
    return accuracyScore(toFlat(t), pred).estimate.value
  }
}

/** Fit an MLP (the shared body of the classifier and the regressor). */
function fitMlp(
  data: LearnData,
  h: MlpHyper,
  options: FitOptions,
  classifier: boolean,
): { params: MlpParams; exposition: Exposition[] } {
  const name = classifier ? 'mlpClassifier' : 'mlpRegressor'
  const key = options.key ?? makeKey(0)
  const prep = prepare(data, h, classifier, key)
  const d = prep.rows.x.shape[1] ?? 1
  const sizes = [d, ...h.hiddenLayerSizes, prep.nOut]
  const net = mlpNetwork(sizes, h.activation, foldIn(key, 1))
  const n = prep.rows.x.shape[0] ?? 0
  const batch = Math.max(1, Math.min(n, h.batchSize > 0 ? h.batchSize : 200))
  const { opt, lr } = optimizerFor(h, net, batch, n)
  const training = train({
    name,
    model: net,
    train: prep.rows,
    ...(prep.validation === undefined ? {} : { validation: prep.validation }),
    loss: lossFor(prep.output),
    score: scoreFor(prep.output),
    optimizer: opt,
    learningRate: lr,
    epochs: h.maxIter,
    batchSize: batch,
    stopping: {
      tol: h.tol,
      nIterNoChange: h.nIterNoChange,
      monitor: h.earlyStopping ? 'validationScore' : 'loss',
      adaptive: h.solver === 'sgd' && h.learningRate === 'adaptive',
    },
    snapshots: h.snapshots,
    fit: { ...options, key: foldIn(key, 2) },
    ...(h.onProgress === undefined
      ? {}
      : { onEpoch: progressReporter(h.onProgress, (s) => [surfaceAt(data, net, prep, h, s.params)]) }),
  })
  const params: MlpParams = {
    sizes,
    activation: h.activation,
    output: prep.output,
    classes: prep.classes,
    layers: toLayers(net, training.params),
    epochs: training.epochs.length,
    loss: training.trainLoss.at(-1) ?? Number.NaN,
  }
  return { params, exposition: mlpExposition(data, net, training, prep, h) }
}

/** The decision surface (or regression curve) of the network under some parameters, over the training rows. */
function surfaceAt(data: LearnData, net: Sequential, prep: Prepared, h: MlpHyper, params: Params): DecisionSurface {
  const x = matrixView(data.x, 'mlp')
  const classNames = prep.output === 'identity' ? undefined : classNamesOf(data, prep.classes)
  return decisionSurfaceOf(snapshotFitted(net, params, prep.output, prep.classes, x.d), inputs(data.x), {
    resolution: h.resolution,
    featureNames: featureNamesOf(data, x.d),
    ...(classNames === undefined ? {} : { classNames }),
  })
}

/**
 * The exposition of a training run: the decision surface (or regression curve) of the fitted network, the loss and
 * score per epoch, and the last hidden layer's representation of the training rows at every snapshot.
 */
function mlpExposition(
  data: LearnData,
  net: Sequential,
  training: Training,
  prep: Prepared,
  h: MlpHyper,
): Exposition[] {
  const x = matrixView(data.x, 'mlp')
  // TODO: once DecisionSurface records steps (the records branch), record the surface at every snapshot.
  const out: Exposition[] = [surfaceAt(data, net, prep, h, training.params), lossCurve(training)]
  const score = scoreCurve(training, prep.output === 'identity' ? 'R²' : 'accuracy')
  if (score !== undefined) out.push(score)
  if (h.hiddenLayerSizes.length > 0) {
    const limit = Math.min(x.n, 500)
    const sample =
      limit === x.n ? inputs(data.x) : (fromStorage(x.data.slice(0, limit * x.d), [limit, x.d], 'float64') as Tensor)
    out.push(
      representationSteps(
        `${net.kind} last hidden layer`,
        training.snapshots,
        (s) => hiddenActivations(net, s.params, sample).at(-1) as Tensor,
        lossAtSnapshots(training),
      ),
    )
  }
  return out
}

function mlpVerbs(p: MlpParams): MlpVerbs {
  const net = mlpNetwork(p.sizes, p.activation, makeKey(0))
  const params = fromLayers(net, p.layers)
  net.setParams(params)
  const nFeaturesIn = p.sizes[0] ?? 0
  return {
    ...verbsFor(net, params, p.output, p.classes),
    ...(p.output === 'identity' ? {} : { classes: p.classes }),
    nFeaturesIn,
    describe: () => describe(net, [1, nFeaturesIn]),
    hidden: (x) => hiddenActivations(net, params, inputs(x)),
  }
}

function withDefaults(hyper: Partial<MlpHyper>): MlpHyper {
  return { ...DEFAULTS, ...hyper }
}

/**
 * A multi-layer perceptron classifier (sklearn.neural_network.MLPClassifier): hidden layers with a nonlinearity, a
 * logistic output for two classes and a softmax for more, trained on the log loss plus an L2 penalty on the weights by
 * Adam or SGD with momentum over shuffled minibatches; scikit-learn's stopping rule (`tol`, `nIterNoChange`), early
 * stopping on a held-out fraction and the 'invscaling' and 'adaptive' learning rates. Exposition: the decision surface,
 * the loss and accuracy per epoch, and the last hidden layer's representation of the training rows at about 40
 * geometrically spaced epochs (an embedding that untangles the classes as training proceeds).
 *
 * @param hyper - Hidden widths, activation, solver, penalty, batching, learning rate and stopping.
 * @returns The estimator.
 */
export function mlpClassifier(
  hyper: Partial<MlpHyper> = {},
): ExplainedEstimator<MlpHyper, Fitted<MlpParams> & MlpVerbs> {
  return defineExplainedEstimator(
    {
      name: 'mlpClassifier',
      kind: 'classifier',
      hyperSpace,
      update: updateHyper,
      fit: (data, h, options) => fitMlp(data, h, options, true),
      methods: mlpVerbs,
    },
    withDefaults(hyper),
  )
}

/**
 * A multi-layer perceptron regressor (sklearn.neural_network.MLPRegressor): as {@link mlpClassifier} with a linear
 * output and the squared loss. Exposition: the regression curve (one feature) or surface (two), the loss and R² per
 * epoch, and the last hidden layer's representation of the training rows at about 40 epochs.
 *
 * @param hyper - As {@link mlpClassifier}.
 * @returns The estimator.
 */
export function mlpRegressor(
  hyper: Partial<MlpHyper> = {},
): ExplainedEstimator<MlpHyper, Fitted<MlpParams> & MlpVerbs> {
  return defineExplainedEstimator(
    {
      name: 'mlpRegressor',
      kind: 'regressor',
      hyperSpace,
      update: updateHyper,
      fit: (data, h, options) => fitMlp(data, h, options, false),
      methods: mlpVerbs,
    },
    withDefaults(hyper),
  )
}

/** The class form of {@link mlpClassifier}. */
export class MLPClassifier extends ExplainedEstimatorClass<MlpHyper, Fitted<MlpParams> & MlpVerbs> {
  /** @param hyper - As {@link mlpClassifier}. */
  constructor(hyper: Partial<MlpHyper> = {}) {
    super(mlpClassifier(hyper))
  }
}

/** The class form of {@link mlpRegressor}. */
export class MLPRegressor extends ExplainedEstimatorClass<MlpHyper, Fitted<MlpParams> & MlpVerbs> {
  /** @param hyper - As {@link mlpRegressor}. */
  constructor(hyper: Partial<MlpHyper> = {}) {
    super(mlpRegressor(hyper))
  }
}
