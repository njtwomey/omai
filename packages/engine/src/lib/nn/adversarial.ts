/**
 * Adversarial examples and adversarial training for any differentiable classifier (a function from a batch of inputs
 * to logits): the fast gradient sign method, projected gradient descent with a random start inside an l-infinity or l2
 * ball (targeted or untargeted, with the cross-entropy or the Carlini-Wagner margin loss) as a function and as a
 * stepped algorithm, the Carlini-Wagner l2 attack, DeepFool, a transfer attack, the black-box Square-style attack that
 * only reads logits, robust accuracy over a sweep of radii, and adversarial training (PGD inner maximisation, optionally
 * with TRADES's trade-off term) as a step over `TrainState`. Certification is in `certify.ts`.
 */

import { grad, valueAndGrad } from '../autodiff/index.ts'
import { MdmlError } from '../base/index.ts'
import { crossEntropy, klDivergence } from '../losses/index.ts'
import { adam, applyUpdates, type GradientTransformation, globalNorm } from '../optim/index.ts'
import { foldIn, type Key, normal, split, uniform } from '../random/index.ts'
import {
  add,
  argmax,
  asTensor,
  clip,
  div,
  fromStorage,
  item,
  max,
  maximum,
  mean,
  minimum,
  mul,
  neg,
  norm,
  oneHot,
  pow,
  primalOf,
  reshape,
  sign,
  square,
  stopGradient,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  event,
  runAlgorithm,
  scalar,
  stateFromJson,
  stateToJson,
  tensor as tensorSource,
} from '../trace/index.ts'
import { concreteTree, type Module, type Params } from './module/module.ts'
import { applyGradients, type TrainState } from './training/state.ts'

/** A classifier as an attack sees it: a batch of inputs `[N, ...]` to logits `[N, C]`. */
export type LogitsFunction = (x: Value) => Value

/** The geometry of the perturbation ball. */
export type AttackNorm = 'linf' | 'l2'

/** The loss an attack ascends: the cross-entropy of the label, the Carlini-Wagner margin, or (TRADES) the KL divergence from the clean prediction. */
export type AttackLoss = 'cross-entropy' | 'margin' | 'kl'

/** Options shared by the gradient attacks. */
export interface AttackOptions {
  /** The radius of the ball around the input. */
  readonly epsilon: number
  /** The ball (default l-infinity). */
  readonly norm?: AttackNorm
  /** Targeted: the labels are the classes to reach, and the attack descends their loss instead of ascending. */
  readonly targeted?: boolean
  /** The loss to push (default cross-entropy). */
  readonly loss?: AttackLoss
  /** The data range the adversarial input is clipped to (the inputs are assumed to lie inside it: clipping a point outside it can leave the ball). */
  readonly range?: readonly [number, number]
  /** Margin saturation: the margin loss stops growing past this value (default unbounded). */
  readonly kappa?: number
}

/**
 * Per-example Carlini-Wagner margin: the largest wrong logit minus the label's logit (positive when the label is
 * lost). `kappa` saturates it, so an attack stops pushing an example that is already misclassified by that confidence.
 *
 * @param logits - `[N, C]` logits.
 * @param labels - The `N` class labels.
 * @param kappa - Saturation (default none).
 * @returns The `N` margins.
 * @concept adversarial-examples
 * @example
 * ```ts
 * import { marginLoss } from 'mdml-engine/nn'
 * marginLoss([[2, 1, 0], [0, 3, 1]], [0, 0]).data // => Float64Array.of(-1, 3)
 * ```
 */
export function marginLoss(logits: TensorLike, labels: TensorLike, kappa: number = Number.POSITIVE_INFINITY): Value {
  const z = asTensor(logits)
  const c = z.shape[z.shape.length - 1] ?? 0
  const hot = oneHot(labels, c)
  const own = sum(mul(z, hot), -1)
  const other = max(sub(z, mul(hot, 1e9)), -1)
  const m = sub(other, own)
  return Number.isFinite(kappa) ? minimum(m, kappa) : m
}

const perExample = (v: Value): Value => reshape(v, [v.shape[0] ?? 1, -1])

/** The l2 norm (or sup norm) of each example's row of a batch, `[N, 1, ...]` for broadcasting. */
function rowNorm(v: Value, kind: AttackNorm): Value {
  const n = norm(perExample(v), kind === 'l2' ? 2 : Number.POSITIVE_INFINITY, 1, true)
  return reshape(n, [v.shape[0] ?? 1, ...v.shape.slice(1).map(() => 1)])
}

/** The per-example loss an attack pushes, as a function of the (perturbed) input; `x0` is the clean input (for `kl`). */
function lossOf(model: LogitsFunction, y: Value, options: AttackOptions, x0: Value): (x: Value) => Value {
  const kind = options.loss ?? 'cross-entropy'
  const reference = kind === 'kl' ? stopGradient(model(x0)) : undefined
  return (x) => {
    const z = model(x)
    if (reference !== undefined) return klDivergence(reference, z, { logits: true, reduction: 'none' })
    return kind === 'margin' ? marginLoss(z, y, options.kappa) : crossEntropy(z, y, { reduction: 'none' })
  }
}

/** The input gradient of the ascent objective: the summed loss, signed by whether the attack is targeted. */
function inputGradient(model: LogitsFunction, x: Value, y: Value, options: AttackOptions, x0: Value): Value {
  const loss = lossOf(model, y, options, x0)
  const sign_ = options.targeted === true ? -1 : 1
  return grad((z: Value) => mul(sum(loss(z)), sign_))(x)
}

/** Project a perturbed input back into the ball around the original and the data range. */
function project(x0: Value, x: Value, options: AttackOptions): Value {
  const norm_ = options.norm ?? 'linf'
  let delta = sub(x, x0)
  if (norm_ === 'linf') delta = clip(delta, -options.epsilon, options.epsilon)
  else {
    const n = rowNorm(delta, 'l2')
    delta = mul(delta, minimum(div(options.epsilon, maximum(n, 1e-30)), 1))
  }
  const out = add(x0, delta)
  return options.range === undefined ? out : clip(out, options.range[0], options.range[1])
}

/** The steepest ascent direction of the ball: the sign of the gradient (l-infinity) or its unit vector (l2). */
function direction(g: Value, kind: AttackNorm): Value {
  return kind === 'l2' ? div(g, maximum(rowNorm(g, 'l2'), 1e-30)) : sign(g)
}

const toTensor = (v: Value): Tensor => asTensor(v) as Tensor

/**
 * The fast gradient sign method (Goodfellow, Shlens and Szegedy 2015): one step of size epsilon along the sign of the
 * input gradient of the loss (the l2 form steps along the gradient's unit vector), clipped to the data range. To first
 * order it raises the loss by epsilon times the gradient's dual norm.
 *
 * @param model - Inputs to logits.
 * @param x - The inputs `[N, ...]`.
 * @param labels - The labels to attack (the classes to reach when targeted).
 * @param options - Radius, norm, targeting, loss and range.
 * @returns The adversarial inputs.
 * @concept adversarial-examples
 * @example
 * ```ts
 * import { fgsm } from 'mdml-engine/nn'
 * import { matmul } from 'mdml-engine/tensor'
 * const model = (x) => matmul(x, [[1, -1], [0, 0]])
 * fgsm(model, [[1, 0]], [0], { epsilon: 0.5 }).data // => Float64Array.of(0.5, 0)
 * ```
 */
export function fgsm(model: LogitsFunction, x: TensorLike, labels: TensorLike, options: AttackOptions): Tensor {
  const x0 = asTensor(x)
  const y = asTensor(labels)
  const g = inputGradient(model, x0, y, options, x0)
  const step = mul(direction(g, options.norm ?? 'linf'), options.epsilon)
  const out = add(x0, step)
  return toTensor(options.range === undefined ? out : clip(out, options.range[0], options.range[1]))
}

/** Options of {@link pgd}. */
export interface PgdOptions extends AttackOptions {
  /** Gradient steps (default 10). */
  readonly steps?: number
  /** The step size (default 2.5 epsilon / steps, Madry et al.'s rule of thumb). */
  readonly stepSize?: number
  /** Start at a uniformly random point of the ball (default true; needs `key`). */
  readonly randomStart?: boolean
  readonly key?: Key
}

/** A uniformly random point of the ball around zero, per example. */
function randomPerturbation(key: Key, shape: readonly number[], options: AttackOptions): Value {
  if ((options.norm ?? 'linf') === 'linf')
    return uniform(key, [...shape], { low: -options.epsilon, high: options.epsilon })
  const [kn, ku] = split(key, 2) as [Key, Key]
  const z = normal(kn, [...shape])
  const d = shape.slice(1).reduce((a, b) => a * b, 1)
  const u = uniform(ku, [shape[0] ?? 1])
  // The radius of a uniform point of a d-ball is epsilon u^(1/d).
  const radius = reshape(mul(pow(u, 1 / d), options.epsilon), [shape[0] ?? 1, ...shape.slice(1).map(() => 1)])
  return mul(div(z, maximum(rowNorm(z, 'l2'), 1e-30)), radius)
}

/** One PGD step from a point: ascend, then project. */
function pgdStep(model: LogitsFunction, x0: Value, x: Value, y: Value, options: PgdOptions, stepSize: number): Value {
  const g = inputGradient(model, x, y, options, x0)
  const moved = add(x, mul(direction(g, options.norm ?? 'linf'), stepSize))
  return project(x0, moved, options)
}

const stepSizeOf = (o: PgdOptions): number => o.stepSize ?? (2.5 * o.epsilon) / (o.steps ?? 10)

function startPoint(x0: Value, options: PgdOptions): Value {
  if (options.randomStart === false) return x0
  if (options.key === undefined) throw new MdmlError('value', 'pgd: a random start needs a key (or randomStart: false)')
  return project(x0, add(x0, randomPerturbation(options.key, x0.shape, options)), options)
}

/**
 * Projected gradient descent (Madry et al. 2018): steps of the fast gradient method, each followed by projection back
 * into the epsilon ball around the input (l-infinity or l2) and the data range, from a uniformly random start in the
 * ball. Untargeted by default (maximise the loss of the label), or targeted (descend the loss of the target class); the
 * loss is the cross-entropy or the margin.
 *
 * @param model - Inputs to logits.
 * @param x - The inputs `[N, ...]`.
 * @param labels - The labels to attack (the target classes when targeted).
 * @param options - FGSM's options, `steps`, `stepSize`, `randomStart` and the `key` for the start.
 * @returns The adversarial inputs, all within epsilon of `x`.
 * @concept adversarial-examples
 * @example
 * ```ts
 * import { pgd } from 'mdml-engine/nn'
 * import { key } from 'mdml-engine/random'
 * import { matmul } from 'mdml-engine/tensor'
 * const model = (x) => matmul(x, [[1, -1], [0, 0]])
 * Math.max(...pgd(model, [[1, 0]], [0], { epsilon: 0.5, steps: 5, key: key(0) }).data.map((v, i) => Math.abs(v - [1, 0][i]))) <= 0.5 // => true
 * ```
 */
export function pgd(model: LogitsFunction, x: TensorLike, labels: TensorLike, options: PgdOptions): Tensor {
  const x0 = asTensor(x)
  const y = asTensor(labels)
  const steps = options.steps ?? 10
  const size = stepSizeOf(options)
  let adv = startPoint(x0, options)
  for (let i = 0; i < steps; i++) adv = pgdStep(model, x0, adv, y, options, size)
  return toTensor(adv)
}

/** The paths of {@link pgdAlgorithm}. */
export type PgdPath =
  | 'attack/step'
  | 'attack/point'
  | 'attack/loss'
  | 'attack/distance'
  | 'attack/successRate'
  | 'attack/succeeded'

/** What a stepped PGD attack logs. */
export const pgdContract: Contract<PgdState, PgdPath> = defineContract({
  'attack/step': scalar({ description: 'the gradient step' }),
  'attack/point': tensorSource({ role: 'state', description: 'the adversarial inputs after the step' }),
  'attack/loss': scalar({ description: 'the mean loss the attack ascends' }),
  'attack/distance': scalar({ description: 'the mean distance from the original inputs, in the attack norm' }),
  'attack/successRate': scalar({
    description: 'the share of inputs the model gets wrong (or, targeted, sends to the target)',
  }),
  'attack/succeeded': event({ description: 'the first step at which the attack succeeds on an input' }),
})

/** The state of a stepped PGD attack: the current adversarial inputs and what the last step measured. */
export interface PgdState {
  readonly x: Tensor
  readonly step: number
  readonly loss: number
  readonly distance: number
  readonly successRate: number
  /** Which inputs have been fooled at some step. */
  readonly everFooled: readonly boolean[]
}

function fooled(logits: Value, y: Value, targeted: boolean): boolean[] {
  const pred = toFlat(argmax(logits, -1))
  const lab = toFlat(y)
  return pred.map((p, i) => (targeted ? p === lab[i] : p !== lab[i]))
}

/**
 * PGD as a stepped algorithm, one gradient step per step, for watching the point walk across the decision boundary:
 * it logs the adversarial inputs, the loss, the distance travelled, the share of inputs fooled, and an event the first
 * time an input is fooled. The state serialises; the random start comes from the run's key.
 *
 * @param model - Inputs to logits.
 * @param x - The inputs.
 * @param labels - The labels to attack.
 * @param options - As {@link pgd}; `steps` is the number of steps the run takes.
 * @returns The algorithm.
 * @concept adversarial-examples
 * @example
 * ```ts
 * import { pgdAlgorithm } from 'mdml-engine/nn'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * import { matmul } from 'mdml-engine/tensor'
 * const model = (x) => matmul(x, [[1, -1], [0, 0]])
 * runAlgorithm(pgdAlgorithm(model, [[0.2, 0]], [0], { epsilon: 0.5, steps: 8 }), { maxSteps: 20 }).state.successRate // => 1
 * ```
 */
export function pgdAlgorithm(
  model: LogitsFunction,
  x: TensorLike,
  labels: TensorLike,
  options: Omit<PgdOptions, 'key'>,
): Algorithm<PgdState, PgdPath> {
  const x0 = asTensor(x)
  const y = asTensor(labels)
  const steps = options.steps ?? 10
  const size = stepSizeOf(options)
  const targeted = options.targeted === true
  const measureLoss = lossOf(model, y, options, x0)
  const measure = (adv: Value, step: number, ever: readonly boolean[]): PgdState => {
    const z = model(adv)
    const f = fooled(z, y, targeted)
    const l = measureLoss(adv)
    return {
      x: toTensor(adv),
      step,
      loss: item(mean(l)),
      distance: item(mean(rowNorm(sub(adv, x0), options.norm ?? 'linf'))),
      successRate: f.filter(Boolean).length / f.length,
      everFooled: f.map((v, i) => v || (ever[i] ?? false)),
    }
  }
  return {
    name: 'pgd',
    config: {
      epsilon: options.epsilon,
      norm: options.norm ?? 'linf',
      steps,
      stepSize: size,
      targeted,
      loss: options.loss ?? 'cross-entropy',
    },
    init: (key) => measure(startPoint(x0, { ...options, key }), 0, []),
    step: (s, ctx) => {
      const adv = pgdStep(model, x0, s.x, y, options, size)
      const next = measure(adv, s.step + 1, s.everFooled)
      next.everFooled.forEach((v, i) => {
        if (v && !(s.everFooled[i] ?? false)) ctx.emit('attack/succeeded', { example: i, step: next.step })
      })
      ctx.log('attack/step', next.step)
      ctx.log('attack/point', next.x)
      ctx.log('attack/loss', next.loss)
      ctx.log('attack/distance', next.distance)
      ctx.log('attack/successRate', next.successRate)
      return next
    },
    done: (s) => (s.step >= steps ? 'completed' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as PgdState,
    observe: (s) => ({ metrics: { loss: s.loss, distance: s.distance, successRate: s.successRate } }),
  }
}

/** The result of an attack that tracks its own success. */
export interface AttackResult {
  /** The adversarial inputs (the original where the attack failed). */
  readonly x: Tensor
  /** Which inputs are now misclassified (or hit their target). */
  readonly success: readonly boolean[]
  /** The distance of each adversarial input from the original, in the attack's norm. */
  readonly distance: readonly number[]
}

/** Options of {@link carliniWagnerL2} and {@link carliniWagnerAlgorithm}. */
export interface CarliniWagnerOptions {
  /** The weight c of the margin term against the squared l2 distance (default 1). */
  readonly c?: number
  /** The confidence kappa: the margin by which the wrong class must win (default 0). */
  readonly kappa?: number
  readonly steps?: number
  /** The Adam learning rate on the perturbation (default 0.05); ignored when `optimiser` is given. */
  readonly learningRate?: number
  /** The optimiser of the perturbation, a gradient transformation (default Adam at `learningRate`); build one from `optimisers`. */
  readonly optimiser?: GradientTransformation
  readonly targeted?: boolean
  readonly range?: readonly [number, number]
}

/** The paths of {@link carliniWagnerAlgorithm}. */
export type CarliniWagnerPath =
  | 'attack/step'
  | 'attack/delta'
  | 'attack/loss'
  | 'attack/margin'
  | 'attack/norm'
  | 'attack/success'

/**
 * The state of {@link carliniWagnerAlgorithm}: the perturbation and the optimiser's state, the closest adversarial
 * point found so far for each input and the distance of each.
 */
export interface CarliniWagnerState {
  readonly step: number
  readonly delta: Tensor
  /** The optimiser's state (a tree of tensors and numbers). */
  readonly optimiser: unknown
  /** The closest adversarial inputs found so far (the original where none yet), flattened per input. */
  readonly best: Float64Array
  /** The l2 distance of the closest adversarial point per input (meaningless where `won` is false). */
  readonly bestDistance: readonly number[]
  readonly won: readonly boolean[]
  readonly loss: number
  readonly margin: number
  readonly norm: number
  readonly successRate: number
}

/** What a stepped Carlini-Wagner attack logs: the perturbation, the objective, the mean margin, the distance and the share fooled. */
export const carliniWagnerContract: Contract<CarliniWagnerState, CarliniWagnerPath> = defineContract({
  'attack/step': scalar({ description: 'the optimiser step' }),
  'attack/delta': tensorSource({ role: 'state', description: 'the perturbation after the step' }),
  'attack/loss': scalar({
    description: 'the mean of ||delta||^2 + c f(x + delta) before the step, the objective minimised',
  }),
  'attack/margin': scalar({
    description: 'the mean margin (largest wrong logit minus the label logit) after the step',
  }),
  'attack/norm': scalar({ description: 'the mean l2 norm of the perturbation after the step' }),
  'attack/success': scalar({ description: 'the share of inputs fooled after the step' }),
})

/**
 * The Carlini-Wagner l2 attack (2017) as a stepped algorithm, one optimiser step on the perturbation per step:
 * minimise ||delta||^2 + c max(z_y - max_{j != y} z_j, -kappa) over delta, with the box constraint enforced by
 * clipping, keeping for each input the closest adversarial point found along the way. It logs the perturbation, the
 * objective, the mean margin, the mean distance and the share of inputs fooled. The optimiser is a slot (default
 * Adam); the state, including the optimiser's, serialises.
 *
 * @param model - Inputs to logits.
 * @param x - The inputs `[N, ...]`.
 * @param labels - True labels (the targets when targeted).
 * @param options - The constant c, the confidence kappa, steps, optimiser or learning rate.
 * @returns The algorithm; it is done after `steps` steps.
 * @concept adversarial-examples
 * @example
 * ```ts
 * import { carliniWagnerAlgorithm } from 'mdml-engine/nn'
 * import { matmul } from 'mdml-engine/tensor'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const model = (x) => matmul(x, [[1, -1], [0, 0]])
 * runAlgorithm(carliniWagnerAlgorithm(model, [[1, 0]], [0], { steps: 200, c: 10 }), { maxSteps: 200 }).state.won // => [true]
 * ```
 */
export function carliniWagnerAlgorithm(
  model: LogitsFunction,
  x: TensorLike,
  labels: TensorLike,
  options: CarliniWagnerOptions = {},
): Algorithm<CarliniWagnerState, CarliniWagnerPath> {
  const x0 = asTensor(x)
  const y = asTensor(labels)
  const targeted = options.targeted === true
  const c = options.c ?? 1
  const kappa = options.kappa ?? 0
  const steps = options.steps ?? 100
  const optimiser = options.optimiser ?? adam({ learningRate: options.learningRate ?? 0.05 })
  const n = x0.shape[0] ?? 1
  const advOf = (delta: Value): Value =>
    options.range === undefined ? add(x0, delta) : clip(add(x0, delta), options.range[0], options.range[1])
  const objective = (delta: Value): Value => {
    const z = model(advOf(delta))
    // Untargeted: z_y - max other (pushed below -kappa); targeted: max other - z_t.
    const m = marginLoss(z, y)
    const f = maximum(targeted ? m : neg(m), -kappa)
    const d2 = sum(square(perExample(delta)), 1)
    return sum(add(d2, mul(f, c)))
  }
  const flat0 = toFlat(x0)
  const d = flat0.length / n
  const zero = (): Value => mul(x0, 0)
  return {
    name: 'carlini-wagner',
    config: { c, kappa, steps, targeted, optimiser: optimiser.name },
    init: () => ({
      step: 0,
      delta: toTensor(zero()),
      optimiser: optimiser.init(zero()),
      best: Float64Array.from(flat0),
      bestDistance: new Array<number>(n).fill(0),
      won: new Array<boolean>(n).fill(false),
      loss: 0,
      margin: 0,
      norm: 0,
      successRate: 0,
    }),
    step: (s, ctx) => {
      const { value, grad: g } = valueAndGrad(objective)(s.delta as Value)
      const r = optimiser.update(g, s.optimiser, s.delta)
      const delta = toTensor(applyUpdates(s.delta, r.updates) as Value)
      const adv = advOf(delta)
      const z = model(adv)
      const f = fooled(z, y, targeted)
      const advFlat = toFlat(adv)
      const dist = toFlat(rowNorm(sub(adv, x0), 'l2'))
      const best = Float64Array.from(s.best)
      const bestDistance = s.bestDistance.slice()
      const won = s.won.slice()
      f.forEach((ok, i) => {
        if (ok && (!won[i] || (dist[i] ?? 0) < (bestDistance[i] ?? 0))) {
          bestDistance[i] = dist[i] ?? 0
          won[i] = true
          for (let j = 0; j < d; j++) best[i * d + j] = advFlat[i * d + j] ?? 0
        }
      })
      const mean_ = (v: readonly number[]): number => v.reduce((a, b) => a + b, 0) / Math.max(1, n)
      const next: CarliniWagnerState = {
        step: s.step + 1,
        delta,
        optimiser: r.state,
        best,
        bestDistance,
        won,
        loss: item(value) / Math.max(1, n),
        margin: mean_(toFlat(marginLoss(z, y))),
        norm: mean_(dist),
        successRate: f.filter(Boolean).length / Math.max(1, f.length),
      }
      ctx.log('attack/step', next.step)
      ctx.log('attack/delta', next.delta)
      ctx.log('attack/loss', next.loss)
      ctx.log('attack/margin', next.margin)
      ctx.log('attack/norm', next.norm)
      ctx.log('attack/success', next.successRate)
      return next
    },
    done: (s) => (s.step >= steps ? 'completed' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as CarliniWagnerState,
    observe: (s) => ({ metrics: { loss: s.loss, margin: s.margin, norm: s.norm, successRate: s.successRate } }),
  }
}

/**
 * The Carlini-Wagner l2 attack (2017): minimise ||delta||^2 + c max(z_y - max_{j != y} z_j, -kappa) over the
 * perturbation delta by Adam, with the box constraint enforced by clipping, and keep for each input the closest
 * adversarial point found along the way. It looks for the smallest perturbation rather than the strongest one in a
 * ball. This is {@link carliniWagnerAlgorithm} run to the end.
 *
 * @param model - Inputs to logits.
 * @param x - The inputs `[N, ...]`.
 * @param labels - True labels (the targets when targeted).
 * @param options - The constant c, the confidence kappa, steps, optimiser or learning rate.
 * @returns The closest adversarial inputs found, which inputs were fooled, and their l2 distances.
 * @concept adversarial-examples
 * @example
 * ```ts
 * import { carliniWagnerL2 } from 'mdml-engine/nn'
 * import { matmul } from 'mdml-engine/tensor'
 * const model = (x) => matmul(x, [[1, -1], [0, 0]])
 * carliniWagnerL2(model, [[1, 0]], [0], { steps: 200, c: 10 }).success // => [true]
 * ```
 */
export function carliniWagnerL2(
  model: LogitsFunction,
  x: TensorLike,
  labels: TensorLike,
  options: CarliniWagnerOptions = {},
): AttackResult {
  const x0 = asTensor(x)
  const s = runAlgorithm(carliniWagnerAlgorithm(model, x0, labels, options), {
    maxSteps: options.steps ?? 100,
  }).state
  const distance = s.best.length === 0 ? [] : s.bestDistance.map((v, i) => (s.won[i] ? v : 0))
  return { x: fromStorage(Float64Array.from(s.best), [...x0.shape], 'float64'), success: [...s.won], distance }
}

/** The rows of a batch as `[1, ...]` tensors. */
function batchRows(x: Value): Tensor[] {
  const n = x.shape[0] ?? 0
  const d = n === 0 ? 0 : toFlat(x).length / n
  const data = toFlat(x)
  return Array.from({ length: n }, (_, i) =>
    fromStorage(Float64Array.from(data.slice(i * d, (i + 1) * d)), [1, ...x.shape.slice(1)], 'float64'),
  )
}

const stackRows = (rows: readonly Value[]): Tensor => {
  const flat = rows.flatMap((r) => toFlat(r))
  const first = rows[0] as Value
  return fromStorage(Float64Array.from(flat), [rows.length, ...first.shape.slice(1)], 'float64')
}

/** Options of {@link deepFool}. */
export interface DeepFoolOptions {
  /** Iteration cap per input (default 50). */
  readonly steps?: number
  /** The final push past the boundary, as a fraction of the perturbation (default 0.02). */
  readonly overshoot?: number
  readonly range?: readonly [number, number]
}

/** The paths of {@link deepFoolAlgorithm}. */
export type DeepFoolPath = 'attack/step' | 'attack/point' | 'attack/margin' | 'attack/norm' | 'attack/success'

/**
 * The state of {@link deepFoolAlgorithm}: the current adversarial inputs, the accumulated perturbation, each input's
 * original class, and which inputs are still being pushed.
 */
export interface DeepFoolState {
  readonly step: number
  readonly x: Tensor
  /** The accumulated (un-overshot) perturbation. */
  readonly total: Tensor
  /** The class the model gave each input before the attack. */
  readonly original: readonly number[]
  /** Inputs still on their original class (the others are done). */
  readonly active: readonly boolean[]
  readonly margin: number
  readonly norm: number
  readonly successRate: number
}

/** What a stepped DeepFool attack logs: the adversarial inputs, the mean margin to the original class, the mean l2 distance and the share fooled. */
export const deepFoolContract: Contract<DeepFoolState, DeepFoolPath> = defineContract({
  'attack/step': scalar({ description: 'the linearisation step' }),
  'attack/point': tensorSource({ role: 'state', description: 'the adversarial inputs after the step' }),
  'attack/margin': scalar({ description: 'the mean margin (largest other logit minus the original class logit)' }),
  'attack/norm': scalar({ description: 'the mean l2 distance from the original inputs' }),
  'attack/success': scalar({ description: 'the share of inputs that left their original class' }),
})

/**
 * DeepFool (Moosavi-Dezfooli, Fawzi and Frossard 2016) as a stepped algorithm: each step linearises the logits at the
 * current point of every input still on its original class, moves it to the closest boundary of the linearised model
 * and pushes a little past it (the overshoot). It logs the adversarial inputs, the mean margin, the mean distance and
 * the share fooled. The state serialises.
 *
 * @param model - Inputs to logits.
 * @param x - The inputs `[N, ...]`.
 * @param options - Iteration cap, overshoot and the data range.
 * @returns The algorithm; it is done when every input is fooled or after `steps` steps.
 * @concept adversarial-examples
 * @example
 * ```ts
 * import { deepFoolAlgorithm } from 'mdml-engine/nn'
 * import { matmul } from 'mdml-engine/tensor'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const model = (x) => matmul(x, [[1, 0], [0, 1]])
 * runAlgorithm(deepFoolAlgorithm(model, [[1, 0]]), { maxSteps: 50 }).state.successRate // => 1
 * ```
 */
export function deepFoolAlgorithm(
  model: LogitsFunction,
  x: TensorLike,
  options: DeepFoolOptions = {},
): Algorithm<DeepFoolState, DeepFoolPath> {
  const x0 = toTensor(asTensor(x))
  const steps = options.steps ?? 50
  const overshoot = options.overshoot ?? 0.02
  const rows = batchRows(x0)
  const n = rows.length
  const classes = n === 0 ? 0 : (model(rows[0] as Value).shape.at(-1) ?? 0)
  const predicted = (v: Value): number => item(argmax(model(v), -1))
  const measure = (s: Omit<DeepFoolState, 'margin' | 'norm' | 'successRate'>): DeepFoolState => {
    const adv = batchRows(s.x)
    const margins = adv.map((r, i) => item(marginLoss(model(r), [s.original[i] ?? 0])))
    const norms = adv.map((r, i) => item(norm(reshape(sub(r, rows[i] as Value), [-1]), 2)))
    const fooledCount = adv.filter((_, i) => !s.active[i]).length
    return {
      ...s,
      margin: margins.reduce((a, b) => a + b, 0) / Math.max(1, n),
      norm: norms.reduce((a, b) => a + b, 0) / Math.max(1, n),
      successRate: fooledCount / Math.max(1, n),
    }
  }
  const advance = (row: Tensor, adv0: Tensor, total0: Tensor, original: number): { adv: Tensor; total: Tensor } => {
    let bestRatio = Number.POSITIVE_INFINITY
    let bestStep: Value = total0
    const gOriginal = grad((v: Value) => sum(mul(model(v), oneHot([original], classes))))(adv0)
    const zOriginal = item(sum(mul(model(adv0), oneHot([original], classes))))
    for (let k = 0; k < classes; k++) {
      if (k === original) continue
      const gk = grad((v: Value) => sum(mul(model(v), oneHot([k], classes))))(adv0)
      const w = sub(gk, gOriginal)
      const f = item(sum(mul(model(adv0), oneHot([k], classes)))) - zOriginal
      const wn = item(norm(reshape(w, [-1]), 2))
      const ratio = Math.abs(f) / Math.max(wn, 1e-30)
      if (ratio < bestRatio) {
        bestRatio = ratio
        bestStep = mul(w, (Math.abs(f) + 1e-4) / Math.max(wn * wn, 1e-30))
      }
    }
    const total = add(total0, bestStep)
    let adv: Value = add(row, mul(total, 1 + overshoot))
    if (options.range !== undefined) adv = clip(adv, options.range[0], options.range[1])
    return { adv: toTensor(adv), total: toTensor(total) }
  }
  return {
    name: 'deepfool',
    config: { steps, overshoot },
    init: () =>
      measure({
        step: 0,
        x: x0,
        total: toTensor(mul(x0, 0)),
        original: rows.map(predicted),
        active: rows.map(() => true),
      }),
    step: (s, ctx) => {
      const adv = batchRows(s.x)
      const total = batchRows(s.total)
      const active = s.active.slice()
      const nextAdv: Tensor[] = []
      const nextTotal: Tensor[] = []
      rows.forEach((row, i) => {
        const a = adv[i] as Tensor
        const t = total[i] as Tensor
        if (!active[i] || predicted(a) !== s.original[i]) {
          active[i] = false
          nextAdv.push(a)
          nextTotal.push(t)
          return
        }
        const r = advance(row, a, t, s.original[i] ?? 0)
        nextAdv.push(r.adv)
        nextTotal.push(r.total)
        active[i] = predicted(r.adv) === s.original[i]
      })
      const next = measure({
        step: s.step + 1,
        x: n === 0 ? x0 : stackRows(nextAdv),
        total: n === 0 ? s.total : stackRows(nextTotal),
        original: s.original,
        active,
      })
      ctx.log('attack/step', next.step)
      ctx.log('attack/point', next.x)
      ctx.log('attack/margin', next.margin)
      ctx.log('attack/norm', next.norm)
      ctx.log('attack/success', next.successRate)
      return next
    },
    done: (s) => (s.step >= steps || s.active.every((a) => !a) ? 'completed' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as DeepFoolState,
    observe: (s) => ({ metrics: { margin: s.margin, norm: s.norm, successRate: s.successRate } }),
  }
}

/**
 * DeepFool (Moosavi-Dezfooli, Fawzi and Frossard 2016): the minimal l2 perturbation to the nearest linearised decision
 * boundary, found by repeatedly linearising the logits at the current point, stepping to the closest boundary of the
 * linearised model, and finishing with a small overshoot. A cheap estimate of an input's distance to the boundary.
 * This is {@link deepFoolAlgorithm} run to the end.
 *
 * @param model - Inputs to logits.
 * @param x - The inputs `[N, ...]`.
 * @param options - Iteration cap, overshoot and the data range.
 * @returns The adversarial inputs, which were fooled and their l2 distances.
 * @concept adversarial-examples
 * @example
 * ```ts
 * import { deepFool } from 'mdml-engine/nn'
 * import { matmul } from 'mdml-engine/tensor'
 * // Logits (x0, x1): the boundary is x0 = x1, at l2 distance 1 / sqrt(2) from (1, 0.3)... here from (1, 0).
 * const r = deepFool((x) => matmul(x, [[1, 0], [0, 1]]), [[1, 0]])
 * r.distance[0] > 0.7071 && r.distance[0] < 0.7071 * 1.05 // => true
 * ```
 */
export function deepFool(model: LogitsFunction, x: TensorLike, options: DeepFoolOptions = {}): AttackResult {
  const x0 = asTensor(x)
  const s = runAlgorithm(deepFoolAlgorithm(model, x0, options), { maxSteps: options.steps ?? 50 }).state
  const rows = batchRows(x0)
  const adv = batchRows(s.x)
  const success = adv.map((r, i) => item(argmax(model(r), -1)) !== s.original[i])
  const distance = adv.map((r, i) => (success[i] ? item(norm(reshape(sub(r, rows[i] as Value), [-1]), 2)) : 0))
  return { x: s.x, success, distance }
}

/** The result of {@link transferAttack}. */
export interface TransferResult {
  /** The adversarial inputs crafted against the source model. */
  readonly x: Tensor
  /** The share of inputs the source model gets right that the attack fools on the source. */
  readonly sourceSuccess: number
  /** The share of inputs the target gets right (and the source gets right) that the crafted inputs fool on the target. */
  readonly transferRate: number
}

/**
 * A transfer attack (Papernot, McDaniel and Goodfellow 2016): craft the adversarial inputs against a source model with
 * PGD and test them on a different target model, which the attacker needs no access to beyond its answers. The
 * transfer rate is the share of inputs both models classify correctly that now fool the target.
 *
 * @param source - The attacked surrogate.
 * @param target - The model the inputs are tested on.
 * @param x - The inputs.
 * @param labels - Their labels.
 * @param options - PGD's options, with the `key` for the random start.
 * @returns The adversarial inputs and the two success rates.
 * @concept adversarial-examples
 * @example
 * ```ts
 * import { transferAttack } from 'mdml-engine/nn'
 * import { matmul } from 'mdml-engine/tensor'
 * const model = (x) => matmul(x, [[1, -1], [0, 0]])
 * transferAttack(model, model, [[1, 0]], [0], { epsilon: 2, steps: 5, randomStart: false }).transferRate // => 1
 * ```
 */
export function transferAttack(
  source: LogitsFunction,
  target: LogitsFunction,
  x: TensorLike,
  labels: TensorLike,
  options: PgdOptions,
): TransferResult {
  const x0 = asTensor(x)
  const y = asTensor(labels)
  const adv = pgd(source, x0, y, options)
  const targeted = options.targeted === true
  const cleanSource = fooled(source(x0), y, false).map((v) => !v)
  const cleanTarget = fooled(target(x0), y, false).map((v) => !v)
  const fooledSource = fooled(source(adv), y, targeted)
  const fooledTarget = fooled(target(adv), y, targeted)
  const base = cleanSource.map((v, i) => v && (cleanTarget[i] ?? false))
  const count = (flags: readonly boolean[], over: readonly boolean[]): number =>
    flags.filter((f, i) => f && (over[i] ?? false)).length
  const denominatorSource = cleanSource.filter(Boolean).length
  const denominator = base.filter(Boolean).length
  return {
    x: adv,
    sourceSuccess: denominatorSource === 0 ? 0 : count(fooledSource, cleanSource) / denominatorSource,
    transferRate: denominator === 0 ? 0 : count(fooledTarget, base) / denominator,
  }
}

/** The result of {@link squareAttack}. */
export interface SquareResult extends AttackResult {
  /** Queries to the model, per input (stopping at the first success). */
  readonly queries: readonly number[]
}

/** Options of {@link squareAttack} and {@link squareAlgorithm}. */
export interface SquareOptions {
  readonly epsilon: number
  /** The query budget per input (default 200). */
  readonly queries?: number
  /** The initial share of coordinates changed per query, halved as the budget is spent (default 0.1). */
  readonly pInit?: number
  readonly range?: readonly [number, number]
  readonly key: Key
}

/** The paths of {@link squareAlgorithm}. */
export type SquarePath =
  | 'attack/step'
  | 'attack/point'
  | 'attack/margin'
  | 'attack/norm'
  | 'attack/success'
  | 'attack/queries'

/** The state of {@link squareAlgorithm}: the current adversarial inputs, their margins and the queries spent. */
export interface SquareState {
  /** Queries issued to the batch so far (the initial corner counts for the first). */
  readonly step: number
  /** The current adversarial inputs, flattened per input. */
  readonly current: Float64Array
  readonly margins: readonly number[]
  readonly queries: readonly number[]
  /** Inputs already fooled (they are left alone). */
  readonly done: readonly boolean[]
  readonly norm: number
  readonly successRate: number
}

/** What a stepped Square attack logs: the adversarial inputs, the mean margin and l-infinity distance, the share fooled and the queries used. */
export const squareContract: Contract<SquareState, SquarePath> = defineContract({
  'attack/step': scalar({ description: 'the query round' }),
  'attack/point': tensorSource({ role: 'state', description: 'the adversarial inputs after the round' }),
  'attack/margin': scalar({ description: 'the mean margin (largest wrong logit minus the label logit)' }),
  'attack/norm': scalar({ description: 'the mean l-infinity distance from the original inputs' }),
  'attack/success': scalar({ description: 'the share of inputs fooled' }),
  'attack/queries': scalar({ description: 'the queries used so far, summed over inputs' }),
})

/**
 * A Square-style black-box attack (Andriushchenko et al. 2020, l-infinity, untargeted) as a stepped algorithm: only the
 * model's logits are read. The input starts at a random corner of the ball (every coordinate moved by +-epsilon); each
 * step re-draws the perturbation on a random subset of coordinates (a share that shrinks as the budget is spent) and
 * keeps the change if the margin loss grows. For images the paper changes a square of pixels, here the subset is of
 * coordinates, which is the same move for any input. The state serialises; the randomness comes from `options.key`.
 *
 * @param model - Inputs to logits (queried, never differentiated).
 * @param x - The inputs `[N, ...]`.
 * @param labels - True labels.
 * @param options - Radius, query budget, range and key.
 * @returns The algorithm; it is done when every input is fooled or the budget is spent.
 * @concept adversarial-examples
 * @example
 * ```ts
 * import { squareAlgorithm } from 'mdml-engine/nn'
 * import { key } from 'mdml-engine/random'
 * import { matmul } from 'mdml-engine/tensor'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const model = (x) => matmul(x, [[1, -1], [0, 0], [0, 0]])
 * runAlgorithm(squareAlgorithm(model, [[0.3, 0, 0]], [0], { epsilon: 0.5, key: key(0) }), { maxSteps: 200 }).state.done // => [true]
 * ```
 */
export function squareAlgorithm(
  model: LogitsFunction,
  x: TensorLike,
  labels: TensorLike,
  options: SquareOptions,
): Algorithm<SquareState, SquarePath> {
  const x0 = asTensor(x)
  const y = asTensor(labels)
  const n = x0.shape[0] ?? 0
  const flat0 = toFlat(x0)
  const d = n === 0 ? 0 : flat0.length / n
  const budget = options.queries ?? 200
  const eps = options.epsilon
  const clipTo = (v: number): number =>
    options.range === undefined ? v : Math.min(options.range[1], Math.max(options.range[0], v))
  const asBatch = (data: Float64Array): Tensor => fromStorage(data, [...x0.shape], 'float64')
  const measure = (
    step: number,
    current: Float64Array,
    margins: readonly number[],
    queries: readonly number[],
    done: readonly boolean[],
  ): SquareState => {
    let dist = 0
    for (let i = 0; i < n; i++) {
      let m = 0
      for (let j = 0; j < d; j++) m = Math.max(m, Math.abs((current[i * d + j] ?? 0) - (flat0[i * d + j] ?? 0)))
      dist += m
    }
    return {
      step,
      current,
      margins,
      queries,
      done,
      norm: dist / Math.max(1, n),
      successRate: margins.filter((m) => m > 0).length / Math.max(1, n),
    }
  }
  return {
    name: 'square-attack',
    config: { epsilon: eps, queries: budget, pInit: options.pInit ?? 0.1 },
    init: () => {
      const current = Float64Array.from(flat0)
      const signs = toFlat(uniform(foldIn(options.key, 0), [n, d]))
      for (let i = 0; i < n * d; i++) current[i] = clipTo((flat0[i] ?? 0) + ((signs[i] ?? 0) < 0.5 ? -eps : eps))
      const margins = toFlat(marginLoss(model(asBatch(current)), y))
      return measure(
        0,
        current,
        Array.from(margins),
        new Array<number>(n).fill(1),
        Array.from(margins, (m) => m > 0),
      )
    },
    step: (s, ctx) => {
      const q = s.step + 1
      const share = Math.max(1 / d, (options.pInit ?? 0.1) * 0.5 ** Math.floor((4 * q) / budget))
      const m = Math.max(1, Math.round(share * d))
      const rank = toFlat(uniform(foldIn(options.key, 2 * q), [n, d]))
      const sgn = toFlat(uniform(foldIn(options.key, 2 * q + 1), [n, d]))
      const candidate = Float64Array.from(s.current)
      for (let i = 0; i < n; i++) {
        if (s.done[i]) continue
        const order = Array.from({ length: d }, (_, j) => j).sort(
          (a, b) => (rank[i * d + a] ?? 0) - (rank[i * d + b] ?? 0),
        )
        for (const j of order.slice(0, m))
          candidate[i * d + j] = clipTo((flat0[i * d + j] ?? 0) + ((sgn[i * d + j] ?? 0) < 0.5 ? -eps : eps))
      }
      const next = toFlat(marginLoss(model(asBatch(candidate)), y))
      const current = Float64Array.from(s.current)
      const margins = s.margins.slice()
      const queries = s.queries.slice()
      const done = s.done.slice()
      for (let i = 0; i < n; i++) {
        if (done[i]) continue
        queries[i] = (queries[i] ?? 0) + 1
        if ((next[i] ?? Number.NEGATIVE_INFINITY) > (margins[i] ?? 0)) {
          for (let j = 0; j < d; j++) current[i * d + j] = candidate[i * d + j] ?? 0
          margins[i] = next[i] as number
        }
        if ((margins[i] ?? 0) > 0) done[i] = true
      }
      const out = measure(q, current, margins, queries, done)
      ctx.log('attack/step', out.step)
      ctx.log('attack/point', asBatch(current))
      ctx.log('attack/margin', margins.reduce((a, b) => a + b, 0) / Math.max(1, n))
      ctx.log('attack/norm', out.norm)
      ctx.log('attack/success', out.successRate)
      ctx.log(
        'attack/queries',
        queries.reduce((a, b) => a + b, 0),
      )
      return out
    },
    done: (s) => (s.step >= budget || s.done.every(Boolean) ? 'completed' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as SquareState,
    observe: (s) => ({ metrics: { norm: s.norm, successRate: s.successRate } }),
  }
}

/**
 * A Square-style black-box attack (Andriushchenko et al. 2020, l-infinity, untargeted): only the model's logits are
 * read. The input starts at a random corner of the ball (every coordinate moved by +-epsilon); each query re-draws the
 * perturbation on a random subset of coordinates (a share that shrinks as the budget is spent) and keeps the change if
 * the margin loss grows. For images the paper changes a square of pixels, here the subset is of coordinates, which is
 * the same move for any input. This is {@link squareAlgorithm} run to the end.
 *
 * @param model - Inputs to logits (queried, never differentiated).
 * @param x - The inputs `[N, ...]`.
 * @param labels - True labels.
 * @param options - Radius, query budget, range and key.
 * @returns The adversarial inputs, success flags, l-infinity distances and queries used per input.
 * @concept adversarial-examples
 * @example
 * ```ts
 * import { squareAttack } from 'mdml-engine/nn'
 * import { key } from 'mdml-engine/random'
 * import { matmul } from 'mdml-engine/tensor'
 * const model = (x) => matmul(x, [[1, -1], [0, 0], [0, 0]])
 * squareAttack(model, [[0.3, 0, 0]], [0], { epsilon: 0.5, key: key(0) }).success // => [true]
 * ```
 */
export function squareAttack(
  model: LogitsFunction,
  x: TensorLike,
  labels: TensorLike,
  options: SquareOptions,
): SquareResult {
  const x0 = asTensor(x)
  const n = x0.shape[0] ?? 0
  const d = n === 0 ? 0 : toFlat(x0).length / n
  const s = runAlgorithm(squareAlgorithm(model, x0, labels, options), { maxSteps: options.queries ?? 200 }).state
  const flat0 = toFlat(x0)
  const distance = Array.from({ length: n }, (_, i) => {
    let m = 0
    for (let j = 0; j < d; j++) m = Math.max(m, Math.abs((s.current[i * d + j] ?? 0) - (flat0[i * d + j] ?? 0)))
    return m
  })
  return {
    x: fromStorage(Float64Array.from(s.current), [...x0.shape], 'float64'),
    success: s.margins.map((m) => m > 0),
    distance,
    queries: [...s.queries],
  }
}

/**
 * Accuracy of a classifier on a batch.
 *
 * @param model - Inputs to logits.
 * @param x - Inputs.
 * @param labels - Labels.
 * @returns The share of correct predictions.
 * @example
 * ```ts
 * import { accuracyOf } from 'mdml-engine/nn'
 * accuracyOf((x) => x, [[2, 1], [0, 1]], [0, 0]) // => 0.5
 * ```
 */
export function accuracyOf(model: LogitsFunction, x: TensorLike, labels: TensorLike): number {
  const wrong = fooled(model(asTensor(x)), asTensor(labels), false)
  return 1 - wrong.filter(Boolean).length / Math.max(1, wrong.length)
}

/**
 * Robust accuracy over a sweep of radii: for each epsilon, the accuracy on the inputs the attack produces for it (the
 * attack is a function of the radius, so the same routine sweeps any of them). The first entry for epsilon 0 is the
 * clean accuracy if `0` is among the radii.
 *
 * @param model - Inputs to logits.
 * @param x - Inputs.
 * @param labels - Labels.
 * @param epsilons - Radii.
 * @param attack - Builds the adversarial inputs for a radius (for example a call to {@link pgd}).
 * @returns Accuracy per radius.
 * @concept adversarial-examples
 * @example
 * ```ts
 * import { robustAccuracy, fgsm } from 'mdml-engine/nn'
 * import { matmul } from 'mdml-engine/tensor'
 * const model = (x) => matmul(x, [[1, -1], [0, 0]])
 * robustAccuracy(model, [[1, 0]], [0], [0, 2], (epsilon) => fgsm(model, [[1, 0]], [0], { epsilon })) // => [{ epsilon: 0, accuracy: 1 }, { epsilon: 2, accuracy: 0 }]
 * ```
 */
export function robustAccuracy(
  model: LogitsFunction,
  x: TensorLike,
  labels: TensorLike,
  epsilons: readonly number[],
  attack: (epsilon: number) => TensorLike,
): { readonly epsilon: number; readonly accuracy: number }[] {
  return epsilons.map((epsilon) => ({
    epsilon,
    accuracy: accuracyOf(model, epsilon === 0 ? x : attack(epsilon), labels),
  }))
}

/** Options of {@link adversarialTrainStep}. */
export interface AdversarialTrainingOptions {
  /** The inner maximisation: radius, norm, steps, loss and range (the key is derived from the state's). */
  readonly attack: Omit<PgdOptions, 'key'>
  /** TRADES (Zhang et al. 2019): train on the clean cross-entropy plus beta times the KL from the clean to the adversarial prediction, the inner attack maximising that KL. */
  readonly trades?: { readonly beta: number }
  /** Weight of the clean loss added to the adversarial one (default 0: adversarial only; ignored with TRADES). */
  readonly cleanWeight?: number
}

/** What an adversarial training step reports besides the new state. */
export interface AdversarialStep {
  readonly state: TrainState
  /** The mean cross-entropy on the clean batch. */
  readonly cleanLoss: number
  /** The mean cross-entropy on the adversarial batch. */
  readonly robustLoss: number
}

/**
 * One step of adversarial training (Madry et al. 2018): craft PGD examples against the current parameters (evaluation
 * mode, the step's key for the random start), then take an optimiser step on their loss; with `trades` the objective
 * is TRADES's clean loss plus a KL term, against an inner attack that maximises that KL. A wrapper over `TrainState`,
 * so any loop that steps a state can use it in place of `trainStep`.
 *
 * @param model - The classifier module (logits out).
 * @param state - The training state.
 * @param x - The minibatch inputs.
 * @param labels - Their labels.
 * @param optimizer - The gradient transformation the state was built with.
 * @param options - The attack, and optionally TRADES or a clean-loss weight.
 * @returns The next state and the clean and robust losses of the batch.
 * @concept adversarial-training
 * @example
 * ```ts
 * import { adversarialTrainStep, trainState, mlp } from 'mdml-engine/nn'
 * import { sgd } from 'mdml-engine/optim'
 * const m = mlp([2, 4, 2])
 * const opt = sgd({ learningRate: 0.1 })
 * adversarialTrainStep(m, trainState(m, opt), [[0.5, -0.5], [-1, 1]], [0, 1], opt, { attack: { epsilon: 0.1, steps: 3 } }).state.step // => 1
 * ```
 */
export function adversarialTrainStep(
  model: Module<Value, Value>,
  state: TrainState,
  x: TensorLike,
  labels: TensorLike,
  optimizer: GradientTransformation,
  options: AdversarialTrainingOptions,
): AdversarialStep {
  const x0 = asTensor(x)
  const y = asTensor(labels)
  const stepKey = foldIn(state.key, state.step)
  const evalModel: LogitsFunction = (v) => model.apply(state.params, v, { train: false, buffers: state.buffers })
  const attack: PgdOptions = {
    ...options.attack,
    ...(options.trades === undefined ? {} : { loss: 'kl' as const }),
    key: foldIn(stepKey, 1),
  }
  const adv = pgd(evalModel, x0, y, attack)
  let next = state.buffers
  let cleanLoss = Number.NaN
  let robustLoss = Number.NaN
  const forward = (p: Params, input: Value, k: number): Value =>
    model.apply(p, input, {
      train: true,
      buffers: state.buffers,
      key: foldIn(stepKey, k),
      onBuffers: (b) => {
        next = b
      },
    })
  const objective = (p: Params): Value => {
    const zAdv = forward(p, adv, 2)
    const ceAdv = crossEntropy(zAdv, y)
    robustLoss = item(primalOf(ceAdv))
    if (options.trades !== undefined) {
      const zClean = forward(p, x0, 3)
      const ce = crossEntropy(zClean, y)
      cleanLoss = item(primalOf(ce))
      return add(ce, mul(klDivergence(zClean, zAdv, { logits: true }), options.trades.beta))
    }
    const w = options.cleanWeight ?? 0
    if (w === 0) {
      cleanLoss = item(primalOf(crossEntropy(model.apply(p, x0, { train: false, buffers: state.buffers }), y)))
      return ceAdv
    }
    const ce = crossEntropy(forward(p, x0, 3), y)
    cleanLoss = item(primalOf(ce))
    return add(mul(ceAdv, 1 - w), mul(ce, w))
  }
  const { value, grad: g } = valueAndGrad(objective)(state.params)
  const mask = model.trainableMask()
  const applied = applyGradients(state, g, optimizer, { mask })
  return {
    state: {
      ...applied.state,
      buffers: concreteTree(next),
      step: state.step + 1,
      loss: item(value),
      gradNorm: item(globalNorm(g)),
    },
    cleanLoss,
    robustLoss,
  }
}
