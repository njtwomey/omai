/**
 * Fair classifiers trained with fairness in the objective or the constraints, on numeric features: the prejudice
 * remover (Kamishima et al., 2012), adversarial debiasing (Zhang, Lemoine and Mitchell, 2018), fairness-constrained
 * logistic regression (Zafar et al., 2017), the meta fair classifier for the statistical rate (Celis et al., 2019) and
 * GerryFair's fictitious play against rich subgroups (Kearns et al., 2018), each as AIF360 offers them.
 */

import { valueAndGrad } from '../autodiff/index.ts'
import { WhiteboxError } from '../base/index.ts'
import { lstsq } from '../numerics/linalg/index.ts'
import { lbfgs, slsqp } from '../optim/index.ts'
import { foldIn, type Key, key as makeKey, normal } from '../random/index.ts'
import {
  abs,
  add,
  fromStorage,
  log,
  matmul,
  mean,
  mul,
  reshape,
  sigmoid,
  sliceAxis,
  softplus,
  square,
  sub,
  sum,
  type Tensor,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import { richSubgroupAudit } from './intersectional.ts'

/** Rows as a matrix tensor with a leading column of ones (the intercept). */
function design(
  name: string,
  x: readonly (readonly number[])[],
): { readonly t: Tensor; readonly n: number; readonly d: number } {
  const n = x.length
  const d = (x[0]?.length ?? 0) + 1
  if (n === 0) throw new WhiteboxError('value', `${name}: needs at least one row`)
  const data = new Float64Array(n * d)
  for (let i = 0; i < n; i++) {
    data[i * d] = 1
    const row = x[i] ?? []
    if (row.length !== d - 1) throw new WhiteboxError('shape', `${name}: rows differ in length`)
    for (let j = 1; j < d; j++) data[i * d + j] = row[j - 1] ?? 0
  }
  return { t: fromStorage(data, [n, d], 'float64'), n, d }
}

const column = (v: readonly number[]): Tensor => fromStorage(Float64Array.from(v), [v.length, 1], 'float64')

/** Mean binary cross-entropy of logits, by softplus: log(1 + e^z) - y z. */
function logisticLoss(logits: Value, y: Tensor): Value {
  return mean(sub(softplus(logits), mul(y, logits)))
}

/** A linear model's decision: P(positive) = sigmoid(intercept + w.x). */
export interface LinearClassifier {
  readonly intercept: number
  readonly weights: readonly number[]
}

/**
 * The positive-class probabilities of a linear classifier.
 *
 * @param model - Its intercept and weights.
 * @param x - Rows `[n, d]`.
 * @returns P(positive) for each row.
 * @example
 * ```ts
 * import { linearProbabilities } from 'whitebox-engine/fairness'
 * linearProbabilities({ intercept: 0, weights: [1] }, [[0]]) // => [0.5]
 * ```
 */
export function linearProbabilities(model: LinearClassifier, x: readonly (readonly number[])[]): number[] {
  return x.map(
    (row) => 1 / (1 + Math.exp(-(model.intercept + row.reduce((s, v, j) => s + v * (model.weights[j] ?? 0), 0)))),
  )
}

/** Options of {@link prejudiceRemover}. */
export interface PrejudiceRemoverOptions {
  /** The weight of the prejudice index (default 1, AIF360's eta). */
  readonly eta?: number
  /** The L2 penalty on the weights (default 1e-2). */
  readonly l2?: number
  readonly maxIterations?: number
}

/** A fitted prejudice remover: one linear classifier per group, and the prejudice index it reached. */
export interface PrejudiceRemover {
  readonly models: readonly LinearClassifier[]
  /** The prejudice index (mutual information between decision and group, as estimated from the probabilities). */
  readonly prejudiceIndex: number
  readonly loss: number
}

/**
 * The prejudice remover (Kamishima, Akaho, Asoh and Sakuma, 2012; AIF360's `PrejudiceRemover`): logistic regression,
 * one per group, fitted by L-BFGS to the log loss plus `eta` times the prejudice index, the mutual information between
 * the predicted label and the group estimated from the predicted probabilities, `sum_i sum_y P(y | x_i) log(P(y | s_i) /
 * P(y))`. Larger `eta` trades accuracy for independence of the decision from the group.
 *
 * @param x - Features `[n, d]` (without the protected attribute; the group picks the model).
 * @param labels - Binary labels `[n]`.
 * @param groups - Group indices `[n]`.
 * @param options - eta, the L2 penalty and the iteration cap.
 * @returns The per-group models, the prejudice index and the loss.
 * @concept algorithmic-fairness
 * @example
 * ```ts
 * import { prejudiceRemover } from 'whitebox-engine/fairness'
 * const x = [[0], [1], [2], [3], [1], [2], [3], [4]]
 * const loose = prejudiceRemover(x, [0, 0, 1, 1, 0, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1], { eta: 0 })
 * const tight = prejudiceRemover(x, [0, 0, 1, 1, 0, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1], { eta: 25 })
 * tight.prejudiceIndex < loose.prejudiceIndex // => true
 * ```
 */
export function prejudiceRemover(
  x: readonly (readonly number[])[],
  labels: readonly number[],
  groups: readonly number[],
  options: PrejudiceRemoverOptions = {},
): PrejudiceRemover {
  const { t: X, n, d } = design('prejudiceRemover', x)
  if (labels.length !== n || groups.length !== n)
    throw new WhiteboxError('shape', 'prejudiceRemover: one label and group per row')
  const k = Math.max(...groups) + 1
  const eta = options.eta ?? 1
  const l2 = options.l2 ?? 1e-2
  const onehot = new Float64Array(n * k)
  const counts = new Array<number>(k).fill(0)
  for (const [i, g] of groups.entries()) {
    onehot[i * k + g] = 1
    counts[g] = (counts[g] ?? 0) + 1
  }
  const G = fromStorage(onehot, [n, k], 'float64')
  const Gt = fromStorage(
    Float64Array.from(
      { length: k * n },
      (_, j) => (onehot[(j % n) * k + Math.floor(j / n)] ?? 0) / Math.max(1, counts[Math.floor(j / n)] ?? 0),
    ),
    [k, n],
    'float64',
  )
  const y = column(labels.map((v) => (v === 1 ? 1 : 0)))
  const parts = (p: Value): { logits: Value; pi: Value } => {
    const W = reshape(p, [k, d])
    // Each row's logit under its own group's model: sum_j X[i, j] W[g_i, j].
    const logits = reshape(sum(mul(X, matmul(G, W)), 1), [n, 1])
    const prob = sigmoid(logits)
    const byGroup = matmul(Gt, prob) // [k, 1]: each group's mean probability
    const own = matmul(G, byGroup) // [n, 1]
    const all = mean(prob)
    const eps = 1e-12
    const pi = mean(
      add(
        mul(prob, sub(log(add(own, eps)), log(add(all, eps)))),
        mul(sub(1, prob), sub(log(add(sub(1, own), eps)), log(add(sub(1, all), eps)))),
      ),
    )
    return { logits, pi }
  }
  const objective = (p: Value): Value => {
    const { logits, pi } = parts(p)
    return add(add(logisticLoss(logits, y), mul(pi, eta)), mul(sum(square(p)), l2 / 2))
  }
  const r = lbfgs(objective, fromStorage(new Float64Array(k * d), [k * d], 'float64'), {
    gradTol: 1e-8,
    maxIterations: options.maxIterations ?? 500,
  })
  const w = toFlat(r.x as Tensor)
  const { pi } = parts(r.x as Tensor)
  return {
    models: Array.from({ length: k }, (_, g) => ({
      intercept: w[g * d] ?? 0,
      weights: Array.from({ length: d - 1 }, (_, j) => w[g * d + j + 1] ?? 0),
    })),
    prejudiceIndex: toFlat(pi as Tensor)[0] ?? Number.NaN,
    loss: r.fun,
  }
}

/** Options of {@link adversarialDebiasing}. */
export interface AdversarialDebiasingOptions {
  /** The adversary's weight in the predictor's update (default 1, AIF360's adversary_loss_weight is 0.1). */
  readonly adversaryWeight?: number
  /** `demographicParity` (the adversary sees the logit) or `equalisedOdds` (it also sees the label). Default the first. */
  readonly constraint?: 'demographicParity' | 'equalisedOdds'
  /** Full-batch gradient steps (default 500). */
  readonly steps?: number
  /** The predictor's step size (default 0.05). */
  readonly learningRate?: number
  /** The adversary's step size (default 1) and its steps per predictor step (default 5), so it keeps up. */
  readonly adversaryLearningRate?: number
  readonly adversarySteps?: number
  readonly key?: Key
}

/** One recorded step of {@link adversarialDebiasing}. */
export interface AdversarialStep {
  readonly step: number
  readonly predictorLoss: number
  readonly adversaryLoss: number
}

/** A classifier trained against an adversary, and the training's history. */
export interface AdversarialDebiasing {
  readonly model: LinearClassifier
  /** The adversary's final weights (on its inputs, then its intercept). */
  readonly adversary: readonly number[]
  readonly steps: readonly AdversarialStep[]
}

/**
 * Adversarial debiasing (Zhang, Lemoine and Mitchell, 2018; AIF360's `AdversarialDebiasing`): a logistic predictor
 * trained against a logistic adversary that tries to recover the group (binary: the privileged group 0 against the
 * rest) from the predictor's output (and, for equalised odds, the label). Each step the adversary descends its loss and
 * the predictor descends its own loss with its gradient's component along the adversary's removed, minus the
 * adversary's gradient times `adversaryWeight` (Zhang et al.'s projection), so the predictor learns what the adversary
 * cannot use. With `adversaryWeight` 0 the predictor trains alone (plain logistic regression), for comparison.
 *
 * @param x - Features `[n, d]`.
 * @param labels - Binary labels `[n]`.
 * @param groups - Group indices `[n]` (0 the privileged group).
 * @param options - The adversary's weight, constraint, steps, step size and key.
 * @returns The predictor, the adversary and the losses over training.
 * @concept algorithmic-fairness
 * @example
 * ```ts
 * import { adversarialDebiasing } from 'whitebox-engine/fairness'
 * const x = [[0, 0], [1, 0], [2, 0], [3, 0], [1, 1], [2, 1], [3, 1], [4, 1]]
 * const r = adversarialDebiasing(x, [0, 0, 1, 1, 0, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1], { steps: 50 })
 * r.steps.length // => 50
 * ```
 */
export function adversarialDebiasing(
  x: readonly (readonly number[])[],
  labels: readonly number[],
  groups: readonly number[],
  options: AdversarialDebiasingOptions = {},
): AdversarialDebiasing {
  const { t: X, n, d } = design('adversarialDebiasing', x)
  if (labels.length !== n || groups.length !== n)
    throw new WhiteboxError('shape', 'adversarialDebiasing: one label and group per row')
  const eo = options.constraint === 'equalisedOdds'
  const alpha = options.adversaryWeight ?? 1
  const lr = options.learningRate ?? 0.05
  const lrA = options.adversaryLearningRate ?? 1
  const inner = options.adversarySteps ?? 5
  const total = options.steps ?? 500
  const y = column(labels.map((v) => (v === 1 ? 1 : 0)))
  const z = column(groups.map((g) => (g === 0 ? 0 : 1)))
  const yv = labels.map((v) => (v === 1 ? 1 : 0))
  const k = options.key ?? makeKey(0)
  let w: Tensor = fromStorage(
    Float64Array.from(toFlat(normal(foldIn(k, 0), [d])), (v) => 0.01 * v),
    [d, 1],
    'float64',
  )
  // The adversary's inputs: s = sigmoid((1 + |c|) logit), and for equalised odds s y and s (1 - y); then an intercept.
  const m = eo ? 3 : 1
  let u: Tensor = fromStorage(new Float64Array(m + 2), [m + 2, 1], 'float64')
  const Y = fromStorage(Float64Array.from(yv), [n, 1], 'float64')
  const adversaryLogits = (weights: Value, adv: Value): Value => {
    const logits = matmul(X, weights)
    const c = sliceAxis(adv, 0, m + 1, m + 2)
    const s = sigmoid(mul(add(1, abs(c)), logits))
    const inputs = eo ? [s, mul(s, Y), mul(s, sub(1, Y))] : [s]
    const ws = sliceAxis(adv, 0, 0, m)
    let out: Value = mul(sliceAxis(adv, 0, m, m + 1), 1)
    for (const [j, inp] of inputs.entries()) out = add(out, mul(inp, sliceAxis(ws, 0, j, j + 1)))
    return out
  }
  const predictorLoss = valueAndGrad((weights: Value) => logisticLoss(matmul(X, weights), y))
  const adversaryOnW = valueAndGrad((weights: Value) => logisticLoss(adversaryLogits(weights, u), z))
  const adversaryOnU = valueAndGrad((adv: Value) => logisticLoss(adversaryLogits(w, adv), z))
  const steps: AdversarialStep[] = []
  for (let t = 0; t < total; t++) {
    let a = adversaryOnU(u)
    for (let s = 0; s < inner; s++) {
      u = sub(u, mul(a.grad as Value, lrA)) as Tensor
      a = adversaryOnU(u)
    }
    const p = predictorLoss(w)
    const q = adversaryOnW(w)
    const gp = toFlat(p.grad as Tensor)
    const ga = toFlat(q.grad as Tensor)
    const na = ga.reduce((s, v) => s + v * v, 0)
    const dot = gp.reduce((s, v, j) => s + v * (ga[j] ?? 0), 0)
    // Zhang et al.: remove the predictor gradient's projection on the adversary's, and push against the adversary;
    // with no adversary weight, plain logistic regression.
    const project = alpha > 0 && na > 0 ? dot / na : 0
    const update = gp.map((v, j) => v - project * (ga[j] ?? 0) - alpha * (ga[j] ?? 0))
    w = fromStorage(
      Float64Array.from(toFlat(w), (v, j) => v - lr * (update[j] ?? 0)),
      [d, 1],
      'float64',
    )
    steps.push({
      step: t + 1,
      predictorLoss: toFlat(p.value as Tensor)[0] ?? Number.NaN,
      adversaryLoss: toFlat(a.value as Tensor)[0] ?? Number.NaN,
    })
  }
  const wf = toFlat(w)
  return {
    model: { intercept: wf[0] ?? 0, weights: Array.from({ length: d - 1 }, (_, j) => wf[j + 1] ?? 0) },
    adversary: Array.from(toFlat(u)),
    steps,
  }
}

/** Options of {@link fairConstrainedLogistic}. */
export interface FairConstrainedOptions {
  /** The bound on the covariance between the group and the decision boundary's signed distance (default 0). */
  readonly covariance?: number
  /** The L2 penalty on the weights (default 1e-3). */
  readonly l2?: number
  readonly maxIterations?: number
}

/** A fairness-constrained logistic regression and its decision-boundary covariance with the group. */
export interface FairConstrainedLogistic {
  readonly model: LinearClassifier
  /** cov(group, intercept + w.x) at the solution. */
  readonly covariance: number
  /** The same for the unconstrained fit. */
  readonly unconstrainedCovariance: number
  readonly loss: number
}

/**
 * Fairness-constrained logistic regression (Zafar, Valera, Gomez-Rodriguez and Gummadi, 2017): minimise the log loss
 * subject to |cov(z, d(x))| <= c, where z is the (binary) group and d(x) = intercept + w.x the signed distance to the
 * decision boundary, a convex proxy for disparate impact; solved by SLSQP. c = 0 asks for a boundary uncorrelated with
 * the group.
 *
 * @param x - Features `[n, d]` (without the protected attribute).
 * @param labels - Binary labels `[n]`.
 * @param groups - Group indices `[n]` (0 the privileged group; others pooled).
 * @param options - The covariance bound, L2 penalty and iteration cap.
 * @returns The model and its covariance with the group, constrained and not.
 * @concept algorithmic-fairness
 * @example
 * ```ts
 * import { fairConstrainedLogistic } from 'whitebox-engine/fairness'
 * const x = [[0], [1], [2], [3], [1], [2], [3], [4]]
 * const r = fairConstrainedLogistic(x, [0, 0, 1, 1, 0, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1])
 * Math.abs(r.covariance) < 1e-6 // => true
 * ```
 */
export function fairConstrainedLogistic(
  x: readonly (readonly number[])[],
  labels: readonly number[],
  groups: readonly number[],
  options: FairConstrainedOptions = {},
): FairConstrainedLogistic {
  const { t: X, n, d } = design('fairConstrainedLogistic', x)
  if (labels.length !== n || groups.length !== n)
    throw new WhiteboxError('shape', 'fairConstrainedLogistic: one label and group per row')
  const y = column(labels.map((v) => (v === 1 ? 1 : 0)))
  const zs = groups.map((g) => (g === 0 ? 0 : 1))
  const zbar = zs.reduce((s: number, v) => s + v, 0) / n
  const zc = fromStorage(
    Float64Array.from(zs, (v) => (v - zbar) / n),
    [1, n],
    'float64',
  )
  const l2 = options.l2 ?? 1e-3
  const c = options.covariance ?? 0
  const loss = (p: Value): Value => add(logisticLoss(matmul(X, reshape(p, [d, 1])), y), mul(sum(square(p)), l2 / 2))
  const cov = (p: Value): Value => reshape(matmul(zc, matmul(X, reshape(p, [d, 1]))), [1])
  const x0 = fromStorage(new Float64Array(d), [d], 'float64')
  const free = lbfgs(loss, x0, { gradTol: 1e-9, maxIterations: options.maxIterations ?? 500 })
  const r = slsqp(loss, free.x as Tensor, {
    constraints: [
      { type: 'ineq', fun: (p: Value) => sub(c, cov(p)) },
      { type: 'ineq', fun: (p: Value) => add(c, cov(p)) },
    ],
    maxIterations: options.maxIterations ?? 500,
  })
  const w = toFlat(r.x as Tensor)
  return {
    model: { intercept: w[0] ?? 0, weights: Array.from({ length: d - 1 }, (_, j) => w[j + 1] ?? 0) },
    covariance: toFlat(cov(r.x as Tensor) as Tensor)[0] ?? Number.NaN,
    unconstrainedCovariance: toFlat(cov(free.x as Tensor) as Tensor)[0] ?? Number.NaN,
    loss: r.fun,
  }
}

/** Options of {@link metaFairClassifier}. */
export interface MetaFairOptions {
  /** The smallest allowed ratio of the lowest to the highest group selection rate (default 0.8, AIF360's tau). */
  readonly tau?: number
  /** Candidate lowest selection rates searched (default 201). */
  readonly grid?: number
}

/** The meta fair classifier's per-group thresholds on the posterior. */
export interface MetaFair {
  /** Each group's threshold: a row is selected when its posterior is at least it. */
  readonly thresholds: readonly number[]
  readonly selectionRates: readonly number[]
  /** The selection-rate ratio reached (min over max). */
  readonly ratio: number
  /** The expected accuracy under the posteriors. */
  readonly expectedAccuracy: number
}

/**
 * The meta fair classifier for the statistical rate (Celis, Huang, Keswani and Vishnoi, 2019; AIF360's
 * `MetaFairClassifier` with the statistical-rate metric): from estimated posteriors P(y = 1 | x, group), the classifier
 * maximising expected accuracy subject to the smallest group selection rate being at least `tau` times the largest.
 * Celis et al. show the optimum thresholds the posterior per group; this searches the band of selection rates
 * [r, r / tau] on a grid of r, giving each group the selection nearest its own optimum within the band (each group's
 * expected accuracy is unimodal in how many of its highest posteriors it selects).
 *
 * @param posteriors - P(y = 1 | x) for each row `[n]` (from any calibrated classifier).
 * @param groups - Group indices `[n]`.
 * @param options - tau and the grid size.
 * @returns Each group's threshold and selection rate, the ratio and the expected accuracy.
 * @concept algorithmic-fairness
 * @example
 * ```ts
 * import { metaFairClassifier } from 'whitebox-engine/fairness'
 * const r = metaFairClassifier([0.9, 0.8, 0.7, 0.3, 0.6, 0.4, 0.2, 0.1], [0, 0, 0, 0, 1, 1, 1, 1], { tau: 1 })
 * r.selectionRates // => [0.5, 0.5]
 * ```
 */
export function metaFairClassifier(
  posteriors: readonly number[],
  groups: readonly number[],
  options: MetaFairOptions = {},
): MetaFair {
  if (posteriors.length !== groups.length) throw new WhiteboxError('shape', 'metaFairClassifier: one group per row')
  const tau = options.tau ?? 0.8
  if (!(tau > 0 && tau <= 1)) throw new WhiteboxError('value', 'metaFairClassifier: tau is in (0, 1]')
  const k = Math.max(...groups) + 1
  const n = posteriors.length
  // Each group's posteriors, highest first, and the expected accuracy gain of selecting its top j: sum of (2 eta - 1).
  const sorted = Array.from({ length: k }, (_, g) => posteriors.filter((_, i) => groups[i] === g).sort((a, b) => b - a))
  const gains = sorted.map((s) => {
    const out = [0]
    for (const p of s) out.push((out[out.length - 1] ?? 0) + 2 * p - 1)
    return out
  })
  const best = gains.map((gain) => gain.indexOf(Math.max(...gain)))
  const baseline = posteriors.reduce((s, p) => s + (1 - p), 0)
  const steps = options.grid ?? 201
  let choice: number[] = best
  let value = Number.NEGATIVE_INFINITY
  for (let a = 0; a < steps; a++) {
    const r = a / (steps - 1)
    const hi = Math.min(1, r / tau)
    const counts = sorted.map((s, g) => {
      const size = s.length
      if (size === 0) return 0
      const lo = Math.ceil(r * size - 1e-9)
      const top = Math.floor(hi * size + 1e-9)
      if (lo > top) return -1
      return Math.min(top, Math.max(lo, best[g] ?? 0))
    })
    if (counts.some((c) => c < 0)) continue
    const v = counts.reduce((s, c, g) => s + (gains[g]?.[c] ?? 0), 0)
    if (v > value + 1e-12) {
      value = v
      choice = counts
    }
  }
  const rates = choice.map((c, g) => c / Math.max(1, sorted[g]?.length ?? 0))
  return {
    thresholds: choice.map((c, g) => (c === 0 ? Number.POSITIVE_INFINITY : (sorted[g]?.[c - 1] ?? 0))),
    selectionRates: rates,
    ratio: Math.max(...rates) === 0 ? 1 : Math.min(...rates) / Math.max(...rates),
    expectedAccuracy: (baseline + choice.reduce((s, c, g) => s + (gains[g]?.[c] ?? 0), 0)) / n,
  }
}

/** Options of {@link gerryFairClassifier}. */
export interface GerryFairOptions {
  /** The error rate kept fair over rich subgroups (default false positives). */
  readonly rate?: 'falsePositive' | 'falseNegative'
  /** The Lagrangian bound on the auditor's multipliers (default 10, AIF360's C). */
  readonly C?: number
  /** The unfairness tolerated, below which the auditor's group costs nothing (default 0.01). */
  readonly gamma?: number
  /** Rounds of fictitious play (default 10, AIF360's max_iters). */
  readonly iterations?: number
}

/** One round of {@link gerryFairClassifier}. */
export interface GerryFairStep {
  readonly iteration: number
  /** The error of the average classifier so far. */
  readonly error: number
  /** The weighted disparity of the worst rich subgroup the auditor found against it. */
  readonly violation: number
}

/** The randomised classifier of GerryFair: an average of linear cost-sensitive classifiers. */
export interface GerryFair {
  /** Each round's classifier: predict 1 where `cost1 . [1, x] < cost0 . [1, x]`. */
  readonly classifiers: readonly { readonly cost0: readonly number[]; readonly cost1: readonly number[] }[]
  /** P(positive decision) for each training row (the average of the classifiers). */
  readonly decisions: readonly number[]
  readonly steps: readonly GerryFairStep[]
}

/**
 * GerryFair (Kearns, Neel, Roth and Wu, 2018; AIF360's `GerryFairClassifier`): fictitious play between a learner, who
 * best-responds to the current costs by cost-sensitive classification (two linear regressions of the costs of predicting
 * 0 and 1 on the features), and an auditor, who finds the rich subgroup (a linear threshold of the protected features)
 * with the largest weighted false-positive (or false-negative) disparity against the learner's average classifier and
 * charges its members. The result is the average of the learner's classifiers, fair to every linearly defined subgroup
 * up to `gamma`. Unlike AIF360's `update_costs`, which charges every row the first group member's indicator (its index
 * is reset inside the loop), each row is charged by its own membership, as in the paper.
 *
 * @param x - All features `[n, d]`.
 * @param protectedFeatures - The protected features `[n, p]` the subgroups are defined on.
 * @param labels - Binary labels `[n]`.
 * @param options - The rate, C, gamma and rounds.
 * @returns The classifiers, the average decisions and the history.
 * @concept algorithmic-fairness
 * @example
 * ```ts
 * import { gerryFairClassifier } from 'whitebox-engine/fairness'
 * const x = [[0, 0], [1, 0], [2, 0], [3, 0], [0, 1], [1, 1], [2, 1], [3, 1]]
 * const r = gerryFairClassifier(x, x.map((r) => [r[1] ?? 0]), [0, 0, 1, 1, 0, 1, 1, 1], { iterations: 4 })
 * r.steps.length // => 3
 * ```
 */
export function gerryFairClassifier(
  x: readonly (readonly number[])[],
  protectedFeatures: readonly (readonly number[])[],
  labels: readonly number[],
  options: GerryFairOptions = {},
): GerryFair {
  const n = x.length
  if (protectedFeatures.length !== n || labels.length !== n)
    throw new WhiteboxError('shape', 'gerryFairClassifier: one protected row and label per row')
  const rate = options.rate ?? 'falsePositive'
  const C = options.C ?? 10
  const gamma = options.gamma ?? 0.01
  const iterations = options.iterations ?? 10
  const y = labels.map((v) => (v === 1 ? 1 : 0))
  const fp = rate === 'falsePositive'
  // The audited class: negatives for false positives, positives for false negatives.
  const audited = y.map((v) => (fp ? v === 0 : v === 1))
  const designRows = x.map((row) => [1, ...row])
  const fit = (target: readonly number[]): number[] =>
    toFlat(
      lstsq(
        designRows,
        target.map((v) => [v]),
      ) as never,
    ) as number[]
  const score = (beta: readonly number[], row: readonly number[]): number =>
    row.reduce((s, v, j) => s + v * (beta[j] ?? 0), 0)
  // AIF360's initial costs: predicting the wrong class costs 1/n.
  let c0 = y.map((v) => (fp ? 0 : (2 * v - 1) / n))
  let c1 = y.map((v) => (fp ? -(2 * v - 1) / n : 0))
  const classifiers: { cost0: number[]; cost1: number[] }[] = []
  let predictions = new Array<number>(n).fill(0)
  const steps: GerryFairStep[] = []
  for (let t = 1; t < iterations; t++) {
    const cost0 = fit(c0)
    const cost1 = fit(c1)
    classifiers.push({ cost0, cost1 })
    const h = designRows.map((row) => (score(cost1, row) < score(cost0, row) ? 1 : 0))
    predictions = predictions.map((p, i) => ((t - 1) / t) * p + (h[i] ?? 0) / t)
    const error = predictions.reduce((s, p, i) => s + Math.abs(p - (y[i] ?? 0)), 0) / n
    const group = richSubgroupAudit(protectedFeatures, y, predictions, { rate })
    steps.push({ iteration: t, error, violation: group.weightedDisparity })
    // Charge the audited rows by their membership of the subgroup, against its weight among them.
    const m = audited.filter(Boolean).length
    const inside = protectedFeatures.filter((_, i) => audited[i] === true && group.members[i] === 1).length
    const weight = m === 0 ? 0 : inside / m
    const charge = group.weightedDisparity < gamma ? 0 : 1
    const next = y.map((_, i) => {
      if (audited[i] !== true) return -1 / n
      const extra = charge * (1 / n) * group.sign * C * (1 / t) * (weight - (group.members[i] ?? 0))
      const old = (fp ? c1[i] : c0[i]) ?? 0
      return (old - 1 / n) * ((t - 1) / t) + extra + 1 / n
    })
    if (fp) c1 = next
    else c0 = next
  }
  return { classifiers, decisions: predictions, steps }
}
