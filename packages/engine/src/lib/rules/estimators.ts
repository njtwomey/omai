/**
 * The paths of CN2 and PRISM: the channels its algorithm reports, renamed to paths by its contract, and the derived `model/rules` every rule learner has.
 *
 * @example
 * ```ts
 * import type { CoveringPath } from 'mdml-engine/rules'
 * const path: CoveringPath = 'rule/literal'
 * path // => 'rule/literal'
 * ```
 */
export type CoveringPath =
  | 'rule/literal'
  | 'rule/coverage'
  | 'rule/precision'
  | 'ruleset/rules'
  | 'ruleset/accuracy'
  | 'ruleset/covered'
  | 'model/rules'

/**
 * The rule learners as classifiers on the iterative base: `cn2Classifier`, `prismClassifier`, `ripperClassifier`,
 * `corelsClassifier`, `bayesianRuleListClassifier`, `oneRClassifier` and `decisionTableClassifier`. A fit is a session
 * of the learner's algorithm (one rule, one prefix, one proposal or one feature per step), so `.method(data)` and
 * `.session(data)` exist beside `.fit` and `fittedAt(t)` is the rule model as it stood at step t. The fitted value
 * predicts, gives Laplace-smoothed class probabilities from the rules' class counts, and carries the model (rules,
 * default, text, the rule-set record).
 */

import type { Json } from '../base/index.ts'
import type { Exposition } from '../exposition/index.ts'
import { type IterativeEstimator, iterative } from '../learn/bases/iterative.ts'
import type { FittedFacts, LearnData, Verbs } from '../learn/estimator.ts'
import { classNamesOf, featureNamesOf } from '../learn/explained.ts'
import { matrixView } from '../learn/validation.ts'
import { accuracyScore } from '../metrics/index.ts'
import { choice, dict, int, real } from '../space/index.ts'
import { fromStorage, type Tensor, type TensorLike } from '../tensor/index.ts'
import { type Contract, defineContract, derived, Log, message, scalar } from '../trace/index.ts'
import { type CoveringOptions, type CoveringState, coveringAlgorithm, coveringModel } from './covering.ts'
import {
  type BayesianRuleListOptions,
  type BayesianRuleListState,
  bayesianRuleListAlgorithm,
  bayesianRuleListModel,
  type CorelsOptions,
  type CorelsState,
  corelsAlgorithm,
  corelsModel,
} from './lists.ts'
import { type RipperOptions, type RipperState, ripperAlgorithm, ripperModel } from './ripper.ts'
import { argmaxOf, modelRecord, modelText, predictCounts, type RuleModel, smoothedProbabilities } from './rule.ts'
import {
  type DecisionTableOptions,
  type DecisionTableState,
  decisionTableAlgorithm,
  decisionTableModel,
  type OneROptions,
  type OneRState,
  oneRAlgorithm,
  oneRModel,
} from './sets.ts'

/** The verbs and facts of a fitted rule learner. */
export interface RuleVerbs extends Verbs, FittedFacts {
  readonly predict: (x: TensorLike) => Tensor
  /** Laplace-smoothed class probabilities from the class counts of the rule that decides (or the rules that fire). */
  readonly predictProba: (x: TensorLike) => Tensor
  readonly score: (data: LearnData) => number
  /** The fitted model. */
  readonly model: RuleModel
  /** The rules as if-then lines, the default last. */
  readonly text: (features?: readonly string[], classNames?: readonly string[]) => string
  /** The rules as a record for the views. */
  readonly record: (features: readonly string[], classNames: readonly string[], method?: string) => Exposition
  /** The index of the rule that decides each row, -1 for the default rule. */
  readonly apply: (x: TensorLike) => Int32Array
}

/** A rule learner's estimator type. */
export type RuleEstimator<H extends object, S, K extends string = string> = IterativeEstimator<
  H,
  S,
  RuleModel,
  RuleVerbs,
  K
>

function ruleVerbs(model: RuleModel, method: string): RuleVerbs {
  const k = model.classes.length
  const d = model.nFeatures
  const counts = (x: TensorLike): { n: number; c: Float64Array } => {
    const m = matrixView(x, method, { nFeatures: d })
    return { n: m.n, c: predictCounts(model, m.data) }
  }
  const proba = (x: TensorLike): Tensor => {
    const { n, c } = counts(x)
    const out = new Float64Array(n * k)
    for (let i = 0; i < n; i++) out.set(smoothedProbabilities(c.subarray(i * k, (i + 1) * k)), i * k)
    return fromStorage(out, [n, k], 'float64')
  }
  const predict = (x: TensorLike): Tensor => {
    const { n, c } = counts(x)
    const out = new Float64Array(n)
    for (let i = 0; i < n; i++) out[i] = model.classes[argmaxOf(c.subarray(i * k, (i + 1) * k))] ?? Number.NaN
    return fromStorage(out, [n], 'float64')
  }
  return {
    classes: model.classes,
    nFeaturesIn: d,
    model,
    predict,
    predictProba: proba,
    score: (data) => accuracyScore(data.y ?? [], predict(data.x)).estimate.value,
    text: (features, classNames) => modelText(model, features, classNames),
    record: (features, classNames, name = method) => modelRecord(model, features, classNames, name),
    apply: (x) => {
      const m = matrixView(x, method, { nFeatures: d })
      return Int32Array.from(firing(model, m.data))
    },
  }
}

import { firingRule as firing } from './rule.ts'

/** The standard probes of a rule learner: the model as a record at any step. */
function ruleContract<S, K extends string>(
  learn: LearnData,
  modelOf: (s: S) => RuleModel,
  aliases: Contract<S, K>,
  method: string,
): Contract<S, K | 'model/rules'> {
  const d = matrixView(learn.x, method).d
  const features = featureNamesOf(learn, d)
  return defineContract({
    ...aliases,
    'model/rules': derived<S>({
      description: 'the rule model so far: rules with precision and coverage, the default last',
      compute: (s) => {
        const model = modelOf(s)
        return modelRecord(model, features, classNamesOf(learn, model.classes), method) as unknown as Json
      },
    }),
  })
}

// ------------------------------------------------------------------------------------------------ CN2 and PRISM

/** Hyperparameters of the covering classifiers. */
export interface CoveringHyper
  extends Required<
    Pick<
      CoveringOptions,
      'mode' | 'heuristic' | 'beamWidth' | 'maxLength' | 'alpha' | 'minCovered' | 'maxRules' | 'm' | 'maxThresholds'
    >
  > {
  readonly nominal: readonly number[]
}

function coveringEstimator(
  name: string,
  defaults: CoveringHyper,
): RuleEstimator<CoveringHyper, CoveringState, CoveringPath> {
  return iterative<CoveringHyper, CoveringState, RuleModel, RuleVerbs, CoveringPath>({
    name,
    kind: 'classifier',
    defaults,
    hyperSpace: (h) =>
      dict({
        beamWidth: int(1, 20, { default: h.beamWidth, label: 'beam width' }),
        maxLength: int(1, 12, { default: h.maxLength, label: 'longest rule' }),
        alpha: real(0.001, 1, { default: h.alpha, label: 'significance level' }),
        heuristic: choice(['entropy', 'laplace', 'accuracy', 'm-estimate', 'wracc'] as const, {
          default: h.heuristic,
          label: 'rule quality',
        }),
      }),
    algorithm: (learn, h) => coveringAlgorithm(learn, h),
    params: (state, learn, h) => coveringModel(learn, state, h),
    verbs: (model) => ruleVerbs(model, name),
    maxSteps: (h) => h.maxRules + 50,
    contract: (learn, h) =>
      ruleContract<CoveringState, Exclude<CoveringPath, 'model/rules'>>(
        learn,
        (s) => coveringModel(learn, s, h),
        {
          'rule/literal': message({
            from: 'rule',
            description: 'the rule just found: its literals in the order added and the rows in play after each',
          }),
          'rule/coverage': scalar({ from: 'coverage', description: 'rows the last rule covers' }),
          'rule/precision': scalar({
            from: 'precision',
            description: 'the last rule’s share of its class among the rows it covers',
          }),
          'ruleset/rules': scalar({ from: 'rules', description: 'rules so far' }),
          'ruleset/accuracy': scalar({
            from: 'accuracy',
            description: 'training accuracy of the rules so far',
            direction: 'higher',
          }),
          'ruleset/covered': scalar({ from: 'covered', description: 'training rows covered by some rule' }),
        },
        name,
      ),
    logs: { 'rule/literal': Log(), 'rule/coverage': Log(), 'ruleset/rules': Log(), 'ruleset/accuracy': Log() },
  })
}

function coveringDefaults(mode: CoveringOptions['mode'], hyper: Partial<CoveringHyper>): CoveringHyper {
  const m = mode ?? 'ordered'
  return {
    mode: m,
    heuristic: hyper.heuristic ?? (m === 'ordered' ? 'entropy' : 'laplace'),
    beamWidth: hyper.beamWidth ?? 5,
    maxLength: hyper.maxLength ?? 6,
    alpha: hyper.alpha ?? 0.05,
    minCovered: hyper.minCovered ?? 2,
    maxRules: hyper.maxRules ?? 100,
    m: hyper.m ?? 2,
    maxThresholds: hyper.maxThresholds ?? 24,
    nominal: hyper.nominal ?? [],
  }
}

/**
 * CN2 (Clark and Niblett 1989; Clark and Boswell 1991), ordered (a decision list, the default) or unordered (rules per
 * class that vote), by sequential covering: one rule per step, found by a beam search over conjunctions scored by
 * entropy or Laplace accuracy and kept if a likelihood-ratio test calls it significant. The session logs `rule/literal`,
 * `rule/coverage`, `rule/precision`, `ruleset/rules`, `ruleset/accuracy` and `ruleset/covered`, and answers the probe
 * `model/rules`.
 *
 * @param hyper - Ordered or unordered (`mode`), heuristic, beam width, rule length, significance level, limits and the
 *   nominal features.
 * @returns The estimator, with `method(data)` and `session(data)` beside `fit`.
 * @example
 * ```ts
 * import { cn2Classifier } from 'mdml-engine/rules'
 * import { toFlat } from 'mdml-engine/tensor'
 * const data = { x: [[0], [1], [2], [3], [4], [5]], y: [0, 0, 0, 1, 1, 1] }
 * const fitted = cn2Classifier({ alpha: 1, minCovered: 1 }).fit(data)
 * toFlat(fitted.predict([[0.5], [4.5]])) // => [0, 1]
 * ```
 */
export function cn2Classifier(
  hyper: Partial<CoveringHyper> = {},
): RuleEstimator<CoveringHyper, CoveringState, CoveringPath> {
  const h = coveringDefaults(hyper.mode === 'unordered' ? 'unordered' : 'ordered', hyper)
  return coveringEstimator(h.mode === 'unordered' ? 'cn2-unordered-classifier' : 'cn2-classifier', h)
}

/**
 * PRISM (Cendrowska 1987): for each class, rules grown greedily on precision until pure, covered rows of the class
 * removed, repeated until the class is covered; the rules vote by their class counts. One rule per step.
 *
 * @param hyper - Rule length, minimum coverage, limits and the nominal features.
 * @returns The estimator.
 * @example
 * ```ts
 * import { prismClassifier } from 'mdml-engine/rules'
 * import { toFlat } from 'mdml-engine/tensor'
 * const data = { x: [[0], [1], [2], [3], [4], [5]], y: [0, 0, 0, 1, 1, 1] }
 * toFlat(prismClassifier().fit(data).predict([[0.5], [4.5]])) // => [0, 1]
 * ```
 */
export function prismClassifier(
  hyper: Partial<CoveringHyper> = {},
): RuleEstimator<CoveringHyper, CoveringState, CoveringPath> {
  return coveringEstimator(
    'prism-classifier',
    coveringDefaults('prism', { ...hyper, minCovered: hyper.minCovered ?? 1 }),
  )
}

// ------------------------------------------------------------------------------------------------ RIPPER

/** Hyperparameters of RIPPER. */
export interface RipperHyper extends Required<Omit<RipperOptions, 'nominal'>> {
  readonly nominal: readonly number[]
}

/**
 * The paths of RIPPER and IREP: the channels its algorithm reports, renamed to paths by its contract, and the derived `model/rules` every rule learner has.
 *
 * @example
 * ```ts
 * import type { RipperPath } from 'mdml-engine/rules'
 * const path: RipperPath = 'pass/revision'
 * path // => 'pass/revision'
 * ```
 */
export type RipperPath = 'rule/descriptionLength' | 'pass/revision' | 'ruleset/size' | 'model/rules'

/**
 * RIPPER (Cohen 1995) or IREP (`mode: 'irep'`): rules per class from the rarest up, each grown by FOIL gain on two thirds
 * of the rows and pruned on the rest, stopped by the 50% error rule and the description length (MDL) rule, then
 * optimised by replacement and revision. The step adds or revisits one rule; the session logs `rule/descriptionLength`,
 * `pass/revision`, `ruleset/size` and `ruleset/accuracy`.
 *
 * @param hyper - Mode, optimisation passes, grow fraction, rule length, slack bits and the nominal features.
 * @returns The estimator.
 * @example
 * ```ts
 * import { ripperClassifier } from 'mdml-engine/rules'
 * const data = { x: [[0], [1], [2], [3], [4], [5], [6], [7]], y: [0, 0, 0, 0, 1, 1, 1, 1] }
 * ripperClassifier().fit(data, { seed: 1 }).classes // => [0, 1]
 * ```
 */
export function ripperClassifier(
  hyper: Partial<RipperHyper> = {},
): RuleEstimator<RipperHyper, RipperState, RipperPath> {
  const mode = hyper.mode ?? 'ripper'
  const defaults: RipperHyper = {
    mode,
    optimisations: hyper.optimisations ?? (mode === 'irep' ? 0 : 2),
    growFraction: hyper.growFraction ?? 2 / 3,
    maxLength: hyper.maxLength ?? 8,
    slack: hyper.slack ?? 64,
    maxThresholds: hyper.maxThresholds ?? 24,
    nominal: hyper.nominal ?? [],
  }
  const name = `${mode}-classifier`
  return iterative<RipperHyper, RipperState, RuleModel, RuleVerbs, RipperPath>({
    name,
    kind: 'classifier',
    defaults,
    hyperSpace: (h) =>
      dict({
        optimisations: int(0, 5, { default: h.optimisations, label: 'optimisation passes' }),
        slack: real(0, 200, { default: h.slack, label: 'description-length slack (bits)' }),
        maxLength: int(1, 12, { default: h.maxLength, label: 'longest rule' }),
      }),
    algorithm: (learn, h) => ripperAlgorithm(learn, h),
    params: (state, learn, h) => ripperModel(learn, state, h),
    verbs: (model) => ruleVerbs(model, name),
    maxSteps: () => 5000,
    contract: (learn, h) =>
      ruleContract<RipperState, Exclude<RipperPath, 'model/rules'>>(
        learn,
        (s) => ripperModel(learn, s, h),
        {
          'rule/descriptionLength': scalar({
            from: 'descriptionLength',
            description: 'description length of the current class’s rule set in bits',
            direction: 'lower',
          }),
          'pass/revision': message({
            from: 'step',
            description: 'what the step did: a rule grown, a stop, a revision kept, replaced or revised',
          }),
          'ruleset/size': scalar({ from: 'rules', description: 'rules so far' }),
        },
        name,
      ),
    logs: { 'rule/descriptionLength': Log(), 'pass/revision': Log(), 'ruleset/size': Log() },
  })
}

// ------------------------------------------------------------------------------------------------ rule lists

/** Hyperparameters of CORELS. */
export interface CorelsHyper extends Required<Omit<CorelsOptions, 'nominal'>> {
  readonly nominal: readonly number[]
}

/**
 * The paths of CORELS: the channels its algorithm reports, renamed to paths by its contract, and the derived `model/rules` every rule learner has.
 *
 * @example
 * ```ts
 * import type { CorelsPath } from 'mdml-engine/rules'
 * const path: CorelsPath = 'search/bound'
 * path // => 'search/bound'
 * ```
 */
export type CorelsPath = 'search/bound' | 'search/incumbent' | 'search/nodes' | 'model/rules'

/**
 * Certifiably optimal rule lists (CORELS-style branch and bound over prefixes of pre-mined antecedents). One prefix is
 * expanded per step; the session logs `search/bound`, `search/incumbent` and `search/nodes`. The fitted model's list is
 * optimal for the objective `misclassification rate + lambda * rules` once the queue is empty.
 *
 * @param hyper - The penalty `lambda`, the node budget and the antecedent mining options.
 * @returns The estimator.
 * @example
 * ```ts
 * import { corelsClassifier } from 'mdml-engine/rules'
 * import { toFlat } from 'mdml-engine/tensor'
 * const data = { x: [[0], [0], [0], [1], [1], [1]], y: [0, 0, 0, 1, 1, 1] }
 * toFlat(corelsClassifier({ minSupport: 0.3, maxCardinality: 1 }).fit(data).predict([[0], [1]])) // => [0, 1]
 * ```
 */
export function corelsClassifier(
  hyper: Partial<CorelsHyper> = {},
): RuleEstimator<CorelsHyper, CorelsState, CorelsPath> {
  const defaults: CorelsHyper = {
    lambda: hyper.lambda ?? 0.01,
    maxNodes: hyper.maxNodes ?? 200_000,
    minSupport: hyper.minSupport ?? 0.05,
    maxCardinality: hyper.maxCardinality ?? 2,
    maxThresholds: hyper.maxThresholds ?? 6,
    nominal: hyper.nominal ?? [],
  }
  return iterative<CorelsHyper, CorelsState, RuleModel, RuleVerbs, CorelsPath>({
    name: 'corels-classifier',
    kind: 'classifier',
    defaults,
    hyperSpace: (h) =>
      dict({
        lambda: real(0, 0.2, { default: h.lambda, label: 'penalty per rule' }),
        maxCardinality: int(1, 4, { default: h.maxCardinality, label: 'literals per antecedent' }),
        minSupport: real(0.001, 0.5, { default: h.minSupport, label: 'antecedent support' }),
      }),
    algorithm: (learn, h) => corelsAlgorithm(learn, h),
    params: (state, learn, h) => corelsModel(learn, state, h),
    verbs: (model) => ruleVerbs(model, 'corels-classifier'),
    maxSteps: (h) => Math.min(h.maxNodes, 20_000),
    contract: (learn, h) =>
      ruleContract<CorelsState, Exclude<CorelsPath, 'model/rules'>>(
        learn,
        (s) => corelsModel(learn, s, h),
        {
          'search/bound': scalar({
            from: 'lowerBound',
            description: 'the smallest lower bound among prefixes still to expand',
            direction: 'higher',
          }),
          'search/incumbent': scalar({
            from: 'objective',
            description: 'the objective of the best list found',
            direction: 'lower',
          }),
          'search/nodes': scalar({ from: 'nodes', description: 'prefixes evaluated' }),
        },
        'corels',
      ),
    logs: { 'search/bound': Log(), 'search/incumbent': Log(), 'search/nodes': Log() },
  })
}

/** Hyperparameters of Bayesian rule lists. */
export interface BayesianRuleListHyper extends Required<Omit<BayesianRuleListOptions, 'nominal'>> {
  readonly nominal: readonly number[]
  /** Metropolis-Hastings proposals (default 2000). */
  readonly steps: number
}

/**
 * The paths of the Bayesian rule list: the channels its algorithm reports, renamed to paths by its contract, and the derived `model/rules` every rule learner has.
 *
 * @example
 * ```ts
 * import type { BayesianRuleListPath } from 'mdml-engine/rules'
 * const path: BayesianRuleListPath = 'chain/list'
 * path // => 'chain/list'
 * ```
 */
export type BayesianRuleListPath = 'chain/logPosterior' | 'chain/accepted' | 'chain/list' | 'model/rules'

/**
 * Bayesian rule lists sampled by Metropolis-Hastings over pre-mined antecedents, one proposal per step; the fitted
 * model is the highest-posterior list visited. The session logs `chain/logPosterior` and `chain/accepted`.
 *
 * @param hyper - The prior means (`listLength`, `ruleSize`), the Dirichlet `alpha`, `steps` and the mining options.
 * @returns The estimator.
 * @example
 * ```ts
 * import { bayesianRuleListClassifier } from 'mdml-engine/rules'
 * const data = { x: [[0], [0], [0], [1], [1], [1]], y: [0, 0, 0, 1, 1, 1] }
 * bayesianRuleListClassifier({ minSupport: 0.3, maxCardinality: 1, steps: 100 }).fit(data, { seed: 1 }).classes // => [0, 1]
 * ```
 */
export function bayesianRuleListClassifier(
  hyper: Partial<BayesianRuleListHyper> = {},
): RuleEstimator<BayesianRuleListHyper, BayesianRuleListState, BayesianRuleListPath> {
  const defaults: BayesianRuleListHyper = {
    listLength: hyper.listLength ?? 3,
    ruleSize: hyper.ruleSize ?? 1,
    alpha: hyper.alpha ?? 1,
    minSupport: hyper.minSupport ?? 0.05,
    maxCardinality: hyper.maxCardinality ?? 2,
    maxThresholds: hyper.maxThresholds ?? 6,
    nominal: hyper.nominal ?? [],
    steps: hyper.steps ?? 2000,
  }
  return iterative<BayesianRuleListHyper, BayesianRuleListState, RuleModel, RuleVerbs, BayesianRuleListPath>({
    name: 'bayesian-rule-list-classifier',
    kind: 'classifier',
    defaults,
    hyperSpace: (h) =>
      dict({
        listLength: real(0.5, 10, { default: h.listLength, label: 'prior list length' }),
        ruleSize: real(0.5, 4, { default: h.ruleSize, label: 'prior rule size' }),
        alpha: real(0.1, 5, { default: h.alpha, label: 'Dirichlet prior' }),
      }),
    algorithm: (learn, h) => bayesianRuleListAlgorithm(learn, h),
    params: (state, learn, h) => bayesianRuleListModel(learn, state, h),
    verbs: (model) => ruleVerbs(model, 'bayesian-rule-list-classifier'),
    maxSteps: (h) => h.steps,
    contract: (learn, h) =>
      ruleContract<BayesianRuleListState, Exclude<BayesianRuleListPath, 'model/rules'>>(
        learn,
        (s) => bayesianRuleListModel(learn, s, h),
        {
          'chain/logPosterior': scalar({
            from: 'logPosterior',
            description: 'log posterior of the current list',
            direction: 'higher',
          }),
          'chain/accepted': scalar({ from: 'accepted', description: 'proposals accepted so far' }),
          'chain/list': derived<BayesianRuleListState>({
            description: 'the current list of antecedent indices',
            compute: (s) => [...s.list],
          }),
        },
        'bayesian-rule-list',
      ),
    logs: { 'chain/logPosterior': Log(), 'chain/accepted': Log() },
  })
}

// ------------------------------------------------------------------------------------------------ OneR and decision table

/** Hyperparameters of OneR. */
export interface OneRHyper extends Required<Omit<OneROptions, 'nominal'>> {
  readonly nominal: readonly number[]
}

/**
 * The paths of OneR: the channels its algorithm reports, renamed to paths by its contract, and the derived `model/rules` every rule learner has.
 *
 * @example
 * ```ts
 * import type { OneRPath } from 'mdml-engine/rules'
 * const path: OneRPath = 'ruleset/errors'
 * path // => 'ruleset/errors'
 * ```
 */
export type OneRPath = 'ruleset/errors' | 'ruleset/bestErrors' | 'model/rules'

/**
 * OneR (Holte 1993): the single feature whose per-value (or per-bucket) majority rules make the fewest training errors;
 * one feature is examined per step.
 *
 * @param hyper - The bucket size and the nominal features.
 * @returns The estimator.
 * @example
 * ```ts
 * import { oneRClassifier } from 'mdml-engine/rules'
 * import { toFlat } from 'mdml-engine/tensor'
 * const data = { x: [[0, 5], [0, 6], [1, 5], [1, 6]], y: [0, 0, 1, 1] }
 * toFlat(oneRClassifier({ nominal: [0, 1] }).fit(data).predict([[0, 6], [1, 5]])) // => [0, 1]
 * ```
 */
export function oneRClassifier(hyper: Partial<OneRHyper> = {}): RuleEstimator<OneRHyper, OneRState, OneRPath> {
  const defaults: OneRHyper = { minBucket: hyper.minBucket ?? 6, nominal: hyper.nominal ?? [] }
  return iterative<OneRHyper, OneRState, RuleModel, RuleVerbs, OneRPath>({
    name: 'one-r-classifier',
    kind: 'classifier',
    defaults,
    hyperSpace: (h) => dict({ minBucket: int(1, 30, { default: h.minBucket, label: 'minimum bucket' }) }),
    algorithm: (learn, h) => oneRAlgorithm(learn, h),
    params: (state, learn, h) => oneRModel(learn, state, h),
    verbs: (model) => ruleVerbs(model, 'one-r-classifier'),
    maxSteps: () => 10_000,
    contract: (learn, h) =>
      ruleContract<OneRState, Exclude<OneRPath, 'model/rules'>>(
        learn,
        (s) => oneRModel(learn, s, h),
        {
          'ruleset/errors': scalar({
            from: 'errors',
            description: 'training errors of the feature just examined',
            direction: 'lower',
          }),
          'ruleset/bestErrors': scalar({
            from: 'bestErrors',
            description: 'training errors of the best feature so far',
            direction: 'lower',
          }),
        },
        'one-r',
      ),
    logs: { 'ruleset/errors': Log(), 'ruleset/bestErrors': Log() },
  })
}

/** Hyperparameters of the decision table. */
export interface DecisionTableHyper extends Required<Omit<DecisionTableOptions, 'nominal' | 'maxFeatures'>> {
  readonly nominal: readonly number[]
  readonly maxFeatures: number
}

/**
 * The paths of the decision table: the channels its algorithm reports, renamed to paths by its contract, and the derived `model/rules` every rule learner has.
 *
 * @example
 * ```ts
 * import type { DecisionTablePath } from 'mdml-engine/rules'
 * const path: DecisionTablePath = 'features/chosen'
 * path // => 'features/chosen'
 * ```
 */
export type DecisionTablePath = 'ruleset/accuracy' | 'features/chosen' | 'model/rules'

/**
 * The decision table majority classifier (Kohavi 1995): class counts looked up by the (binned) values of the features
 * that a forward search chose by leave-one-out accuracy, one feature added per step.
 *
 * @param hyper - Bins, nominal features and the size limit.
 * @returns The estimator.
 * @example
 * ```ts
 * import { decisionTableClassifier } from 'mdml-engine/rules'
 * import { toFlat } from 'mdml-engine/tensor'
 * const data = { x: [[0, 1], [0, 0], [1, 1], [1, 0], [0, 1], [1, 0]], y: [0, 0, 1, 1, 0, 1] }
 * toFlat(decisionTableClassifier({ nominal: [0, 1] }).fit(data).predict([[0, 0], [1, 1]])) // => [0, 1]
 * ```
 */
export function decisionTableClassifier(
  hyper: Partial<DecisionTableHyper> = {},
): RuleEstimator<DecisionTableHyper, DecisionTableState, DecisionTablePath> {
  const defaults: DecisionTableHyper = {
    bins: hyper.bins ?? 5,
    nominal: hyper.nominal ?? [],
    maxFeatures: hyper.maxFeatures ?? 1000,
  }
  return iterative<DecisionTableHyper, DecisionTableState, RuleModel, RuleVerbs, DecisionTablePath>({
    name: 'decision-table-classifier',
    kind: 'classifier',
    defaults,
    hyperSpace: (h) => dict({ bins: int(2, 20, { default: h.bins, label: 'bins per numeric feature' }) }),
    algorithm: (learn, h) => decisionTableAlgorithm(learn, h),
    params: (state, learn, h) => decisionTableModel(learn, state, h),
    verbs: (model) => ruleVerbs(model, 'decision-table-classifier'),
    maxSteps: () => 1000,
    contract: (learn, h) =>
      ruleContract<DecisionTableState, Exclude<DecisionTablePath, 'model/rules'>>(
        learn,
        (s) => decisionTableModel(learn, s, h),
        {
          'ruleset/accuracy': scalar({
            from: 'accuracy',
            description: 'leave-one-out accuracy of the table',
            direction: 'higher',
          }),
          'features/chosen': derived<DecisionTableState>({
            description: 'the features in the table',
            compute: (s) => [...s.subset],
          }),
        },
        'decision-table',
      ),
    logs: { 'ruleset/accuracy': Log() },
  })
}
