/**
 * Compiling a bound model spec to a factor graph for message passing: `toFactorGraph` expands the spec's named
 * dimensions (plates) into the scalar and vector variables the engine of `omai-engine/message-passing` supports, maps
 * distribution families and operators to the engine's message families and factor operators, and reports exactly what it
 * cannot map and why. The engine has no plates of its own, so a plate is compiled one of two ways. An observed node whose
 * rows are i.i.d. draws of one regression on a weight vector (a Gaussian with one known scale, or a Bernoulli with a
 * probit or logistic link, of `X.matmul(w)` with `w` a vector latent, missing rows left out) is ONE plated factor
 * (`linear-regression-plate`, `probit-plate`, `logistic-plate`) holding the rows as its parameters and computing its
 * message over all of them in one call: the graph has no variable or factor per row, does not grow with the data, and
 * compiles and reads back in time that does not depend on the rows but for copying the design matrix once. Every other plate
 * is an expansion: N copies of the factors over it, with the data folded into the factors' constants (a hierarchy, rows
 * with their own variables or scales, scalar weights, discrete tables over a plate). `plates: 'expand'` forces the
 * expansion everywhere (the same posterior, a message per row, so finer anytime steps).
 *
 * What it maps. Latents: `normal` (a Gaussian variable per element, or one `gaussian-mv` vector variable when a
 * `matmul` reads it, so the posterior keeps its correlations), `gamma`, `exponential`, `beta`, `dirichlet`, and
 * `bernoulli` or `categorical` with known probabilities (or a Dirichlet latent for them: the conjugate `discrete` factor),
 * and a `multivariateNormal` with a known mean and covariance (or `scaleTril`): one `gaussian-mv` variable with a dense prior, whose
 * components a likelihood reads (a Gaussian process prior over the inputs of the data).
 * An `inverseGamma` latent (a variance) is a gamma precision: its site is the reciprocal of the gamma variable, and `sqrt(sigma2)` (or
 * `sqrt(sigma2 / kappa)`, a known multiple) is the precision idiom. A Gaussian mean whose variance is a multiple of the observed
 * Gaussian's (`mu ~ N(m, sigma2 / kappa)`, `y ~ N(mu, sigma2)`, nothing else reading either) is the pair of a mean and a precision,
 * the classic normal-inverse-gamma (or normal-gamma) prior: one `normal-gamma` variable and a `gaussian-normal-gamma` factor per
 * observation, so expectation propagation is exact (the marginal of the mean a Student t, of the variance an inverse gamma).
 * A Gaussian whose scale is the idiom `1 / sqrt(tau)` (or `tau ** -0.5`) of a gamma or exponential latent `tau` is the
 * `gaussian-gamma-precision` factor, for observations and latents (variational message passing, and expectation propagation when the
 * Gaussian is observed: the linear regression with an unknown noise level), and a scale `exp(s)` with `s` affine in Gaussian latents
 * is the `gaussian-log-scale` factor (expectation propagation, an observed value and a known mean: stochastic volatility, through
 * the unrolled `scan`). Values: expressions that are affine in Gaussian latents (`add`,
 * `sub`, `neg`, `mul` and `div` by a known value, `sum`, `mean`, `at` by index data, `matmul` with data) become the
 * affine operators of the engine (`plus`, `difference`, `times-constant`, `sum`, `linear`, `dot-constant`); the product of
 * two random values is the `product` factor on a new variable, and `at(components, z)` with a categorical latent `z` (or
 * `where(z, a, b)` with a Bernoulli latent) is the gate `select`, on a new variable of the components' family (expectation
 * propagation); a lookup of known values by a discrete latent (`at(means, z)`, `z * 6 - 3`) is the mean of a Gaussian observation (a
 * `table` over `z`) or of a Gaussian latent (a `gate` of one Gaussian prior per state, the gate of Minka and Winn); a summed-out
 * mixture (`normalOutliers`, `normalSpikeSlab`) with a random share (a beta latent) or random precisions (the gamma idiom) is the same
 * gate over a hidden Bernoulli switch, one per observation or element (the exact mixture site stays for known ones), and
 * `categorical` with probabilities `at(A, z)` (a known matrix `A`, a discrete latent `z`) is a `table` over the two (a Markov
 * chain), and when the rows of `A` are Dirichlet latents (a transition matrix to learn) one `gate` per step whose branch `j` is the
 * conjugate `discrete` factor on row `j`. A Gaussian latent's constraint (`ordered`, `positive`, bounds) is comparison factors with an observed true outcome
 * (a truncated Gaussian under moment matching). A Gaussian whose precision is a known multiple of a gamma latent plus a known
 * value (`lambda d_j + n_j`) is the `gaussian-gamma-precision` factor with a `scale` and an `offset`, and when a vector read by a
 * `matmul` has the precisions of one shared gamma latent (a smoothing parameter) it is one `gaussian-mv` variable with a
 * `gaussian-mv-precision-scale` factor (option `sharedScale: 'elements'` for the per-coefficient form). Likelihoods:
 * observed `normal` with a known scale (`gaussian`, or `linear-regression-row` for a regression row), `bernoulli` with
 * `probit(affine)` (`probit`), `sigmoid(affine)` or `logits` (`logistic`), `gt(affine, c)` (`greater-than`, a hard
 * comparison; expectation propagation only), or `probs` a beta latent (`bernoulli`); `binomial`, `poisson` (with an exposure) and `categorical` on a
 * beta, gamma or Dirichlet latent, an `exponential` or `gamma` value (known shape) with a gamma rate and a `geometric` or
 * `negativeBinomial` count (known successes) with a beta probability (the conjugate operators); and a Bernoulli network over scalar nodes, as `table`
 * factors (what belief propagation runs on). Expectation propagation also maps the numeric likelihoods of the registry of
 * `message-passing/operators/likelihoods.ts` (`poisson` with `exposure * exp(affine)`, `binomial` with `sigmoid` or `probit`, a
 * `negativeBinomial` in the NB2 idiom with a known or a random positive dispersion, `weibull` with `scale = exp(affine)`,
 * `studentT`, `laplace`) and, on a log-normal outcome, the exact Gaussian rule on its log; a unary function of a random value
 * (`exp`, `square`, `tanh`, `softplus`, `abs`, `sigmoid`, and `max` or `min` with a known operand as a rectifier) is the `apply`
 * factor on a new variable. A prior with no message family (`halfNormal`, `halfCauchy`, `logNormal`, `uniform`,
 * `truncatedNormal`) is the exact numeric `prior-site` on a Gaussian variable: a positive scale on its log (the value `tau` is `exp` of
 * that variable), a bounded real value on itself, so the graph holds the prior and EP matches it against the cavity every sweep. Such a scale (or an `exp` of a
 * Gaussian) is the scale of a latent normal in its non-centred form (a product with a standard normal), and the scale of an
 * observed normal through the `gaussian-scale` factor (the mean may be random too); a `clip` on a scale is read as a floor
 * that never binds. With `packScalars` the scalar normal latents an observed node reads together are one joint vector.
 * Discrete structure under Dirichlet-type weights and linear dynamics: `ops.stickBreaking(v)` of beta latents (the weights of a truncated
 * Dirichlet process) is the `stick-breaking` factor on each assignment that reads it; a categorical read through a discrete index of
 * Dirichlet rows (`categorical({ probs: phi.at(z) })`, the words of a topic model) is one `gate` on the index whose branches are the
 * conjugate `discrete` factors on the rows; `ops.pick(values, z)` (the prediction of the class `z` picks, a mixture of regressions) is a
 * lookup of the branches read by the observation as one gate of likelihoods, and a mean and a log spread `exp(at(logSd, z))` chosen by one `z`
 * are one gate of `gaussian-scale` branches; a `hurdlePoisson` count (a zero, or a zero-truncated Poisson) is a conjugate Bernoulli on its
 * beta zero probability and a `poisson-log-truncated` site on the log rate; and a multivariate normal whose mean is `ops.matvec(F, previous)`
 * of another Gaussian vector (the step of a vector-state `scan`) with a known covariance is one Gaussian vector and a dense
 * `gaussian-transition` to the previous one, its readings `ops.dot(h, state)` being regression rows on it. Anything
 * else is reported with a reason, never silently dropped.
 *
 * Ids follow `ids.ts`: a variable is its site's name, `<site>[<i>]` for an element of a site over dimensions, and
 * `<site>#<k>[<i>]` for an auxiliary variable; a factor is `factor:` and the id of the variable it defines (an observed
 * element's likelihood is `factor:y[3]`); `siteOfId` maps every id back to the spec node it belongs to, so a view can paint a
 * run's marginals on `toStructure(spec)` by site name. Results are read back by `factorGraphPosterior`, which builds
 * the library's `Posterior` keyed by site name, in the spec's shapes.
 *
 * @index function toFactorGraph: a bound ModelSpec as a message-passing factor graph, plates expanded.
 */

import { type Json, OmaiError } from '../../base/index.ts'
import {
  bernoulli,
  beta,
  categorical,
  type Distribution,
  dirichlet,
  gamma,
  inverseGamma,
  multivariateNormal,
  normal,
  studentT,
} from '../../dists/index.ts'
import {
  bernoulliEta,
  betaEta,
  type CompiledGraph,
  compileGraph,
  dirichletPrior,
  discretePrior,
  evidenceDefects,
  evidenceOf,
  type FactorGraphSpec,
  type FactorOperator,
  type FactorSpec,
  type FamilyKind,
  defaultOperators as factorOperators,
  gammaPrior,
  gaussianMvPrior,
  gaussianPrior,
  likelihoods,
  type MpState,
  marginalsOf,
  normalGammaPrior,
  pairLikelihoods,
  priorFactor,
  projectionOf,
  unaryFunctions,
  type VariableSpec,
} from '../../message-passing/index.ts'
import { asTensor, fromStorage, type NestedArray, toFlat, type Value } from '../../tensor/index.ts'
import { definePosterior, type Posterior } from '../posterior.ts'
import type { SpecAnalysis } from './analysis.ts'
import type { BoundModel } from './bind.ts'
import { distributionFamilies } from './dists/families.ts'
import { eventToDims } from './event-dims.ts'
import { auxId, elementId, factorId, siteOfId } from './ids.ts'
import { defaultOperators, type NamedValue, type OperatorTable } from './operators.ts'
import { ScanUnrollError, unrollScans } from './scan.ts'
import { type DistSpec, type ExprSpec, exprRefs, type ModelNode, nodeRefs } from './spec.ts'

/** Why a part of a model could not be put on a factor graph: the kind says what to look at. */
export type FactorIssueKind =
  /** A `custom` call: a function the graph cannot see through. */
  | 'opaque'
  /** A distribution family with no message family in the engine. */
  | 'family'
  /** An operator with no message rule on a random value, or an expression that is not affine. */
  | 'operator'
  /** A shape of model the first version does not expand (event axes, plated discrete tables). */
  | 'structure'
  /** Data that is not bound, a missing size, missing values. */
  | 'data'
  /** A constraint beyond the family's own support. */
  | 'constraint'
  /** A kind of node with no counterpart on a factor graph (`param`, a `factor` term) or one that depends on a node that failed. */
  | 'node'

/** One thing a model has that the compiler cannot map, with the node it is on and the reason in words. */
export interface FactorIssue {
  readonly node: string
  readonly kind: FactorIssueKind
  readonly reason: string
}

/** How a site of the spec is laid out in the graph, so results can be read back in the spec's shapes. */
export interface SiteLayout {
  readonly site: string
  readonly kind: 'latent' | 'observed' | 'derived'
  readonly dims: readonly string[]
  /** The sizes of `dims`. */
  readonly shape: readonly number[]
  /** The engine family of the site's variables. */
  readonly family: FamilyKind | undefined
  /**
   * `scalar`: one variable named `site`. `elements`: a variable per element, `site[i]`, in row-major order of `dims`.
   * `packed`: one vector variable named `site` over the elements. `observed`: the variables hold the data (no marginal).
   * `constant`: a derived value with no random part, so no variable.
   */
  readonly layout: 'scalar' | 'elements' | 'packed' | 'observed' | 'constant'
  /** The variable of each element in row-major order (`null` where an element has none: a constant or a missing value). */
  readonly variables: readonly (string | null)[]
  /** Whether the site is one of a posterior's sites (latents and recorded derived values). */
  readonly posterior: boolean
  /**
   * A site that is the reciprocal of the gamma variable that carries it (an inverse-gamma variance whose precision is the
   * variable): the posterior reads it as an inverse gamma.
   */
  readonly reciprocal?: boolean
  /**
   * For a site read from a normal-gamma variable shared with another site (the mean and the precision of a Gaussian, the
   * conjugate pair): which part of the joint it is, `mean` (a Student t marginal) or `precision` (a gamma marginal).
   */
  readonly part?: 'mean' | 'precision'
}

/** What a compilation to a factor graph reports. */
export interface FactorGraphCompilation {
  readonly name: string
  /** The graph, or `undefined` when something could not be mapped (see `issues`). */
  readonly graph: FactorGraphSpec | undefined
  readonly issues: readonly FactorIssue[]
  /** The nodes message passing cannot see through (`analyseSpec`'s `opaque`: a `custom` call or a family with no exponential-family form). */
  readonly opaque: readonly string[]
  readonly sites: readonly SiteLayout[]
  /** The auxiliary variables the expansion introduced (`y#0[3]`). */
  readonly auxiliary: readonly string[]
  /** The observed sites compiled as one plated factor over all their rows (no variable or factor per row), by site name. */
  readonly plated: readonly string[]
  /** What the expansion made: the plates that were unrolled and the sizes of the result. */
  readonly expanded: {
    readonly plates: readonly { readonly dim: string; readonly size: number }[]
    readonly variables: number
    readonly factors: number
  }
}

class Refusal extends Error {
  constructor(
    readonly kind: FactorIssueKind,
    reason: string,
  ) {
    super(reason)
  }
}

// ----- symbolic values: tensors of affine forms over the graph's variables -----

/** One random scalar of the graph an affine form is made of: a variable, or a coordinate of a vector variable. */
type Atom =
  | { readonly kind: 'var'; readonly name: string }
  | { readonly kind: 'comp'; readonly vector: string; readonly index: number; readonly size: number }

/** An affine form `offset + sum of coefficient times atom`; a constant has no terms. */
interface Aff {
  readonly terms: ReadonlyMap<string, number>
  readonly offset: number
}

/** An element of a symbolic tensor: an affine form of Gaussian atoms, or a variable of another family (not arithmetic). */
type Cell =
  | { readonly kind: 'aff'; readonly aff: Aff }
  | { readonly kind: 'rv'; readonly variable: string; readonly family: FamilyKind }
  // The stick-breaking weights of beta variables, `v_k prod_{j<k} (1 - v_j)` and the remainder (one more weight than breaks): the
  // probabilities of a categorical, which reads them through the `stick-breaking` factor (the truncated Dirichlet process).
  | { readonly kind: 'sticks'; readonly variables: readonly string[] }
  // A known multiple of a gamma variable plus a known offset, `scale tau + offset`: the precision of a Gaussian that shares
  // one smoothing parameter or one global scale (it has no other use: it is not a Gaussian value).
  | { readonly kind: 'scaled'; readonly variable: string; readonly scale: number; readonly offset: number }
  // A known multiple of the reciprocal of a gamma variable, `scale / tau`: an inverse-gamma variance (the variable is its
  // precision), which is mapped only as the square of the scale of a Gaussian (the precision idiom).
  | { readonly kind: 'recip'; readonly variable: string; readonly scale: number }
  // A known function of a discrete variable, one number per state (`at(means, z)`, `z * 6 - 3`): a value that is not Gaussian
  // but selects among known ones, which a likelihood or a prior reads through a table or a gate.
  // `items[k]`, when present, is a random Gaussian form that state k takes in place of the known `values[k]` (a mixture of a known
  // value and a random one: the clutter problem's `where(z, 0, theta)`).
  // `precisions[k]`, when present, is the gamma precision `scale tau + offset` that state k takes (a precision per component of a
  // mixture, read by a Gaussian observation as the gate of `gaussian-gamma-precision` branches); `values[k]` is then the known
  // precision of a state that has none.
  | {
      readonly kind: 'lookup'
      readonly variable: string
      readonly values: readonly number[]
      readonly items?: readonly (Aff | undefined)[]
      readonly precisions?: readonly (Precision | undefined)[]
    }

/** One step of an `apply` chain: a function of the registry by name, or `mul`, `add`, `pow` with its constant. */
type ApplyStep = readonly [string, number?]

/** A gamma variable's multiple plus an offset: the precision `scale tau + offset`. */
interface Precision {
  readonly variable: string
  readonly scale: number
  readonly offset: number
}

/** A tensor over named dimensions whose elements are cells, row-major in the order of `dims`. */
interface Sym {
  readonly dims: readonly string[]
  readonly cells: readonly Cell[]
}

const constAff = (offset: number): Aff => ({ terms: new Map(), offset })
const constCell = (v: number): Cell => ({ kind: 'aff', aff: constAff(v) })
const isConstCell = (c: Cell): c is Cell & { kind: 'aff' } => c.kind === 'aff' && c.aff.terms.size === 0

function combine(a: Aff, b: Aff, kb: number): Aff {
  const terms = new Map(a.terms)
  for (const [k, c] of b.terms) {
    const v = (terms.get(k) ?? 0) + kb * c
    if (v === 0) terms.delete(k)
    else terms.set(k, v)
  }
  return { terms, offset: a.offset + kb * b.offset }
}

function scaleAff(a: Aff, k: number): Aff {
  if (k === 0) return constAff(0)
  return { terms: new Map([...a.terms].map(([key, c]) => [key, c * k])), offset: a.offset * k }
}

function affOf(c: Cell): Aff {
  if (c.kind === 'lookup')
    throw new Refusal(
      'operator',
      `the value looked up by the discrete variable '${c.variable}' is not a Gaussian one: it is mapped as the mean of a Gaussian observation or prior only`,
    )
  if (c.kind === 'scaled')
    throw new Refusal(
      'operator',
      `arithmetic on the scaled precision of '${c.variable}': only a known multiple plus a known offset of a gamma variable is mapped, and only as the precision of a Gaussian`,
    )
  if (c.kind === 'recip')
    throw new Refusal(
      'operator',
      `arithmetic on the inverse-gamma variable behind '${c.variable}': only a known multiple of it is mapped, and only as the squared scale of a Gaussian (the precision idiom)`,
    )
  if (c.kind === 'sticks')
    throw new Refusal(
      'operator',
      'arithmetic on stick-breaking weights: they are only the probabilities of a categorical',
    )
  if (c.kind !== 'aff')
    throw new Refusal('operator', `arithmetic on the ${c.family} variable '${c.variable}': only Gaussian values add up`)
  return c.aff
}

/** A gamma variable or a known multiple of one plus an offset, as `scale tau + offset`; undefined for any other cell. */
function scaledOf(c: Cell): { readonly variable: string; readonly scale: number; readonly offset: number } | undefined {
  if (c.kind === 'scaled') return c
  if (c.kind === 'rv' && c.family === 'gamma') return { variable: c.variable, scale: 1, offset: 0 }
  return undefined
}

/**
 * The arithmetic of a known function of a discrete variable: a discrete or Bernoulli variable (its states as numbers) or a
 * lookup, combined with a known value or with another lookup of the same variable, state by state. Undefined when neither
 * operand is one, so the other rules apply.
 */
function lookupArithmetic(op: string, x: Cell, y: Cell, states: (variable: string) => number): Cell | undefined {
  const asLookup = (c: Cell): (Cell & { kind: 'lookup' }) | undefined => {
    if (c.kind === 'lookup') return c
    if (c.kind === 'rv' && (c.family === 'discrete' || c.family === 'bernoulli')) {
      const k = c.family === 'bernoulli' ? 2 : states(c.variable)
      return { kind: 'lookup', variable: c.variable, values: Array.from({ length: k }, (_, i) => i) }
    }
    return undefined
  }
  const a = asLookup(x)
  const b = asLookup(y)
  if (a === undefined && b === undefined) return undefined
  if ((a?.items ?? b?.items)?.some((v) => v !== undefined))
    throw new Refusal(
      'operator',
      `'${op}' of a value that is random in some states of a discrete variable has no message rule`,
    )
  const f = (u: number, v: number): number =>
    op === 'add' ? u + v : op === 'sub' ? u - v : op === 'mul' ? u * v : u / v
  if (!['add', 'sub', 'mul', 'div'].includes(op))
    throw new Refusal('operator', `'${op}' of a discrete variable has no message rule`)
  if (a !== undefined && b !== undefined) {
    if (a.variable !== b.variable)
      throw new Refusal(
        'operator',
        `'${op}' of two different discrete variables ('${a.variable}', '${b.variable}') has no message rule`,
      )
    return { kind: 'lookup', variable: a.variable, values: a.values.map((u, i) => f(u, b.values[i] as number)) }
  }
  const [look, other, flip] = a !== undefined ? [a, y, false] : [b as Cell & { kind: 'lookup' }, x, true]
  if (!isConstCell(other))
    throw new Refusal('operator', `'${op}' of a discrete variable and a random value has no message rule`)
  const k = other.aff.offset
  return { kind: 'lookup', variable: look.variable, values: look.values.map((u) => (flip ? f(k, u) : f(u, k))) }
}

/**
 * The arithmetic of an inverse-gamma variance `scale / tau`: a known multiple (`mul`, `div` by a known value) stays one; a known
 * value over it is the precision `value tau / scale`, a scaled gamma variable. Undefined when neither operand is one.
 */
function recipArithmetic(op: string, x: Cell, y: Cell): Cell | undefined {
  if (x.kind !== 'recip' && y.kind !== 'recip') return undefined
  const other = x.kind === 'recip' ? y : x
  const r = (x.kind === 'recip' ? x : y) as Cell & { kind: 'recip' }
  const known = isConstCell(other) ? other.aff.offset : undefined
  if (known === undefined || (x.kind === 'recip' && y.kind === 'recip') || !(known > 0))
    throw new Refusal(
      'operator',
      `'${op}' of an inverse-gamma variance and a random value (or a non-positive one) has no message rule: only a positive known multiple of it, as the squared scale of a Gaussian`,
    )
  if (op === 'mul') return { kind: 'recip', variable: r.variable, scale: r.scale * known }
  if (op === 'div' && x.kind === 'recip') return { kind: 'recip', variable: r.variable, scale: r.scale / known }
  if (op === 'div') return { kind: 'scaled', variable: r.variable, scale: known / r.scale, offset: 0 }
  throw new Refusal('operator', `'${op}' of an inverse-gamma variance has no message rule`)
}

/**
 * The arithmetic of a precision: a gamma variable (or a scaled one) times, plus, minus or over a known value. Undefined when
 * neither operand is a gamma-based cell, so the ordinary affine rules apply.
 */
function scaledArithmetic(op: string, x: Cell, y: Cell): Cell | undefined {
  const left = scaledOf(x)
  const right = scaledOf(y)
  if (left === undefined && right === undefined) return undefined
  const other = left === undefined ? x : y
  const known = isConstCell(other) ? other.aff.offset : undefined
  const g = (left ?? right) as { variable: string; scale: number; offset: number }
  const refuse = (): never => {
    throw new Refusal(
      'operator',
      `'${op}' of a gamma variable and a random value has no message rule: a precision is a known multiple of one gamma variable plus a known offset`,
    )
  }
  if (known === undefined || (left !== undefined && right !== undefined)) return refuse()
  switch (op) {
    case 'mul':
      return { kind: 'scaled', variable: g.variable, scale: g.scale * known, offset: g.offset * known }
    case 'div':
      if (left === undefined) return refuse()
      return { kind: 'scaled', variable: g.variable, scale: g.scale / known, offset: g.offset / known }
    case 'add':
      return { kind: 'scaled', variable: g.variable, scale: g.scale, offset: g.offset + known }
    case 'sub':
      if (left === undefined) return refuse()
      return { kind: 'scaled', variable: g.variable, scale: g.scale, offset: g.offset - known }
    default:
      return refuse()
  }
}

const prod = (xs: readonly number[]): number => xs.reduce((a, b) => a * b, 1)

/** The hidden symbol of the log of a positive latent carried on its log (not a node of the spec, so no site can clash with it). */
const logName = (site: string): string => `${site}#log`

function unflatten(flat: number, shape: readonly number[]): number[] {
  const idx: number[] = new Array<number>(shape.length).fill(0)
  let rest = flat
  for (let i = shape.length - 1; i >= 0; i--) {
    const n = shape[i] as number
    idx[i] = rest % n
    rest = Math.floor(rest / n)
  }
  return idx
}

/** A factor-graph name for the family of a distribution family, for the families the first version maps. */
const MESSAGE_FAMILY: Readonly<Record<string, FamilyKind>> = {
  normal: 'gaussian',
  normalSpikeSlab: 'gaussian',
  gamma: 'gamma',
  inverseGamma: 'gamma',
  exponential: 'gamma',
  beta: 'beta',
  dirichlet: 'dirichlet',
  bernoulli: 'bernoulli',
  categorical: 'discrete',
  multivariateNormal: 'gaussian-mv',
}

/**
 * The families of a prior with no message family that are projected onto a Gaussian once (`latentProjected`): a positive
 * scale (half-normal, half-Cauchy, log-normal, uniform or truncated normal on the positive half line) as a Gaussian on its
 * log, any other bounded or truncated value as a Gaussian on the value.
 */
const PROJECTED_PRIORS: ReadonlySet<string> = new Set([
  'halfNormal',
  'halfCauchy',
  'logNormal',
  'uniform',
  'truncatedNormal',
])

/** The factor operator a registry name maps to: the model operator's own `factor`, else its name. */
function factorName(table: OperatorTable, op: string): string {
  return table[op]?.factor ?? op
}

/** The `gaussian-gamma-precision` factor of a Gaussian `name` with a mean and the precision `scale tau + offset`. */
function precisionFactor(
  name: string,
  mean: string | { readonly const: number },
  tau: { readonly variable: string; readonly scale: number; readonly offset: number },
): FactorSpec {
  const params: Record<string, Json> = {}
  if (tau.scale !== 1) params['scale'] = tau.scale
  if (tau.offset !== 0) params['offset'] = tau.offset
  return {
    operator: 'gaussian-gamma-precision',
    args: [name, mean, tau.variable],
    ...(Object.keys(params).length === 0 ? {} : { params }),
  }
}

/** A Gaussian mean whose variance is a known multiple of the precision's reciprocal and the observations that read both: one normal-gamma variable. */
interface NormalGammaPlan {
  /** The node of the precision (a gamma or inverse-gamma latent) and of the mean (a normal latent). */
  readonly precision: string
  readonly mean: string
  /** The variable (named as the mean's node) and whether the precision's node is an inverse gamma (a variance). */
  readonly variable: string
  readonly inverse: boolean
  /** The known multiple `kappa` of the mean's precision over the shared one: `mu | lambda ~ N(m, 1 / (kappa lambda))`. */
  readonly kappa: number
  readonly observed: readonly string[]
}

/** One component of a random concentration, `weight * alpha + offset`, with the variable of `alpha` (the variable of its log when `log`). */
interface ConcentrationPart {
  readonly variable?: string
  readonly log?: boolean
  readonly weight: number
  readonly offset: number
}

class Compiler {
  readonly variables: VariableSpec[] = []
  readonly factors: FactorSpec[] = []
  readonly issues: FactorIssue[] = []
  readonly sites: SiteLayout[] = []
  readonly auxiliary: string[] = []
  readonly plated: string[] = []
  private readonly consts = new Map<string, NamedValue>()
  private readonly syms = new Map<string, Sym>()
  private readonly lazy = new Map<string, ExprSpec>()
  private readonly failed = new Set<string>()
  private readonly unbound: ReadonlySet<string>
  private readonly atoms = new Map<string, Atom>()
  private readonly auxCount = new Map<string, number>()
  private readonly packed = new Set<string>()
  /** The derived vectors `z / S(x)` of independent standard normals over a scale of one latent (non-centred), by name, with the name of `z`. */
  private readonly nonCentred = new Map<string, string>()
  /** Scalar normal latents that share a predictor and are compiled as the coordinates of one joint Gaussian vector (`packScalars`), by member name. */
  private readonly groups = new Map<string, { readonly vector: string; readonly members: readonly string[] }>()
  private readonly groupsMade = new Set<string>()
  /** The Gaussian log-scale variable behind a positive scale variable (the value variable `tau` is `exp` of it), by the value variable's name. */
  private readonly logScale = new Map<string, string>()
  private readonly components = new Map<string, string>()
  /** The apply factors made so far: the input variable and the chain of steps behind each output, so arithmetic on an output collapses into one chain. */
  private readonly applied = new Map<string, { readonly input: string; readonly steps: readonly ApplyStep[] }>()
  /** The normal-gamma pairs (a Gaussian mean and the precision its variance shares), by the name of the precision's node and of the mean's. */
  private readonly normalGammaOf = new Map<string, NormalGammaPlan>()
  private readonly normalGammaByMean = new Map<string, NormalGammaPlan>()
  /** The observed Gaussian nodes of a normal-gamma pair, with the known multiple of the precision each carries. */
  private readonly normalGammaObserved = new Map<
    string,
    { readonly plan: NormalGammaPlan; readonly multiple: number }
  >()
  private readonly names = new Set<string>()
  /** The custom functions the compiler has evaluated or read as a matrix, so a node that calls only these is not opaque. */
  readonly handledCustoms = new Set<string>()
  private readonly nodes = new Map<string, ModelNode>()
  private readonly families = distributionFamilies()
  /** The node being compiled: the site that auxiliary variables of its expressions are named after. */
  private current = ''
  /** Set while the mean and the precision of an observation chosen by one discrete variable are read: a gate keeps its components as a lookup. */
  private lookupMode = false

  constructor(
    private readonly bound: BoundModel,
    private readonly modelOps: OperatorTable,
    private readonly ops: Readonly<Record<string, FactorOperator>>,
    private readonly vectorise: boolean,
    private readonly packScalars: boolean,
    private readonly denseScales: boolean = true,
  ) {
    this.unbound = new Set(bound.unbound)
    for (const n of bound.spec.nodes) if (n.kind !== 'observed') this.nodes.set(n.name, n)
    this.findPacked()
    this.findNonCentred()
    if (this.packScalars) this.findGroups()
  }

  // ----- the plan: which vector latents stay one vector variable -----

  /** A latent over one dimension that a `matmul` reads stays a vector variable, so its posterior keeps the correlations. */
  private findPacked(): void {
    const walk = (e: ExprSpec): void => {
      if (e.kind !== 'op' && e.kind !== 'custom') return
      // A custom function that is a matrix of a normal vector (read as one by the compiler) keeps that vector joint too.
      if (e.kind === 'op' ? e.op === 'matmul' : this.bound.customs[e.name] !== undefined)
        for (const a of e.args) {
          const target = a.kind === 'ref' ? this.nodes.get(a.name) : undefined
          // A normal with a random scale (ARD: a precision of its own for each coefficient) cannot be one joint Gaussian vector:
          // its elements stay separate variables, each with its own gaussian-gamma-precision factor.
          const scale = target?.kind === 'latent' ? target.dist.params['scale'] : undefined
          const randomScale =
            scale !== undefined &&
            this.dependsOnLatent(scale) &&
            !(this.denseScales && this.sharedGamma(scale) !== undefined)
          if (
            target?.kind === 'latent' &&
            target.dims.length === 1 &&
            ((target.dist.family === 'normal' && !randomScale) || target.dist.family === 'normalSpikeSlab')
          )
            this.packed.add(target.name)
        }
      for (const a of e.args) walk(a)
    }
    for (const n of this.bound.spec.nodes) {
      if (n.kind === 'latent' || n.kind === 'observed' || n.kind === 'generated')
        for (const e of Object.values(n.dist.params)) walk(e)
      if (n.kind === 'derived' || n.kind === 'factor') walk(n.expr)
    }
  }

  /**
   * The non-centred vectors: a derived node `D = z / S` whose numerator `z` is a latent vector of independent standard normals
   * that nothing else reads, and whose divisor depends on a latent (the smoothing parameter of the non-centred penalised spline,
   * `z / sqrt(exp(x) d + n)`). Compiled as one Gaussian vector with a `gaussian-mv-function-scale` factor when the divisor is a
   * chain of unary steps from one variable (`denseNonCentred`), so the posterior keeps the correlations of the coefficients; `z` is
   * then not a variable of the graph (the factor integrates it out).
   */
  private findNonCentred(): void {
    if (!this.denseScales) return
    const readers = new Map<string, number>()
    for (const n of this.bound.spec.nodes) for (const r of nodeRefs(n)) readers.set(r, (readers.get(r) ?? 0) + 1)
    const is = (e: ExprSpec | undefined, value: number): boolean => e?.kind === 'const' && e.value === value
    for (const n of this.bound.spec.nodes) {
      if (n.kind !== 'derived' || n.expr.kind !== 'op' || n.expr.op !== 'div' || n.dims.length !== 1) continue
      const [numerator, divisor] = n.expr.args
      const z = numerator?.kind === 'ref' ? this.nodes.get(numerator.name) : undefined
      if (z?.kind !== 'latent' || z.dist.family !== 'normal' || z.dims.length !== 1 || z.dims[0] !== n.dims[0]) continue
      if (z.constraint !== undefined && z.constraint.kind !== 'real') continue
      if (!is(z.dist.params['loc'], 0) || !is(z.dist.params['scale'], 1) || readers.get(z.name) !== 1) continue
      if (divisor === undefined || !this.dependsOnLatent(divisor)) continue
      this.nonCentred.set(n.name, z.name)
    }
  }

  /**
   * With `packScalars`: the scalar normal latents (known prior) that an observed node reads together, directly or through
   * derived nodes, are grouped: each group is one joint Gaussian vector, so a likelihood on `a + b x` sees the correlation of
   * `a` and `b` instead of a loop of scalar messages (which keeps the means and loses the correlations: marginal sds too small).
   */
  private findGroups(): void {
    const staticExpr = (e: ExprSpec): boolean =>
      e.kind === 'const' ||
      (e.kind === 'ref' && this.nodes.get(e.name)?.kind === 'data') ||
      (e.kind === 'op' && e.args.every(staticExpr))
    const candidate = (n: ModelNode | undefined): n is ModelNode & { readonly kind: 'latent' } =>
      n?.kind === 'latent' &&
      n.dims.length === 0 &&
      n.dist.family === 'normal' &&
      !this.packed.has(n.name) &&
      Object.values(n.dist.params).every(staticExpr)
    const parent = new Map<string, string>()
    const find = (x: string): string => {
      let r = x
      while ((parent.get(r) ?? r) !== r) r = parent.get(r) as string
      parent.set(x, r)
      return r
    }
    for (const n of this.bound.spec.nodes) {
      if (n.kind !== 'observed') continue
      const found: string[] = []
      const seen = new Set<string>()
      const visit = (name: string): void => {
        if (seen.has(name)) return
        seen.add(name)
        const t = this.nodes.get(name)
        if (candidate(t)) found.push(name)
        else if (t?.kind === 'derived') for (const r of nodeRefs(t)) visit(r)
      }
      for (const r of nodeRefs(n)) visit(r)
      for (const f of found) {
        if (!parent.has(f)) parent.set(f, f)
        if (found[0] !== undefined) parent.set(find(f), find(found[0]))
      }
    }
    const byRoot = new Map<string, string[]>()
    for (const name of parent.keys()) byRoot.set(find(name), [...(byRoot.get(find(name)) ?? []), name])
    const order = new Map(this.bound.spec.nodes.map((n, i) => [n.name, i]))
    for (const members of byRoot.values()) {
      if (members.length < 2) continue
      members.sort((a, b) => (order.get(a) as number) - (order.get(b) as number))
      const vector = auxId(members[0] as string, 8)
      for (const m of members) this.groups.set(m, { vector, members })
    }
  }

  /**
   * The one scalar gamma (or exponential) latent that an expression depends on, through derived nodes, when it is the only
   * latent it reads: the shared smoothing parameter or ridge strength of a vector whose coefficients' precisions are
   * known multiples of it. Absent when there are several latents or the one is not a scalar gamma.
   */
  private sharedGamma(e: ExprSpec): string | undefined {
    const found = new Set<string>()
    const visit = (x: ExprSpec): void => {
      if (x.kind === 'ref') {
        const n = this.nodes.get(x.name)
        if (n?.kind === 'latent') found.add(n.name)
        else if (n?.kind === 'derived') visit(n.expr)
        return
      }
      if (x.kind === 'op' || x.kind === 'custom') for (const a of x.args) visit(a)
    }
    visit(e)
    if (found.size !== 1) return undefined
    const n = this.nodes.get([...found][0] as string)
    return n?.kind === 'latent' && n.dims.length === 0 && (n.dist.family === 'gamma' || n.dist.family === 'exponential')
      ? n.name
      : undefined
  }

  /** Whether an expression refers (through derived nodes) to a latent: a random value, where data and constants are known. */
  private dependsOnLatent(e: ExprSpec): boolean {
    if (e.kind === 'ref') {
      const n = this.nodes.get(e.name)
      return n?.kind === 'latent' || (n?.kind === 'derived' && this.dependsOnLatent(n.expr))
    }
    return (e.kind === 'op' || e.kind === 'custom') && e.args.some((a) => this.dependsOnLatent(a))
  }

  // ----- sizes and indexing -----

  private size(dim: string): number {
    const n = this.bound.sizes[dim]
    if (n === undefined)
      throw new Refusal('data', `the dimension '${dim}' has no size (bind data for it or fix its size)`)
    return n
  }

  private shapeOf(dims: readonly string[]): number[] {
    return dims.map((d) => this.size(d))
  }

  private cellAt(sym: Sym, outDims: readonly string[], idx: readonly number[]): Cell {
    let flat = 0
    for (const d of sym.dims) {
      const at = outDims.indexOf(d)
      if (at < 0)
        throw new Refusal('structure', `a value over '${d}' does not fit the dimensions [${outDims.join(', ')}]`)
      flat = flat * this.size(d) + (idx[at] as number)
    }
    return sym.cells[flat] as Cell
  }

  private joinDims(syms: readonly Sym[]): string[] {
    const dims: string[] = []
    for (const s of syms) for (const d of s.dims) if (!dims.includes(d)) dims.push(d)
    return dims
  }

  private zip(syms: readonly Sym[], f: (cells: readonly Cell[], index: number) => Cell): Sym {
    const dims = this.joinDims(syms)
    const shape = this.shapeOf(dims)
    const cells: Cell[] = []
    for (let i = 0; i < prod(shape); i++) {
      const idx = unflatten(i, shape)
      cells.push(
        f(
          syms.map((s) => this.cellAt(s, dims, idx)),
          i,
        ),
      )
    }
    return { dims, cells }
  }

  /** The cells of a symbolic value laid out over a node's dimensions (broadcasting what it lacks). */
  private over(sym: Sym, dims: readonly string[]): Cell[] {
    const shape = this.shapeOf(dims)
    const out: Cell[] = []
    for (let i = 0; i < prod(shape); i++) out.push(this.cellAt(sym, dims, unflatten(i, shape)))
    return out
  }

  // ----- values -----

  private fromNamed(nv: NamedValue): Sym {
    if (nv.value.shape.length !== nv.dims.length)
      throw new Refusal('structure', 'a value with event axes cannot stand in a scalar expression')
    return { dims: nv.dims, cells: toFlat(nv.value).map(constCell) }
  }

  private toNamed(sym: Sym): NamedValue {
    const shape = this.shapeOf(sym.dims)
    const values = sym.cells.map((c) => {
      if (!isConstCell(c)) throw new Refusal('node', 'a random value where a known one is needed')
      return c.aff.offset
    })
    return { value: fromStorage(Float64Array.from(values), shape, 'float64'), dims: sym.dims }
  }

  private isConst(e: ExprSpec): boolean {
    switch (e.kind) {
      case 'const':
        return true
      case 'ref':
        return this.consts.has(e.name)
      case 'op':
        return e.args.every((a) => this.isConst(a))
      case 'custom':
        // A custom function of known values is a known value (the Gram matrix of a kernel whose length scale is fixed).
        return this.bound.customs[e.name] !== undefined && e.args.every((a) => this.isConst(a))
    }
  }

  /** The value of an expression that has no random part, by the operator registry's own evaluation. */
  evalNamed(e: ExprSpec, override?: ReadonlyMap<string, NamedValue>): NamedValue {
    switch (e.kind) {
      case 'const':
        return { value: asTensor(e.value as NestedArray, 'float64'), dims: e.dims ?? [] }
      case 'ref': {
        const o = override?.get(e.name) ?? this.consts.get(e.name)
        if (o !== undefined) return o
        const lazy = this.lazy.get(e.name)
        if (lazy !== undefined) return this.evalNamed(lazy, override)
        if (this.unbound.has(e.name)) throw new Refusal('data', `data '${e.name}' is not bound`)
        if (this.failed.has(e.name)) throw new Refusal('node', `depends on '${e.name}', which could not be compiled`)
        throw new Refusal('node', `'${e.name}' is random where a known value is needed`)
      }
      case 'op': {
        const def = this.modelOps[e.op]
        if (def === undefined) throw new Refusal('operator', `unknown operator '${e.op}'`)
        return def.evaluate(
          e.args.map((a) => this.evalNamed(a, override)),
          e.params ?? {},
        )
      }
      case 'custom': {
        const fn = this.bound.customs[e.name]
        if (fn === undefined) throw new Refusal('opaque', `the custom function '${e.name}' was not supplied`)
        const value = fn(e.args.map((a) => this.evalNamed(a, override)))
        this.handledCustoms.add(e.name)
        return { value, dims: e.dims }
      }
    }
  }

  /**
   * A custom function of known values and one random vector that is affine in it, `f(z) = f(0) + A z`: the matrix is read off by
   * calling the function on `0` and the unit vectors (and checked at two other points, so a function that is not affine is
   * refused rather than mistaken for one). This is what the whitened Gaussian process `f = L(ell) z` is once the length scale
   * is known: the Cholesky factor is a matrix the graph can use.
   */
  private affineCustom(e: ExprSpec & { readonly kind: 'custom' }): Sym {
    const fn = this.bound.customs[e.name]
    if (fn === undefined) throw new Refusal('opaque', `the custom function '${e.name}' was not supplied`)
    const random = e.args.map((a, i) => (this.isConst(a) ? -1 : i)).filter((i) => i >= 0)
    if (random.length !== 1 || e.eventRank !== 0 || e.dims.length !== 1)
      throw new Refusal(
        'opaque',
        `the custom function '${e.name}' is opaque to message passing (it is read as a matrix only when it has one random vector argument, known others and a vector result)`,
      )
    const at = random[0] as number
    const z = this.evalSym(e.args[at] as ExprSpec)
    const n = z.cells.length
    const m = this.size(e.dims[0] as string)
    const known = e.args.map((a, i) => (i === at ? undefined : this.evalNamed(a)))
    const call = (v: readonly number[]): Float64Array => {
      const input = { value: fromStorage(Float64Array.from(v), [n], 'float64'), dims: z.dims }
      const out = toFlat(fn(known.map((k, i) => (i === at ? input : (k as NamedValue)))))
      if (out.length !== m)
        throw new Refusal('structure', `the custom function '${e.name}' returned ${out.length} values, not ${m}`)
      return Float64Array.from(out)
    }
    const f0 = call(new Array<number>(n).fill(0))
    const columns: Float64Array[] = []
    for (let j = 0; j < n; j++) {
      const unit = new Array<number>(n).fill(0)
      unit[j] = 1
      columns.push(call(unit).map((v, i) => v - (f0[i] ?? 0)))
    }
    for (const probe of [0.37, -1.21]) {
      const point = Array.from({ length: n }, (_, j) => probe * Math.cos(1 + 2.3 * j))
      const got = call(point)
      for (let i = 0; i < m; i++) {
        let want = f0[i] ?? 0
        for (let j = 0; j < n; j++) want += ((columns[j] as Float64Array)[i] ?? 0) * (point[j] ?? 0)
        if (Math.abs((got[i] ?? 0) - want) > 1e-8 * (1 + Math.abs(want)))
          throw new Refusal('opaque', `the custom function '${e.name}' is not affine in its random argument`)
      }
    }
    this.handledCustoms.add(e.name)
    const cells: Cell[] = []
    for (let i = 0; i < m; i++) {
      let acc = constAff(f0[i] ?? 0)
      for (let j = 0; j < n; j++) {
        const a = (columns[j] as Float64Array)[i] ?? 0
        if (a !== 0) acc = combine(acc, affOf(z.cells[j] as Cell), a)
      }
      cells.push({ kind: 'aff', aff: acc })
    }
    return { dims: e.dims, cells }
  }

  /** An expression as a tensor of affine forms, by the registry where it is known and symbolically where it is random. */
  evalSym(e: ExprSpec): Sym {
    if (this.isConst(e)) return this.fromNamed(this.evalNamed(e))
    switch (e.kind) {
      case 'ref': {
        const s = this.syms.get(e.name)
        if (s !== undefined) return s
        if (this.lazy.has(e.name))
          throw new Refusal(
            'structure',
            `'${e.name}' depends on a discrete variable: only a table over scalar nodes handles that`,
          )
        if (this.unbound.has(e.name)) throw new Refusal('data', `data '${e.name}' is not bound`)
        throw new Refusal('node', `depends on '${e.name}', which could not be compiled`)
      }
      case 'custom':
        return this.affineCustom(e)
      case 'const':
        return this.fromNamed(this.evalNamed(e))
      case 'op':
        if (e.op === 'matvec' || e.op === 'dot') return this.eventLinear(e)
        return this.symOp(
          e.op,
          e.args.map((a) => this.evalSym(a)),
          e.params ?? {},
        )
    }
  }

  /**
   * `matvec(A, x)` and `dot(a, x)`: a known matrix or vector applied to a vector `x` (a Gaussian vector variable, its coordinates
   * the cells of the symbol), as affine forms of its coordinates: the step `A x` of a linear dynamical system (a vector of as many
   * forms as `A` has rows), and the scalar reading `a . x`.
   */
  private eventLinear(e: ExprSpec & { readonly kind: 'op' }): Sym {
    const known = this.evalNamed(e.args[0] as ExprSpec)
    const x = this.evalSym(e.args[1] as ExprSpec)
    const values = Array.from(toFlat(known.value))
    const n = x.cells.length
    const rows = e.op === 'dot' ? 1 : values.length / n
    if (!Number.isInteger(rows) || rows < 1)
      throw new Refusal('structure', `${e.op}: the matrix has ${values.length} entries for a vector of ${n}`)
    const cells: Cell[] = []
    for (let r = 0; r < rows; r++) {
      let acc = constAff(0)
      for (let j = 0; j < n; j++) {
        const a = values[r * n + j] as number
        if (a !== 0) acc = combine(acc, affOf(x.cells[j] as Cell), a)
      }
      cells.push({ kind: 'aff', aff: acc })
    }
    return e.op === 'dot' ? { dims: [], cells } : { dims: x.dims, cells }
  }

  /** Arithmetic on a discrete variable's lookup or on a gamma variable's precision: undefined when neither operand is one. */
  private special(op: string, x: Cell, y: Cell, index?: number): Cell | undefined {
    const shifted = this.shiftedByState(op, x, y)
    if (shifted !== undefined) return shifted
    const look = lookupArithmetic(
      op,
      x,
      y,
      (v) => (this.variables.find((u) => u.name === v)?.family.size ?? 2) as number,
    )
    if (look !== undefined) return look
    const inverse = recipArithmetic(op, x, y)
    if (inverse !== undefined) return inverse
    const left = scaledOf(x)
    const right = scaledOf(y)
    if (op === 'mul' && left !== undefined && right !== undefined && left.offset === 0 && right.offset === 0) {
      if (left.variable === right.variable)
        throw new Refusal(
          'operator',
          `the square of the gamma variable '${left.variable}' as a precision has no message rule`,
        )
      // The product of two gamma variables (a local times a global precision): a gamma variable defined by `gamma-product`.
      const site = this.current
      const out = this.auxName(site, index)
      this.need('gamma-product')
      this.addVariable({ name: out, family: { kind: 'gamma' } })
      this.addFactor({ operator: 'gamma-product', args: [out, left.variable, right.variable] }, out)
      return { kind: 'scaled', variable: out, scale: left.scale * right.scale, offset: 0 }
    }
    return scaledArithmetic(op, x, y)
  }

  /**
   * A Gaussian value shifted by a known function of a discrete variable (`g + offset[c]`, `offset[c] - g`, a wrapped normal's
   * copies a period apart): a gate that selects among the shifted forms `g + offset[k]` by the variable, the `select` factor on a
   * new variable. Undefined when the operation is not an addition or subtraction of a lookup and a random Gaussian form.
   */
  private shiftedByState(op: string, x: Cell, y: Cell): Cell | undefined {
    if (op !== 'add' && op !== 'sub') return undefined
    const states = (v: string): number => (this.variables.find((u) => u.name === v)?.family.size ?? 2) as number
    const asLookup = (c: Cell): (Cell & { kind: 'lookup' }) | undefined => {
      if (c.kind === 'lookup') return c.items === undefined ? c : undefined
      if (c.kind === 'rv' && (c.family === 'discrete' || c.family === 'bernoulli'))
        return {
          kind: 'lookup',
          variable: c.variable,
          values: Array.from({ length: c.family === 'bernoulli' ? 2 : states(c.variable) }, (_, i) => i),
        }
      return undefined
    }
    const random = (c: Cell): boolean => c.kind === 'aff' && !isConstCell(c)
    const [look, other, lookFirst] =
      asLookup(x) !== undefined && random(y)
        ? [asLookup(x) as Cell & { kind: 'lookup' }, y, true]
        : asLookup(y) !== undefined && random(x)
          ? [asLookup(y) as Cell & { kind: 'lookup' }, x, false]
          : [undefined, x, false]
    if (look === undefined) return undefined
    const sign = op === 'add' ? 1 : -1
    const components = look.values.map((v): Cell => {
      const shifted = lookFirst ? combine(constAff(v), affOf(other), sign) : combine(affOf(other), constAff(v), sign)
      return { kind: 'aff', aff: shifted }
    })
    return this.gate(components, look.variable)
  }

  /** A comparison of a known function of a discrete variable with a known value: a lookup of 0 and 1 over its states. */
  private compare(op: string, x: Cell, y: Cell): Cell {
    const asLookup = (c: Cell): (Cell & { kind: 'lookup' }) | undefined =>
      lookupArithmetic(
        'add',
        c,
        constCell(0),
        (v) => (this.variables.find((u) => u.name === v)?.family.size ?? 2) as number,
      ) as (Cell & { kind: 'lookup' }) | undefined
    const a = asLookup(x)
    const b = asLookup(y)
    const f = (u: number, v: number): number => Number(op === 'gt' ? u > v : op === 'lt' ? u < v : u === v)
    if (a !== undefined && isConstCell(y))
      return { kind: 'lookup', variable: a.variable, values: a.values.map((u) => f(u, y.aff.offset)) }
    if (b !== undefined && isConstCell(x))
      return { kind: 'lookup', variable: b.variable, values: b.values.map((u) => f(x.aff.offset, u)) }
    if (isConstCell(x) && isConstCell(y)) return constCell(f(x.aff.offset, y.aff.offset))
    throw new Refusal(
      'operator',
      `'${op}' of random values has no message rule (only a discrete variable against a known value is mapped, as a gate)`,
    )
  }

  private symOp(op: string, args: readonly Sym[], params: { readonly [k: string]: Json }): Sym {
    const [a, b] = args as [Sym, Sym]
    switch (op) {
      case 'add':
        return this.zip(
          [a, b],
          ([x, y]) =>
            this.special(op, x as Cell, y as Cell) ?? {
              kind: 'aff',
              aff: combine(affOf(x as Cell), affOf(y as Cell), 1),
            },
        )
      case 'sub':
        return this.zip(
          [a, b],
          ([x, y]) =>
            this.special(op, x as Cell, y as Cell) ?? {
              kind: 'aff',
              aff: combine(affOf(x as Cell), affOf(y as Cell), -1),
            },
        )
      case 'neg':
        return {
          dims: a.dims,
          cells: a.cells.map((x): Cell => {
            const look = this.special('mul', x, constCell(-1))
            return look?.kind === 'lookup' ? look : { kind: 'aff', aff: scaleAff(affOf(x), -1) }
          }),
        }
      case 'mul':
        return this.zip(
          [a, b],
          ([x, y], i) =>
            this.special(op, x as Cell, y as Cell, i) ?? {
              kind: 'aff',
              aff: this.mulAff(affOf(x as Cell), affOf(y as Cell), i),
            },
        )
      case 'div':
        return this.zip([a, b], ([x, y], i) => {
          const scaled = this.special(op, x as Cell, y as Cell)
          if (scaled !== undefined) return scaled
          if (!isConstCell(y as Cell)) {
            // A random divisor that is a function of one latent (`z / sqrt(precision)`, the scale of a non-centred coefficient): the
            // quotient is one `apply-product` factor, the numerator times the reciprocal of the chain behind the divisor.
            const quotient =
              (y as Cell).kind === 'aff' ? this.quotientByChain(affOf(x as Cell), affOf(y as Cell), i) : undefined
            if (quotient === undefined)
              throw new Refusal('operator', 'division by a random value has no Gaussian message')
            return { kind: 'aff', aff: quotient }
          }
          const d = (y as Cell & { kind: 'aff' }).aff.offset
          return { kind: 'aff', aff: scaleAff(affOf(x as Cell), 1 / d) }
        })
      case 'sum':
      case 'mean':
        return this.reduce(a, typeof params['dim'] === 'string' ? params['dim'] : undefined, op === 'mean')
      case 'at':
        return this.gather(a, b)
      case 'where':
        return this.whereGate(args as [Sym, Sym, Sym])
      case 'gt':
      case 'lt':
      case 'eq':
        return this.zip([a, b], ([x, y]) => this.compare(op, x as Cell, y as Cell))
      case 'maximum':
      case 'minimum':
        return this.zip([a, b], ([x, y], i) => ({
          kind: 'aff',
          aff: this.extremum(op === 'maximum' ? 'max' : 'min', affOf(x as Cell), affOf(y as Cell), i),
        }))
      case 'matmul':
        return this.matmul(a, b)
      case 'stickBreaking':
        return this.stickBreaking(a)
      case 'pick':
        return this.pick(a, b)
      case 'exp':
      case 'square':
      case 'tanh':
      case 'softplus':
      case 'abs':
      case 'sigmoid':
        return { dims: a.dims, cells: a.cells.map((c, i) => ({ kind: 'aff', aff: this.applyAff(op, affOf(c), i) })) }
      case 'sqrt':
        // The square root of a Gaussian value, or of a known multiple of a gamma variable plus a known offset (a precision).
        return {
          dims: a.dims,
          cells: a.cells.map((c, i): Cell => {
            const gammaArg = scaledOf(c)
            return {
              kind: 'aff',
              aff: gammaArg === undefined ? this.applyAff(op, affOf(c), i) : this.applyGamma(op, gammaArg, i),
            }
          }),
        }
      default:
        throw new Refusal(
          'operator',
          `operator '${op}' applied to a random value has no message rule (message passing handles affine maps of Gaussians, exp, square, tanh, softplus, abs, sigmoid and the rectifier of a random value, and a link such as probit only as the likelihood of a Bernoulli observation)`,
        )
    }
  }

  /**
   * The stick-breaking weights of beta latents along the last dimension: one `sticks` cell for each index of the leading
   * dimensions, holding the beta variables of its breaks (the categorical that reads it makes the `stick-breaking` factor).
   */
  private stickBreaking(a: Sym): Sym {
    const sticks = a.dims.length === 0 ? 0 : this.size(a.dims[a.dims.length - 1] as string)
    if (sticks === 0) throw new Refusal('structure', 'stickBreaking needs a vector of breaks over a named dimension')
    const cells: Cell[] = []
    for (let i = 0; i < a.cells.length; i += sticks) {
      const breaks = a.cells.slice(i, i + sticks)
      if (breaks.some((c) => c.kind !== 'rv' || c.family !== 'beta'))
        throw new Refusal('operator', 'the breaks of a stick-breaking process must be beta latents')
      cells.push({ kind: 'sticks', variables: breaks.map((c) => (c as { readonly variable: string }).variable) })
    }
    return { dims: a.dims.slice(0, -1), cells }
  }

  /**
   * The product of two affine forms: a scaling when one is known, else the `product` factor of the engine on a variable for
   * each (made by `materialise`), whose output is a new Gaussian variable (the moment-matched product of two Gaussians:
   * expectation propagation only).
   */
  private mulAff(a: Aff, b: Aff, index: number): Aff {
    if (a.terms.size === 0) return scaleAff(b, a.offset)
    if (b.terms.size === 0) return scaleAff(a, b.offset)
    const site = this.current
    const left = this.materialise(a, site, undefined) as string
    const right = this.materialise(b, site, undefined) as string
    if (left === right)
      throw new Refusal(
        'operator',
        `a random value times itself (a square): the product factor takes two independent variables, and '${left}' is both`,
      )
    const out = this.auxName(site, index)
    this.need('product')
    this.addVariable({ name: out, family: { kind: 'gaussian' } })
    this.addFactor({ operator: 'product', args: [out, left, right] }, out)
    const key = this.atomKey({ kind: 'var', name: out })
    this.atoms.set(key, { kind: 'var', name: out })
    return { terms: new Map([[key, 1]]), offset: 0 }
  }

  /**
   * A unary function of an affine value: a known value is evaluated; a random one becomes a new Gaussian variable defined
   * by the `apply` factor of the engine (`exp`, `square`, `tanh`, `softplus`, `abs`, `sigmoid`, `relu`, `sqrt`, the reciprocal:
   * the output matched by moments under the input's cavity, expectation propagation only). When the value is a known affine
   * map of the output of an earlier `apply` (`sqrt(d exp(x) + n)`), the new factor is the whole chain from the earlier one's
   * input, not a second Gaussian approximation on top of the first.
   */
  private applyAff(fn: string, a: Aff, index: number): Aff {
    if (a.terms.size === 0) return constAff((unaryFunctions()[fn] as { f: (x: number) => number }).f(a.offset))
    return this.applyChain(this.chainOf(a), [fn], index)
  }

  /**
   * `numerator / divisor` for a divisor that is a known affine map of the output of an `apply` (a function of a latent): the
   * `apply-product` factor of the numerator and the divisor's chain with the reciprocal appended. Undefined for any other divisor.
   */
  private quotientByChain(numerator: Aff, divisor: Aff, index: number): Aff | undefined {
    const terms = [...divisor.terms].filter(([, c]) => c !== 0)
    const atom = terms.length === 1 ? (this.atoms.get(terms[0]?.[0] as string) as Atom) : undefined
    if (atom?.kind !== 'var' || !this.applied.has(atom.name)) return undefined
    const base = this.chainOf(divisor)
    const steps = [...base.steps, ['recip'] as ApplyStep]
    if (numerator.terms.size === 0) return scaleAff(this.applyChain(base, ['recip'], index), numerator.offset)
    const z = this.materialise(numerator, this.current, undefined) as string
    const out = this.auxName(this.current, index)
    this.need('apply-product')
    this.addVariable({ name: out, family: { kind: 'gaussian' } })
    this.addFactor(
      { operator: 'apply-product', args: [out, z, base.input], params: { steps: steps as unknown as Json } },
      out,
    )
    const key = this.atomKey({ kind: 'var', name: out })
    this.atoms.set(key, { kind: 'var', name: out })
    return { terms: new Map([[key, 1]]), offset: 0 }
  }

  /** A known multiple of a gamma variable plus an offset (a precision) through a unary function: an `apply` factor on the gamma variable. */
  private applyGamma(fn: string, g: Precision, index: number): Aff {
    const steps: ApplyStep[] = []
    if (g.scale !== 1) steps.push(['mul', g.scale])
    if (g.offset !== 0) steps.push(['add', g.offset])
    return this.applyChain({ input: g.variable, steps }, [fn], index)
  }

  /**
   * The variable an affine value is read from and the steps from it: the output of an earlier `apply` under a known affine map is
   * followed back to that factor's input (its steps, then the map), any other value is a variable of its own (no steps).
   */
  private chainOf(a: Aff): { readonly input: string; readonly steps: readonly ApplyStep[] } {
    const terms = [...a.terms].filter(([, c]) => c !== 0)
    const [key, coefficient] = terms[0] as [string, number]
    const atom = terms.length === 1 ? (this.atoms.get(key) as Atom) : undefined
    const earlier = atom?.kind === 'var' ? this.applied.get(atom.name) : undefined
    if (earlier === undefined) return { input: this.materialise(a, this.current, undefined) as string, steps: [] }
    const steps: ApplyStep[] = [...earlier.steps]
    if (coefficient !== 1) steps.push(['mul', coefficient])
    if (a.offset !== 0) steps.push(['add', a.offset])
    return { input: earlier.input, steps }
  }

  private applyChain(
    base: { readonly input: string; readonly steps: readonly ApplyStep[] },
    more: readonly (string | ApplyStep)[],
    index: number,
  ): Aff {
    const steps = [...base.steps, ...more.map((m): ApplyStep => (typeof m === 'string' ? [m] : m))]
    const site = this.current
    const out = this.auxName(site, index)
    this.need('apply')
    this.addVariable({ name: out, family: { kind: 'gaussian' } })
    const single = steps.length === 1 && steps[0]?.[1] === undefined ? steps[0]?.[0] : undefined
    this.addFactor(
      {
        operator: 'apply',
        args: [out, base.input],
        params: single === undefined ? { steps: steps as unknown as Json } : { fn: single },
      },
      out,
    )
    this.applied.set(out, { input: base.input, steps })
    const key = this.atomKey({ kind: 'var', name: out })
    this.atoms.set(key, { kind: 'var', name: out })
    return { terms: new Map([[key, 1]]), offset: 0 }
  }

  /** The larger or smaller of two random values: the `max` or `min` factor of the engine on a new variable (expectation propagation, Clark's moments). */
  private extremum(operator: 'max' | 'min', a: Aff, b: Aff, index: number): Aff {
    if (a.terms.size === 0 || b.terms.size === 0) {
      // max(x, k) = k + relu(x - k) and min(x, k) = x - relu(x - k): a rectifier of an affine value (the `apply` factor).
      const [k, other] = a.terms.size === 0 ? [a.offset, b] : [b.offset, a]
      const r = this.applyAff('relu', combine(other, constAff(k), -1), index)
      return operator === 'max' ? combine(r, constAff(k), 1) : combine(other, r, -1)
    }
    const site = this.current
    const left = this.materialise(a, site, undefined) as string
    const right = this.materialise(b, site, undefined) as string
    if (left === right)
      throw new Refusal('operator', `the ${operator === 'max' ? 'maximum' : 'minimum'} of a random value with itself`)
    const out = this.auxName(site, index)
    this.need(operator)
    this.addVariable({ name: out, family: { kind: 'gaussian' } })
    this.addFactor({ operator, args: [out, left, right] }, out)
    const key = this.atomKey({ kind: 'var', name: out })
    this.atoms.set(key, { kind: 'var', name: out })
    return { terms: new Map([[key, 1]]), offset: 0 }
  }

  private reduce(a: Sym, dim: string | undefined, mean: boolean): Sym {
    const outDims = dim === undefined ? [] : a.dims.filter((d) => d !== dim)
    const shape = this.shapeOf(a.dims)
    const outShape = this.shapeOf(outDims)
    const groups: Aff[] = new Array<Aff>(prod(outShape)).fill(constAff(0))
    const counts: number[] = new Array<number>(prod(outShape)).fill(0)
    for (let i = 0; i < a.cells.length; i++) {
      const idx = unflatten(i, shape)
      const out = outDims.map((d) => idx[a.dims.indexOf(d)] as number)
      let flat = 0
      outDims.forEach((_, k) => {
        flat = flat * (outShape[k] as number) + (out[k] as number)
      })
      groups[flat] = combine(groups[flat] as Aff, affOf(a.cells[i] as Cell), 1)
      counts[flat] = (counts[flat] ?? 0) + 1
    }
    return {
      dims: outDims,
      cells: groups.map((g, k) => ({ kind: 'aff', aff: mean ? scaleAff(g, 1 / (counts[k] as number)) : g })),
    }
  }

  private gather(a: Sym, idx: Sym): Sym {
    const first = a.dims[0]
    if (first === undefined) throw new Refusal('structure', 'at: the first operand needs a dimension to index')
    const rest = a.dims.slice(1)
    const dims = [...idx.dims, ...rest]
    const shape = this.shapeOf(dims)
    const restShape = this.shapeOf(rest)
    const cells: Cell[] = []
    for (let i = 0; i < prod(shape); i++) {
      const at = unflatten(i, shape)
      const code = this.cellAt(idx, dims, at)
      let restFlat = 0
      for (let k = 0; k < rest.length; k++)
        restFlat = restFlat * (restShape[k] as number) + (at[idx.dims.length + k] as number)
      cells.push(this.choose(code, this.size(first), (j) => a.cells[j * prod(restShape) + restFlat] as Cell, first))
    }
    return { dims, cells }
  }

  /**
   * The cell chosen by an index cell among `states` components: a known index picks one, an index that is a known function of a
   * discrete variable (a change point's `t > tau`) is a lookup of the components, and a categorical or Bernoulli latent is the
   * gate that selects among them.
   */
  private choose(code: Cell, states: number, component: (state: number) => Cell, dim: string, asLookup = false): Cell {
    if (isConstCell(code)) {
      const c = code.aff.offset
      if (!Number.isInteger(c) || c < 0 || c >= states)
        throw new Refusal('data', `at: index ${c} is out of range for '${dim}'`)
      return component(c)
    }
    if (code.kind === 'lookup' && code.items === undefined) {
      const comps = code.values.map((v) => {
        if (!Number.isInteger(v) || v < 0 || v >= states)
          throw new Refusal('data', `at: index ${v} is out of range for '${dim}'`)
        return component(v)
      })
      return this.lookupOf(comps, code.variable)
    }
    if (code.kind !== 'rv' || code.family !== 'discrete')
      throw new Refusal(
        'operator',
        'at: the index must be data or a categorical latent (a gate); a computed random index has no message rule',
      )
    const comps = Array.from({ length: states }, (_, k) => component(k))
    // The branches of a `pick` stay the states of one lookup (their random Gaussian forms are the branches of the gate the observation
    // is), instead of one `select` variable: the observation is a gate of likelihoods, exact in the discrete variable.
    if (asLookup && comps.every((c) => c.kind === 'aff')) return this.lookupOf(comps, code.variable)
    return this.gate(comps, code.variable)
  }

  /**
   * `pick(values, index)`: along the last dimension of `values`, the entry that `index` names, for every element of the other
   * dimensions (a different regression per class of a mixture of regressions: `values` the predictions of each class, `index` the
   * class of each row). A known index selects, a categorical latent is the gate over the branches.
   */
  private pick(values: Sym, idx: Sym): Sym {
    const component = values.dims[values.dims.length - 1]
    if (component === undefined) throw new Refusal('structure', 'pick: the values need a dimension to pick along')
    const states = this.size(component)
    const kept = values.dims.slice(0, -1)
    const dims = this.joinDims([{ dims: kept, cells: [] }, idx])
    const shape = this.shapeOf(dims)
    const cells: Cell[] = []
    for (let i = 0; i < prod(shape); i++) {
      const at = unflatten(i, shape)
      const code = this.cellAt(idx, dims, at)
      const compAt = (k: number): Cell => {
        let flat = 0
        for (const d of kept) flat = flat * this.size(d) + (at[dims.indexOf(d)] as number)
        return values.cells[flat * states + k] as Cell
      }
      cells.push(this.choose(code, states, compAt, component, true))
    }
    return { dims, cells }
  }

  /**
   * `y = c_z`: a gate that picks one of several random values by a discrete (or Bernoulli) variable, the `select` factor of
   * the engine on a new variable of the components' family. The components are Gaussian (an affine value becomes a variable)
   * or variables of one other family; a known component has no variable to select and is refused.
   */
  private gate(components: readonly Cell[], z: string): Cell {
    const site = this.current
    // Known components: the value is a known function of the discrete variable (a lookup), not a random value to select.
    if (components.every(isConstCell))
      return { kind: 'lookup', variable: z, values: components.map((c) => (c as Cell & { kind: 'aff' }).aff.offset) }
    // A mix of known and random Gaussian forms: a lookup whose random states carry their forms (a gate of Gaussian branches).
    if (components.every((c) => c.kind === 'aff') && components.some(isConstCell))
      return {
        kind: 'lookup',
        variable: z,
        values: components.map((c) => (isConstCell(c) ? c.aff.offset : 0)),
        items: components.map((c) => (isConstCell(c) ? undefined : (c as Cell & { kind: 'aff' }).aff)),
      }
    // Read by an observation whose precision is chosen by the same discrete variable: the components stay states of one lookup, so
    // the observation is one gate over its branches (random means and gamma precisions of each state), not a select per value.
    if (this.lookupMode) {
      if (components.every((c) => c.kind === 'aff'))
        return {
          kind: 'lookup',
          variable: z,
          values: components.map((c) => (isConstCell(c) ? c.aff.offset : 0)),
          items: components.map((c) => (isConstCell(c) ? undefined : (c as Cell & { kind: 'aff' }).aff)),
        }
      const gammas = components.map((c) => (isConstCell(c) ? undefined : scaledOf(c)))
      if (components.every((c, k) => isConstCell(c) || gammas[k] !== undefined))
        return {
          kind: 'lookup',
          variable: z,
          values: components.map((c) => (isConstCell(c) ? c.aff.offset : 0)),
          precisions: gammas,
        }
    }
    const first = components[0] as Cell
    const rv = first.kind === 'rv'
    const kind: FamilyKind = rv ? first.family : 'gaussian'
    const names = components.map((c) => {
      if (rv) {
        if (c.kind !== 'rv' || c.family !== kind)
          throw new Refusal('operator', `a gate selects among values of one family: they are not all ${kind}`)
        return c.variable
      }
      const v = this.materialise(affOf(c), site, undefined)
      if (typeof v === 'number')
        throw new Refusal(
          'operator',
          'a gate selects among random values: a known component has no variable (give it a prior)',
        )
      return v
    })
    const out = this.auxName(site, undefined)
    this.need('select')
    this.addVariable({ name: out, family: { kind } })
    this.addFactor({ operator: 'select', args: [out, z, ...names], params: { components: names.length } }, out)
    if (rv) return { kind: 'rv', variable: out, family: kind }
    const key = this.atomKey({ kind: 'var', name: out })
    this.atoms.set(key, { kind: 'var', name: out })
    return { kind: 'aff', aff: { terms: new Map([[key, 1]]), offset: 0 } }
  }

  /**
   * The value that is `components[k]` in state `k` of the discrete variable `z`, as a lookup: known numbers, or Gaussian forms
   * that are random in some states (read by a gate of Gaussian branches, over the variables the branches share).
   */
  private lookupOf(components: readonly Cell[], z: string): Cell {
    if (!components.every((c) => c.kind === 'aff'))
      throw new Refusal(
        'operator',
        'a value chosen by a discrete variable must be known or a Gaussian form in every state',
      )
    const values = components.map((c) => (isConstCell(c) ? c.aff.offset : 0))
    if (components.every(isConstCell)) return { kind: 'lookup', variable: z, values }
    return {
      kind: 'lookup',
      variable: z,
      values,
      items: components.map((c) => (isConstCell(c) ? undefined : (c as Cell & { kind: 'aff' }).aff)),
    }
  }

  /** `where(c, a, b)`: a known condition picks a branch; a Bernoulli latent condition is the gate `c ? a : b`. */
  private whereGate([cond, a, b]: [Sym, Sym, Sym]): Sym {
    return this.zip([cond, a, b], ([c, x, y]) => {
      if (isConstCell(c as Cell)) return (c as Cell & { kind: 'aff' }).aff.offset !== 0 ? (x as Cell) : (y as Cell)
      const gate = c as Cell
      if (gate.kind === 'lookup' && gate.items === undefined)
        return this.lookupOf(
          gate.values.map((v) => (v !== 0 ? (x as Cell) : (y as Cell))),
          gate.variable,
        )
      if (gate.kind !== 'rv' || gate.family !== 'bernoulli')
        throw new Refusal(
          'operator',
          'where: the condition must be known or a Bernoulli latent; a comparison of random values has no message rule',
        )
      return this.gate([y as Cell, x as Cell], gate.variable)
    })
  }

  private matmul(a: Sym, b: Sym): Sym {
    const inner = a.dims[a.dims.length - 1]
    if (inner === undefined || inner !== b.dims[0] || b.dims.length > 2)
      throw new Refusal('structure', `matmul: need [..., d] times [d] or [d, k], got [${a.dims}] and [${b.dims}]`)
    const left = a.dims.slice(0, -1)
    const right = b.dims.slice(1)
    const dims = [...left, ...right]
    const shape = this.shapeOf(dims)
    const d = this.size(inner)
    const cells: Cell[] = []
    for (let i = 0; i < prod(shape); i++) {
      const at = unflatten(i, shape)
      let acc = constAff(0)
      for (let j = 0; j < d; j++) {
        const x = this.cellAt({ dims: a.dims, cells: a.cells }, a.dims, [...at.slice(0, left.length), j])
        const y = this.cellAt({ dims: b.dims, cells: b.cells }, b.dims, [j, ...at.slice(left.length)])
        acc = combine(acc, this.mulAff(affOf(x), affOf(y), i), 1)
      }
      cells.push({ kind: 'aff', aff: acc })
    }
    return { dims, cells }
  }

  // ----- the graph -----

  private need(name: string): FactorOperator {
    const op = this.ops[name]
    if (op === undefined) throw new Refusal('operator', `the message-passing registry has no factor operator '${name}'`)
    return op
  }

  private addVariable(v: VariableSpec): string {
    if (this.names.has(v.name)) throw new OmaiError('value', `factor graph: the variable '${v.name}' is made twice`)
    this.names.add(v.name)
    this.variables.push(v)
    return v.name
  }

  private addFactor(f: FactorSpec, variable: string): void {
    this.need(f.operator)
    this.factors.push({ ...f, id: factorId(variable) })
  }

  /**
   * Drops the `apply` factors whose output nothing reads (the Gaussian approximation an arithmetic chain passed through before it was
   * collapsed into one factor from the chain's input): a flat output sends nothing back, so they only cost time.
   */
  prune(): void {
    const kept = new Set(this.sites.flatMap((s) => s.variables.filter((v): v is string => v !== null)))
    for (const v of this.variables) if (v.observed !== undefined) kept.add(v.name)
    for (;;) {
      const used = new Set<string>(kept)
      for (const f of this.factors)
        for (const [i, a] of f.args.entries())
          if (typeof a === 'string' && !(f.operator === 'apply' && i === 0)) used.add(a)
      const dead = this.factors.filter((f) => f.operator === 'apply' && !used.has(f.args[0] as string))
      if (dead.length === 0) return
      const gone = new Set(dead.map((f) => f.args[0] as string))
      this.factors.splice(0, this.factors.length, ...this.factors.filter((f) => !dead.includes(f)))
      this.variables.splice(0, this.variables.length, ...this.variables.filter((v) => !gone.has(v.name)))
      this.auxiliary.splice(0, this.auxiliary.length, ...this.auxiliary.filter((n) => !gone.has(n)))
    }
  }

  private auxName(site: string, index: number | undefined): string {
    const key = `${site}|${index ?? ''}`
    const k = this.auxCount.get(key) ?? 0
    this.auxCount.set(key, k + 1)
    const name = auxId(site, k, index)
    this.auxiliary.push(name)
    return name
  }

  private atomKey(a: Atom): string {
    return a.kind === 'var' ? `v:${a.name}` : `c:${a.vector}:${a.index}`
  }

  private varAtom(name: string): Cell {
    const atom: Atom = { kind: 'var', name }
    const key = this.atomKey(atom)
    this.atoms.set(key, atom)
    return { kind: 'aff', aff: { terms: new Map([[key, 1]]), offset: 0 } }
  }

  private compAtom(vector: string, index: number, size: number): Cell {
    const atom: Atom = { kind: 'comp', vector, index, size }
    const key = this.atomKey(atom)
    this.atoms.set(key, atom)
    return { kind: 'aff', aff: { terms: new Map([[key, 1]]), offset: 0 } }
  }

  /** The scalar variable that stands for coordinate `index` of a vector variable (made on first use, a unit `dot-constant`). */
  private component(vector: string, index: number, size: number): string {
    const name = elementId(vector, index)
    if (!this.components.has(name)) {
      this.components.set(name, name)
      this.addVariable({ name, family: { kind: 'gaussian' } })
      const a = new Array<number>(size).fill(0)
      a[index] = 1
      this.addFactor({ operator: 'dot-constant', args: [name, vector], params: { a } }, name)
    }
    return name
  }

  /**
   * A variable (or a number, when it is constant) for an affine form: the variable itself when the form is one atom, else
   * an auxiliary variable defined by the affine operators of the engine. With `as`, a variable of that name is made.
   */
  private materialise(aff: Aff, site: string, index: number | undefined, as?: string): string | number {
    const terms = [...aff.terms].filter(([, c]) => c !== 0)
    if (terms.length === 0) return aff.offset
    const first = terms[0] as [string, number]
    if (terms.length === 1 && aff.offset === 0 && first[1] === 1 && as === undefined) {
      const atom = this.atoms.get(first[0]) as Atom
      return atom.kind === 'var' ? atom.name : this.component(atom.vector, atom.index, atom.size)
    }
    const parts: [string, number][] = []
    const groups = new Map<string, { size: number; a: number[] }>()
    for (const [key, c] of terms) {
      const atom = this.atoms.get(key) as Atom
      if (atom.kind === 'var') parts.push([atom.name, c])
      else {
        const g = groups.get(atom.vector) ?? { size: atom.size, a: new Array<number>(atom.size).fill(0) }
        g.a[atom.index] = c
        groups.set(atom.vector, g)
      }
    }
    const onlyVector = groups.size === 1 && parts.length === 0 && aff.offset === 0
    for (const [vector, g] of groups) {
      const name = onlyVector && as !== undefined ? as : this.auxName(site, index)
      this.addVariable({ name, family: { kind: 'gaussian' } })
      this.addFactor({ operator: 'dot-constant', args: [name, vector], params: { a: g.a } }, name)
      if (onlyVector) return name
      parts.push([name, 1])
    }
    const out = as ?? this.auxName(site, index)
    this.addVariable({ name: out, family: { kind: 'gaussian' } })
    const unit = parts.every(([, c]) => c === 1)
    const plus = factorName(this.modelOps, 'add')
    const minus = factorName(this.modelOps, 'sub')
    const args = [out, ...parts.map(([n]) => n)]
    if (aff.offset === 0 && unit && parts.length === 2) this.addFactor({ operator: plus, args }, out)
    else if (aff.offset === 0 && parts.length === 2 && parts[0]?.[1] === 1 && parts[1]?.[1] === -1)
      this.addFactor({ operator: minus, args }, out)
    else if (aff.offset === 0 && unit && parts.length > 2)
      this.addFactor({ operator: 'sum', args, params: { n: parts.length } }, out)
    else if (aff.offset === 0 && parts.length === 1 && (parts[0] as [string, number])[1] !== 1)
      this.addFactor({ operator: 'times-constant', args, params: { factor: (parts[0] as [string, number])[1] } }, out)
    else if (aff.offset === 0 && parts.length === 1) this.addFactor({ operator: 'equal', args }, out)
    else
      this.addFactor({ operator: 'linear', args, params: { coeffs: parts.map(([, c]) => c), offset: aff.offset } }, out)
    return out
  }

  // ----- the nodes -----

  run(): void {
    for (const node of this.bound.spec.nodes) {
      this.current = node.name
      try {
        this.node(node)
      } catch (e) {
        if (!(e instanceof Refusal)) throw e
        this.issues.push({ node: node.name, kind: e.kind, reason: e.message })
        this.failed.add(node.name)
      }
    }
  }

  private node(node: ModelNode): void {
    switch (node.kind) {
      case 'data': {
        const t = this.bound.data[node.name]
        if (t !== undefined) this.consts.set(node.name, { value: t, dims: node.dims })
        return
      }
      case 'latent':
        // A standard normal vector that a non-centred derived node absorbs is compiled by that node only if the dense factor does not apply.
        if ([...this.nonCentred.values()].includes(node.name)) return
        this.latent(node)
        return
      case 'derived':
        this.derived(node)
        return
      case 'observed':
        this.observed(node)
        return
      case 'generated':
        return
      case 'param':
        throw new Refusal(
          'node',
          `'${node.name}' is a param (a learnable value, not a random variable): it has no factor`,
        )
      case 'factor':
        throw new Refusal(
          'node',
          `'${node.name}' is an arbitrary log-density term: no message operator exists for it (a numeric factor operator could be given)`,
        )
    }
  }

  private paramSym(dist: DistSpec, name: string): Sym {
    const e = dist.params[name]
    if (e !== undefined) return this.evalSym(e)
    const d = this.families[dist.family]?.defaults[name]
    if (d === undefined) throw new Refusal('structure', `${dist.family}: the parameter '${name}' is missing`)
    return this.fromNamed({ value: asTensor(d, 'float64'), dims: [] })
  }

  private constParam(node: ModelNode & { readonly dist: DistSpec }, name: string, what: string): number[] {
    const sym = this.paramSym(node.dist, name)
    return this.over(sym, node.dims).map((c) => {
      if (!isConstCell(c))
        throw new Refusal(
          'operator',
          `the ${name} of '${node.name}' depends on a random value: ${what} must be known for a message-passing prior`,
        )
      return c.aff.offset
    })
  }

  /**
   * The expression a Gaussian's scale takes when it is the idiom of a random precision: `1 / sqrt(tau)`, `tau ** -0.5` or
   * `sqrt(1 / tau)`, followed through derived nodes. Absent when the scale is not that shape.
   */
  private precisionExpr(scale: ExprSpec): ExprSpec | undefined {
    const e = this.unwrap(scale)
    if (e.kind !== 'op') return undefined
    const isOne = (x: ExprSpec): boolean => {
      if (!this.isConst(x)) return false
      return toFlat(this.evalNamed(x).value).every((v) => v === 1)
    }
    if (e.op === 'div') {
      const [num, den] = e.args as [ExprSpec, ExprSpec]
      const d = this.unwrap(den)
      if (isOne(num) && d.kind === 'op' && d.op === 'sqrt') return (d.args as [ExprSpec])[0]
    }
    if (e.op === 'pow') {
      const [base, power] = e.args as [ExprSpec, ExprSpec]
      if (this.isConst(power) && toFlat(this.evalNamed(power).value).every((v) => v === -0.5)) return base
    }
    if (e.op === 'sqrt') {
      const inner = this.unwrap((e.args as [ExprSpec])[0])
      if (inner.kind === 'op' && inner.op === 'div' && isOne(inner.args[0] as ExprSpec)) return inner.args[1]
      // The scale is the square root of an inverse-gamma variance (a known multiple of it): the precision is its reciprocal.
      if (this.isRecip(inner)) return { kind: 'op', op: 'div', args: [{ kind: 'const', value: 1 }, inner] }
    }
    return undefined
  }

  /** Whether an expression is an inverse-gamma variance or a known multiple of one (`sigma2`, `sigma2 / kappa`, `2 * sigma2`). */
  private isRecip(e: ExprSpec): boolean {
    const u = this.unwrap(e)
    if (u.kind === 'ref') {
      const cells = this.syms.get(u.name)?.cells
      return cells !== undefined && cells.length > 0 && cells.every((c) => c.kind === 'recip')
    }
    if (u.kind === 'op' && (u.op === 'mul' || u.op === 'div')) {
      const [a, b] = u.args as [ExprSpec, ExprSpec]
      return (this.isRecip(a) && this.isConst(b)) || (u.op === 'mul' && this.isConst(a) && this.isRecip(b))
    }
    return false
  }

  /**
   * The gamma variables that are the precisions of a Gaussian node whose scale is random, element by element: the scale
   * must be the idiom `1 / sqrt(tau)` of a gamma (or exponential) latent. Anything else is refused with the idiom named.
   */
  private randomPrecisions(
    node: ModelNode & { readonly dist: DistSpec },
    param = 'scale',
  ): { readonly variable: string; readonly scale: number; readonly offset: number }[] {
    const scale = node.dist.params[param]
    const idiom =
      'write the scale as 1 / sqrt(tau) with tau a gamma or exponential latent (the conjugate precision; variational message passing only)'
    const tau = scale === undefined ? undefined : this.precisionExpr(scale)
    if (tau === undefined)
      throw new Refusal(
        'operator',
        `the ${param} of '${node.name}' depends on a random value that is not the precision idiom: ${idiom}`,
      )
    let sym: Sym
    try {
      sym = this.evalSym(tau)
    } catch (e) {
      if (e instanceof Refusal && e.kind === 'operator')
        throw new Refusal(
          'operator',
          `the precision of '${node.name}' must be a known multiple of a gamma or exponential latent plus a known value (${e.message}); ${idiom}`,
        )
      throw e
    }
    return this.over(sym, node.dims).map((c) => {
      const scaled = scaledOf(c)
      if (scaled === undefined)
        throw new Refusal(
          'operator',
          `the precision of '${node.name}' must be a gamma or exponential latent, not a ${c.kind === 'rv' ? `${c.family} variable` : 'value computed from other latents'}: ${idiom}`,
        )
      if (!(scaled.scale >= 0 && scaled.offset >= 0 && scaled.scale + scaled.offset > 0))
        throw new Refusal(
          'data',
          `the precision of '${node.name}' is ${scaled.scale} times a gamma variable plus ${scaled.offset}: both must be non-negative, not both zero`,
        )
      return scaled
    })
  }

  private elementName(site: string, dims: readonly string[], i: number): string {
    return dims.length === 0 ? site : elementId(site, i)
  }

  private layout(
    node: ModelNode,
    kind: SiteLayout['kind'],
    family: FamilyKind | undefined,
    layout: SiteLayout['layout'],
    variables: readonly (string | null)[],
    posterior: boolean,
    extra: Partial<Pick<SiteLayout, 'reciprocal' | 'part'>> = {},
  ): void {
    this.sites.push({
      site: node.name,
      kind,
      dims: node.dims,
      shape: this.shapeOf(node.dims),
      family,
      layout,
      variables,
      posterior,
      ...extra,
    })
  }

  private latent(node: ModelNode & { readonly kind: 'latent' }): void {
    const fam = node.dist.family
    const truncated = node.constraint !== undefined && node.constraint.kind !== 'real' && fam === 'normal'
    if (node.constraint !== undefined && node.constraint.kind !== 'real' && !truncated)
      throw new Refusal(
        'constraint',
        `'${node.name}' has the constraint '${node.constraint.kind}': a message family lives on the family's own support, so a constraint beyond it is not expressible`,
      )
    if (PROJECTED_PRIORS.has(fam)) {
      this.latentProjected(node)
      return
    }
    const meanOf = this.normalGammaByMean.get(node.name)
    if (meanOf !== undefined) {
      this.layout(node, 'latent', 'normal-gamma', 'scalar', [meanOf.variable], true, { part: 'mean' })
      return
    }
    if ((fam === 'gamma' || fam === 'inverseGamma') && this.latentNormalGamma(node)) return
    const kind = MESSAGE_FAMILY[fam]
    if (kind === undefined)
      throw new Refusal(
        'family',
        `'${node.name}' is ${fam}: the engine has message families gaussian, gamma, beta, bernoulli, discrete and dirichlet, and none for ${fam}`,
      )
    if (fam === 'normal') {
      this.latentNormal(node)
      if (truncated) this.truncate(node)
      return
    }
    if (fam === 'multivariateNormal') {
      this.latentMultivariateNormal(node)
      return
    }
    if (fam === 'normalSpikeSlab') {
      this.latentSpikeSlab(node)
      return
    }
    if (fam === 'bernoulli' || fam === 'categorical') {
      this.latentDiscrete(node)
      return
    }
    if (fam === 'dirichlet') {
      const concentration = node.dist.params['concentration']
      if (concentration !== undefined && !this.isConst(concentration)) this.latentRandomConcentration(node)
      else this.latentDirichlet(node)
      return
    }
    if (fam === 'beta' && ['a', 'b'].some((k) => !this.isConst(node.dist.params[k] as ExprSpec))) {
      this.latentRandomConcentration(node)
      return
    }
    // An inverse gamma `x` with scale `b` is the precision `1 / x` ~ Gamma(a, rate b): the variable is that gamma, the site its reciprocal.
    const inverse = fam === 'inverseGamma'
    const rateName = inverse ? 'scale' : 'rate'
    const rateExpr = node.dist.params[rateName]
    if ((fam === 'gamma' || inverse) && rateExpr !== undefined && !this.isConst(rateExpr)) {
      this.latentGammaRate(node)
      return
    }
    const shape = this.shapeOf(node.dims)
    const n = prod(shape)
    const params =
      fam === 'gamma' || inverse
        ? [this.constParam(node, 'concentration', 'the shape'), this.constParam(node, rateName, 'the rate')]
        : fam === 'exponential'
          ? [new Array<number>(n).fill(1), this.constParam(node, 'rate', 'the rate')]
          : [this.constParam(node, 'a', 'the shape'), this.constParam(node, 'b', 'the shape')]
    const cells: Cell[] = []
    const vars: string[] = []
    for (let i = 0; i < n; i++) {
      const name = this.addVariable({ name: this.elementName(node.name, node.dims, i), family: { kind } })
      const p = [params[0]?.[i] as number, params[1]?.[i] as number]
      const f =
        fam === 'beta'
          ? priorFactor(name, betaEta(p[0] as number, p[1] as number))
          : gammaPrior(name, p[0] as number, p[1] as number)
      this.addFactor(f, name)
      cells.push(inverse ? { kind: 'recip', variable: name, scale: 1 } : { kind: 'rv', variable: name, family: kind })
      vars.push(name)
    }
    this.syms.set(node.name, { dims: node.dims, cells })
    this.layout(
      node,
      'latent',
      kind,
      node.dims.length === 0 ? 'scalar' : 'elements',
      vars,
      true,
      inverse ? { reciprocal: true } : {},
    )
  }

  /**
   * The known multiple `c` of a value in a node: `tau`, `c * tau`, `tau / c` (and nested), 1 for the node itself. Absent when the
   * expression is anything else.
   */
  private multipleOf(e: ExprSpec, name: string): number | undefined {
    const u = this.unwrap(e)
    if (u.kind === 'ref') return u.name === name ? 1 : undefined
    if (u.kind !== 'op' || (u.op !== 'mul' && u.op !== 'div')) return undefined
    const [a, b] = u.args as [ExprSpec, ExprSpec]
    const known = (x: ExprSpec): number | undefined => {
      if (!this.isConst(x)) return undefined
      const v = toFlat(this.evalNamed(x).value)
      return v.length === 1 && Number.isFinite(v[0]) ? (v[0] as number) : undefined
    }
    const ka = known(a)
    const kb = known(b)
    if (u.op === 'mul') {
      const m = ka !== undefined ? this.multipleOf(b, name) : kb !== undefined ? this.multipleOf(a, name) : undefined
      const k = ka ?? kb
      return m === undefined || k === undefined ? undefined : m * k
    }
    const m = kb !== undefined ? this.multipleOf(a, name) : undefined
    return m === undefined || kb === undefined ? undefined : m / kb
  }

  /**
   * The known multiple of the precision `tau` that a Gaussian's scale stands for, when it is the idiom of one: `1 / sqrt(c tau)`
   * (a gamma `tau`), or `sqrt(c sigma2)` with `sigma2` an inverse gamma (so the precision is `tau / c`). Absent otherwise.
   */
  private precisionMultiple(scale: ExprSpec, tau: string, inverse: boolean): number | undefined {
    const e = this.unwrap(scale)
    if (e.kind !== 'op') return undefined
    if (inverse) {
      const c = e.op === 'sqrt' ? this.multipleOf(e.args[0] as ExprSpec, tau) : undefined
      return c === undefined || !(c > 0) ? undefined : 1 / c
    }
    const p = this.precisionExpr(scale)
    const c = p === undefined ? undefined : this.multipleOf(p, tau)
    return c === undefined || !(c > 0) ? undefined : c
  }

  /**
   * The normal-gamma pair of a gamma (or inverse-gamma) latent `T` (a precision or a variance): a Gaussian latent `M` with a
   * known mean and the variance `1 / (kappa lambda)` (`mu ~ N(m, sigma2 / kappa)`) and observed Gaussians `y ~ N(M, 1 / (c lambda))`,
   * with nothing else reading either of them. The pair is one `normal-gamma` variable with a prior of both and a
   * `gaussian-normal-gamma` factor per observation, so the joint posterior is exactly the conjugate normal-gamma (the classic
   * normal-inverse-gamma prior on a mean and a variance). Returns false (and changes nothing) when the model is not that shape: the
   * mean and precision then compile to a Gaussian with a gamma precision each.
   */
  private latentNormalGamma(node: ModelNode & { readonly kind: 'latent' }): boolean {
    if (node.dims.length !== 0) return false
    const inverse = node.dist.family === 'inverseGamma'
    const rateName = inverse ? 'scale' : 'rate'
    if (
      !this.isConst(node.dist.params['concentration'] as ExprSpec) ||
      !this.isConst(node.dist.params[rateName] as ExprSpec)
    )
      return false
    const readers = this.bound.spec.nodes.filter((n) => n.name !== node.name && nodeRefs(n).includes(node.name))
    const means = readers.filter((n) => n.kind === 'latent')
    const mean = means[0]
    if (mean?.kind !== 'latent' || means.length !== 1) return false
    const loc = mean.dist.params['loc']
    const scale = mean.dist.params['scale']
    if (mean.dims.length !== 0 || mean.dist.family !== 'normal' || loc === undefined || scale === undefined)
      return false
    if (!this.isConst(loc) || mean.constraint !== undefined) return false
    const kappa = this.precisionMultiple(scale, node.name, inverse)
    if (kappa === undefined) return false
    const observed: (ModelNode & { readonly kind: 'observed' })[] = []
    const multiples = new Map<string, number>()
    for (const n of this.bound.spec.nodes) {
      if (n.name === node.name || n.name === mean.name) continue
      const refs = nodeRefs(n)
      if (!refs.includes(node.name) && !refs.includes(mean.name)) continue
      if (n.kind !== 'observed' || n.dist.family !== 'normal') return false
      const l = n.dist.params['loc']
      const sc = n.dist.params['scale']
      if (l === undefined || sc === undefined || this.unwrap(l).kind !== 'ref') return false
      if ((this.unwrap(l) as { readonly name: string }).name !== mean.name) return false
      if (exprRefs(n.value).some((r) => r === node.name || r === mean.name)) return false
      const c = this.precisionMultiple(sc, node.name, inverse)
      if (c === undefined) return false
      if (this.bound.batch !== undefined && n.dims.includes(this.bound.batch.dim)) return false
      observed.push(n)
      multiples.set(n.name, c)
    }
    if (observed.length === 0) return false
    this.need('gaussian-normal-gamma')
    const a = this.constParam(node, 'concentration', 'the shape')[0] as number
    const b = this.constParam(node, rateName, 'the rate')[0] as number
    const m = this.constParam(mean, 'loc', 'the mean')[0] as number
    const plan: NormalGammaPlan = {
      precision: node.name,
      mean: mean.name,
      variable: mean.name,
      inverse,
      kappa,
      observed: observed.map((n) => n.name),
    }
    this.normalGammaOf.set(node.name, plan)
    this.normalGammaByMean.set(mean.name, plan)
    for (const n of observed) this.normalGammaObserved.set(n.name, { plan, multiple: multiples.get(n.name) as number })
    this.addVariable({ name: plan.variable, family: { kind: 'normal-gamma' } })
    this.addFactor(normalGammaPrior(plan.variable, m, kappa, a, b), plan.variable)
    this.layout(node, 'latent', 'normal-gamma', 'scalar', [plan.variable], true, {
      part: 'precision',
      ...(inverse ? { reciprocal: true } : {}),
    })
    return true
  }

  /** The observed Gaussian rows of a normal-gamma pair: a `gaussian-normal-gamma` factor on the shared variable per row. */
  private observedNormalGamma(
    node: ModelNode & { readonly kind: 'observed' },
    ng: { readonly plan: NormalGammaPlan; readonly multiple: number },
    values: readonly number[],
  ): void {
    const vars: (string | null)[] = []
    values.forEach((y, i) => {
      if (Number.isNaN(y)) {
        vars.push(null)
        return
      }
      const name = this.elementName(node.name, node.dims, i)
      this.addVariable({ name, family: { kind: 'gaussian' }, observed: y })
      this.addFactor(
        {
          operator: 'gaussian-normal-gamma',
          args: [name, ng.plan.variable],
          ...(ng.multiple === 1 ? {} : { params: { precision: ng.multiple } }),
        },
        name,
      )
      vars.push(name)
    })
    this.layout(node, 'observed', undefined, node.dims.length === 0 ? 'scalar' : 'observed', vars, false)
  }

  /**
   * A multivariate normal latent with a known mean and covariance (a Gaussian process prior over the inputs of the data: the
   * covariance a constant Gram matrix, or given by its Cholesky factor `scaleTril`): one `gaussian-mv` variable with a
   * dense `gaussian-mv` prior, whose components the likelihood reads (`probit(f)`, `logits`, a Gaussian row), so the
   * correlations between the elements are kept and the engines see a dense Gaussian vector.
   */
  private latentMultivariateNormal(node: ModelNode & { readonly kind: 'latent' }): void {
    const shape = this.shapeOf(node.dims)
    const n = prod(shape)
    if (node.dims.length !== 1)
      throw new Refusal('structure', `'${node.name}' is a multivariateNormal: one vector over one dimension is mapped`)
    const params = node.dist.params
    // A random mean that is a linear map of another Gaussian vector (`matvec(F, previous)`): one dense Gaussian step between the two.
    if (params['loc'] !== undefined && !this.isConst(params['loc'])) {
      this.latentTransition(node, params['loc'])
      return
    }
    const known = (e: ExprSpec | undefined, what: string): number[] => {
      if (e === undefined || !this.isConst(e))
        throw new Refusal(
          'operator',
          `the ${what} of '${node.name}' depends on a random value: a known matrix is mapped`,
        )
      return Array.from(toFlat(this.evalNamed(e).value))
    }
    const loc = known(params['loc'], 'mean')
    const mean = loc.length === 1 ? new Array<number>(n).fill(loc[0] ?? 0) : loc
    let cov: number[]
    if (params['covariance'] !== undefined) cov = known(params['covariance'], 'covariance')
    else if (params['scaleTril'] !== undefined) {
      const L = known(params['scaleTril'], 'scale factor')
      cov = new Array<number>(n * n).fill(0)
      for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++) {
          let v = 0
          for (let k = 0; k <= Math.min(i, j); k++) v += (L[i * n + k] ?? 0) * (L[j * n + k] ?? 0)
          cov[i * n + j] = v
        }
    } else
      throw new Refusal(
        'structure',
        `'${node.name}' is a multivariateNormal with a precision: give a covariance or scaleTril`,
      )
    if (cov.length !== n * n) throw new Refusal('structure', `the covariance of '${node.name}' is not ${n} by ${n}`)
    this.addVariable({ name: node.name, family: { kind: 'gaussian-mv', size: n } })
    this.addFactor(gaussianMvPrior(node.name, mean, cov), node.name)
    this.syms.set(node.name, {
      dims: node.dims,
      cells: Array.from({ length: n }, (_, j) => this.compAtom(node.name, j, n)),
    })
    this.layout(node, 'latent', 'gaussian-mv', 'packed', [node.name], true)
  }

  /**
   * A multivariate normal latent whose mean is `F w + c` for a known matrix and another Gaussian vector `w` (the step of a linear
   * dynamical system, `x_t ~ N(F x_(t-1), Q)`), with a known covariance: one `gaussian-vector` variable and a `gaussian-transition`
   * factor between it and `w`, so a chain of steps with linear Gaussian readings is the Kalman smoother.
   */
  private latentTransition(node: ModelNode & { readonly kind: 'latent' }, locExpr: ExprSpec): void {
    if (node.dims.length !== 1)
      throw new Refusal('structure', `'${node.name}' is a multivariateNormal: one vector over one dimension is mapped`)
    const n = this.size(node.dims[0] as string)
    const params = node.dist.params
    const covarianceExpr = params['covariance']
    if (covarianceExpr === undefined || !this.isConst(covarianceExpr))
      throw new Refusal('operator', `the covariance of '${node.name}' must be a known matrix when its mean is random`)
    const covariance = Array.from(toFlat(this.evalNamed(covarianceExpr).value))
    if (covariance.length !== n * n)
      throw new Refusal('structure', `the covariance of '${node.name}' is not ${n} by ${n}`)
    const loc = this.over(this.evalSym(locExpr), node.dims)
    let vector: string | undefined
    let columns = 0
    const forms = loc.map((c) => {
      if (c.kind !== 'aff')
        throw new Refusal('operator', `the mean of '${node.name}' must be a linear map of another Gaussian vector`)
      const terms = [...c.aff.terms].filter(([, k]) => k !== 0)
      for (const [key] of terms) {
        const atom = this.atoms.get(key)
        if (atom?.kind !== 'comp' || (vector !== undefined && atom.vector !== vector))
          throw new Refusal('operator', `the mean of '${node.name}' must be a linear map of one Gaussian vector`)
        vector = atom.vector
        columns = atom.size
      }
      return { aff: c.aff, terms }
    })
    if (vector === undefined)
      throw new Refusal('operator', `the mean of '${node.name}' does not depend on a Gaussian vector`)
    const matrix = new Array<number>(n * columns).fill(0)
    const offset = forms.map(({ aff, terms }, i) => {
      for (const [key, k] of terms) matrix[i * columns + (this.atoms.get(key) as Atom & { kind: 'comp' }).index] = k
      return aff.offset
    })
    this.addVariable({ name: node.name, family: { kind: 'gaussian-mv', size: n } })
    this.addFactor(
      {
        operator: 'gaussian-transition',
        args: [node.name, vector],
        params: { matrix, covariance, ...(offset.some((v) => v !== 0) ? { offset } : {}) },
      },
      node.name,
    )
    this.syms.set(node.name, {
      dims: node.dims,
      cells: Array.from({ length: n }, (_, j) => this.compAtom(node.name, j, n)),
    })
    this.layout(node, 'latent', 'gaussian-mv', 'packed', [node.name], true)
  }

  /**
   * A prior with no message family (half-normal, half-Cauchy, log-normal, uniform, truncated normal), the exact numeric
   * `prior-site` on a Gaussian variable (a log-normal is the exact Gaussian on its log). A density on the positive half line is
   * carried on its log: a Gaussian variable `l = log tau` with the density of `l` as its site, and the value `tau` is the variable
   * `exp(l)` of an `apply` factor, whose Gaussian belief is the moment match of a log-normal. A density on a bounded or truncated
   * real range is the site on the value itself. The site's message is the moment match against the current cavity at every sweep,
   * so the graph holds the prior itself; the belief of the variable is still one Gaussian.
   */
  private latentProjected(node: ModelNode & { readonly kind: 'latent' }): void {
    const fam = node.dist.family
    const n = prod(this.shapeOf(node.dims))
    const names: Readonly<Record<string, readonly string[]>> = {
      halfNormal: ['scale'],
      halfCauchy: ['scale'],
      logNormal: ['mu', 'sigma'],
      uniform: ['low', 'high'],
      truncatedNormal: ['loc', 'scale', 'low', 'high'],
    }
    const bases: Readonly<Record<string, string>> = {
      halfNormal: 'half-normal',
      halfCauchy: 'half-cauchy',
      uniform: 'uniform',
      truncatedNormal: 'truncated-normal',
    }
    const values = Object.fromEntries((names[fam] ?? []).map((p) => [p, this.constParam(node, p, `the ${p}`)]))
    const cells: Cell[] = []
    const vars: string[] = []
    const logCells: Cell[] = []
    const site = node.dims.length === 0 ? undefined : (i: number): number => i
    for (let i = 0; i < n; i++) {
      const p = (k: string): number => (values[k] as number[])[i] as number
      const name = this.addVariable({ name: this.elementName(node.name, node.dims, i), family: { kind: 'gaussian' } })
      const low = fam === 'uniform' || fam === 'truncatedNormal' ? p('low') : 0
      const high = fam === 'uniform' || fam === 'truncatedNormal' ? p('high') : Number.POSITIVE_INFINITY
      if (fam === 'logNormal' && !(p('sigma') > 0))
        throw new Refusal('data', `the sigma of '${node.name}' must be positive`)
      const positive = fam === 'halfNormal' || fam === 'halfCauchy' || fam === 'logNormal' || low >= 0
      // The exact density as a numeric site (`prior-site`): its message is the moment match against the current cavity at every
      // sweep, so the graph holds the prior itself. Infinite bounds are left out of the parameters (they are JSON).
      const priorFactor = (variable: string, log: boolean): FactorSpec => ({
        operator: 'prior-site',
        args: [variable],
        params: {
          y: 0,
          base: bases[fam] as string,
          log: log ? 1 : 0,
          ...(names[fam]?.includes('scale') ? { scale: p('scale') } : {}),
          ...(fam === 'truncatedNormal' ? { loc: p('loc') } : {}),
          ...(Number.isFinite(low) && (fam === 'uniform' || fam === 'truncatedNormal') ? { low } : {}),
          ...(Number.isFinite(high) && (fam === 'uniform' || fam === 'truncatedNormal') ? { high } : {}),
        },
      })
      if (positive) {
        const l = this.auxName(node.name, site === undefined ? undefined : site(i))
        this.addVariable({ name: l, family: { kind: 'gaussian' } })
        if (fam === 'logNormal') this.addFactor(gaussianPrior(l, p('mu'), p('sigma') ** 2), l)
        else this.addFactor(priorFactor(l, true), l)
        this.need('apply')
        this.addFactor({ operator: 'apply', args: [name, l], params: { fn: 'exp' } }, name)
        this.logScale.set(name, l)
        logCells.push(this.varAtom(l))
      } else this.addFactor(priorFactor(name, false), name)
      cells.push(this.varAtom(name))
      vars.push(name)
    }
    this.syms.set(node.name, { dims: node.dims, cells })
    // A positive value carried on its log (`tau = exp(l)`): the log is addressable, so a log link reads it directly (a rate or a scale).
    if (logCells.length > 0) this.syms.set(logName(node.name), { dims: node.dims, cells: logCells })
    this.layout(node, 'latent', 'gaussian', node.dims.length === 0 ? 'scalar' : 'elements', vars, true)
  }

  /**
   * The constraint of a Gaussian latent as comparison factors on its variables: `positive`, `nonNegative`, `greaterThan`,
   * `lessThan` and `interval` with known scalar bounds truncate each element (a `greater-than` or `is-between` factor with an
   * observed true outcome, whose moment matching is the truncated Gaussian), and `ordered` says each element is above the
   * previous one (a `greater-than` on the difference of neighbours). The prior is not renormalised, as in Stan.
   */
  private truncate(node: ModelNode & { readonly kind: 'latent' }): void {
    const c = node.constraint
    if (c === undefined) return
    const cells = this.syms.get(node.name)?.cells ?? []
    const flag = (
      x: string,
      operator: 'greater-than' | 'is-between',
      params: Record<string, Json>,
      i: number,
    ): void => {
      const name = this.auxName(node.name, i)
      this.need(operator)
      this.addVariable({ name, family: { kind: 'bernoulli' }, observed: 1 })
      this.addFactor({ operator, args: [name, x], params }, name)
    }
    const number = (v: Json): number => {
      if (typeof v !== 'number')
        throw new Refusal('constraint', `the bound of the constraint of '${node.name}' must be one known number`)
      return v
    }
    if (c.kind === 'ordered') {
      if (node.dims.length !== 1)
        throw new Refusal('constraint', `'${node.name}' is ordered over [${node.dims}]: one dimension is mapped`)
      for (let i = 0; i + 1 < cells.length; i++) {
        const gap = combine(affOf(cells[i + 1] as Cell), affOf(cells[i] as Cell), -1)
        flag(this.materialise(gap, node.name, i) as string, 'greater-than', { threshold: 0 }, i)
      }
      return
    }
    const bounds =
      c.kind === 'positive' || c.kind === 'nonNegative'
        ? { lower: 0, upper: Number.POSITIVE_INFINITY }
        : c.kind === 'greaterThan'
          ? { lower: number(c.low), upper: Number.POSITIVE_INFINITY }
          : c.kind === 'lessThan'
            ? { lower: Number.NEGATIVE_INFINITY, upper: number(c.high) }
            : c.kind === 'interval'
              ? { lower: number(c.low), upper: number(c.high) }
              : undefined
    if (bounds === undefined)
      throw new Refusal(
        'constraint',
        `'${node.name}' has the constraint '${c.kind}': only ordered and bounds on a Gaussian are mapped`,
      )
    for (const [i, cell] of cells.entries()) {
      const x = this.materialise(affOf(cell), node.name, node.dims.length === 0 ? undefined : i) as string
      if (bounds.upper === Number.POSITIVE_INFINITY) flag(x, 'greater-than', { threshold: bounds.lower }, i)
      else flag(x, 'is-between', { lower: Math.max(bounds.lower, -1e12), upper: bounds.upper }, i)
    }
  }

  private latentNormal(node: ModelNode & { readonly kind: 'latent' }): void {
    const shape = this.shapeOf(node.dims)
    const n = prod(shape)
    const loc = this.over(this.paramSym(node.dist, 'loc'), node.dims)
    const scaleExpr = node.dist.params['scale']
    if (scaleExpr !== undefined && !this.isConst(scaleExpr)) {
      if (this.packed.has(node.name)) this.latentSharedPrecision(node, loc)
      else if (this.precisionExpr(scaleExpr) === undefined) this.latentRandomScale(node, loc, scaleExpr)
      else this.latentRandomPrecision(node, loc)
      return
    }
    const scale = this.constParam(node, 'scale', 'the scale')
    for (const s of scale)
      if (!(s > 0)) throw new Refusal('data', `the scale of '${node.name}' is ${s}: it must be positive`)
    const group = this.groups.get(node.name)
    if (group !== undefined) {
      this.latentGrouped(node, group)
      return
    }
    if (this.packed.has(node.name)) {
      const mean = loc.map((c) => {
        if (!isConstCell(c))
          throw new Refusal(
            'operator',
            `'${node.name}' is read as a vector by a matmul, so its prior mean must be known`,
          )
        return c.aff.offset
      })
      const cov = new Array<number>(n * n).fill(0)
      for (let i = 0; i < n; i++) cov[i * n + i] = (scale[i] as number) ** 2
      this.addVariable({ name: node.name, family: { kind: 'gaussian-mv', size: n } })
      this.addFactor(gaussianMvPrior(node.name, mean, cov), node.name)
      this.syms.set(node.name, {
        dims: node.dims,
        cells: Array.from({ length: n }, (_, j) => this.compAtom(node.name, j, n)),
      })
      this.layout(node, 'latent', 'gaussian-mv', 'packed', [node.name], true)
      return
    }
    const cells: Cell[] = []
    const vars: string[] = []
    for (let i = 0; i < n; i++) {
      const name = this.addVariable({ name: this.elementName(node.name, node.dims, i), family: { kind: 'gaussian' } })
      const l = loc[i] as Cell
      const variance = (scale[i] as number) ** 2
      if (l.kind === 'lookup') {
        // A mean chosen by a discrete variable (a mixture of components): a gate over one Gaussian prior per state, over the
        // variable and the random means of some states.
        this.need('gate')
        const others: string[] = []
        const branches = l.values.map((v, k) => {
          const item = l.items?.[k]
          let mean: number | { readonly const: number } = { const: v }
          if (item !== undefined && item.terms.size > 0) {
            const m = this.materialise(item, node.name, node.dims.length === 0 ? undefined : i) as string
            let at = others.indexOf(m)
            if (at < 0) {
              at = others.length
              others.push(m)
            }
            mean = at + 1
          }
          return { operator: 'gaussian', args: [0, mean], params: { precision: 1 / variance } }
        })
        this.addFactor(
          { operator: 'gate', args: [l.variable, name, ...others], params: { arity: 1 + others.length, branches } },
          name,
        )
      } else if (isConstCell(l)) this.addFactor(gaussianPrior(name, l.aff.offset, variance), name)
      else {
        const m = this.materialise(affOf(l), node.name, node.dims.length === 0 ? undefined : i)
        this.addFactor({ operator: 'gaussian', args: [name, m as string], params: { precision: 1 / variance } }, name)
      }
      cells.push(this.varAtom(name))
      vars.push(name)
    }
    this.syms.set(node.name, { dims: node.dims, cells })
    this.layout(node, 'latent', 'gaussian', node.dims.length === 0 ? 'scalar' : 'elements', vars, true)
  }

  /**
   * A scalar normal latent that belongs to a group (`findGroups`): the first member made creates the joint Gaussian vector of
   * the group with a diagonal prior and a scalar readout variable for each member (the marginal of its coordinate, which is
   * what the posterior reads), and the members' cells are coordinates of the vector, so a predictor over several of them is
   * one dot product on it.
   */
  private latentGrouped(
    node: ModelNode & { readonly kind: 'latent' },
    group: { readonly vector: string; readonly members: readonly string[] },
  ): void {
    if (!this.groupsMade.has(group.vector)) {
      this.groupsMade.add(group.vector)
      const size = group.members.length
      const mean: number[] = []
      const cov = new Array<number>(size * size).fill(0)
      group.members.forEach((m, k) => {
        const member = this.nodes.get(m) as ModelNode & { readonly kind: 'latent' }
        const loc = this.constParam(member, 'loc', 'the mean')[0] as number
        const sd = this.constParam(member, 'scale', 'the scale')[0] as number
        if (!(sd > 0)) throw new Refusal('data', `the scale of '${m}' is ${sd}: it must be positive`)
        mean.push(loc)
        cov[k * size + k] = sd * sd
      })
      this.addVariable({ name: group.vector, family: { kind: 'gaussian-mv', size } })
      this.auxiliary.push(group.vector)
      this.addFactor(gaussianMvPrior(group.vector, mean, cov), group.vector)
      group.members.forEach((m, k) => {
        this.addVariable({ name: m, family: { kind: 'gaussian' } })
        const a = new Array<number>(size).fill(0)
        a[k] = 1
        this.addFactor({ operator: 'dot-constant', args: [m, group.vector], params: { a } }, m)
        this.syms.set(m, { dims: [], cells: [this.compAtom(group.vector, k, size)] })
      })
    }
    this.layout(node, 'latent', 'gaussian', 'scalar', [node.name], true)
  }

  /**
   * A spike and slab latent (`normalSpikeSlab`: a mixture of a wide and a narrow zero-mean normal on every element): each
   * element is a Gaussian variable (or a coordinate of one packed Gaussian vector, when a `matmul` reads it) with a
   * `mixture-site` (`mixture-row`) factor, whose EP message is exact. A scalar site speaks first through the engine's flat-cavity
   * policy (`needsDensity`); the row on a packed vector starts with a very wide Gaussian message (a hundred slab standard
   * deviations), so the other factors have a proper cavity to begin from, and its own first update replaces it.
   */
  private latentSpikeSlab(node: ModelNode & { readonly kind: 'latent' }): void {
    const n = prod(this.shapeOf(node.dims))
    if (['inclusion', 'slab', 'spike'].some((p) => !this.paramIsConst(node, p))) {
      this.latentSpikeSlabSwitch(node, n)
      return
    }
    const inclusion = this.constParam(node, 'inclusion', 'the inclusion probability')
    const slab = this.constParam(node, 'slab', 'the slab scale')
    const spike = this.constParam(node, 'spike', 'the spike scale')
    for (let i = 0; i < n; i++)
      if (!(slab[i] !== undefined && spike[i] !== undefined && (slab[i] as number) > 0 && (spike[i] as number) > 0))
        throw new Refusal('data', `the scales of '${node.name}' must be positive`)
    const mixture = (i: number): Record<string, Json> => ({
      weights: [inclusion[i] as number, 1 - (inclusion[i] as number)],
      means: [0, 0],
      sds: [slab[i] as number, spike[i] as number],
    })
    const wide = (i: number): number => (100 * Math.max(slab[i] as number, spike[i] as number)) ** 2
    if (this.packed.has(node.name)) {
      this.need('mixture-row')
      this.addVariable({ name: node.name, family: { kind: 'gaussian-mv', size: n } })
      for (let i = 0; i < n; i++) {
        const a = new Array<number>(n).fill(0)
        a[i] = 1
        // The row starts with a vague message in its own coordinate, so the other factors have a proper cavity to begin from.
        const start = [...new Array<number>(n).fill(0), ...new Array<number>(n * n).fill(0)]
        start[n + i * n + i] = 1 / wide(i)
        this.addFactor(
          { operator: 'mixture-row', args: [node.name], params: { a, ...mixture(i) }, start: [start] },
          this.elementName(node.name, node.dims, i),
        )
      }
      this.syms.set(node.name, {
        dims: node.dims,
        cells: Array.from({ length: n }, (_, j) => this.compAtom(node.name, j, n)),
      })
      this.layout(node, 'latent', 'gaussian-mv', 'packed', [node.name], true)
      return
    }
    const cells: Cell[] = []
    const vars: string[] = []
    for (let i = 0; i < n; i++) {
      const name = this.addVariable({ name: this.elementName(node.name, node.dims, i), family: { kind: 'gaussian' } })
      this.addFactor({ operator: 'mixture-site', args: [name], params: mixture(i) }, name)
      cells.push(this.varAtom(name))
      vars.push(name)
    }
    this.syms.set(node.name, { dims: node.dims, cells })
    this.layout(node, 'latent', 'gaussian', node.dims.length === 0 ? 'scalar' : 'elements', vars, true)
  }

  /**
   * A spike and slab latent with a random inclusion probability (a beta latent) or random slab or spike widths (the precision idiom
   * of a gamma latent), the switch summed out: each element is a Gaussian variable under the gate of its spike and slab Gaussians
   * over a hidden Bernoulli switch (`hiddenSwitch`), the same graph as the explicit-switch form with the switch left out of the
   * posterior. A vector that a `matmul` reads (one joint Gaussian) has one covariance, so it is refused.
   */
  private latentSpikeSlabSwitch(node: ModelNode & { readonly kind: 'latent' }, n: number): void {
    if (this.packed.has(node.name))
      throw new Refusal(
        'operator',
        `'${node.name}' is read as a vector by a matmul and has a random inclusion or scale: a joint Gaussian vector has one covariance, so they must be known`,
      )
    const inclusion = this.over(this.paramSym(node.dist, 'inclusion'), node.dims)
    // The precision of each element's slab and spike: a known one, or the gamma variable's multiple.
    const precisions = (param: 'slab' | 'spike'): ({ readonly sd: number } | Precision)[] =>
      this.paramIsConst(node, param)
        ? this.constParam(node, param, `the ${param} scale`).map((sd) => {
            if (!(sd > 0))
              throw new Refusal('data', `the ${param} scale of '${node.name}' is ${sd}: it must be positive`)
            return { sd }
          })
        : this.randomPrecisions(node, param)
    const slab = precisions('slab')
    const spike = precisions('spike')
    const cells: Cell[] = []
    const vars: string[] = []
    for (let i = 0; i < n; i++) {
      const index = node.dims.length === 0 ? undefined : i
      const name = this.addVariable({ name: this.elementName(node.name, node.dims, i), family: { kind: 'gaussian' } })
      const z = this.hiddenSwitch(node, inclusion[i] as Cell, index)
      const variables = [name]
      const branch = (p: { readonly sd: number } | Precision): Record<string, Json> => {
        if ('sd' in p) return { operator: 'gaussian', args: [0, { const: 0 }], params: { precision: 1 / p.sd ** 2 } }
        let at = variables.indexOf(p.variable)
        if (at < 0) {
          at = variables.length
          variables.push(p.variable)
        }
        this.need('gaussian-gamma-precision')
        return {
          operator: 'gaussian-gamma-precision',
          args: [0, { const: 0 }, at],
          params: { scale: p.scale, offset: p.offset },
        }
      }
      // State 0 of the switch is the spike, state 1 the slab (the probability of the Bernoulli is the inclusion).
      const branches = [branch(spike[i] as never), branch(slab[i] as never)]
      this.need('gate')
      this.addFactor(
        {
          operator: 'gate',
          args: [z, ...variables],
          params: { arity: variables.length, branches: branches as unknown as Json },
        },
        name,
      )
      cells.push(this.varAtom(name))
      vars.push(name)
    }
    this.syms.set(node.name, { dims: node.dims, cells })
    this.layout(node, 'latent', 'gaussian', node.dims.length === 0 ? 'scalar' : 'elements', vars, true)
  }

  /** A scale with a modelling floor `clip(tau, lo, hi)` read as the scale itself: the floor never binds for a positive scale variable. */
  private stripClip(e: ExprSpec): ExprSpec {
    let cur = this.unwrap(e)
    while (
      cur.kind === 'op' &&
      cur.op === 'clip' &&
      this.isConst(cur.args[1] as ExprSpec) &&
      this.isConst(cur.args[2] as ExprSpec)
    )
      cur = this.unwrap(cur.args[0] as ExprSpec)
    return cur
  }

  /**
   * A Gaussian latent whose scale is a random positive value (a half-normal or half-Cauchy sd of a hierarchy, `exp` of a
   * Gaussian), written in its non-centred form: `x = loc + scale * z` with a standard normal `z`, the product of two Gaussian
   * variables (the `product` factor of the engine). A scale that is not an affine Gaussian value is refused with the
   * precision idiom named.
   */
  private latentRandomScale(
    node: ModelNode & { readonly kind: 'latent' },
    loc: readonly Cell[],
    scaleExpr: ExprSpec,
  ): void {
    let scales: Cell[]
    try {
      scales = this.over(this.evalSym(this.stripClip(scaleExpr)), node.dims)
    } catch (e) {
      if (!(e instanceof Refusal)) throw e
      this.latentRandomPrecision(node, loc)
      return
    }
    if (scales.some((c) => c.kind !== 'aff')) {
      this.latentRandomPrecision(node, loc)
      return
    }
    this.need('product')
    const cells: Cell[] = []
    const vars: string[] = []
    for (const [i, c] of scales.entries()) {
      const index = node.dims.length === 0 ? undefined : i
      const name = this.elementName(node.name, node.dims, i)
      const z = this.auxName(node.name, index)
      this.addVariable({ name: z, family: { kind: 'gaussian' } })
      this.addFactor(gaussianPrior(z, 0, 1), z)
      const zAtom = this.atomKey({ kind: 'var', name: z })
      this.atoms.set(zAtom, { kind: 'var', name: z })
      const scale = affOf(c)
      const spread = this.mulAff(scale, { terms: new Map([[zAtom, 1]]), offset: 0 }, i)
      this.materialise(combine(affOf(loc[i] as Cell), spread, 1), node.name, index, name)
      cells.push(this.varAtom(name))
      vars.push(name)
    }
    this.syms.set(node.name, { dims: node.dims, cells })
    this.layout(node, 'latent', 'gaussian', node.dims.length === 0 ? 'scalar' : 'elements', vars, true)
  }

  /**
   * A Gaussian vector (read as a vector by a matmul) whose coefficients have the precisions `s_j lambda + o_j` of one shared
   * gamma variable (a smoothing parameter): one `gaussian-mv` variable with one `gaussian-mv-precision-scale` factor, so the
   * posterior keeps the dense correlations of the coefficients and the multiplier is inferred from them.
   */
  private latentSharedPrecision(node: ModelNode & { readonly kind: 'latent' }, loc: readonly Cell[]): void {
    if (!loc.every((c) => isConstCell(c) && c.aff.offset === 0))
      throw new Refusal(
        'operator',
        `'${node.name}' has a shared random precision and a mean: the vector with a smoothing parameter must have a zero known mean`,
      )
    const taus = this.randomPrecisions(node)
    const shared = taus[0]?.variable
    if (shared === undefined || taus.some((t) => t.variable !== shared))
      throw new Refusal(
        'operator',
        `the precisions of '${node.name}' must be multiples of one gamma variable plus constants (a vector with several random precisions is the ARD form, one variable per coefficient)`,
      )
    this.need('gaussian-mv-precision-scale')
    const n = taus.length
    this.addVariable({ name: node.name, family: { kind: 'gaussian-mv', size: n } })
    this.addFactor(
      {
        operator: 'gaussian-mv-precision-scale',
        args: [node.name, shared],
        params: { scale: taus.map((t) => t.scale), offset: taus.map((t) => t.offset) },
      },
      node.name,
    )
    this.syms.set(node.name, {
      dims: node.dims,
      cells: Array.from({ length: n }, (_, j) => this.compAtom(node.name, j, n)),
    })
    this.layout(node, 'latent', 'gaussian-mv', 'packed', [node.name], true)
  }

  /**
   * A gamma latent whose rate is a known multiple of another gamma latent (`gamma(0.5, b)` with `b` gamma: the
   * Makalic-Schmidt half-Cauchy, or a Student-t's mixing precision): a `gamma-rate` factor per element, whose shape is known.
   */
  private latentGammaRate(node: ModelNode & { readonly kind: 'latent' }): void {
    const inverse = node.dist.family === 'inverseGamma'
    const shapes = this.constParam(node, 'concentration', 'the shape')
    const rate = node.dist.params[inverse ? 'scale' : 'rate'] as ExprSpec
    const cells = this.over(this.evalSym(rate), node.dims)
    const out: Cell[] = []
    const vars: string[] = []
    for (const [i, c] of cells.entries()) {
      const r = scaledOf(c)
      if (r === undefined || r.offset !== 0 || !(r.scale > 0))
        throw new Refusal(
          'operator',
          `the rate of '${node.name}' must be a known multiple of a gamma latent (the message rule integrates the gamma's rate); it is not`,
        )
      const name = this.addVariable({ name: this.elementName(node.name, node.dims, i), family: { kind: 'gamma' } })
      this.need('gamma-rate')
      this.addFactor(
        {
          operator: 'gamma-rate',
          args: [name, r.variable],
          params: { shape: shapes[i] as number, ...(r.scale === 1 ? {} : { scale: r.scale }) },
        },
        name,
      )
      out.push(inverse ? { kind: 'recip', variable: name, scale: 1 } : { kind: 'rv', variable: name, family: 'gamma' })
      vars.push(name)
    }
    this.syms.set(node.name, { dims: node.dims, cells: out })
    this.layout(
      node,
      'latent',
      'gamma',
      node.dims.length === 0 ? 'scalar' : 'elements',
      vars,
      true,
      inverse ? { reciprocal: true } : {},
    )
  }

  /** A Gaussian latent with a gamma precision: a `gaussian-gamma-precision` factor per element (the mean-field hierarchy of Bishop 2006, 10.2). */
  private latentRandomPrecision(node: ModelNode & { readonly kind: 'latent' }, loc: readonly Cell[]): void {
    if (this.packed.has(node.name))
      throw new Refusal(
        'operator',
        `'${node.name}' is read as a vector by a matmul and has a random scale: a joint Gaussian vector has one covariance, so its scale must be known`,
      )
    const taus = this.randomPrecisions(node)
    const cells: Cell[] = []
    const vars: string[] = []
    for (const [i, tau] of taus.entries()) {
      const name = this.addVariable({ name: this.elementName(node.name, node.dims, i), family: { kind: 'gaussian' } })
      const l = loc[i] as Cell
      const mean = isConstCell(l)
        ? { const: l.aff.offset }
        : (this.materialise(affOf(l), node.name, node.dims.length === 0 ? undefined : i) as string)
      this.addFactor(precisionFactor(name, mean, tau), name)
      cells.push(this.varAtom(name))
      vars.push(name)
    }
    this.syms.set(node.name, { dims: node.dims, cells })
    this.layout(node, 'latent', 'gaussian', node.dims.length === 0 ? 'scalar' : 'elements', vars, true)
  }

  /** The vector at one element of a const parameter that carries an event axis (a Dirichlet's concentrations). */
  private eventVector(nv: NamedValue, dims: readonly string[], idx: readonly number[]): number[] {
    const k = nv.value.shape[nv.value.shape.length - 1] as number
    let flat = 0
    nv.dims.forEach((d, axis) => {
      const at = dims.indexOf(d)
      if (at < 0) throw new Refusal('structure', `a parameter over '${d}' does not fit [${dims.join(', ')}]`)
      flat = flat * (nv.value.shape[axis] as number) + (idx[at] as number)
    })
    return toFlat(nv.value).slice(flat * k, (flat + 1) * k)
  }

  private latentDirichlet(node: ModelNode & { readonly kind: 'latent' }): void {
    const e = node.dist.params['concentration']
    if (e === undefined) throw new Refusal('structure', `'${node.name}': the concentration is missing`)
    let nv: NamedValue
    try {
      nv = this.evalNamed(e)
    } catch (err) {
      if (err instanceof Refusal)
        throw new Refusal('operator', `the concentration of '${node.name}' depends on a random value: it must be known`)
      throw err
    }
    const shape = this.shapeOf(node.dims)
    const cells: Cell[] = []
    const vars: string[] = []
    for (let i = 0; i < prod(shape); i++) {
      const alpha = this.eventVector(nv, node.dims, unflatten(i, shape))
      const name = this.addVariable({
        name: this.elementName(node.name, node.dims, i),
        family: { kind: 'dirichlet', size: alpha.length },
      })
      this.addFactor(dirichletPrior(name, alpha), name)
      cells.push({ kind: 'rv', variable: name, family: 'dirichlet' })
      vars.push(name)
    }
    this.syms.set(node.name, { dims: node.dims, cells })
    this.layout(node, 'latent', 'dirichlet', node.dims.length === 0 ? 'scalar' : 'elements', vars, true)
  }

  /**
   * A beta or Dirichlet latent whose concentrations are known multiples of one random positive value plus known offsets
   * (`beta(alpha, alpha)`, `beta(alpha, 2)`, `dirichlet(alpha * ones)`: the strength of the prior learned from the data) with `alpha`
   * a gamma latent or a log-normal one (carried on its log): a `dirichlet-concentration` factor per element, which is the
   * prior of the variable (a numeric rule: a grid over `log alpha`).
   */
  private latentRandomConcentration(node: ModelNode & { readonly kind: 'latent' }): void {
    const isBeta = node.dist.family === 'beta'
    const shape = this.shapeOf(node.dims)
    const n = prod(shape)
    // The concentration of each element: one cell per component (a beta has two parameters, a Dirichlet an event axis).
    let perElement: ConcentrationPart[][]
    if (isBeta) {
      const a = this.over(this.evalSym(node.dist.params['a'] as ExprSpec), node.dims)
      const b = this.over(this.evalSym(node.dist.params['b'] as ExprSpec), node.dims)
      perElement = Array.from({ length: n }, (_, i) => [
        this.concentrationOf(node, a[i] as Cell),
        this.concentrationOf(node, b[i] as Cell),
      ])
    } else {
      // `concentration = S * V + O` with `S` random (a gamma or log-normal latent, a known multiple of one plus a known value) and `V`, `O`
      // known vectors over the event axis: the cell of each component is `V_k S + O_k`.
      const parsed = this.concentrationForm(node.dist.params['concentration'] as ExprSpec)
      if (parsed === undefined)
        throw new Refusal(
          'operator',
          `the concentration of '${node.name}' is not a random value times a known vector plus a known vector`,
        )
      const scalar = this.over(this.evalSym(parsed.random), node.dims)
      perElement = scalar.map((s) => {
        const base = this.concentrationOf(node, s)
        return parsed.scale.map(
          (v, k): ConcentrationPart => ({
            ...base,
            weight: v * base.weight,
            offset: v * base.offset + (parsed.offset[k] as number),
          }),
        )
      })
    }
    const cells: Cell[] = []
    const vars: string[] = []
    this.need('dirichlet-concentration')
    for (const [i, parts] of perElement.entries()) {
      const comps = parts
      const random = parts.filter((p) => p.variable !== undefined)
      const variable = random[0]?.variable
      const log = random[0]?.log === true
      // Two separate random shapes, each exactly a gamma or log-normal latent: the two-dimensional rule.
      const [pa, pb] = comps
      if (
        isBeta &&
        pa?.variable !== undefined &&
        pb?.variable !== undefined &&
        pa.variable !== pb.variable &&
        [pa, pb].every((q) => q.weight === 1 && q.offset === 0)
      ) {
        this.need('beta-two-shapes')
        const name = this.addVariable({ name: this.elementName(node.name, node.dims, i), family: { kind: 'beta' } })
        this.addFactor({ operator: 'beta-two-shapes', args: [name, pa.variable, pb.variable] }, name)
        cells.push({ kind: 'rv', variable: name, family: 'beta' })
        vars.push(name)
        continue
      }
      if (variable === undefined || random.some((p) => p.variable !== variable || (p.log === true) !== log))
        throw new Refusal(
          'operator',
          `the concentration of '${node.name}' must be known multiples of one random positive value (a gamma or log-normal latent) plus known offsets`,
        )
      const family: FamilyKind = isBeta ? 'beta' : 'dirichlet'
      const name = this.addVariable({
        name: this.elementName(node.name, node.dims, i),
        family: isBeta ? { kind: 'beta' } : { kind: 'dirichlet', size: comps.length },
      })
      this.addFactor(
        {
          operator: 'dirichlet-concentration',
          args: [name, variable],
          params: {
            weights: parts.map((p) => p.weight),
            offsets: parts.map((p) => p.offset),
            ...(log ? { log: true } : {}),
          },
        },
        name,
      )
      cells.push({ kind: 'rv', variable: name, family })
      vars.push(name)
    }
    this.syms.set(node.name, { dims: node.dims, cells })
    this.layout(
      node,
      'latent',
      isBeta ? 'beta' : 'dirichlet',
      node.dims.length === 0 ? 'scalar' : 'elements',
      vars,
      true,
    )
  }

  /** `S * V + O` for known vectors `V` and `O` over an event axis and a random scalar expression `S`: the form of a Dirichlet's random concentration. */
  private concentrationForm(
    e: ExprSpec,
  ): { readonly random: ExprSpec; readonly scale: readonly number[]; readonly offset: readonly number[] } | undefined {
    const u = this.unwrap(e)
    const vector = (x: ExprSpec): number[] | undefined => {
      if (!this.isConst(x)) return undefined
      const nv = this.evalNamed(x)
      return nv.value.shape.length > nv.dims.length ? toFlat(nv.value) : undefined
    }
    if (u.kind !== 'op') return undefined
    const [a, b] = u.args as [ExprSpec, ExprSpec]
    if (u.op === 'mul') {
      const va = vector(a)
      const vb = vector(b)
      if (va !== undefined && vb === undefined) return { random: b, scale: va, offset: va.map(() => 0) }
      if (vb !== undefined && va === undefined) return { random: a, scale: vb, offset: vb.map(() => 0) }
      return undefined
    }
    if (u.op === 'add') {
      const va = vector(a)
      const vb = vector(b)
      const [inner, o] = va !== undefined ? [b, va] : vb !== undefined ? [a, vb] : [undefined, undefined]
      if (inner === undefined || o === undefined) return undefined
      const base = this.concentrationForm(inner)
      return base === undefined ? undefined : { ...base, offset: base.offset.map((v, k) => v + (o[k] as number)) }
    }
    return undefined
  }

  /**
   * One component of a concentration as `weight * alpha + offset`: a known value, a gamma variable (or a known multiple of one
   * plus an offset), or a positive latent carried on its log (`alpha = exp(l)`, the variable returned is `l`, with `log`).
   */
  private concentrationOf(node: ModelNode, c: Cell): ConcentrationPart {
    if (isConstCell(c)) return { weight: 0, offset: c.aff.offset }
    const g = scaledOf(c)
    if (g !== undefined) return { variable: g.variable, weight: g.scale, offset: g.offset }
    if (c.kind === 'aff' && c.aff.terms.size === 1) {
      const [key, coefficient] = [...c.aff.terms][0] as [string, number]
      const atom = this.atoms.get(key)
      const l = atom?.kind === 'var' ? this.logScale.get(atom.name) : undefined
      if (l !== undefined) return { variable: l, log: true, weight: coefficient, offset: c.aff.offset }
    }
    throw new Refusal(
      'operator',
      `a concentration of '${node.name}' is not a known multiple of a gamma or log-normal latent plus a known value`,
    )
  }

  /** The variable of a scalar discrete latent. */
  private rvName(site: string): string {
    const cell = this.syms.get(site)?.cells[0]
    if (cell?.kind !== 'rv') throw new Refusal('structure', `'${site}' has no discrete variable`)
    return cell.variable
  }

  /** The discrete latents (Bernoulli) a node's parameters depend on, through derived nodes. */
  private discreteParents(node: ModelNode): string[] {
    const found: string[] = []
    const seen = new Set<string>()
    const visit = (name: string): void => {
      if (seen.has(name)) return
      seen.add(name)
      const n = this.nodes.get(name)
      if (n === undefined) return
      if (n.kind === 'latent' && n.dist.family === 'bernoulli' && this.syms.has(name)) found.push(name)
      else if (n.kind === 'derived') for (const r of nodeRefs(n)) visit(r)
    }
    for (const r of nodeRefs(node)) visit(r)
    return found
  }

  /** The probability of 1 of a Bernoulli node as a function of its discrete parents, by enumeration, as a table over (parents..., node). */
  private bernoulliTable(node: ModelNode & { readonly dist: DistSpec }, parents: readonly string[]): number[] {
    const e = node.dist.params['probs'] ?? node.dist.params['logits']
    if (e === undefined) throw new Refusal('structure', `'${node.name}': the probability is missing`)
    const logits = node.dist.params['probs'] === undefined
    const values: number[] = []
    for (let a = 0; a < 2 ** parents.length; a++) {
      const override = new Map<string, NamedValue>()
      parents.forEach((p, k) => {
        const bit = (a >> (parents.length - 1 - k)) & 1
        override.set(p, { value: asTensor(bit, 'float64'), dims: [] })
      })
      const raw = toFlat(this.evalNamed(e, override).value)[0] as number
      const p1 = logits ? 1 / (1 + Math.exp(-raw)) : raw
      if (!(p1 >= 0 && p1 <= 1)) throw new Refusal('data', `'${node.name}': a probability of ${p1} is outside [0, 1]`)
      values.push(1 - p1, p1)
    }
    return values
  }

  /**
   * A categorical latent: a `discrete` variable per element with a known probability vector (a `prior` factor), or with a
   * Dirichlet latent as its probabilities (the conjugate `discrete` factor, a mixture weight). Its role in a model is as
   * the gate of `at(components, z)` or the index of a conjugate observation.
   */
  private latentCategorical(node: ModelNode & { readonly kind: 'latent' }): void {
    const shape = this.shapeOf(node.dims)
    const n = prod(shape)
    const probsExpr = node.dist.params['probs']
    const logitsExpr = node.dist.params['logits']
    const expr = probsExpr ?? logitsExpr
    if (expr === undefined) throw new Refusal('structure', `'${node.name}': give probs or logits`)
    const cells: Cell[] = []
    const vars: string[] = []
    const chain = logitsExpr === undefined && !this.isConst(expr) ? this.transitionRow(expr) : undefined
    if (chain !== undefined) {
      // The probabilities are a known row of a matrix chosen by another discrete variable (a Markov chain's transition): a table over the two.
      const parents = this.over(this.evalSym(chain.index), node.dims)
      const matrix = toFlat(this.evalNamed(chain.matrix).value)
      for (const [i, p] of parents.entries()) {
        if (p.kind !== 'rv' || (p.family !== 'discrete' && p.family !== 'bernoulli'))
          throw new Refusal(
            'operator',
            `the row of '${node.name}' must be chosen by a discrete latent, not a computed value`,
          )
        const from =
          p.family === 'bernoulli'
            ? 2
            : ((this.variables.find((v) => v.name === p.variable)?.family.size ?? 2) as number)
        const to = matrix.length / from
        if (!Number.isInteger(to) || to < 1)
          throw new Refusal(
            'data',
            `the transition matrix of '${node.name}' has ${matrix.length} entries for ${from} states`,
          )
        const name = this.addVariable({
          name: this.elementName(node.name, node.dims, i),
          family: { kind: 'discrete', size: to },
        })
        this.need('table')
        this.addFactor(
          { operator: 'table', args: [p.variable, name], params: { states: [from, to], values: Array.from(matrix) } },
          name,
        )
        cells.push({ kind: 'rv', variable: name, family: 'discrete' })
        vars.push(name)
      }
    } else if (!this.isConst(expr) && this.randomTransitionRow(expr) !== undefined) {
      // The rows of the matrix are Dirichlet latents (a transition matrix to learn), chosen by another discrete variable: one
      // `gate` per element, whose state is the previous state and whose branch `j` is the conjugate `discrete` factor of this
      // state on row `j`. Each row then hears only the transitions out of its state, weighted by the belief that the chain was there.
      const chosen = this.randomTransitionRow(expr) as { readonly matrix: ExprSpec; readonly index: ExprSpec }
      const matrix = this.evalSym(chosen.matrix)
      const rows = matrix.cells
      const parents = this.over(this.evalSym(chosen.index), node.dims)
      if (matrix.dims.length !== 1 || rows.some((r) => r.kind !== 'rv' || r.family !== 'dirichlet'))
        throw new Refusal(
          'operator',
          `the rows of the matrix behind '${node.name}' must be one Dirichlet latent each (a latent over the dimension of the states)`,
        )
      const rowVars = rows.map((r) => (r as { readonly variable: string }).variable)
      const size = (this.variables.find((v) => v.name === rowVars[0])?.family.size ?? 2) as number
      for (const [i, p] of parents.entries()) {
        if (p.kind !== 'rv' || (p.family !== 'discrete' && p.family !== 'bernoulli'))
          throw new Refusal(
            'operator',
            `the row of '${node.name}' must be chosen by a discrete latent, not a computed value`,
          )
        const from =
          p.family === 'bernoulli'
            ? 2
            : ((this.variables.find((v) => v.name === p.variable)?.family.size ?? 2) as number)
        if (from !== rowVars.length)
          throw new Refusal('data', `'${node.name}' has ${from} previous states and its matrix ${rowVars.length} rows`)
        const name = this.addVariable({
          name: this.elementName(node.name, node.dims, i),
          family: { kind: 'discrete', size },
        })
        this.need('gate')
        this.need('discrete')
        const branches = rowVars.map((_, j) => ({ operator: 'discrete', args: [0, 1 + j], params: {} }))
        this.addFactor(
          { operator: 'gate', args: [p.variable, name, ...rowVars], params: { arity: 1 + rowVars.length, branches } },
          name,
        )
        cells.push({ kind: 'rv', variable: name, family: 'discrete' })
        vars.push(name)
      }
    } else if (!this.isConst(expr)) {
      if (logitsExpr !== undefined)
        throw new Refusal(
          'operator',
          `the logits of '${node.name}' depend on a random value: only probs from a Dirichlet latent map`,
        )
      const over = this.over(this.evalSym(expr), node.dims)
      for (const [i, c] of over.entries()) {
        if (c.kind === 'sticks') {
          // The weights are the stick-breaking of beta breaks: a `stick-breaking` factor on the assignment and the breaks.
          const name = this.addVariable({
            name: this.elementName(node.name, node.dims, i),
            family: { kind: 'discrete', size: c.variables.length + 1 },
          })
          this.need('stick-breaking')
          this.addFactor(
            { operator: 'stick-breaking', args: [name, ...c.variables], params: { sticks: c.variables.length } },
            name,
          )
          cells.push({ kind: 'rv', variable: name, family: 'discrete' })
          vars.push(name)
          continue
        }
        if (c.kind !== 'rv' || c.family !== 'dirichlet')
          throw new Refusal(
            'operator',
            `the probabilities of '${node.name}' must be known or a Dirichlet latent for the conjugate 'discrete' operator`,
          )
        const size = (this.variables.find((v) => v.name === c.variable)?.family.size ?? 2) as number
        const name = this.addVariable({
          name: this.elementName(node.name, node.dims, i),
          family: { kind: 'discrete', size },
        })
        this.addFactor({ operator: 'discrete', args: [name, c.variable] }, name)
        cells.push({ kind: 'rv', variable: name, family: 'discrete' })
        vars.push(name)
      }
    } else {
      const nv = this.evalNamed(expr)
      for (let i = 0; i < n; i++) {
        const raw = this.eventVector(nv, node.dims, unflatten(i, shape))
        const probs =
          logitsExpr === undefined
            ? raw
            : (() => {
                const top = Math.max(...raw)
                const e = raw.map((v) => Math.exp(v - top))
                const t = e.reduce((a, b) => a + b, 0)
                return e.map((v) => v / t)
              })()
        const name = this.addVariable({
          name: this.elementName(node.name, node.dims, i),
          family: { kind: 'discrete', size: probs.length },
        })
        this.addFactor(discretePrior(name, probs), name)
        cells.push({ kind: 'rv', variable: name, family: 'discrete' })
        vars.push(name)
      }
    }
    this.syms.set(node.name, { dims: node.dims, cells })
    this.layout(node, 'latent', 'discrete', node.dims.length === 0 ? 'scalar' : 'elements', vars, true)
  }

  /**
   * The matrix and the index of `at(A, z)` when it is the probability row of a categorical: `A` known (rows are the states
   * of the index, columns the outcomes), the index an expression of discrete latents. Absent for any other expression.
   */
  private transitionRow(expr: ExprSpec): { readonly matrix: ExprSpec; readonly index: ExprSpec } | undefined {
    const e = this.unwrap(expr)
    if (e.kind !== 'op' || e.op !== 'at') return undefined
    const [matrix, index] = e.args as [ExprSpec, ExprSpec]
    return this.isConst(matrix) && !this.isConst(index) ? { matrix, index } : undefined
  }

  /**
   * The matrix and the index of `at(A, z)` when it is the probability row of a categorical and the matrix is random (its
   * rows are latents), the index an expression of discrete latents. Absent for any other expression.
   */
  private randomTransitionRow(expr: ExprSpec): { readonly matrix: ExprSpec; readonly index: ExprSpec } | undefined {
    const e = this.unwrap(expr)
    if (e.kind !== 'op' || e.op !== 'at') return undefined
    const [matrix, index] = e.args as [ExprSpec, ExprSpec]
    return !this.isConst(matrix) && !this.isConst(index) ? { matrix, index } : undefined
  }

  private latentDiscrete(node: ModelNode & { readonly kind: 'latent' }): void {
    if (node.dist.family === 'categorical') {
      this.latentCategorical(node)
      return
    }
    const parents = this.discreteParents(node)
    if (node.dims.length > 0 && parents.length > 0)
      throw new Refusal(
        'structure',
        `'${node.name}' is a Bernoulli over [${node.dims}] with discrete parents: tables over plated nodes are not expanded yet`,
      )
    const shape = this.shapeOf(node.dims)
    const n = prod(shape)
    const cells: Cell[] = []
    const vars: string[] = []
    const probsExpr = node.dist.params['probs']
    if (parents.length === 0 && probsExpr !== undefined && !this.isConst(probsExpr)) {
      // A Bernoulli whose probability is a beta latent (an unknown mixing weight, a gate's prior): the conjugate `bernoulli` factor,
      // whose message to the probability is the moment-matched mixture of the two conjugate updates.
      this.need('bernoulli')
      const over = this.over(this.evalSym(probsExpr), node.dims)
      for (const [i, c] of over.entries()) {
        if (c.kind !== 'rv' || c.family !== 'beta')
          throw new Refusal(
            'operator',
            `the probability of '${node.name}' must be known or a beta latent for the conjugate 'bernoulli' operator`,
          )
        const name = this.addVariable({
          name: this.elementName(node.name, node.dims, i),
          family: { kind: 'bernoulli' },
        })
        this.addFactor({ operator: 'bernoulli', args: [name, c.variable] }, name)
        cells.push({ kind: 'rv', variable: name, family: 'bernoulli' })
        vars.push(name)
      }
    } else if (parents.length === 0) {
      const probs = this.constParam(
        node,
        node.dist.params['probs'] === undefined ? 'logits' : 'probs',
        'the probability',
      )
      const p = node.dist.params['probs'] === undefined ? probs.map((l) => 1 / (1 + Math.exp(-l))) : probs
      for (let i = 0; i < n; i++) {
        const name = this.addVariable({
          name: this.elementName(node.name, node.dims, i),
          family: { kind: 'bernoulli' },
        })
        this.addFactor({ operator: 'prior', args: [name], params: { eta: [...bernoulliEta(p[i] as number)] } }, name)
        cells.push({ kind: 'rv', variable: name, family: 'bernoulli' })
        vars.push(name)
      }
    } else {
      const name = this.addVariable({ name: node.name, family: { kind: 'bernoulli' } })
      const pv = parents.map((p) => this.rvName(p))
      this.addFactor(
        {
          operator: 'table',
          args: [...pv, name],
          params: { states: new Array<number>(parents.length + 1).fill(2), values: this.bernoulliTable(node, parents) },
        },
        name,
      )
      cells.push({ kind: 'rv', variable: name, family: 'bernoulli' })
      vars.push(name)
    }
    this.syms.set(node.name, { dims: node.dims, cells })
    this.layout(node, 'latent', 'bernoulli', node.dims.length === 0 ? 'scalar' : 'elements', vars, true)
  }

  private derived(node: ModelNode & { readonly kind: 'derived' }): void {
    const absorbed = this.nonCentred.get(node.name)
    if (absorbed !== undefined) {
      if (this.denseNonCentred(node)) return
      // The dense factor does not apply to this divisor: the standard normals are compiled as the scalar variables they would have been.
      const z = this.nodes.get(absorbed) as ModelNode & { readonly kind: 'latent' }
      this.nonCentred.delete(node.name)
      this.current = z.name
      this.latent(z)
      this.current = node.name
    }
    if (this.discreteParents(node).length > 0) {
      this.lazy.set(node.name, node.expr)
      return
    }
    const sym = this.evalSym(node.expr)
    const shape = this.shapeOf(node.dims)
    const cells = this.over(sym, node.dims)
    if (cells.every(isConstCell)) {
      this.consts.set(node.name, this.toNamed({ dims: node.dims, cells }))
      this.layout(node, 'derived', undefined, 'constant', new Array<null>(prod(shape)).fill(null), false)
      return
    }
    // Stick-breaking weights are not a random variable of the graph (the assignments read the breaks through one factor), so there is
    // nothing to record: the weights are read back from the breaks.
    if (!node.record || cells.some((c) => c.kind === 'sticks')) {
      this.syms.set(node.name, { dims: node.dims, cells })
      return
    }
    const vars: (string | null)[] = []
    const out: Cell[] = cells.map((c, i) => {
      if (c.kind === 'rv') {
        vars.push(c.variable)
        return c
      }
      if (c.kind === 'sticks') throw new Refusal('structure', `'${node.name}' is a stick-breaking weight vector`)
      if (c.kind === 'scaled' || c.kind === 'lookup' || c.kind === 'recip')
        throw new Refusal(
          'structure',
          `'${node.name}' is a ${c.kind === 'scaled' ? 'scaled precision' : c.kind === 'recip' ? 'multiple of an inverse-gamma variance' : 'lookup by a discrete variable'}: it cannot be recorded as a value`,
        )
      const name = this.elementName(node.name, node.dims, i)
      const v = this.materialise(c.aff, node.name, node.dims.length === 0 ? undefined : i, name)
      if (typeof v === 'number') {
        vars.push(null)
        return c
      }
      vars.push(v)
      return this.varAtom(v)
    })
    this.syms.set(node.name, { dims: node.dims, cells: out })
    this.layout(node, 'derived', 'gaussian', node.dims.length === 0 ? 'scalar' : 'elements', vars, true)
  }

  /**
   * `D = z / S(x)` as one Gaussian vector variable `D` and a `gaussian-mv-function-scale` factor with the variable `x` the divisor
   * is a chain of unary steps of (`1 / sqrt(exp(x) d_j + n_j)`: each coefficient's chain, a known divisor a constant). The standard
   * normals `z` never become variables: given `x` the vector is Gaussian with the diagonal covariance `1 / S_j(x)^2`, which the
   * factor integrates against the cavity of the whole vector. False (and nothing made) when the divisors are not chains of one variable.
   */
  private denseNonCentred(node: ModelNode & { readonly kind: 'derived' }): boolean {
    const divisor = node.expr.kind === 'op' ? node.expr.args[1] : undefined
    if (divisor === undefined) return false
    let cells: Cell[]
    try {
      cells = this.over(this.evalSym(divisor), node.dims)
    } catch (e) {
      if (e instanceof Refusal) return false
      throw e
    }
    let input: string | undefined
    const chains: ApplyStep[][] = []
    for (const cell of cells) {
      if (isConstCell(cell)) {
        const s = cell.aff.offset
        if (!(s > 0 && Number.isFinite(s))) return false
        chains.push([
          ['mul', 0],
          ['add', 1 / s],
        ])
        continue
      }
      if (cell.kind !== 'aff') return false
      const terms = [...cell.aff.terms].filter(([, c]) => c !== 0)
      const atom = terms.length === 1 ? this.atoms.get(terms[0]?.[0] as string) : undefined
      if (atom?.kind !== 'var' || !this.applied.has(atom.name)) return false
      const base = this.chainOf(cell.aff)
      if (input !== undefined && input !== base.input) return false
      input = base.input
      chains.push([...base.steps, ['recip']])
    }
    if (input === undefined) return false
    const n = cells.length
    this.need('gaussian-mv-function-scale')
    this.addVariable({ name: node.name, family: { kind: 'gaussian-mv', size: n } })
    this.addFactor(
      {
        operator: 'gaussian-mv-function-scale',
        args: [node.name, input],
        params: { steps: chains as unknown as Json },
      },
      node.name,
    )
    this.syms.set(node.name, {
      dims: node.dims,
      cells: Array.from({ length: n }, (_, j) => this.compAtom(node.name, j, n)),
    })
    if (node.record) this.layout(node, 'derived', 'gaussian-mv', 'packed', [node.name], true)
    return true
  }

  /** The expression a derived node stands for, followed through references to derived nodes (to see a link applied to it). */
  private unwrap(e: ExprSpec): ExprSpec {
    let cur = e
    for (let k = 0; k < 64 && cur.kind === 'ref'; k++) {
      const n = this.nodes.get(cur.name)
      if (n?.kind !== 'derived') break
      cur = n.expr
    }
    return cur
  }

  /**
   * A Gaussian observation of a value that is a known function of a discrete variable (a reading moved a period early or late by a
   * copy index): `N(v; m, s)` is `N(0; m - v, s)`, so the observed value moves across to the mean, which then is a gate over the
   * shifted forms. Undefined when the observed value is known, or the family is not the normal.
   */
  private valueMovedToMean(
    node: ModelNode & { readonly kind: 'observed' },
  ): (ModelNode & { readonly kind: 'observed' }) | undefined {
    const loc = node.dist.params['loc']
    if (node.dist.family !== 'normal' || loc === undefined || this.isConst(node.value)) return undefined
    if (!this.mentionsDiscrete(node.value)) return undefined
    const fused = this.gatedMean(node, loc)
    if (fused !== undefined) return { ...node, value: { kind: 'const', value: 0 } }
    return {
      ...node,
      dist: { ...node.dist, params: { ...node.dist.params, loc: { kind: 'op', op: 'sub', args: [loc, node.value] } } },
      value: { kind: 'const', value: 0 },
    }
  }

  /**
   * The mean `at(table, z)` of an observation whose value is a known function of the same discrete variable `z`: one gate over the
   * table's entries each moved by what the value is in that state (`table[k] - value[k]`), so the observed value becomes zero and
   * the per-state shifts sit in the one gate (a gate of the shifted forms, not a gate of the mean followed by another of the
   * value). Stores the cells of the mean for the observation to read; false when the mean is not of that form.
   */
  private gatedMean(node: ModelNode & { readonly kind: 'observed' }, loc: ExprSpec): boolean {
    const e = this.unwrap(loc)
    if (e.kind !== 'op' || e.op !== 'at') return false
    const [first, index] = e.args as [ExprSpec, ExprSpec]
    const table = this.evalSym(first)
    if (table.dims.length !== 1) return false
    const values = this.over(this.evalSym(node.value), node.dims)
    const picks = this.over(this.evalSym(index), node.dims)
    const cells: Cell[] = []
    for (const [i, v] of values.entries()) {
      const pick = picks[i] as Cell
      if (v.kind !== 'lookup' || v.items !== undefined || pick.kind !== 'rv' || pick.variable !== v.variable)
        return false
      if (table.cells.length !== v.values.length || !table.cells.every((c) => c.kind === 'aff')) return false
      const shifted = table.cells.map(
        (c, k): Cell => ({
          kind: 'aff',
          aff: combine(affOf(c), constAff(v.values[k] as number), -1),
        }),
      )
      cells.push(this.gate(shifted, v.variable))
    }
    this.cells.set(`${node.name}|loc`, cells)
    return true
  }

  private observed(node: ModelNode & { readonly kind: 'observed' }): void {
    const moved = this.valueMovedToMean(node)
    if (moved !== undefined) {
      this.observed(moved)
      return
    }
    const fam = node.dist.family
    const shape = this.shapeOf(node.dims)
    const n = prod(shape)
    const info = this.bound.missing[node.name]
    if (info !== undefined && info.count > 0 && info.policy !== 'ignore')
      throw new Refusal(
        'data',
        `'${node.name}' has ${info.count} missing value(s) with the policy '${info.policy}': only 'ignore' is mapped (the entries drop out of the graph)`,
      )
    if (this.bound.batch !== undefined && node.dims.includes(this.bound.batch.dim))
      throw new Refusal(
        'structure',
        `'${node.name}' is subsampled by the batch option: a factor graph is built over all the rows`,
      )
    const values = this.over(this.fromNamed(this.evalNamedValue(node)), node.dims).map(
      (c) => (c as Cell & { kind: 'aff' }).aff.offset,
    )
    const normalGamma = this.normalGammaObserved.get(node.name)
    if (normalGamma !== undefined) {
      this.observedNormalGamma(node, normalGamma, values)
      return
    }
    const vars: (string | null)[] = []
    const discreteParents = this.discreteParents(node)
    if (this.vectorise && node.dims.length > 0 && discreteParents.length === 0 && this.plate(node, fam, values)) return
    if (fam === 'bernoulli' && discreteParents.length > 0) {
      if (node.dims.length > 0)
        throw new Refusal(
          'structure',
          `'${node.name}' is a Bernoulli over [${node.dims}] with discrete parents: tables over plated nodes are not expanded yet`,
        )
      const name = this.addVariable({ name: node.name, family: { kind: 'bernoulli' }, observed: values[0] as number })
      const pv = discreteParents.map((p) => this.rvName(p))
      this.addFactor(
        {
          operator: 'table',
          args: [...pv, name],
          params: {
            states: new Array<number>(pv.length + 1).fill(2),
            values: this.bernoulliTable(node, discreteParents),
          },
        },
        name,
      )
      this.layout(node, 'observed', 'bernoulli', 'scalar', [name], false)
      return
    }
    for (let i = 0; i < n; i++) {
      const y = values[i] as number
      if (Number.isNaN(y)) {
        vars.push(null)
        continue
      }
      const name = this.elementName(node.name, node.dims, i)
      vars.push(this.observedElement(node, fam, name, y, i))
    }
    this.layout(node, 'observed', undefined, node.dims.length === 0 ? 'scalar' : 'observed', vars, false)
  }

  private evalNamedValue(node: ModelNode & { readonly kind: 'observed' }): NamedValue {
    try {
      return this.evalNamed(node.value)
    } catch (e) {
      if (e instanceof Refusal && e.kind === 'node')
        throw new Refusal('node', `the value observed by '${node.name}' is random: ${e.message}`)
      throw e
    }
  }

  private readonly cells = new Map<string, Cell[]>()

  /** The cells of a distribution parameter or a link's argument of an observed node over its dimensions, made once. */
  private cellsOf(node: ModelNode, key: string, make: () => Sym): Cell[] {
    const id = `${node.name}|${key}`
    let cells = this.cells.get(id)
    if (cells === undefined) {
      cells = this.over(make(), node.dims)
      this.cells.set(id, cells)
    }
    return cells
  }

  /** The cell of a parameter of an observed node at element `i`. */
  private paramCell(node: ModelNode & { readonly kind: 'observed' }, name: string, i: number): Cell {
    return this.cellsOf(node, name, () => this.paramSym(node.dist, name))[i] as Cell
  }

  private observedElement(
    node: ModelNode & { readonly kind: 'observed' },
    fam: string,
    name: string,
    y: number,
    i: number,
  ): string | null {
    const index = node.dims.length === 0 ? undefined : i
    switch (fam) {
      case 'normal': {
        const scaleExpr = node.dist.params['scale']
        // A log scale (and a mean) chosen by one discrete variable, each state with random ones of its own: one gate of `gaussian-scale` branches.
        if (this.observedSwitchLogScale(node, y, i, index)) return null
        // A mean or a scale chosen by a discrete variable (a lookup): a table or a gate over that variable.
        const switched = this.switchedCells(node, i)
        if (switched !== undefined) return this.observedSwitch(node, name, y, index, switched.loc, switched.scale)
        if (scaleExpr !== undefined && !this.isConst(scaleExpr)) {
          const e = this.unwrap(scaleExpr)
          const logScale =
            e.kind === 'op' && e.op === 'exp' ? this.observedLogScale(node, name, y, i, index) : undefined
          if (logScale !== undefined) return logScale
          if (this.precisionExpr(scaleExpr) === undefined)
            return this.observedRandomScale(node, name, y, i, index, scaleExpr)
          return this.observedRandomPrecision(node, name, y, i, index)
        }
        return this.observedGaussian(node, name, y, i, index, 'loc', 'scale')
      }
      case 'studentT':
      case 'laplace':
      case 'normalOutliers':
        return this.observedHeavy(node, fam, y, i, index)
      case 'beta':
        return this.observedPair(node, 'beta-shapes', y, i, index)
      case 'orderedProbit':
      case 'orderedLogistic':
      case 'asymmetricLaplace': {
        const numeric = this.observedNumeric(node, fam, y, i, index)
        if (numeric === undefined)
          throw new Refusal(
            'operator',
            `the parameters of '${node.name}' (${fam}) are not in a form with a message rule`,
          )
        return numeric
      }
      case 'logNormal':
        // log y is Gaussian around the predictor: the exact Gaussian rule on the log of the outcome.
        if (!(y > 0)) throw new Refusal('data', `'${node.name}' is log-normal: the outcome ${y} must be positive`)
        return this.observedGaussian(node, name, Math.log(y), i, index, 'mu', 'sigma')
      case 'bernoulli':
        return this.observedBernoulli(node, name, y, i, index)
      case 'binomial':
      case 'poisson': {
        const numeric = this.observedNumeric(node, fam, y, i, index)
        return numeric !== undefined ? numeric : this.observedConjugate(node, fam, name, y, i)
      }
      case 'categorical':
        if (node.dist.params['logits'] !== undefined)
          return this.observedMulticlass(node, y, i, index, 'logits', 'logit')
        return this.observedConjugate(node, fam, name, y, i)
      case 'categoricalProbit':
        return this.observedMulticlass(node, y, i, index, 'scores', 'probit')
      case 'zeroTruncatedPoisson': {
        const numeric = this.observedNumeric(node, fam, y, i, index)
        if (numeric === undefined)
          throw new Refusal(
            'operator',
            `the rate of '${node.name}' must be \`exposure * exp(affine)\` for the zero-truncated Poisson`,
          )
        return numeric
      }
      case 'hurdlePoisson':
        return this.observedHurdle(node, y, i, index)
      case 'normalPair':
        return this.observedMixtureWeight(node, name, y, i)
      case 'exponential':
      case 'gamma':
        return this.observedGammaRate(node, fam, name, y, i)
      case 'geometric':
        return this.observedFailures(node, fam, name, y, i)
      case 'negativeBinomial':
      case 'weibull': {
        const numeric = this.observedNumeric(node, fam, y, i, index)
        if (numeric === undefined && fam === 'negativeBinomial') return this.observedFailures(node, fam, name, y, i)
        if (numeric === undefined)
          throw new Refusal(
            'operator',
            `the parameters of '${node.name}' (${fam}) are not in a form with a message rule (see the log-link idioms of its observedNumeric)`,
          )
        return numeric
      }
      default:
        throw new Refusal(
          'family',
          `'${node.name}' is observed as ${fam}: the likelihoods mapped are normal, logNormal, studentT, laplace, asymmetricLaplace, orderedProbit, orderedLogistic, categoricalProbit, normalOutliers, bernoulli, binomial, poisson, normalPair, exponential, gamma, geometric, negativeBinomial, weibull and categorical`,
        )
    }
  }

  /** The value of a known expression at element `i` of an observed node. */
  private constAt(node: ModelNode, e: ExprSpec, i: number): number {
    return (this.over(this.fromNamed(this.evalNamed(e)), node.dims)[i] as Cell & { kind: 'aff' }).aff.offset
  }

  /** `exp(L)` or `k * exp(L)` (with `k` known), followed through derived nodes: the log-link idiom of a rate, mean or scale. */
  private scaledExp(
    e: ExprSpec,
    randomScale = false,
  ): { readonly inner: ExprSpec; readonly scale: ExprSpec | undefined } | undefined {
    const u = this.unwrap(e)
    // A positive latent carried on its log (a log-normal, half-normal or half-Cauchy rate or scale) is `exp` of that log.
    const logOf = (x: ExprSpec): ExprSpec | undefined =>
      x.kind === 'ref' && this.syms.has(logName(x.name)) ? { kind: 'ref', name: logName(x.name) } : undefined
    const direct = logOf(u)
    if (direct !== undefined) return { inner: direct, scale: undefined }
    if (u.kind !== 'op') return undefined
    if (u.op === 'exp') return { inner: u.args[0] as ExprSpec, scale: undefined }
    if (u.op === 'mul') {
      const [a, b] = u.args as [ExprSpec, ExprSpec]
      const ea = this.unwrap(a)
      const eb = this.unwrap(b)
      const la = logOf(ea)
      const lb = logOf(eb)
      // With `randomScale` the factor may be random too (a known value chosen by a discrete variable: a gate over the rate).
      if ((randomScale || this.isConst(a)) && lb !== undefined) return { inner: lb, scale: a }
      if ((randomScale || this.isConst(b)) && la !== undefined) return { inner: la, scale: b }
      if ((randomScale || this.isConst(a)) && eb.kind === 'op' && eb.op === 'exp')
        return { inner: eb.args[0] as ExprSpec, scale: a }
      if ((randomScale || this.isConst(b)) && ea.kind === 'op' && ea.op === 'exp')
        return { inner: ea.args[0] as ExprSpec, scale: b }
    }
    return undefined
  }

  /**
   * A Poisson count whose rate is `k_z exp(L)`: `L` an affine form of Gaussian variables and `k_z` a known exposure chosen by a
   * discrete variable `z` (zero for a state that is a structural zero: zero inflation). The state of `z` selects which
   * likelihood holds, so it is a `gate` whose branches are the Poisson likelihood of `L` (a state with exposure zero is the
   * constant one, a count of zero) with a `table` that rules out the states in which a positive count has probability zero; when
   * `L` is known the gate is a table over `z`. The weights of `z` may be known or a Dirichlet latent, as for any discrete variable.
   */
  private observedPoissonGate(
    node: ModelNode & { readonly kind: 'observed' },
    y: number,
    i: number,
    index: number | undefined,
    inner: ExprSpec,
    exposure: ExprSpec,
  ): null {
    const cell = this.cellsOf(node, 'exposure', () => this.evalSym(exposure))[i] as Cell
    if (cell.kind !== 'lookup' || cell.items !== undefined)
      throw new Refusal(
        'operator',
        `the exposure of '${node.name}' is random: the poisson likelihood has a message rule for a known exposure, or a known value chosen by a discrete variable (a gate)`,
      )
    const z = cell.variable
    const exposures = cell.values
    if (!exposures.every((e) => e >= 0 && Number.isFinite(e)))
      throw new Refusal('data', `the exposure of '${node.name}' must be non-negative`)
    const why = likelihoods()['poisson-log']?.check?.(y, { likelihood: 'poisson-log', y, exposure: 1 })
    if (why !== undefined) throw new Refusal('data', `'${node.name}' (poisson-log): ${why}`)
    const predictor = this.cellsOf(node, 'predictor', () => this.evalSym(inner))[i] as Cell
    const id = this.elementName(node.name, node.dims, i)
    const impossible = exposures.map((e) => e === 0 && y > 0)
    const logFactorial = Array.from({ length: y }, (_, k) => Math.log(k + 1)).reduce((a, b) => a + b, 0)
    if (isConstCell(predictor)) {
      // Nothing random but the state: the likelihood of the count under each state.
      const pmf = (e: number): number => {
        const rate = e * Math.exp(predictor.aff.offset)
        return rate === 0 ? (y === 0 ? 1 : 0) : Math.exp(-rate + y * Math.log(rate) - logFactorial)
      }
      this.addFactor(
        { operator: 'table', args: [z], params: { states: [exposures.length], values: exposures.map(pmf) } },
        id,
      )
      return null
    }
    const v = this.materialise(affOf(predictor), node.name, index) as string
    if (impossible.some(Boolean))
      this.addFactor(
        {
          operator: 'table',
          args: [z],
          params: { states: [exposures.length], values: impossible.map((bad) => (bad ? 0 : 1)) },
        },
        `${id}|support`,
      )
    const branches = exposures.map((e) =>
      e === 0 ? {} : { operator: 'likelihood-site', args: [0], params: { likelihood: 'poisson-log', y, exposure: e } },
    )
    this.addFactor({ operator: 'gate', args: [z, v], params: { arity: 1, branches } }, id)
    return null
  }

  /**
   * The negative binomial in the NB2 idiom with a random dispersion `a`: `n = 1 / a`, `probs = 1 / (1 + a * mean)` with `mean`
   * `exp(L)` (times a known exposure) and `a` a positive latent (carried on its log by `latentProjected`): the
   * `negative-binomial-dispersion` factor on the predictor `L` and the log of `a`. `undefined` when the parameters are not that
   * idiom or the dispersion is known (the registry's `negative-binomial-log` takes that).
   */
  private negBinomialRandomDispersion(
    node: ModelNode & { readonly kind: 'observed' },
    y: number,
    i: number,
    index: number | undefined,
  ): string | null | undefined {
    const dist = node.dist.params
    const nExpr = dist['n'] === undefined ? undefined : this.unwrap(dist['n'])
    const pExpr = dist['probs'] === undefined ? undefined : this.unwrap(dist['probs'])
    if (nExpr?.kind !== 'op' || nExpr.op !== 'div' || pExpr?.kind !== 'op' || pExpr.op !== 'div') return undefined
    const isOne = (e: ExprSpec): boolean => this.isConst(e) && this.constAt(node, e, i) === 1
    const [one, a] = nExpr.args as [ExprSpec, ExprSpec]
    if (!isOne(one) || this.isConst(a)) return undefined
    const [num, den] = pExpr.args as [ExprSpec, ExprSpec]
    const d = this.unwrap(den)
    if (!isOne(num) || d.kind !== 'op' || d.op !== 'add') return undefined
    const [p, q] = d.args as [ExprSpec, ExprSpec]
    const [unit, product] = this.isConst(p) ? [p, this.unwrap(q)] : [q, this.unwrap(p)]
    if (!isOne(unit) || product.kind !== 'op' || product.op !== 'mul') return undefined
    const same = (x: ExprSpec, z: ExprSpec): boolean =>
      JSON.stringify(this.unwrap(x)) === JSON.stringify(this.unwrap(z))
    const [m1, m2] = product.args as [ExprSpec, ExprSpec]
    const mean = same(m1, a) ? m2 : same(m2, a) ? m1 : undefined
    if (mean === undefined) return undefined
    const m = this.scaledExp(mean)
    if (m === undefined) return undefined
    const aCell = this.cellsOf(node, 'dispersion', () => this.evalSym(a))[i] as Cell
    const only =
      aCell.kind === 'aff' && aCell.aff.offset === 0 && aCell.aff.terms.size === 1 ? [...aCell.aff.terms] : []
    const atom = only.length === 1 && only[0]?.[1] === 1 ? this.atoms.get(only[0][0]) : undefined
    const logDispersion = atom?.kind === 'var' ? this.logScale.get(atom.name) : undefined
    if (logDispersion === undefined)
      throw new Refusal(
        'operator',
        `the dispersion of '${node.name}' is random but not a positive latent (a half-normal, half-Cauchy, log-normal or uniform prior carried on its log): the negative-binomial-dispersion factor needs the log of it`,
      )
    const params: Record<string, Json> = { y, exposure: m.scale === undefined ? 1 : this.constAt(node, m.scale, i) }
    if (!(Number.isInteger(y) && y >= 0))
      throw new Refusal('data', `'${node.name}' (negative binomial): the outcome must be a non-negative integer`)
    const cell = this.cellsOf(node, 'predictor', () => this.evalSym(m.inner))[i] as Cell
    if (isConstCell(cell)) return null
    const u = this.materialise(affOf(cell), node.name, index) as string
    this.addFactor(
      { operator: 'negative-binomial-dispersion', args: [u, logDispersion], params },
      this.elementName(node.name, node.dims, i),
    )
    return null
  }

  /**
   * The likelihoods of the registry of `message-passing/operators/likelihoods.ts` for `observe y ~ D(g(affine))`: Poisson with
   * `rate = exposure * exp(affine)`, binomial with `probs = sigmoid(affine)` or `probit(affine)`, negative binomial with a log
   * link and a known dispersion (`n = r`, `probs = r / (r + mean)`, the mean `exposure * exp(affine)`), and a Weibull with
   * `scale = exp(affine)` and a known shape. Returns `undefined` when the parameters are not that shape (the caller tries the
   * conjugate rule or refuses), else the factor's outcome (always `null`: the outcome is a parameter of the factor).
   */
  private observedNumeric(
    node: ModelNode & { readonly kind: 'observed' },
    fam: string,
    y: number,
    i: number,
    index: number | undefined,
  ): string | null | undefined {
    const dist = node.dist.params
    const known = (p: string, what: string): number => {
      let c: Cell
      try {
        c = this.paramCell(node, p, i)
      } catch (e) {
        // A parameter computed from a random value (1 / alpha with alpha a latent) is random.
        if (e instanceof Refusal && e.kind === 'operator')
          throw new Refusal(
            'operator',
            `the ${what} of '${node.name}' is random (${e.message}): the ${fam} likelihood has a message rule for a known ${what} only`,
          )
        throw e
      }
      if (!isConstCell(c))
        throw new Refusal(
          'operator',
          `the ${what} of '${node.name}' is random: the ${fam} likelihood has a message rule for a known ${what} only`,
        )
      return c.aff.offset
    }
    const emit = (likelihood: string, expr: ExprSpec, extra: Record<string, Json>): string | null => {
      const params: Record<string, Json> = { likelihood, y, ...extra }
      const why = likelihoods()[likelihood]?.check?.(y, params)
      if (why !== undefined) throw new Refusal('data', `'${node.name}' (${likelihood}): ${why}`)
      const cell = this.cellsOf(node, 'predictor', () => this.evalSym(expr))[i] as Cell
      return this.emitSite(node, cell, i, index, 'likelihood-row', 'likelihood-site', params)
    }
    if (fam === 'zeroTruncatedPoisson' || fam === 'hurdlePoisson') {
      const m = dist['rate'] === undefined ? undefined : this.scaledExp(dist['rate'])
      if (m === undefined) return undefined
      return emit('poisson-log-truncated', m.inner, {
        exposure: m.scale === undefined ? 1 : this.constAt(node, m.scale, i),
      })
    }
    if (fam === 'poisson') {
      const m = dist['rate'] === undefined ? undefined : this.scaledExp(dist['rate'], true)
      if (m === undefined) return undefined
      if (m.scale !== undefined && !this.isConst(m.scale))
        return this.observedPoissonGate(node, y, i, index, m.inner, m.scale)
      return emit('poisson-log', m.inner, { exposure: m.scale === undefined ? 1 : this.constAt(node, m.scale, i) })
    }
    if (fam === 'binomial') {
      const probs = dist['probs'] === undefined ? undefined : this.unwrap(dist['probs'])
      if (probs?.kind !== 'op' || (probs.op !== 'sigmoid' && probs.op !== 'probit')) return undefined
      return emit(probs.op === 'sigmoid' ? 'binomial-logit' : 'binomial-probit', probs.args[0] as ExprSpec, {
        n: known('n', 'number of trials'),
      })
    }
    if (fam === 'orderedProbit' || fam === 'orderedLogistic') {
      const cutsExpr = dist['cutpoints'] as ExprSpec
      if (!this.isConst(cutsExpr))
        throw new Refusal(
          'operator',
          `the cut points of '${node.name}' are random: the ordinal likelihoods have a message rule for known cut points only`,
        )
      const cuts = Array.from(toFlat(this.evalNamed(cutsExpr).value))
      return emit(fam === 'orderedProbit' ? 'ordinal-probit' : 'ordinal-logit', dist['predictor'] as ExprSpec, { cuts })
    }
    if (fam === 'asymmetricLaplace') {
      if (!isConstCell(this.paramCell(node, 'scale', i)))
        return this.observedPair(node, 'asymmetric-laplace-scale', y, i, index)
      return emit('asymmetric-laplace', dist['loc'] as ExprSpec, {
        scale: known('scale', 'scale'),
        tau: known('tau', 'quantile level'),
      })
    }
    if (fam === 'weibull') {
      const m = dist['scale'] === undefined ? undefined : this.scaledExp(dist['scale'])
      if (m === undefined || m.scale !== undefined) return undefined
      if (!isConstCell(this.paramCell(node, 'concentration', i)))
        return this.observedPair(node, 'weibull-shape', y, i, index)
      return emit('weibull-log', m.inner, { shape: known('concentration', 'shape') })
    }
    if (fam === 'negativeBinomial') {
      const random = this.negBinomialRandomDispersion(node, y, i, index)
      if (random !== undefined) return random
      // The negative binomial with a log link: n = r, probs = r / (r + mean) = A / (A + K exp(L)) with mean = n K exp(L) / A.
      const r = known('n', 'dispersion (the number of successes n = 1 / dispersion)')
      const probs = dist['probs'] === undefined ? undefined : this.unwrap(dist['probs'])
      if (probs?.kind !== 'op' || probs.op !== 'div') return undefined
      const [num, den] = probs.args as [ExprSpec, ExprSpec]
      const d = this.unwrap(den)
      if (!this.isConst(num) || d.kind !== 'op' || d.op !== 'add') return undefined
      const [p, q] = d.args as [ExprSpec, ExprSpec]
      const [one, rest] = this.isConst(p) ? [p, q] : [q, p]
      if (!this.isConst(one)) return undefined
      const A = this.constAt(node, num, i)
      if (Math.abs(this.constAt(node, one, i) - A) > 1e-12 * Math.abs(A)) return undefined
      const m = this.scaledExp(rest)
      if (m === undefined) return undefined
      const K = m.scale === undefined ? 1 : this.constAt(node, m.scale, i)
      return emit('negative-binomial-log', m.inner, { dispersion: 1 / r, exposure: (r * K) / A })
    }
    return undefined
  }

  /**
   * A categorical outcome whose class scores are affine forms of Gaussian variables over one extra dimension (the classes): a
   * `logits` expression `X W` with `W` over (feature, class) for the softmax, or `scores` for the multinomial probit. One
   * `multiclass-site` factor per observation on the `K` scores (each a Gaussian variable, or a known value). The outcome is a
   * parameter of the factor.
   */
  private observedMulticlass(
    node: ModelNode & { readonly kind: 'observed' },
    y: number,
    i: number,
    index: number | undefined,
    param: string,
    link: 'logit' | 'probit',
  ): string | null {
    const key = `${node.name}|${param}`
    let cells = this.cells.get(key)
    if (cells === undefined) {
      const sym = this.evalSym(node.dist.params[param] as ExprSpec)
      const extra = sym.dims.filter((dim) => (node.dist.eventDims ?? []).includes(dim))
      if (extra.length !== 1)
        throw new Refusal(
          'structure',
          `the ${param} of '${node.name}' must carry the class dimension declared by \`over\` (categorical(..., { over })), it has [${sym.dims}]`,
        )
      cells = this.over(sym, [...node.dims, extra[0] as string])
      this.cells.set(key, cells)
    }
    const classes = cells.length / Math.max(1, prod(this.shapeOf(node.dims)))
    if (!(Number.isInteger(y) && y >= 0 && y < classes))
      throw new Refusal('data', `'${node.name}': the class ${y} is outside 0 to ${classes - 1}`)
    const args = Array.from({ length: classes }, (_, k) => {
      const cell = cells?.[i * classes + k] as Cell
      if (isConstCell(cell)) return { const: cell.aff.offset }
      return this.materialise(affOf(cell), node.name, index) as string
    })
    if (args.every((a) => typeof a !== 'string')) return null
    this.addFactor(
      { operator: 'multiclass-site', args, params: { y, classes, link } },
      this.elementName(node.name, node.dims, i),
    )
    return null
  }

  /**
   * A positive random parameter of an observed node as one port of a `likelihood-pair` factor, in the coordinate the rule reads
   * (the log of the value): a gamma variable (its own variable), a positive latent carried on its log (a log-normal, half-normal or
   * half-Cauchy prior: the log variable), or `exp(affine)` (a Gaussian variable made of the affine form). A known value is the
   * constant coordinate `log value`.
   */
  private positivePort(
    node: ModelNode & { readonly kind: 'observed' },
    key: string,
    i: number,
    index: number | undefined,
  ): string | { readonly const: number } {
    const expr = node.dist.params[key]
    const e = expr === undefined ? undefined : this.stripClip(expr)
    if (e?.kind === 'op' && e.op === 'exp') {
      const cell = this.cellsOf(node, `${key}:log`, () => this.evalSym(e.args[0] as ExprSpec))[i] as Cell
      if (cell.kind === 'aff') {
        const m = this.materialise(cell.aff, node.name, index)
        return typeof m === 'string' ? m : { const: m }
      }
    }
    const c = this.paramCell(node, key, i)
    if (isConstCell(c)) {
      if (!(c.aff.offset > 0))
        throw new Refusal('data', `the ${key} of '${node.name}' is ${c.aff.offset}: it must be positive`)
      return { const: Math.log(c.aff.offset) }
    }
    if (c.kind === 'rv' && c.family === 'gamma') return c.variable
    if (c.kind === 'aff' && c.aff.offset === 0 && c.aff.terms.size === 1) {
      const only = [...c.aff.terms][0] as [string, number]
      const atom = only[1] === 1 ? this.atoms.get(only[0]) : undefined
      const logVar = atom?.kind === 'var' ? this.logScale.get(atom.name) : undefined
      if (logVar !== undefined) return logVar
    }
    throw new Refusal(
      'operator',
      `the ${key} of '${node.name}' is random but not a gamma or exponential latent, a positive latent carried on its log (a half-normal, half-Cauchy, log-normal or uniform prior) or exp of an affine value: the likelihood-pair factor needs one of those`,
    )
  }

  /**
   * A numeric likelihood over two random scalars (`likelihood-pair`): a Laplace with a random scale, a Student-t with random
   * degrees of freedom (known scale), a Weibull with a random shape (`scale = exp(affine)`) and a beta outcome with two random
   * shapes. The outcome is a parameter of the factor. A parameter that is known is a constant coordinate; nothing random at
   * all says nothing.
   */
  private observedPair(
    node: ModelNode & { readonly kind: 'observed' },
    likelihood: string,
    y: number,
    i: number,
    index: number | undefined,
  ): string | null {
    const extra: Record<string, Json> = {}
    const predictor = (cell: Cell): string | { readonly const: number } => {
      if (isConstCell(cell)) return { const: cell.aff.offset }
      return this.materialise(affOf(cell), node.name, index) as string
    }
    let args: [string | { readonly const: number }, string | { readonly const: number }]
    if (likelihood === 'laplace-scale' || likelihood === 'asymmetric-laplace-scale') {
      if (likelihood === 'asymmetric-laplace-scale') {
        const tau = this.paramCell(node, 'tau', i)
        if (!isConstCell(tau))
          throw new Refusal('operator', `the quantile level of '${node.name}' is random: it must be known`)
        extra['tau'] = tau.aff.offset
      }
      args = [predictor(this.paramCell(node, 'loc', i)), this.positivePort(node, 'scale', i, index)]
    } else if (likelihood === 'student-t-df') {
      const scale = this.paramCell(node, 'scale', i)
      if (!isConstCell(scale))
        throw new Refusal(
          'operator',
          `the degrees of freedom and the scale of '${node.name}' are both random: the likelihood-pair factor takes one random scale or shape beside the location`,
        )
      extra['scale'] = scale.aff.offset
      args = [predictor(this.paramCell(node, 'loc', i)), this.positivePort(node, 'df', i, index)]
    } else if (likelihood === 'weibull-shape') {
      const m = node.dist.params['scale'] === undefined ? undefined : this.scaledExp(node.dist.params['scale'])
      if (m === undefined || m.scale !== undefined)
        throw new Refusal(
          'operator',
          `the scale of '${node.name}' is not exp of an affine value: the Weibull with a random shape needs the log link`,
        )
      args = [
        predictor(this.cellsOf(node, 'predictor', () => this.evalSym(m.inner))[i] as Cell),
        this.positivePort(node, 'concentration', i, index),
      ]
    } else {
      args = [this.positivePort(node, 'a', i, index), this.positivePort(node, 'b', i, index)]
    }
    const params: Record<string, Json> = { likelihood, y, ...extra }
    const why = pairLikelihoods()[likelihood]?.check?.(y, params)
    if (why !== undefined) throw new Refusal('data', `'${node.name}' (${likelihood}): ${why}`)
    if (typeof args[0] !== 'string' && typeof args[1] !== 'string') return null
    this.addFactor({ operator: 'likelihood-pair', args, params }, this.elementName(node.name, node.dims, i))
    return null
  }

  /** An observed Gaussian with a known scale around an affine mean, named by the node's parameters (`loc`/`scale`, or a log-normal's `mu`/`sigma`). */
  private observedGaussian(
    node: ModelNode & { readonly kind: 'observed' },
    name: string,
    y: number,
    i: number,
    index: number | undefined,
    locName: string,
    scaleName: string,
  ): string | null {
    const scale = this.paramCell(node, scaleName, i)
    if (!isConstCell(scale)) throw new Refusal('operator', `the scale of '${node.name}' is not known`)
    if (!(scale.aff.offset > 0))
      throw new Refusal('data', `the scale of '${node.name}' is ${scale.aff.offset}: it must be positive`)
    const precision = 1 / scale.aff.offset ** 2
    const loc = this.paramCell(node, locName, i)
    if (isConstCell(loc)) return null
    const aff = affOf(loc)
    const terms = [...aff.terms].filter(([, c]) => c !== 0)
    const comps = terms.map(([k]) => this.atoms.get(k) as Atom)
    const oneVector =
      aff.offset === 0 &&
      comps.length > 0 &&
      comps.every((a) => a.kind === 'comp' && a.vector === (comps[0] as Atom & { kind: 'comp' }).vector)
    this.addVariable({ name, family: { kind: 'gaussian' }, observed: y })
    if (oneVector) {
      const first = comps[0] as Atom & { kind: 'comp' }
      const a = new Array<number>(first.size).fill(0)
      for (const [k, c] of terms) a[(this.atoms.get(k) as Atom & { kind: 'comp' }).index] = c
      this.addFactor({ operator: 'linear-regression-row', args: [name, first.vector], params: { a, precision } }, name)
    } else {
      const m = this.materialise(aff, node.name, index)
      this.addFactor({ operator: 'gaussian', args: [name, m as string], params: { precision } }, name)
    }
    return name
  }

  /**
   * A heavy-tailed likelihood of an affine predictor with a known scale: `studentT` (the degrees of freedom known),
   * `laplace`, or `normalOutliers` (a mixture of two normal noise levels). One row factor on the packed weights when the
   * predictor is a single dot product with one vector, else a site factor on an auxiliary predictor variable. The outcome
   * is a parameter of the factor, so the observed value has no variable of its own.
   */
  private observedHeavy(
    node: ModelNode & { readonly kind: 'observed' },
    fam: string,
    y: number,
    i: number,
    index: number | undefined,
  ): string | null {
    const known = (p: string): number => {
      const c = this.paramCell(node, p, i)
      if (!isConstCell(c))
        throw new Refusal(
          'operator',
          `the ${p} of '${node.name}' is random: the ${fam} likelihood has a message rule for a known ${p} only`,
        )
      return c.aff.offset
    }
    if (fam === 'laplace' && !isConstCell(this.paramCell(node, 'scale', i)))
      return this.observedPair(node, 'laplace-scale', y, i, index)
    if (fam === 'studentT' && !isConstCell(this.paramCell(node, 'df', i)))
      return this.observedPair(node, 'student-t-df', y, i, index)
    if (fam === 'normalOutliers' && ['scale', 'outlier'].some((p) => !this.paramIsConst(node, p)))
      return this.observedOutlierSwitch(node, y, i, index)
    const scale = known('scale')
    if (!(scale > 0)) throw new Refusal('data', `the scale of '${node.name}' is ${scale}: it must be positive`)
    let params: Record<string, Json>
    let row: string
    let site: string
    if (fam === 'studentT') {
      params = { y, scale, df: known('df') }
      row = 'student-t-row'
      site = 'student-t-site'
    } else if (fam === 'laplace') {
      params = { y, scale }
      row = 'laplace-row'
      site = 'laplace-site'
    } else {
      const outlier = known('outlier')
      params = { weights: [1 - outlier, outlier], means: [y, y], sds: [scale, scale * known('width')] }
      row = 'mixture-row'
      site = 'mixture-site'
    }
    return this.emitSite(node, this.paramCell(node, 'loc', i), i, index, row, site, params)
  }

  /** Whether a parameter of a node's distribution is known (a default counts). */
  private paramIsConst(node: ModelNode & { readonly dist: DistSpec }, name: string): boolean {
    const e = node.dist.params[name]
    return e === undefined || this.isConst(e)
  }

  /**
   * A Bernoulli variable no model node names, the switch that a summed-out mixture (`normalOutliers`, `normalSpikeSlab`) with a
   * random share or random precisions is read through: the mixture is the gate of its components over this switch, which is the
   * same graph as the explicit-switch form (`z ~ bernoulli(share)`, `where(z, ...)`) with the switch left out of the posterior.
   * A beta share is the conjugate `bernoulli` factor, a known one a prior on the switch.
   */
  private hiddenSwitch(node: ModelNode, probs: Cell, index: number | undefined): string {
    if (probs.kind === 'rv' && probs.family === 'beta') {
      const name = this.addVariable({ name: this.auxName(node.name, index), family: { kind: 'bernoulli' } })
      this.addFactor({ operator: 'bernoulli', args: [name, probs.variable] }, name)
      return name
    }
    if (!isConstCell(probs))
      throw new Refusal(
        'operator',
        `the share of '${node.name}' must be known or a beta latent (the conjugate 'bernoulli' operator on the switch of a mixture)`,
      )
    const p = probs.aff.offset
    if (!(p > 0 && p < 1))
      throw new Refusal('data', `the share of '${node.name}' is ${p}: it must be strictly between 0 and 1`)
    const name = this.addVariable({ name: this.auxName(node.name, index), family: { kind: 'bernoulli' } })
    this.addFactor({ operator: 'prior', args: [name], params: { eta: [...bernoulliEta(p)] } }, name)
    return name
  }

  /**
   * An observation `normalOutliers(loc, scale, share, width)` with a random scale (the precision idiom of a gamma latent) or a
   * random share (a beta latent), the switch summed out: the gate of an inlier and an outlier branch over a hidden Bernoulli
   * switch, read through the same route as an observation whose precision is chosen by a discrete variable (`observedSwitch`):
   * the inlier has the precision `tau`, the outlier `tau / width^2`, both `gaussian-gamma-precision` branches over the centre and
   * the gamma precision.
   */
  private observedOutlierSwitch(
    node: ModelNode & { readonly kind: 'observed' },
    y: number,
    i: number,
    index: number | undefined,
  ): string | null {
    const widthCell = this.paramCell(node, 'width', i)
    if (!isConstCell(widthCell))
      throw new Refusal(
        'operator',
        `the width of '${node.name}' is random: the outlier mixture has a message rule for a known width`,
      )
    const width = widthCell.aff.offset
    if (!(width > 0)) throw new Refusal('data', `the width of '${node.name}' is ${width}: it must be positive`)
    const z = this.hiddenSwitch(node, this.paramCell(node, 'outlier', i), index)
    const loc = this.paramCell(node, 'loc', i)
    let scale: Cell
    if (this.paramIsConst(node, 'scale')) {
      const sd = (this.paramCell(node, 'scale', i) as Cell & { kind: 'aff' }).aff.offset
      scale = { kind: 'lookup', variable: z, values: [sd, sd * width] }
    } else {
      const scaleExpr = node.dist.params['scale'] as ExprSpec
      const precision = this.precisionExpr(scaleExpr)
      const tau =
        precision === undefined
          ? undefined
          : scaledOf(this.cellsOf(node, 'precisionProbe', () => this.evalSym(precision))[i] as Cell)
      if (tau === undefined)
        throw new Refusal(
          'operator',
          `the scale of '${node.name}' must be the precision idiom 1 / sqrt(tau) of a gamma or exponential latent for the outlier mixture`,
        )
      const w2 = width * width
      scale = {
        kind: 'lookup',
        variable: z,
        values: [0, 0],
        precisions: [tau, { ...tau, scale: tau.scale / w2, offset: tau.offset / w2 }],
      }
    }
    return this.observedSwitch(node, this.elementName(node.name, node.dims, i), y, index, loc, scale)
  }

  /**
   * The factor of a numeric likelihood of an affine predictor: one row factor on the packed weights when the predictor is a
   * single dot product with one vector, else a site factor on an auxiliary predictor variable. The outcome is a parameter of
   * the factor, so the observed value has no variable of its own. A predictor with no random part says nothing (no factor).
   */
  private emitSite(
    node: ModelNode & { readonly kind: 'observed' },
    predictor: Cell,
    i: number,
    index: number | undefined,
    row: string,
    site: string,
    params: Record<string, Json>,
  ): string | null {
    if (isConstCell(predictor)) return null
    const aff = affOf(predictor)
    const terms = [...aff.terms].filter(([, c]) => c !== 0)
    const comps = terms.map(([k]) => this.atoms.get(k) as Atom)
    const oneVector =
      aff.offset === 0 &&
      comps.length > 0 &&
      comps.every((a) => a.kind === 'comp' && a.vector === (comps[0] as Atom & { kind: 'comp' }).vector)
    const id = this.elementName(node.name, node.dims, i)
    if (oneVector) {
      const first = comps[0] as Atom & { kind: 'comp' }
      const a = new Array<number>(first.size).fill(0)
      for (const [k, c] of terms) a[(this.atoms.get(k) as Atom & { kind: 'comp' }).index] = c
      this.addFactor({ operator: row, args: [first.vector], params: { a, ...params } }, id)
    } else {
      const m = this.materialise(aff, node.name, index)
      if (typeof m !== 'string') return null
      this.addFactor({ operator: site, args: [m], params }, id)
    }
    return null
  }

  /**
   * The rows of an observed node as a design matrix on one packed weight vector, when every row is `a . w` (plus a known
   * offset where `offset` is allowed) of the same vector `w`: the vector's name, its size, and the row (`undefined` for a row
   * with no random part, which says nothing about `w`). Absent when the rows are not of that shape.
   */
  private designRows(
    cells: readonly Cell[],
    offset: boolean,
  ):
    | {
        readonly vector: string
        readonly size: number
        readonly rows: readonly ({ a: number[]; offset: number } | undefined)[]
      }
    | undefined {
    let vector: string | undefined
    let size = 0
    const rows: ({ a: number[]; offset: number } | undefined)[] = []
    for (const c of cells) {
      if (c.kind !== 'aff') return undefined
      const terms = [...c.aff.terms].filter(([, k]) => k !== 0)
      if (terms.length === 0) {
        rows.push(undefined)
        continue
      }
      if (c.aff.offset !== 0 && !offset) return undefined
      const atoms = terms.map(([k]) => this.atoms.get(k) as Atom)
      for (const a of atoms) {
        if (a.kind !== 'comp') return undefined
        if (vector === undefined) {
          vector = a.vector
          size = a.size
        } else if (a.vector !== vector) return undefined
      }
      const a = new Array<number>(size).fill(0)
      for (const [k, coefficient] of terms) a[(this.atoms.get(k) as Atom & { kind: 'comp' }).index] = coefficient
      rows.push({ a, offset: c.aff.offset })
    }
    return vector === undefined ? undefined : { vector, size, rows }
  }

  /**
   * An observed node over a plate as one plated factor on its weight vector, when its rows are i.i.d. draws of one
   * regression on a packed vector `w`: a Gaussian likelihood with one known scale (the exact `linear-regression-plate`, by
   * the sufficient statistics of the rows) or a Bernoulli with a probit or logistic link of `a . w` (`probit-plate`,
   * `logistic-plate`, the design matrix and outcomes as parameters). Anything else, or an engine registry without the
   * plate operators, is not handled here and falls to the expansion. Returns whether the node was compiled.
   */
  private plate(node: ModelNode & { readonly kind: 'observed' }, fam: string, values: readonly number[]): boolean {
    if (fam === 'normal') return this.plateNormal(node, values)
    if (fam === 'bernoulli') return this.plateBernoulli(node, values)
    return false
  }

  private addPlated(node: ModelNode & { readonly kind: 'observed' }, f: FactorSpec): void {
    this.addFactor(f, node.name)
    this.plated.push(node.name)
    this.layout(
      node,
      'observed',
      undefined,
      'observed',
      new Array<null>(prod(this.shapeOf(node.dims))).fill(null),
      false,
    )
  }

  private plateNormal(node: ModelNode & { readonly kind: 'observed' }, values: readonly number[]): boolean {
    if (!('linear-regression-plate' in this.ops)) return false
    const scaleExpr = node.dist.params['scale']
    if (scaleExpr !== undefined && !this.isConst(scaleExpr)) return false
    const scales = this.over(this.paramSym(node.dist, 'scale'), node.dims).map(
      (c) => (c as Cell & { kind: 'aff' }).aff.offset,
    )
    const scale = scales[0] as number
    if (!(scale > 0) || scales.some((v) => v !== scale)) return false
    const locs = this.cellsOf(node, 'loc', () => this.paramSym(node.dist, 'loc'))
    const design = this.designRows(locs, true)
    if (design === undefined) return false
    const d = design.size
    const xtx = new Array<number>(d * d).fill(0)
    const xty = new Array<number>(d).fill(0)
    let yty = 0
    let n = 0
    for (const [i, row] of design.rows.entries()) {
      const y = (values[i] as number) - (row?.offset ?? 0)
      if (row === undefined || Number.isNaN(values[i])) continue
      n++
      yty += y * y
      for (let a = 0; a < d; a++) {
        xty[a] = (xty[a] as number) + (row.a[a] as number) * y
        for (let b = 0; b < d; b++)
          xtx[a * d + b] = (xtx[a * d + b] as number) + (row.a[a] as number) * (row.a[b] as number)
      }
    }
    if (n === 0) return false
    this.addPlated(node, {
      operator: 'linear-regression-plate',
      args: [design.vector],
      params: { xtx, xty, yty, n, precision: 1 / scale ** 2 },
    })
    return true
  }

  private plateBernoulli(node: ModelNode & { readonly kind: 'observed' }, values: readonly number[]): boolean {
    const probs = node.dist.params['probs']
    const logits = node.dist.params['logits']
    let link: 'probit' | 'sigmoid' | undefined
    let argument: ExprSpec | undefined
    if (logits !== undefined) {
      link = 'sigmoid'
      argument = logits
    } else if (probs !== undefined) {
      const e = this.unwrap(probs)
      if (e.kind === 'op' && (e.op === 'probit' || e.op === 'sigmoid')) {
        link = e.op
        argument = e.args[0] as ExprSpec
      }
    }
    if (link === undefined || argument === undefined) return false
    const operator = link === 'probit' ? 'probit-plate' : 'logistic-plate'
    if (!(operator in this.ops)) return false
    const arg = argument
    const cells = this.cellsOf(node, `link:${link}`, () => this.evalSym(arg))
    const design = this.designRows(cells, false)
    if (design === undefined) return false
    const x: number[] = []
    const y: number[] = []
    for (const [i, row] of design.rows.entries()) {
      if (row === undefined || Number.isNaN(values[i])) continue
      x.push(...row.a)
      y.push(values[i] as number)
    }
    if (y.length === 0) return false
    this.addPlated(node, { operator, args: [design.vector], params: { X: x, y } })
    return true
  }

  /** Whether an expression depends, through derived nodes, on a Bernoulli or categorical latent. */
  private mentionsDiscrete(e: ExprSpec): boolean {
    if (e.kind === 'ref') {
      const n = this.nodes.get(e.name)
      if (n?.kind === 'latent') return n.dist.family === 'bernoulli' || n.dist.family === 'categorical'
      return n?.kind === 'derived' && this.mentionsDiscrete(n.expr)
    }
    return (e.kind === 'op' || e.kind === 'custom') && e.args.some((a) => this.mentionsDiscrete(a))
  }

  /** The location and scale cells of an observed normal element when one of them is a lookup by a discrete variable; absent otherwise. */
  private switchedCells(
    node: ModelNode & { readonly kind: 'observed' },
    i: number,
  ): { readonly loc: Cell; readonly scale: Cell } | undefined {
    const locExpr = node.dist.params['loc']
    const scaleExpr = node.dist.params['scale']
    const discrete = (e: ExprSpec | undefined): boolean =>
      e !== undefined && !this.isConst(e) && this.mentionsDiscrete(e)
    if (!discrete(locExpr) && !discrete(scaleExpr)) return undefined
    // A precision chosen by a discrete variable (`1 / sqrt(where(z, tau_b, tau_a))`, `1 / sqrt(at(tau, z))`): the mean and the
    // precision are read as lookups of that variable, so the observation is one gate of `gaussian-gamma-precision` branches.
    const precisionExpr = scaleExpr === undefined ? undefined : this.precisionExpr(scaleExpr)
    if (precisionExpr !== undefined && discrete(precisionExpr)) {
      this.lookupMode = true
      try {
        const loc = this.paramCell(node, 'loc', i)
        const precision = this.cellsOf(node, 'precisionLookup', () => this.evalSym(precisionExpr))[i] as Cell
        if (precision.kind === 'lookup' && precision.precisions !== undefined) return { loc, scale: precision }
      } finally {
        this.lookupMode = false
      }
    }
    const loc = this.paramCell(node, 'loc', i)
    // The precision idiom `1 / sqrt(tau)` where `tau` is a gamma variable, or one chosen among gamma variables by a discrete variable.
    const randomPrecision = (): boolean => {
      const p = scaleExpr === undefined ? undefined : this.precisionExpr(scaleExpr)
      if (p === undefined) return false
      try {
        const c = this.cellsOf(node, 'precisionProbe', () => this.evalSym(p))[i] as Cell
        return c.kind === 'scaled' || (c.kind === 'rv' && c.family === 'gamma')
      } catch (e) {
        if (e instanceof Refusal) return false
        throw e
      }
    }
    // A mean that mentions a discrete variable but is a random Gaussian after its gates (a `select`), with a scale that does not
    // (a precision idiom `1 / sqrt(tau)`), is an ordinary observation: the scale is not evaluated as a lookup.
    if (loc.kind !== 'lookup' && (!discrete(scaleExpr) || randomPrecision())) return undefined
    const scale = this.paramCell(node, 'scale', i)
    return loc.kind === 'lookup' || scale.kind === 'lookup' ? { loc, scale } : undefined
  }

  /**
   * An observed Gaussian whose location and (or) scale is a lookup by one discrete variable `z`: when the location is known in
   * every state, a table over `z` (the likelihood of the value under each state: a hidden Markov model's emission); when it is
   * random in some states, a `gate` of one Gaussian likelihood per state, over the random variables of the branches (the clutter
   * problem, a mixture with a known component, a change point).
   */
  private observedSwitch(
    node: ModelNode & { readonly kind: 'observed' },
    name: string,
    y: number,
    index: number | undefined,
    loc: Cell,
    scale: Cell,
  ): null {
    const looks = [loc, scale].filter((c): c is Cell & { kind: 'lookup' } => c.kind === 'lookup')
    const z = (looks[0] as Cell & { kind: 'lookup' }).variable
    if (looks.some((l) => l.variable !== z))
      throw new Refusal(
        'operator',
        `the location and the scale of '${node.name}' are chosen by different discrete variables`,
      )
    const states = (looks[0] as Cell & { kind: 'lookup' }).values.length
    if (looks.some((l) => l.values.length !== states))
      throw new Refusal('operator', `the lookups of '${node.name}' have different numbers of states`)
    const precisions = scale.kind === 'lookup' ? scale.precisions : undefined
    if (scale.kind === 'lookup' && scale.items !== undefined)
      throw new Refusal(
        'operator',
        `the scale of '${node.name}' is random in some states of a discrete variable: it must be known`,
      )
    if (scale.kind !== 'lookup' && !isConstCell(scale))
      throw new Refusal('operator', `the scale of '${node.name}' is not known`)
    // With a precision lookup, `values` are the known precisions of the states that have no gamma variable, not scales.
    const sdAt = (k: number): number =>
      precisions !== undefined
        ? 1 / Math.sqrt((scale as Cell & { kind: 'lookup' }).values[k] as number)
        : scale.kind === 'lookup'
          ? (scale.values[k] as number)
          : (scale as Cell & { kind: 'aff' }).aff.offset
    for (let k = 0; k < states; k++) {
      if (precisions?.[k] !== undefined) {
        const p = precisions[k] as Precision
        if (!(p.scale >= 0 && p.offset >= 0 && p.scale + p.offset > 0))
          throw new Refusal(
            'data',
            `the precision of '${node.name}' must be a non-negative multiple of a gamma variable`,
          )
        continue
      }
      if (!(sdAt(k) > 0)) throw new Refusal('data', `the scale of '${node.name}' is ${sdAt(k)}: it must be positive`)
    }
    const locAt = (k: number): Aff | number => {
      if (loc.kind === 'lookup') return loc.items?.[k] ?? (loc.values[k] as number)
      return affOf(loc)
    }
    const known =
      precisions === undefined &&
      Array.from({ length: states }, (_, k) => locAt(k)).every((m) => typeof m === 'number' || m.terms.size === 0)
    const pdf = (m: number, sd: number): number => Math.exp(-0.5 * ((y - m) / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI))
    const value = (m: Aff | number): number => (typeof m === 'number' ? m : m.offset)
    if (known) {
      this.need('table')
      this.addFactor(
        {
          operator: 'table',
          args: [z],
          params: { states: [states], values: Array.from({ length: states }, (_, k) => pdf(value(locAt(k)), sdAt(k))) },
        },
        name,
      )
      return null
    }
    this.need('gate')
    const variables: string[] = []
    const branches = Array.from({ length: states }, (_, k) => {
      const m = locAt(k)
      let mean: number | { readonly const: number }
      if (typeof m === 'number' || m.terms.size === 0) mean = { const: value(m) }
      else {
        const v = this.materialise(m, node.name, index) as string
        let at = variables.indexOf(v)
        if (at < 0) {
          at = variables.length
          variables.push(v)
        }
        mean = at
      }
      const gamma = precisions?.[k]
      if (gamma !== undefined) {
        let at = variables.indexOf(gamma.variable)
        if (at < 0) {
          at = variables.length
          variables.push(gamma.variable)
        }
        this.need('gaussian-gamma-precision')
        return {
          operator: 'gaussian-gamma-precision',
          args: [{ const: y }, mean, at],
          params: { scale: gamma.scale, offset: gamma.offset },
        }
      }
      return {
        operator: 'gaussian',
        args: [{ const: y }, typeof mean === 'number' ? mean : mean],
        params: { precision: 1 / sdAt(k) ** 2 },
      }
    })
    this.addFactor({ operator: 'gate', args: [z, ...variables], params: { arity: variables.length, branches } }, name)
    return null
  }

  /**
   * An observed Gaussian whose scale is `exp(s)` with `s` chosen by a discrete variable `z` among Gaussian forms of its own (a spread
   * per component of a mixture, `exp(at(logSd, z))`), whose mean may be chosen by the same `z` among random ones: ONE `gate` on `z`
   * whose branch `k` is the `gaussian-scale` factor of the observation on the mean and the log scale of state `k`, so the
   * component that explains the reading is the one whose mean and spread both move (a `select` of each would let EP lose which
   * component the pair came from, and a spread per component run off). False when the scale is not of that form (the other
   * routes are tried).
   */
  private observedSwitchLogScale(
    node: ModelNode & { readonly kind: 'observed' },
    y: number,
    i: number,
    index: number | undefined,
  ): boolean {
    const scaleExpr = node.dist.params['scale']
    if (scaleExpr === undefined || this.isConst(scaleExpr)) return false
    const e = this.stripClip(scaleExpr)
    if (e.kind !== 'op' || e.op !== 'exp' || !this.mentionsDiscrete(e.args[0] as ExprSpec)) return false
    let loc: Cell
    let logScale: Cell
    this.lookupMode = true
    try {
      loc = this.paramCell(node, 'loc', i)
      logScale = this.cellsOf(node, 'logScaleLookup', () => this.evalSym(e.args[0] as ExprSpec))[i] as Cell
    } catch (err) {
      if (err instanceof Refusal) return false
      throw err
    } finally {
      this.lookupMode = false
    }
    if (logScale.kind !== 'lookup' || logScale.precisions !== undefined) return false
    const z = logScale.variable
    const states = logScale.values.length
    if (loc.kind === 'lookup' && (loc.variable !== z || loc.values.length !== states || loc.precisions !== undefined))
      throw new Refusal(
        'operator',
        `the mean and the spread of '${node.name}' are chosen by different discrete variables`,
      )
    if (loc.kind !== 'lookup' && loc.kind !== 'aff')
      throw new Refusal('operator', `the mean of '${node.name}' is not a Gaussian form or a lookup`)
    const locAt = (k: number): Aff | number => {
      if (loc.kind === 'lookup') return loc.items?.[k] ?? (loc.values[k] as number)
      return affOf(loc)
    }
    const variables: string[] = []
    const argOf = (form: Aff | number | undefined, fallback: number): number | { readonly const: number } => {
      if (form === undefined || typeof form === 'number' || form.terms.size === 0)
        return { const: form === undefined ? fallback : typeof form === 'number' ? form : form.offset }
      const v = this.materialise(form, node.name, index) as string
      let at = variables.indexOf(v)
      if (at < 0) {
        at = variables.length
        variables.push(v)
      }
      return at
    }
    this.need('gate')
    this.need('gaussian-scale')
    const branches = Array.from({ length: states }, (_, k) => ({
      operator: 'gaussian-scale',
      args: [{ const: y }, argOf(locAt(k), 0), argOf(logScale.items?.[k], logScale.values[k] as number)],
      params: {},
    }))
    this.addFactor(
      { operator: 'gate', args: [z, ...variables], params: { arity: variables.length, branches } },
      this.elementName(node.name, node.dims, i),
    )
    return true
  }

  /**
   * An observed Gaussian whose scale is `exp(s)` with `s` an affine form of Gaussian variables (a `gaussian-log-scale`
   * factor: the observation of a stochastic volatility model); the mean must be known.
   */
  private observedLogScale(
    node: ModelNode & { readonly kind: 'observed' },
    name: string,
    y: number,
    i: number,
    index: number | undefined,
  ): string | undefined {
    const inner = this.unwrap(node.dist.params['scale'] as ExprSpec) as ExprSpec & { readonly kind: 'op' }
    const loc = this.paramCell(node, 'loc', i)
    if (!isConstCell(loc)) return undefined
    let cell: Cell
    try {
      cell = this.cellsOf(node, 'logScale', () => this.evalSym(inner.args[0] as ExprSpec))[i] as Cell
    } catch (e) {
      // Not an affine form of Gaussian variables (a product of random values, a gamma latent): the precision idiom is tried next.
      if (e instanceof Refusal) return undefined
      throw e
    }
    if (isConstCell(cell) || cell.kind !== 'aff') return undefined
    const s = this.materialise(affOf(cell), node.name, index) as string
    this.addVariable({ name, family: { kind: 'gaussian' }, observed: y })
    this.addFactor({ operator: 'gaussian-log-scale', args: [name, s], params: { loc: loc.aff.offset } }, name)
    return name
  }

  /**
   * An observed Gaussian whose scale is a positive latent (the `exp` of a Gaussian log-scale variable: a half-normal or
   * half-Cauchy sd projected by `latentProjected`, or `exp(affine)`) and whose mean may be random too: the `gaussian-scale`
   * factor, a one-dimensional quadrature over the log scale. Anything else is refused with the precision idiom named.
   */
  private observedRandomScale(
    node: ModelNode & { readonly kind: 'observed' },
    name: string,
    y: number,
    i: number,
    index: number | undefined,
    scaleExpr: ExprSpec,
  ): string {
    const stripped = this.stripClip(scaleExpr)
    let logVar: string | undefined
    try {
      if (stripped.kind === 'op' && stripped.op === 'exp') {
        const cell = this.cellsOf(node, 'logScale', () => this.evalSym(stripped.args[0] as ExprSpec))[i] as Cell
        if (cell.kind === 'aff' && !isConstCell(cell)) logVar = this.materialise(cell.aff, node.name, index) as string
      } else {
        const cell = this.cellsOf(node, 'scaleValue', () => this.evalSym(stripped))[i] as Cell
        const only =
          cell.kind === 'aff' && cell.aff.offset === 0 && cell.aff.terms.size === 1 ? [...cell.aff.terms] : []
        const atom = only.length === 1 && only[0]?.[1] === 1 ? this.atoms.get(only[0][0]) : undefined
        if (atom?.kind === 'var') logVar = this.logScale.get(atom.name)
      }
    } catch (e) {
      if (!(e instanceof Refusal)) throw e
    }
    if (logVar === undefined) this.randomPrecisions(node)
    const loc = this.paramCell(node, 'loc', i)
    const mean = isConstCell(loc)
      ? { const: loc.aff.offset }
      : (this.materialise(affOf(loc), node.name, index) as string)
    this.need('gaussian-scale')
    this.addVariable({ name, family: { kind: 'gaussian' }, observed: y })
    this.addFactor({ operator: 'gaussian-scale', args: [name, mean, logVar as string] }, name)
    return name
  }

  /** An observed Gaussian with a gamma precision (`gaussian-gamma-precision`): the mean is a variable or a known value. */
  private observedRandomPrecision(
    node: ModelNode & { readonly kind: 'observed' },
    name: string,
    y: number,
    i: number,
    index: number | undefined,
  ): string {
    const tau = this.cellsOf(node, 'precision', () => ({
      dims: node.dims,
      cells: this.randomPrecisions(node).map((p): Cell => ({ kind: 'scaled', ...p })),
    }))[i] as Cell & { kind: 'scaled' }
    const loc = this.paramCell(node, 'loc', i)
    const mean = isConstCell(loc)
      ? { const: loc.aff.offset }
      : (this.materialise(affOf(loc), node.name, index) as string)
    this.addVariable({ name, family: { kind: 'gaussian' }, observed: y })
    this.addFactor(precisionFactor(name, mean, tau), name)
    return name
  }

  private observedBernoulli(
    node: ModelNode & { readonly kind: 'observed' },
    name: string,
    y: number,
    i: number,
    index: number | undefined,
  ): string | null {
    const probs = node.dist.params['probs']
    const logits = node.dist.params['logits']
    const link = (op: string, arg: ExprSpec, params: Record<string, Json> = {}): string | null => {
      const factor = factorName(this.modelOps, op)
      this.need(factor)
      const c = this.cellsOf(node, `link:${op}`, () => this.evalSym(arg))[i] as Cell
      if (isConstCell(c)) return null
      const x = this.materialise(affOf(c), node.name, index) as string
      this.addVariable({ name, family: { kind: 'bernoulli' }, observed: y })
      this.addFactor({ operator: factor, args: [name, x], params }, name)
      return name
    }
    if (logits !== undefined) return link('sigmoid', logits)
    if (probs === undefined) throw new Refusal('structure', `'${node.name}': give probs or logits`)
    const e = this.unwrap(probs)
    if (e.kind === 'op' && e.op === 'probit') return link('probit', e.args[0] as ExprSpec)
    if (e.kind === 'op' && e.op === 'sigmoid') return link('sigmoid', e.args[0] as ExprSpec)
    if (e.kind === 'op' && e.op === 'gt') {
      const threshold = this.evalNamed(e.args[1] as ExprSpec)
      const cell = this.over(this.fromNamed(threshold), node.dims)[i] as Cell & { kind: 'aff' }
      return link('gt', e.args[0] as ExprSpec, { threshold: cell.aff.offset })
    }
    const p = this.paramCell(node, 'probs', i)
    if (isConstCell(p)) return null
    if (p.kind === 'rv' && p.family === 'beta') {
      this.addVariable({ name, family: { kind: 'bernoulli' }, observed: y })
      this.addFactor({ operator: 'bernoulli', args: [name, p.variable] }, name)
      return name
    }
    throw new Refusal(
      'operator',
      `the probability of '${node.name}' is not a link of an affine value (probit, sigmoid, a comparison) or a beta latent: no message rule`,
    )
  }

  /**
   * An observed exponential or gamma value (a known shape) whose rate is a gamma latent or a known multiple of one: the conjugate
   * `gamma-observed` factor, a gamma message to the rate (a waiting time with a gamma prior on the rate).
   */
  private observedGammaRate(
    node: ModelNode & { readonly kind: 'observed' },
    fam: string,
    name: string,
    y: number,
    i: number,
  ): string | null {
    if (!(y > 0)) throw new Refusal('data', `'${node.name}' is ${fam}: the value ${y} must be positive`)
    const shape = fam === 'gamma' ? this.constAt(node, node.dist.params['concentration'] as ExprSpec, i) : 1
    const target = this.paramCell(node, 'rate', i)
    if (isConstCell(target)) return null
    const tau = scaledOf(target)
    if (tau === undefined || tau.offset !== 0 || !(tau.scale > 0))
      throw new Refusal(
        'operator',
        `the rate of '${node.name}' must be a gamma latent or a known multiple of one for the conjugate '${fam}' operator, it is not`,
      )
    this.need('gamma-observed')
    this.addVariable({ name, family: { kind: 'gaussian' }, observed: y })
    this.addFactor(
      {
        operator: 'gamma-observed',
        args: [name, tau.variable],
        params: { shape, ...(tau.scale === 1 ? {} : { scale: tau.scale }) },
      },
      name,
    )
    return name
  }

  /**
   * An observed two-component mixture with known means and scale and a beta weight (`normalPair`, the switch summed out): the
   * `mixture-weight` factor on the weight, with the likelihood of the value under each component.
   */
  private observedMixtureWeight(
    node: ModelNode & { readonly kind: 'observed' },
    name: string,
    y: number,
    i: number,
  ): string | null {
    const target = this.paramCell(node, 'weight', i)
    if (isConstCell(target)) return null
    if (target.kind !== 'rv' || target.family !== 'beta')
      throw new Refusal(
        'operator',
        `the weight of '${node.name}' must be a beta latent for the 'mixture-weight' operator`,
      )
    const [low, high, scale] = ['low', 'high', 'scale'].map((k) =>
      this.constAt(node, node.dist.params[k] as ExprSpec, i),
    )
    const density = (m: number): number =>
      Math.exp(-0.5 * ((y - m) / (scale as number)) ** 2) / ((scale as number) * Math.sqrt(2 * Math.PI))
    this.need('mixture-weight')
    this.addFactor(
      {
        operator: 'mixture-weight',
        args: [target.variable],
        params: { l1: density(high as number), l0: density(low as number) },
      },
      name,
    )
    return null
  }

  /**
   * An observed geometric (trials up to the first success) or negative binomial (failures before the `n`-th success, `n` known)
   * count whose success probability is a beta latent: the conjugate `negative-binomial-beta` factor.
   */
  private observedFailures(
    node: ModelNode & { readonly kind: 'observed' },
    fam: string,
    name: string,
    y: number,
    i: number,
  ): string | null {
    const geometric = fam === 'geometric'
    const failures = geometric ? y - 1 : y
    if (!(failures >= 0 && Number.isInteger(failures)))
      throw new Refusal('data', `'${node.name}' is ${fam}: the count ${y} is not on its support`)
    const successes = geometric ? 1 : this.constAt(node, node.dist.params['n'] as ExprSpec, i)
    const target = this.paramCell(node, 'probs', i)
    if (isConstCell(target)) return null
    if (target.kind !== 'rv' || target.family !== 'beta')
      throw new Refusal(
        'operator',
        `the probs of '${node.name}' must be a beta latent for the conjugate '${fam}' operator, it is not`,
      )
    this.need('negative-binomial-beta')
    this.addVariable({ name, family: { kind: 'gaussian' }, observed: failures })
    this.addFactor({ operator: 'negative-binomial-beta', args: [name, target.variable], params: { successes } }, name)
    return name
  }

  /**
   * An observed categorical whose probabilities are the row of a matrix of Dirichlet latents chosen by a discrete latent
   * (`categorical({ probs: phi.at(z) })`, the words of a topic model: `phi_k` the word distribution of topic `k`, `z` the topic of
   * the token). It is one `gate` on the index whose branch `k` is the conjugate `discrete` factor of the observed outcome on row
   * `k`: each row hears the tokens, weighted by the belief that the token chose it, and the index hears how well each row
   * explains the outcome. The same gate as an unknown transition row, with an observed outcome.
   */
  private observedRowGate(
    node: ModelNode & { readonly kind: 'observed' },
    name: string,
    y: number,
    i: number,
    expr: ExprSpec,
  ): string | null {
    const chosen = this.randomTransitionRow(expr) as { readonly matrix: ExprSpec; readonly index: ExprSpec }
    const matrix = this.evalSym(chosen.matrix)
    if (matrix.dims.length !== 1 || matrix.cells.some((r) => r.kind !== 'rv' || r.family !== 'dirichlet'))
      throw new Refusal(
        'operator',
        `the rows of the matrix behind '${node.name}' must be one Dirichlet latent each (a latent over the dimension of the states)`,
      )
    const rowVars = matrix.cells.map((r) => (r as { readonly variable: string }).variable)
    const size = (this.variables.find((v) => v.name === rowVars[0])?.family.size ?? 2) as number
    const parent = this.over(this.evalSym(chosen.index), node.dims)[i] as Cell
    if (parent.kind !== 'rv' || (parent.family !== 'discrete' && parent.family !== 'bernoulli'))
      throw new Refusal(
        'operator',
        `the row of '${node.name}' must be chosen by a discrete latent, not a computed value`,
      )
    const from =
      parent.family === 'bernoulli'
        ? 2
        : ((this.variables.find((v) => v.name === parent.variable)?.family.size ?? 2) as number)
    if (from !== rowVars.length)
      throw new Refusal('data', `'${node.name}' has ${from} choices of row and its matrix ${rowVars.length} rows`)
    if (!(Number.isInteger(y) && y >= 0 && y < size))
      throw new Refusal('data', `'${node.name}' observes ${y}, outside its ${size} categories`)
    this.addVariable({ name, family: { kind: 'discrete', size }, observed: y })
    this.need('gate')
    this.need('discrete')
    const branches = rowVars.map((_, j) => ({ operator: 'discrete', args: [0, 1 + j], params: {} }))
    this.addFactor(
      { operator: 'gate', args: [parent.variable, name, ...rowVars], params: { arity: 1 + rowVars.length, branches } },
      name,
    )
    return name
  }

  /**
   * An observed hurdle count: zero with probability `zero`, else a zero-truncated Poisson of `exposure * exp(affine)`. The state
   * of the hurdle (a structural zero, or a positive count) is determined by the count and summed out exactly: a zero is an
   * observation of the Bernoulli `zero`, with a beta probability the conjugate `bernoulli` factor, and a positive count is the same
   * Bernoulli observed false and the zero-truncated Poisson site on the log rate (the gate of {point mass at zero, truncated
   * Poisson}, with its switch observed). The positive part and the zero part share no variable, so each is exact.
   */
  private observedHurdle(
    node: ModelNode & { readonly kind: 'observed' },
    y: number,
    i: number,
    index: number | undefined,
  ): string | null {
    if (!(Number.isInteger(y) && y >= 0))
      throw new Refusal('data', `'${node.name}' is a hurdle count: ${y} is not a count`)
    const zero = this.paramCell(node, 'zero', i)
    if (!isConstCell(zero)) {
      if (zero.kind !== 'rv' || zero.family !== 'beta')
        throw new Refusal('operator', `the zero probability of '${node.name}' must be known or a beta latent`)
      const crossed = this.auxName(node.name, index)
      this.addVariable({ name: crossed, family: { kind: 'bernoulli' }, observed: y === 0 ? 1 : 0 })
      this.addFactor({ operator: 'bernoulli', args: [crossed, zero.variable] }, crossed)
    }
    if (y === 0) return null
    const numeric = this.observedNumeric(node, 'hurdlePoisson', y, i, index)
    if (numeric === undefined)
      throw new Refusal(
        'operator',
        `the rate of '${node.name}' must be \`exposure * exp(affine)\` for the hurdle Poisson`,
      )
    return numeric
  }

  private observedConjugate(
    node: ModelNode & { readonly kind: 'observed' },
    fam: string,
    name: string,
    y: number,
    i: number,
  ): string | null {
    const param = fam === 'poisson' ? 'rate' : 'probs'
    const expr = node.dist.params[param]
    if (expr === undefined) throw new Refusal('structure', `'${node.name}': the ${param} is missing`)
    // A word read through a discrete index of Dirichlet rows (`phi.at(z)`, a topic model): a gate over the index.
    if (fam === 'categorical' && this.randomTransitionRow(expr) !== undefined)
      return this.observedRowGate(node, name, y, i, expr)
    let exposure = 1
    let target: Cell
    const e = this.unwrap(expr)
    if (fam === 'poisson' && e.kind === 'op' && e.op === 'mul') {
      const [a, b] = e.args as [ExprSpec, ExprSpec]
      const [k, r] = this.isConst(a) ? [a, b] : [b, a]
      if (!this.isConst(k))
        throw new Refusal('operator', `the rate of '${node.name}' is a product of two random values`)
      exposure = (this.over(this.fromNamed(this.evalNamed(k)), node.dims)[i] as Cell & { kind: 'aff' }).aff.offset
      target = this.over(this.evalSym(r), node.dims)[i] as Cell
    } else target = this.paramCell(node, param, i)
    if (isConstCell(target)) return null
    const want: FamilyKind = fam === 'poisson' ? 'gamma' : fam === 'binomial' ? 'beta' : 'dirichlet'
    if (target.kind !== 'rv' || target.family !== want)
      throw new Refusal(
        'operator',
        `the ${param} of '${node.name}' must be a ${want} latent for the conjugate '${fam}' operator, it is not`,
      )
    if (fam === 'poisson') {
      this.addVariable({ name, family: { kind: 'gaussian' }, observed: y })
      this.addFactor({ operator: 'poisson', args: [name, target.variable], params: { exposure } }, name)
    } else if (fam === 'binomial') {
      const trials = (
        this.over(this.fromNamed(this.evalNamed(node.dist.params['n'] as ExprSpec)), node.dims)[i] as Cell & {
          kind: 'aff'
        }
      ).aff.offset
      this.addVariable({ name, family: { kind: 'gaussian' }, observed: y })
      this.addFactor({ operator: 'binomial', args: [name, target.variable], params: { n: trials } }, name)
    } else {
      const size = (this.variables.find((v) => v.name === target.variable)?.family.size ?? 2) as number
      this.addVariable({ name, family: { kind: 'discrete', size }, observed: y })
      this.addFactor({ operator: 'discrete', args: [name, target.variable] }, name)
    }
    return name
  }
}

/**
 * Compile a bound model to a factor graph for message passing, by expanding its plates. The result carries the graph (or
 * `undefined` with the reasons in `issues` when something cannot be mapped), the layout of each site for reading results
 * back, the auxiliary variables, the opaque nodes `analyseSpec` found, and the size of the expansion. Observed data is
 * folded into the graph (the observed variables' values and the factors' constants), so the graph belongs to the bound
 * model: rebind and compile again for new data.
 *
 * Nodes that the engine has no counterpart for are never dropped silently. A `param` node, a `factor` (log-density)
 * term, a family with no message family, a `custom` call, a random scale that is not the precision idiom (a product of two
 * random precisions is not mapped), a square of a random value, a non-affine operator on a random value
 * (except as a Bernoulli link), a gate over a mix of known and random components, a computed random index, a constraint beyond
 * the support of a Gaussian (a simplex, a positive-definite matrix), missing data that is not ignored, a subsampling batch and a plated table over discrete nodes each become an
 * issue naming the node and the reason, and no graph is produced.
 *
 * @param bound - The bound model.
 * @param options - The factor operators to compile against (default the engine's `defaultOperators()`), the model's operator table, and `plates`: `vectorise` (default) or `expand`.
 * @returns The compilation.
 * @concept model-spec
 * @index function Compile a bound ModelSpec to a message-passing factor graph, plates expanded.
 * @example
 * ```ts
 * import { defineModel, ops, toFactorGraph } from 'omai-engine/probprog/model'
 * import { bernoulli, normal } from 'omai-engine/probprog/model/dists'
 * const m = defineModel('probit', (b) => {
 *   const n = b.range('row'), f = b.range('feature')
 *   const X = b.data('X', { dims: [n, f] })
 *   const y = b.data('y', { dims: [n], support: 'binary' })
 *   const w = b.latent('w', normal(0, 2), { dims: [f] })
 *   b.observe('y', bernoulli({ probs: ops.probit(X.matmul(w)) }), y)
 * })
 * const bound = m.bind({ X: [[1, 0.5], [1, -1]], y: [1, 0] })
 * const plated = toFactorGraph(bound)
 * plated.graph?.factors.map((f) => f.id) // => ['factor:w', 'factor:y']
 * const rows = toFactorGraph(bound, { plates: 'expand' })
 * rows.graph?.variables.map((v) => v.name) // => ['w', 'y#0[0]', 'y[0]', 'y#0[1]', 'y[1]']
 * rows.graph?.factors.map((f) => f.id) // => ['factor:w', 'factor:y#0[0]', 'factor:y[0]', 'factor:y#0[1]', 'factor:y[1]']
 * ```
 */
export function toFactorGraph(
  bound: BoundModel,
  options: {
    readonly operators?: Readonly<Record<string, FactorOperator>>
    readonly modelOperators?: OperatorTable
    readonly analysis?: SpecAnalysis
    /** `vectorise` (default): an observed node whose rows are one regression on a weight vector is one plated factor; `expand`: a factor and a variable per row. */
    readonly plates?: 'vectorise' | 'expand'
    /**
     * Compile the scalar normal latents that an observed node reads together (the intercept and slope of `a + b x`) as the
     * coordinates of one joint Gaussian vector, so message passing keeps their correlation (default false: scalar variables,
     * whose loop of messages keeps the means and loses the correlation, so the marginal sds come out too small).
     */
    readonly packScalars?: boolean
    /**
     * A vector whose coefficients have the precisions of one shared gamma variable (a smoothing parameter): `dense` (default)
     * is one Gaussian vector with a `gaussian-mv-precision-scale` factor (the posterior keeps its correlations), `elements` a
     * scalar variable and a `gaussian-gamma-precision` factor per coefficient (the same prior, a mean-field-like posterior).
     */
    readonly sharedScale?: 'dense' | 'elements'
  } = {},
): FactorGraphCompilation {
  // A vector written as one event (a multivariate normal with no named dimension) is read as a vector over a dimension of its size.
  const eventless = eventToDims(bound)
  if (eventless !== bound) return toFactorGraph(eventless, options)
  if (bound.spec.nodes.some((n) => n.kind === 'scan')) {
    // A chain has no plate: the engine reads it written out, one variable and factor per step.
    let unrolled: BoundModel
    try {
      unrolled = unrollScans(bound)
    } catch (e) {
      if (!(e instanceof ScanUnrollError)) throw e
      return {
        name: bound.spec.name,
        graph: undefined,
        issues: [{ node: e.node, kind: 'structure', reason: e.message }],
        opaque: [],
        sites: [],
        auxiliary: [],
        plated: [],
        expanded: { plates: [], variables: 0, factors: 0 },
      }
    }
    return toFactorGraph(unrolled, options)
  }
  const c = new Compiler(
    bound,
    options.modelOperators ?? defaultOperators(),
    options.operators ?? factorOperators(),
    options.plates !== 'expand',
    options.packScalars === true,
    options.sharedScale !== 'elements',
  )
  c.run()
  c.prune()
  const graph: FactorGraphSpec = { variables: c.variables, factors: c.factors }
  const opaque = (options.analysis?.opaque ?? opaqueNodes(bound)).filter((name) => {
    const node = bound.spec.nodes.find((n) => n.name === name && n.kind !== 'data')
    if (node === undefined) return true
    const called = customNames(node)
    // A node whose custom calls were all evaluated (known arguments) or read as a matrix (affine in one random vector) is not opaque.
    return called.length === 0 || !called.every((n) => c.handledCustoms.has(n))
  })
  for (const name of opaque)
    if (!c.issues.some((i) => i.node === name))
      c.issues.push({
        node: name,
        kind: 'opaque',
        reason: `'${name}' is opaque to message passing (a custom call, or a family with no exponential-family form)`,
      })
  const plates = bound.spec.dims
    .filter((d) => c.sites.some((s) => s.dims.includes(d.name)))
    .map((d) => ({ dim: d.name, size: bound.sizes[d.name] as number }))
  return {
    name: bound.spec.name,
    graph: c.issues.length === 0 ? graph : undefined,
    issues: c.issues,
    opaque,
    sites: c.sites,
    auxiliary: c.auxiliary,
    plated: c.plated,
    expanded: { plates, variables: graph.variables.length, factors: graph.factors.length },
  }
}

/** The names of the custom functions a node calls. */
function customNames(node: ModelNode): string[] {
  const out = new Set<string>()
  const walk = (e: ExprSpec): void => {
    if (e.kind === 'custom') out.add(e.name)
    if (e.kind === 'op' || e.kind === 'custom') for (const a of e.args) walk(a)
  }
  if (node.kind === 'latent' || node.kind === 'observed' || node.kind === 'generated') {
    for (const e of Object.values(node.dist.params)) walk(e)
    if (node.kind === 'observed') walk(node.value)
  } else if (node.kind === 'derived' || node.kind === 'factor') walk(node.expr)
  return [...out]
}

/** The nodes with a `custom` call, found without the analysis (for a caller that gives none). */
function opaqueNodes(bound: BoundModel): string[] {
  const custom = (e: ExprSpec): boolean => e.kind === 'custom' || (e.kind === 'op' && e.args.some(custom))
  return bound.spec.nodes
    .filter((n) => {
      if (n.kind === 'latent' || n.kind === 'observed' || n.kind === 'generated')
        return Object.values(n.dist.params).some(custom) || (n.kind === 'observed' && custom(n.value))
      return n.kind === 'derived' || n.kind === 'factor' ? custom(n.expr) : false
    })
    .map((n) => n.name)
}

/** The message rule a method asks of every operator of a graph. */
export type MessageRule = 'ep' | 'vmp' | 'bp'

/**
 * The factors of a compiled graph whose operator has no message rule for a method (an operator with only an EP rule, such as a hard comparison, cannot
 * run variational message passing; one with no `tilted` and no `ep` cannot run EP), with the operator's name: what a
 * method's capability check reports.
 *
 * @param g - The compiled graph.
 * @param rule - `ep`, `vmp` or `bp`.
 * @returns The names of the operators that lack the rule, without repeats.
 * @example
 * ```ts
 * import { compileGraph, defaultOperators } from 'omai-engine/message-passing'
 * import { operatorsWithout } from 'omai-engine/probprog/model'
 * const g = compileGraph({ variables: [{ name: 'x', family: { kind: 'gaussian' } }, { name: 'b', family: { kind: 'bernoulli' }, observed: 1 }], factors: [{ operator: 'greater-than', args: ['b', 'x'], params: { threshold: 0 } }, { operator: 'prior', args: ['x'], params: { eta: [0, 1] } }] }, defaultOperators())
 * [operatorsWithout(g, 'ep'), operatorsWithout(g, 'vmp')] // => [[], ['greater-than']]
 * ```
 */
export function operatorsWithout(g: CompiledGraph, rule: MessageRule): string[] {
  const out = new Set<string>()
  for (const f of g.factors) {
    const op = f.operator
    const has = rule === 'ep' ? op.ep !== undefined || op.tilted !== undefined : op[rule] !== undefined
    if (!has || op.unsupported?.(rule, f.params) !== undefined) out.add(op.name)
  }
  return [...out]
}

const num = (v: Json | undefined): number => (typeof v === 'number' ? v : Number.NaN)
const nums = (v: Json | undefined): number[] => (Array.isArray(v) ? (v as number[]) : [])

/**
 * The library's posterior of a message-passing state, keyed by site name: for each latent and recorded derived site of the
 * spec, the marginal as a distribution of the spec's shape (`normal` with the dims' batch shape, a `multivariateNormal`
 * for a vector site the compiler kept packed, `gamma`, `beta`, `bernoulli`, `categorical`, `dirichlet`). Variables with no
 * density yet (a state early in a run) read as an unbounded spread around zero and are listed in
 * `diagnostics.improper`; the diagnostics carry the sweeps, the last change, the improper-message counts and the
 * evidence or bound.
 *
 * @param c - The compilation (it must have a graph) and the compiled graph.
 * @param g - The graph compiled from `c.graph`.
 * @param state - A state of the run.
 * @param options - The method's name, the message rule (for the evidence), the status.
 * @returns The posterior (form `variational`).
 * @throws OmaiError with code `value` when the compilation has no graph.
 * @concept model-spec
 * @index function The Posterior of a message-passing state, by site name, in the spec's shapes.
 * @example
 * ```ts
 * import { compileGraph } from 'omai-engine/message-passing'
 * import { defineModel, factorGraphPosterior, toFactorGraph } from 'omai-engine/probprog/model'
 * import { normal } from 'omai-engine/probprog/model/dists'
 * import { defaultOperators, epAlgorithm } from 'omai-engine/message-passing'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const m = defineModel('m', (b) => { b.observe('y', normal(b.latent('mu', normal(0, 2)), 1), b.data('y', { dims: ['row'] })) })
 * const c = toFactorGraph(m.bind({ y: [1, 2, 3] }))
 * const g = compileGraph(c.graph as never, defaultOperators())
 * const r = runAlgorithm(epAlgorithm(g), { maxSteps: 50 })
 * Number(factorGraphPosterior(c, g, r.state).mean('mu').data[0]).toFixed(4) // => '1.8462'
 * ```
 */
export function factorGraphPosterior(
  c: FactorGraphCompilation,
  g: CompiledGraph,
  state: MpState,
  options: {
    readonly engine?: string
    readonly rule?: MessageRule
    readonly status?: 'completed' | 'converged' | 'stopped'
    /** Why the run did not converge, when it did not (see `mpVerdict`): carried in the diagnostics. */
    readonly reason?: string
  } = {},
): Posterior {
  if (c.graph === undefined)
    throw new OmaiError('value', `factor graph of '${c.name}' has issues: ${c.issues.map((i) => i.reason).join('; ')}`)
  const rule = options.rule ?? 'ep'
  const marginals = marginalsOf(g, state)
  const out: Record<string, Distribution> = {}
  const improper: string[] = []
  const sites: string[] = []
  const T = (xs: readonly number[], shape: readonly number[]): Value =>
    fromStorage(Float64Array.from(xs), shape, 'float64')
  for (const s of c.sites) {
    if (!s.posterior) continue
    sites.push(s.site)
    const m = (name: string | null): Record<string, Json> => marginals[name ?? ''] ?? {}
    const shape = s.shape
    if (s.layout === 'packed') {
      const mm = m(s.site)
      const mean = nums(mm['mean'])
      const cov = (mm['covariance'] as number[][] | undefined) ?? []
      if (mean.length === 0) {
        improper.push(s.site)
        const d = s.variables.length > 0 ? (g.variables[g.index.get(s.site) as number]?.family.dim ?? 1) : 1
        out[s.site] = multivariateNormal(new Array<number>(d).fill(0), {
          covariance: Array.from({ length: d }, (_, i) => Array.from({ length: d }, (_, j) => (i === j ? 1e12 : 0))),
        })
      } else out[s.site] = multivariateNormal(mean, { covariance: cov })
      continue
    }
    const vs = s.variables.map((v) => m(v))
    const bad = vs.some((x) => x['improper'] === true)
    if (bad) improper.push(s.site)
    switch (s.family) {
      case 'gaussian':
        out[s.site] = normal(
          T(
            vs.map((x) => (x['improper'] === true ? 0 : num(x['mean']))),
            shape,
          ),
          T(
            vs.map((x) => (x['improper'] === true ? 1e6 : Math.sqrt(num(x['variance'])))),
            shape,
          ),
        )
        break
      case 'gamma': {
        const shapes = T(
          vs.map((x) => (x['improper'] === true ? 1 : num(x['shape']))),
          shape,
        )
        const rates = T(
          vs.map((x) => (x['improper'] === true ? 1e-6 : num(x['rate']))),
          shape,
        )
        out[s.site] = s.reciprocal === true ? inverseGamma(shapes, rates) : gamma(shapes, rates)
        break
      }
      case 'normal-gamma': {
        // The joint of a mean and a precision shares one variable: the mean's marginal is a Student t (2a degrees of freedom around m,
        // scale sqrt(b / (a beta))), the precision's a gamma (an inverse gamma for a variance).
        const x = vs[0] ?? {}
        const proper = x['improper'] !== true
        const [m, beta, a, b] = [num(x['m']), num(x['beta']), num(x['a']), num(x['b'])]
        if (s.part === 'mean') out[s.site] = proper ? studentT(2 * a, m, Math.sqrt(b / (a * beta))) : normal(0, 1e6)
        else {
          const [sh, rt] = proper ? [a, b] : [1, 1e-6]
          out[s.site] = s.reciprocal === true ? inverseGamma(sh, rt) : gamma(sh, rt)
        }
        break
      }
      case 'beta':
        out[s.site] = beta(
          T(
            vs.map((x) => (x['improper'] === true ? 1 : num(x['a']))),
            shape,
          ),
          T(
            vs.map((x) => (x['improper'] === true ? 1 : num(x['b']))),
            shape,
          ),
        )
        break
      case 'bernoulli':
        out[s.site] = bernoulli({
          probs: T(
            vs.map((x) => num(x['p'])),
            shape,
          ),
        })
        break
      case 'discrete': {
        const k = nums(vs[0]?.['probs']).length
        out[s.site] = categorical({
          probs: T(
            vs.flatMap((x) => (nums(x['probs']).length === k ? nums(x['probs']) : new Array<number>(k).fill(1 / k))),
            [...shape, k],
          ),
        })
        break
      }
      case 'dirichlet': {
        const k = nums(vs[0]?.['alpha']).length
        out[s.site] = dirichlet(
          T(
            vs.flatMap((x) => nums(x['alpha'])),
            [...shape, k],
          ),
        )
        break
      }
      default:
        break
    }
  }
  const evidence = evidenceOf(g, { projection: projectionOf(rule) }, state)
  return definePosterior({
    form: 'variational',
    engine: options.engine ?? rule,
    sites: sites.filter((s) => out[s] !== undefined),
    status: options.status ?? 'completed',
    marginals: out,
    ...(Number.isFinite(evidence)
      ? { logEvidence: { value: evidence, method: rule === 'vmp' ? 'elbo' : rule === 'bp' ? 'bethe' : 'ep-energy' } }
      : {}),
    diagnostics: {
      sweeps: state.sweeps,
      updates: state.updates,
      change: Number.isFinite(state.change) ? state.change : null,
      improper: improper,
      improperMessages: state.improper.events,
      repaired: { skipped: state.improper.skipped, damped: state.improper.damped, reset: state.improper.reset },
      unresolved: state.unresolved ?? 0,
      restarts: state.restarts ?? 0,
      ...(options.reason === undefined ? {} : { reason: options.reason }),
      // Why there is no evidence (when there is none): the factors with no part of it. A plated factor whose cavity is improper
      // (a negative-precision site on the weights) is the usual one; `plates: 'expand'` has a proper cavity per row.
      ...(rule === 'ep' && !Number.isFinite(evidence)
        ? {
            evidenceUndefinedAt: evidenceDefects(g, { projection: projectionOf(rule) }, state).map(
              (d) => `${d.factor}: ${d.reason}`,
            ),
          }
        : {}),
      variables: g.variables.length,
      factors: g.factors.length,
    },
  })
}

/**
 * Where a graph's ids sit on the spec's picture: each variable id and each factor id of the compiled graph mapped to the id
 * of the spec node or factor site it belongs to (`w[2]` and `y#0[3]` to `w` and `y`, `factor:y[3]` to `factor:y`). A view
 * that draws `toStructure(spec)` uses it to paint the marginals of a run, keyed by variable id, on the spec's nodes.
 *
 * @param graph - The factor graph of a compilation.
 * @returns The two maps.
 * @example
 * ```ts
 * import { graphOnSpec } from 'omai-engine/probprog/model'
 * const g = graphOnSpec({ variables: [{ name: 'y#0[3]', family: { kind: 'gaussian' } }], factors: [{ operator: 'prior', args: ['y#0[3]'], params: { eta: [0, 1] }, id: 'factor:y#0[3]' }] })
 * [g.variables['y#0[3]'], g.factors['factor:y#0[3]']] // => ['y', 'factor:y']
 * ```
 */
export function graphOnSpec(graph: FactorGraphSpec): {
  readonly variables: Readonly<Record<string, string>>
  readonly factors: Readonly<Record<string, string>>
} {
  const variables: Record<string, string> = {}
  for (const v of graph.variables) variables[v.name] = siteOfId(v.name)
  const factors: Record<string, string> = {}
  for (const f of graph.factors) if (f.id !== undefined) factors[f.id] = siteOfId(f.id)
  return { variables, factors }
}

/**
 * Whether a bound model has a factor graph a message-passing method can run, and exactly why not: the static check behind
 * `ep`, `vmp` and `bp` as inference methods. It reads the compilation (the issues, with node and reason) and, for a rule, the
 * operators of the compiled graph that lack it.
 *
 * @param c - The compilation.
 * @param g - The compiled graph, when the compilation has a graph.
 * @param rule - The method's rule.
 * @returns `ok` with a reason, or the reasons it does not apply and the missing features.
 * @example
 * ```ts
 * import { defineModel, factorGraphVerdict, toFactorGraph } from 'omai-engine/probprog/model'
 * import { normal } from 'omai-engine/probprog/model/dists'
 * const m = defineModel('m', (b) => { b.observe('y', normal(b.latent('mu', normal(0, 2)), b.latent('s', normal(1, 1))), b.data('y', { dims: ['row'] })) })
 * factorGraphVerdict(toFactorGraph(m.bind({ y: [1, 2] })), undefined, 'ep').ok // => false
 * ```
 */
export function factorGraphVerdict(
  c: FactorGraphCompilation,
  g: CompiledGraph | undefined,
  rule: MessageRule,
): { readonly ok: boolean; readonly reason: string; readonly missing: readonly string[] } {
  if (c.graph === undefined || g === undefined) {
    const reasons = c.issues.map((i) => `${i.node}: ${i.reason}`)
    const missing = [
      ...new Set(
        c.issues.map((i) =>
          i.kind === 'opaque' || i.kind === 'family'
            ? 'exponential-family factors'
            : `a factor graph without ${i.kind} issues`,
        ),
      ),
    ]
    return { ok: false, reason: reasons.join('; '), missing }
  }
  const lacking = operatorsWithout(g, rule)
  if (lacking.length > 0) {
    const why = lacking.flatMap((name) => {
      const factors = g.factors.filter((f) => f.operator.name === name)
      const note =
        factors[0]?.operator.noRule?.[rule] ??
        factors.map((f) => f.operator.unsupported?.(rule, f.params)).find((v) => v !== undefined)
      return note === undefined ? [] : [`${name}: ${note}`]
    })
    return {
      ok: false,
      reason: `the operator${lacking.length > 1 ? 's' : ''} ${lacking.join(', ')} ${lacking.length > 1 ? 'have' : 'has'} no ${rule === 'ep' ? 'expectation propagation' : rule === 'vmp' ? 'variational' : 'sum-product'} message rule${why.length > 0 ? ` (${why.join('; ')})` : ''}`,
      missing: [`${rule} message rules`],
    }
  }
  return {
    ok: true,
    reason: `a factor graph of ${g.variables.length} variables and ${g.factors.length} factors`,
    missing: [],
  }
}

/**
 * The compiled graph (indexed, ready for the engine) of a compilation, against the engine's operators or a registry of
 * your own.
 *
 * @param c - The compilation.
 * @param operators - The factor operators by name (default the engine's `defaultOperators()`).
 * @returns The compiled graph.
 * @throws OmaiError with code `value` when the compilation has issues (no graph), naming each node and reason.
 * @concept model-spec
 * @example
 * ```ts
 * import { compileFactorGraph, defineModel, toFactorGraph } from 'omai-engine/probprog/model'
 * import { normal } from 'omai-engine/probprog/model/dists'
 * const m = defineModel('m', (b) => { b.observe('y', normal(b.latent('mu', normal(0, 2)), 1), b.data('y', { dims: ['row'] })) })
 * const g = compileFactorGraph(toFactorGraph(m.bind({ y: [1, 2, 3] })))
 * [g.variables.length, g.factors.length, g.edges.length] // => [4, 4, 4]
 * ```
 */
export function compileFactorGraph(
  c: FactorGraphCompilation,
  operators: Readonly<Record<string, FactorOperator>> = factorOperators(),
): CompiledGraph {
  if (c.graph === undefined)
    throw new OmaiError(
      'value',
      `'${c.name}' does not compile to a factor graph: ${c.issues.map((i) => `${i.node}: ${i.reason}`).join('; ')}`,
    )
  return compileGraph(c.graph, operators)
}
