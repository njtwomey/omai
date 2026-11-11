/**
 * The contracts of message passing: the message families (an exponential family in natural parameters, with product,
 * quotient, moments, divergence), the factor operator (what a factor says to each of its arguments, given what the
 * others say to it), and the factor graph the engine runs on, as JSON. Nothing here computes; the families are in
 * `families.ts` and `families-vector.ts`, the operators in `operators/`, the engine in `engine.ts`.
 *
 * Three words carry the whole design. A **message** is a number array: the natural parameters of an exponential family
 * member, up to a constant that is never tracked. A **family** knows what to do with messages (add, subtract, scale,
 * turn them into moments and back). A **factor operator** is a function from the messages of the other arguments of a
 * factor to the message for one argument; which function (the moment-matching one of EP, the expected-log one of VMP,
 * the sum-product one of BP) is what the `projection` slot of the engine picks. A factor with only a density can be
 * given as an operator whose messages come from a numeric `tilted` function; the engine then projects the moments.
 *
 * ## Ids: one convention for the graph, the tape and the diagram
 *
 * Everything a run reports is addressed by ids, the same ones the picture of the model uses (`pgm/structure.ts`
 * `toStructure` for a `pgm` factor graph; `probprog/model/ids.ts` for a model spec, whose compiler fills them in), so a
 * diagram can be coloured and annotated by a run with no lookup:
 * - a **variable** is its own `name` (the name of the model's variable or site; observed variables included, so a
 *   spec's `data:<name>` nodes keep that name here);
 * - a **factor** is `FactorSpec.id` when it has one (a model spec's compiler gives `factor:<site>`), else `f<i>` with
 *   `i` its index in `FactorGraphSpec.factors` (what `toStructure` gives the factors of a `pgm` factor graph);
 *   `FactorSpec.name` is a label, never an id; ids are unique;
 * - an **edge** is named by its ends, `<factor id>><variable id>`; a **message** is directed along it, of kind
 *   `factorToVariable` (from the factor id to the variable id), the form of the `message` events of
 *   `pgm/inference/belief-propagation.ts` (`from` and `to` set);
 * - the marginals of a run are keyed by variable id, in `MpSnapshot.marginals`, in the `mp/marginals` message and in the
 *   `frame` of each observation (`{ marginals, sent }`, `sent` the messages of the last step as `{ kind, from, to }`).
 * `factorId`, `variableId` and `graphStructure` (in `graph.ts`) are where the defaults are made; the compiled graph
 * carries each factor's id.
 */

import type { Json } from '../base/index.ts'

/** A message: the natural parameters of a member of a family, in the layout the family documents. */
export type Eta = readonly number[]

/** The names of the families. */
export type FamilyKind =
  | 'gaussian'
  | 'gaussian-mv'
  | 'gamma'
  | 'beta'
  | 'bernoulli'
  | 'discrete'
  | 'dirichlet'
  | 'normal-gamma'

/** A family as plain data, so a factor graph is JSON: its kind and, where it has one, its size. */
export interface FamilySpec {
  readonly kind: FamilyKind
  /** The length of a multivariate Gaussian or a Dirichlet, the number of states of a discrete variable. */
  readonly size?: number
}

/** The result of dividing one message by another: the message, and whether it is not a normalisable member. */
export interface Quotient {
  readonly eta: Eta
  /** True when the quotient has no density (a negative precision, a non-positive shape or rate). */
  readonly improper: boolean
}

/**
 * An exponential family p(x) = exp(eta . T(x) - A(eta)) whose members are the messages of a variable. A message is the
 * natural parameters `eta` only; the constant it is defined up to never appears. For a Gaussian the natural parameters
 * are in precision form (`h = precision * mean`, `J = precision`), so `log q = h x - J x^2 / 2 - A`; the other families
 * use the usual ones (see each family's own documentation).
 *
 * Product, quotient and power are the vector operations `a + b`, `a - b` and `p a` on natural parameters (a discrete
 * variable's log weights are renormalised after each, so they stay finite); the family adds the meaning:
 * whether the result is a proper distribution, its moments, the member with given moments (the KL projection), and the
 * divergence.
 *
 * @concept exponential-families
 * @index structure A message family: natural parameters, moments, quotient, KL.
 */
export interface Family {
  readonly kind: FamilyKind
  readonly spec: FamilySpec
  /** The size of the family (vector length, number of states; 1 for a scalar family). */
  readonly dim: number
  /** The number of entries of a message. */
  readonly size: number
  /** The message that says nothing: all zeros (improper for the continuous families, uniform for the discrete ones). */
  readonly flat: Eta
  /** Whether a message has a density: a positive precision, positive shape and rate, a positive definite matrix. */
  isProper(eta: Eta): boolean
  /** The message a - the message b added: the product of the two densities, up to a constant. */
  product(a: Eta, b: Eta): Eta
  /** The message a divided by b, with the flag that it is improper. */
  quotient(a: Eta, b: Eta): Quotient
  /** The message a to the power p (fractional EP; p = 0 is the flat message). */
  power(a: Eta, p: number): Eta
  /** The expected sufficient statistics E[T(x)] of a proper message. */
  moments(eta: Eta): Eta
  /** The member whose expected sufficient statistics are these (the KL(p || q) projection of anything with those moments). */
  fromMoments(moments: Eta): Eta
  /** The pairing eta . T(x) when the sufficient statistics have these expectations: what the log density is linear in. */
  pair(eta: Eta, moments: Eta): number
  /** A(eta) = log of the integral of exp(eta . T(x)): the log of the mass of the unnormalised message. */
  logPartition(eta: Eta): number
  /** The entropy of a proper message. */
  entropy(eta: Eta): number
  /** KL(p || q) for proper messages of the family. */
  kl(p: Eta, q: Eta): number
  /** How much a message moved, dimensionless (moments in units of the standard deviation; probabilities); for the tape and the stopping rule. */
  change(previous: Eta, next: Eta): number
  /** A message in words (`{ mean, variance }`, `{ probs }`), for the tape and the snapshot; `improper: true` when it has no density. */
  summary(eta: Eta): Readonly<Record<string, Json>>
  /** For a family on a finite set (Bernoulli, discrete): the log weights of the states, normalised. */
  logProbs?(eta: Eta): readonly number[]
  /** The inverse of `logProbs`. */
  fromLogProbs?(logProbs: readonly number[]): Eta
}

/** A value a factor takes as an argument that is not random: an observation or a constant, one number or a vector. */
export type Observed = number | readonly number[]

/**
 * What a factor operator sees of one of its arguments: the family of the variable, and either the incoming message
 * or, when the variable is observed (or the argument a constant), its value.
 *
 * What the message is depends on the projection: under `ep` and `bp` it is the message from the variable to the factor
 * (the product of the other factors' messages to the variable, the cavity); under `vmp` it is the variable's current
 * belief q (everything it has heard).
 */
export interface Arg {
  readonly family: Family
  /** The incoming message; `undefined` when the argument is observed. */
  readonly eta: Eta | undefined
  /** The value, when the argument is observed or constant. */
  readonly value: Observed | undefined
}

/** Which message rule an operator is asked for. `ep` moment matching, `vmp` expected log factor, `bp` sum-product. */
export type Mode = 'ep' | 'vmp' | 'bp'

/** How a message rule relates to the true message: closed form, a bound (a variational lower bound), or numeric (quadrature). */
export type Support = 'exact' | 'bound' | 'numeric'

/** One argument of a factor: its name, the families it takes, and whether it may be observed or constant. */
export interface Port {
  readonly name: string
  readonly accepts: readonly FamilyKind[]
  /** Whether the operator handles an observed value here (`Arg.value`). */
  readonly constant?: boolean
  /** True when the port must be observed or a constant (the operator never sends to it and cannot take a message there). */
  readonly observed?: boolean
}

/** The parameters of a factor: JSON constants (a coefficient vector, a threshold, a table). */
export type Params = Readonly<Record<string, Json>>

/** One argument of a numeric problem: a Gaussian (the cavity of a Gaussian port), or a point (variance 0: an observed value). */
export interface GaussianBase {
  readonly mean: number
  readonly variance: number
}

/** What a numeric rule integrates: a log density against Gaussian bases, and which argument's moments are asked for. */
export interface TiltedProblem {
  /** The argument of every port: a Gaussian cavity, or a point for an observed value (variance 0). */
  readonly bases: readonly GaussianBase[]
  /** The argument whose marginal moments are asked for. */
  readonly to: number
  /** log of the (powered) factor at a point, one coordinate per argument. */
  readonly logf: (x: readonly number[]) => number
}

/**
 * A numeric rule: how the moments of a factor against Gaussian cavities are found when there is no closed form
 * (quadrature, Laplace's mode and curvature, importance samples, sigma points). It is what the numeric operator asks
 * of the slot `Projection.numeric` of the engine, so one factor graph runs under any of them.
 */
export interface TiltedRule {
  readonly name: string
  /**
   * The log of the integral of the factor against the (normalised) Gaussian bases, and the mean and variance of the
   * marginal of argument `to` of the density they define.
   */
  moments(problem: TiltedProblem): { readonly logMass: number; readonly mean: number; readonly variance: number }
}

/** What an operator is asked: send a message to port `to`, given the others. */
export interface OperatorContext {
  readonly mode: Mode
  /** The port to send to. */
  readonly to: number
  /** Every port's argument, the target's included (a non-closed-form operator reads its own incoming message). */
  readonly args: readonly Arg[]
  readonly params: Params
  /** The power of fractional EP (1: ordinary EP). Only operators with `power: true` read it. */
  readonly power: number
  /** The numeric rule the run asks of operators that integrate numerically (`Projection.numeric`); absent: the operator's own. */
  readonly rule?: TiltedRule
}

/** The message an operator sends to the target port. */
export interface OperatorResult {
  readonly message: Eta
}

/** The moments of the tilted distribution at a port (cavity times factor, the other ports integrated out) and its mass. */
export interface TiltedMoments {
  /** E[T(x)] at the target port under the tilted distribution. */
  readonly moments: Eta
  /** log of the integral of the factor times every incoming message (the same for every target). */
  readonly logZ: number
}

/**
 * A factor operator: the messages of one kind of factor (a sum, a probit link, a Gaussian likelihood, a table).
 *
 * `ep`, `vmp` and `bp` are the closed-form message rules; an operator gives the ones it has, and `support` says what
 * each is (`exact`, `bound`, `numeric`). An operator with only `tilted` is projected by the engine: the numeric
 * moments become a message by the family's `fromMoments`, divided by the incoming message. `logEvidence` and
 * `expectedLog` are what the evidence estimate and the variational bound need from the factor.
 *
 * The operator is a value, never registered by importing it: `defaultOperators()` collects the library's, and a graph
 * is compiled against the collection it is given.
 *
 * @concept factor-operators
 * @index structure A factor operator: the message to each argument of a factor.
 */
export interface FactorOperator {
  readonly name: string
  readonly description: string
  /** The arguments, in order; a function of the parameters for a factor of variable arity (a sum, a table). */
  readonly ports: readonly Port[] | ((params: Params) => readonly Port[])
  /** What each rule is; a rule that is absent is unsupported. */
  readonly support: Readonly<Partial<Record<Mode, Support>>>
  /**
   * True when the message to a port depends on that port's own incoming message (a moment-matching operator: the
   * tilted distribution is the factor times the cavity). The engine then sends nothing to a port whose incoming message
   * has no density, since there is nothing to match.
   */
  readonly readsTarget?: boolean | ((to: number) => boolean)
  /**
   * True when the operator needs a density at every Gaussian port (it integrates over the cavities) and cannot read one that is
   * still flat (no information yet). The engine's one policy for improper first messages then applies: a flat Gaussian cavity is
   * shown to the operator as a very wide one (`FLAT_CAVITY_PRECISION`), and the message it returns is taken back against the flat
   * cavity, so the factor speaks on a variable that nothing else has spoken of yet and the others' first messages decide.
   * Absent: the operator reads a flat cavity itself (a pushforward, a prior) or sends nothing (`readsTarget`).
   */
  readonly needsDensity?: boolean
  /**
   * True when the operator reads `OperatorContext.power` (fractional EP); the engine refuses a power on the others. A function
   * of the parameters for an operator that takes one only for some of its configurations (a gate, by its branches); see `takesPower`.
   */
  readonly power?: boolean | ((params: Params) => boolean)
  /**
   * The port this factor defines as a deterministic function of the others (the output of a sum, a projection, a copy), if
   * it does. Under variational message passing a variable so defined is not a free variable of the mean field: its belief
   * is the pushforward its definer sends (children's messages never enter it), and the definer is shown the cavity of the
   * output, the messages its other factors would send, which it passes to the inputs. Absent: a stochastic factor.
   */
  readonly defines?: number
  /** Why a rule has no message, in words, for the capability check (`greater-than` has no variational rule: a hard step has no finite KL). */
  readonly noRule?: Readonly<Partial<Record<Mode, string>>>
  /** A reason the factor cannot be built from these parameters and families, or `undefined` when it can. */
  check?(params: Params, families: readonly Family[]): string | undefined
  ep?(c: OperatorContext): OperatorResult
  vmp?(c: OperatorContext): OperatorResult
  bp?(c: OperatorContext): OperatorResult
  /** The numeric fallback for `ep`: moments of the tilted distribution, projected by the engine. */
  tilted?(c: OperatorContext): TiltedMoments
  /** log of the integral of the factor times the incoming messages of every port (under `ep` and `bp`). */
  logEvidence?(c: OperatorContext): number
  /** E_q[log f] under the beliefs of the ports (under `vmp`). */
  expectedLog?(c: OperatorContext): number
}

/** A variable of a factor graph. An observed variable sends no message and is read as a value by its factors. */
export interface VariableSpec {
  readonly name: string
  readonly family: FamilySpec
  readonly observed?: Observed
}

/** An argument of a factor: the name of a variable, or a constant. */
export type ArgSpec = string | { readonly const: Observed }

/** A factor: an operator by name, its arguments, its parameters, and (power EP) its power. */
export interface FactorSpec {
  readonly operator: string
  readonly args: readonly ArgSpec[]
  readonly params?: Params
  /** The power of fractional EP for this factor (default 1). */
  readonly power?: number
  /** A label for a drawing (default the operator's name). */
  readonly name?: string
  /** The factor's id on the tape and in a drawing (default `f<index>`); unique within the graph. */
  readonly id?: string
  /**
   * The message each port starts with, in the order of `args` (`null`: the flat one; the entries of observed or constant
   * arguments are ignored). A start message is replaced by the factor's own first update, so it only gives the other factors a
   * proper cavity to begin from: a vague Gaussian where nothing yet says anything about a variable, in place of a vague
   * prior factor that would stay in the model.
   */
  readonly start?: readonly (readonly number[] | null)[]
}

/** A factor graph as JSON: variables with their families, factors with their operators by name. */
export interface FactorGraphSpec {
  readonly variables: readonly VariableSpec[]
  readonly factors: readonly FactorSpec[]
}

/**
 * Whether a factor of this operator and these parameters takes a power other than 1 (fractional EP, an alpha-divergence).
 *
 * @param op - The operator.
 * @param params - The factor's parameters.
 * @returns True when the operator declares a power for them.
 */
export function takesPower(op: FactorOperator, params: Params): boolean {
  return typeof op.power === 'function' ? op.power(params) : op.power === true
}
