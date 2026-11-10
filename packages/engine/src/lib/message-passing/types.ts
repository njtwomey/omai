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
}

/** The parameters of a factor: JSON constants (a coefficient vector, a threshold, a table). */
export type Params = Readonly<Record<string, Json>>

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
  /** True when the operator reads `OperatorContext.power` (fractional EP); the engine refuses a power on the others. */
  readonly power?: boolean
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
  readonly name?: string
}

/** A factor graph as JSON: variables with their families, factors with their operators by name. */
export interface FactorGraphSpec {
  readonly variables: readonly VariableSpec[]
  readonly factors: readonly FactorSpec[]
}
