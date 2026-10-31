/**
 * Fuzzy logic over tensors, in the manner of Logic Tensor Networks (Badreddine et al. 2022): truth values are numbers in
 * [0, 1] held in tensors, connectives are differentiable tensor operations from a t-norm family (product, Lukasiewicz,
 * Goedel, Hamacher, each with its dual t-conorm, standard negation, residuated and material implications), quantifiers
 * are aggregators over the axis of a variable (the generalised mean and its error form, the extremes, a t-norm), and a
 * formula's satisfaction is differentiable in everything it is computed from. A theory is a set of first-order axioms
 * over predicates whose truth values are learnable, a table of logits over the constants or a small network on their
 * features; the satisfaction of the axioms is the training objective, stated as a `SmoothProblem` so the optimiser is
 * the solver slot of `smoothSolvers`. The same connectives evaluate propositional formulas over a network's outputs.
 */

import { MdmlError } from '../base/index.ts'
import {
  type GradientPath,
  gradientContract,
  loggedSolver,
  type SmoothProblem,
  type SolverPath,
  smoothSolverConfig,
  solverContract,
} from '../optim/index.ts'
import { type Key, key as makeKey, normal } from '../random/index.ts'
import type { SpaceValue } from '../space/index.ts'
import {
  add,
  asTensor,
  broadcastTo,
  clip,
  concat,
  div,
  einsum,
  fromStorage,
  item,
  lessEqual,
  matmul,
  maximum,
  max as maxReduce,
  mean as meanReduce,
  minimum,
  min as minReduce,
  mul,
  pow,
  prod,
  reshape,
  sigmoid,
  sliceAxis,
  squeeze,
  sub,
  sum,
  type TensorLike,
  tanh,
  toFlat,
  transpose,
  type Value,
  where,
} from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  contractSchema,
  defineContract,
  runAlgorithm,
  scalar,
  tensor as tensorSource,
} from '../trace/index.ts'
import { type FolFormula, parseFol } from './first-order.ts'
import { type Formula, parseFormula } from './propositional.ts'

const EPS = 1e-9

// ------------------------------------------------------------------------------------------------ t-norm families

/** The names of the t-norm families. */
export type NormName = 'product' | 'lukasiewicz' | 'goedel' | 'hamacher'

/**
 * A t-norm family as differentiable tensor operations: the conjunction (t-norm), its dual disjunction (t-conorm
 * `1 - T(1 - a, 1 - b)`), the standard negation `1 - a`, the residuated implication (`R(a, b) = sup{z : T(a, z) <= b}`)
 * and the material (S-)implication `S(1 - a, b)`, which for the product family is Reichenbach's `1 - a + ab`, the
 * implication whose gradient does not vanish where the antecedent is false. `reduceAnd` and `reduceOr` fold the
 * connective along an axis.
 */
export interface NormFamily {
  readonly name: string
  and(a: TensorLike, b: TensorLike): Value
  or(a: TensorLike, b: TensorLike): Value
  not(a: TensorLike): Value
  /** The residuated implication. */
  residuum(a: TensorLike, b: TensorLike): Value
  /** The material implication `or(not(a), b)`. */
  implies(a: TensorLike, b: TensorLike): Value
  reduceAnd(x: TensorLike, axis: number): Value
  reduceOr(x: TensorLike, axis: number): Value
}

const one = (x: TensorLike): Value => sub(1, x)

/** Fold a binary connective along an axis. */
function foldAxis(x: TensorLike, axis: number, op: (a: Value, b: Value) => Value, unit: number): Value {
  const v = asTensor(x, 'float64')
  const ax = axis < 0 ? axis + v.shape.length : axis
  const n = v.shape[ax] ?? 0
  const rest = v.shape.filter((_, i) => i !== ax)
  if (n === 0) return broadcastTo(unit, rest)
  let acc = squeeze(sliceAxis(v, ax, 0, 1), ax)
  for (let i = 1; i < n; i++) acc = op(acc, squeeze(sliceAxis(v, ax, i, i + 1), ax))
  return acc
}

function makeFamily(
  name: string,
  and: (a: TensorLike, b: TensorLike) => Value,
  residuum: (a: TensorLike, b: TensorLike) => Value,
  reduceAnd: (x: TensorLike, axis: number) => Value,
  reduceDual: (x: TensorLike, axis: number) => Value,
): NormFamily {
  const or = (a: TensorLike, b: TensorLike): Value => one(and(one(a), one(b)))
  return {
    name,
    and,
    or,
    not: one,
    residuum,
    implies: (a, b) => or(one(a), b),
    reduceAnd,
    reduceOr: reduceDual,
  }
}

const product = makeFamily(
  'product',
  mul,
  (a, b) => where(lessEqual(a, b), 1, div(b, maximum(a, EPS))),
  (x, axis) => prod(x, axis),
  (x, axis) => one(prod(one(x), axis)),
)

const lukasiewicz = makeFamily(
  'lukasiewicz',
  (a, b) => maximum(0, sub(add(a, b), 1)),
  (a, b) => minimum(1, add(one(a), b)),
  (x, axis) => {
    const n = asTensor(x, 'float64').shape[axis < 0 ? axis + asTensor(x).shape.length : axis] ?? 1
    return maximum(0, sub(sum(x, axis), n - 1))
  },
  (x, axis) => minimum(1, sum(x, axis)),
)

const goedel = makeFamily(
  'goedel',
  minimum,
  (a, b) => where(lessEqual(a, b), 1, b),
  (x, axis) => minReduce(x, axis),
  (x, axis) => maxReduce(x, axis),
)

/**
 * The Hamacher family with parameter `lambda >= 0`: `T(a, b) = ab / (lambda + (1 - lambda)(a + b - ab))`, the product
 * for `lambda = 1` and the Hamacher product for `lambda = 0`. It has no closed-form residuum here: `residuum` is the
 * material implication.
 *
 * @param lambda - The parameter (default 0).
 * @returns The family.
 * @throws MdmlError with code `value` for a negative parameter.
 * @example
 * ```ts
 * import { hamacherFamily } from 'mdml-engine/logic'
 * Number(hamacherFamily(0).and(0.5, 0.5).data[0]) // => 1 / 3
 * ```
 */
export function hamacherFamily(lambda = 0): NormFamily {
  if (!(lambda >= 0)) throw new MdmlError('value', 'hamacherFamily: lambda must be non-negative')
  const and = (a: TensorLike, b: TensorLike): Value => {
    const ab = mul(a, b)
    const den = add(lambda, mul(1 - lambda, sub(add(a, b), ab)))
    return div(ab, maximum(den, 1e-12))
  }
  const fold = (x: TensorLike, axis: number): Value => foldAxis(x, axis, and, 1)
  const family = makeFamily(
    'hamacher',
    and,
    (a, b) => add(one(a), mul(a, b)),
    fold,
    (x, axis) => one(fold(one(x), axis)),
  )
  return { ...family, residuum: family.implies }
}

/**
 * A t-norm family by name.
 *
 * @param name - `product`, `lukasiewicz`, `goedel` or `hamacher` (lambda 0).
 * @returns The family.
 * @throws MdmlError with code `value` for an unknown name.
 * @example
 * ```ts
 * import { normFamily } from 'mdml-engine/logic'
 * Number(normFamily('lukasiewicz').and(0.6, 0.7).data[0].toFixed(6)) // => 0.3
 * ```
 */
export function normFamily(name: NormName): NormFamily {
  switch (name) {
    case 'product':
      return product
    case 'lukasiewicz':
      return lukasiewicz
    case 'goedel':
      return goedel
    case 'hamacher':
      return hamacherFamily(0)
    default:
      throw new MdmlError('value', `normFamily: unknown family '${String(name)}'`)
  }
}

// ------------------------------------------------------------------------------------------------ aggregators

/** A quantifier: reduces a truth tensor along one axis to a truth value in [0, 1]. */
export type Aggregator = (x: TensorLike, axis: number) => Value

/**
 * The generalised (power) mean `(mean x^p)^(1/p)` along an axis: the mean for `p = 1`, approaching the maximum as `p`
 * grows, so it reads as an existential quantifier. Inputs are clipped to a small positive floor so the gradient is
 * finite.
 *
 * @param p - The exponent (at least 1).
 * @returns The aggregator.
 * @throws MdmlError with code `value` for `p < 1`.
 * @example
 * ```ts
 * import { pMean } from 'mdml-engine/logic'
 * Number(pMean(2)([0, 1], 0).data[0].toFixed(6)) // => 0.707107
 * ```
 */
export function pMean(p: number): Aggregator {
  if (!(p >= 1)) throw new MdmlError('value', 'pMean: p must be at least 1')
  return (x, axis) => pow(meanReduce(pow(clip(x, EPS, 1), p), axis), 1 / p)
}

/**
 * The generalised mean of the errors, `1 - (mean (1 - x)^p)^(1/p)`: the mean for `p = 1`, approaching the minimum as
 * `p` grows, so it reads as a universal quantifier that one counterexample pulls down without silencing the rest.
 *
 * @param p - The exponent (at least 1).
 * @returns The aggregator.
 * @throws MdmlError with code `value` for `p < 1`.
 * @example
 * ```ts
 * import { pMeanError } from 'mdml-engine/logic'
 * Number(pMeanError(2)([0, 1], 0).data[0].toFixed(6)) // => 0.292893
 * ```
 */
export function pMeanError(p: number): Aggregator {
  if (!(p >= 1)) throw new MdmlError('value', 'pMeanError: p must be at least 1')
  return (x, axis) => one(pow(meanReduce(pow(clip(one(x), EPS, 1), p), axis), 1 / p))
}

/** Aggregators beyond the power means: the extremes, the mean and the t-norm and t-conorm folds of a family. */
export const aggregators: {
  readonly mean: Aggregator
  readonly min: Aggregator
  readonly max: Aggregator
  tnorm(family: NormFamily): Aggregator
  tconorm(family: NormFamily): Aggregator
} = {
  mean: (x, axis) => meanReduce(x, axis),
  min: (x, axis) => minReduce(x, axis),
  max: (x, axis) => maxReduce(x, axis),
  tnorm: (family) => (x, axis) => family.reduceAnd(x, axis),
  tconorm: (family) => (x, axis) => family.reduceOr(x, axis),
}

/** The semantics of a theory: the family, the quantifiers and the implication connectives use. */
export interface FuzzySemantics {
  readonly norm: NormFamily
  readonly forall: Aggregator
  readonly exists: Aggregator
  /** `material` (the family's S-implication, default) or `residuum`. */
  readonly implication: 'material' | 'residuum'
}

/** Options of {@link fuzzySemantics}. */
export interface FuzzySemanticsOptions {
  /** The t-norm family (default product). */
  readonly norm?: NormName | NormFamily
  /** The universal quantifier (default the p-mean error with p = 2). */
  readonly forall?: Aggregator
  /** The existential quantifier (default the p-mean with p = 2). */
  readonly exists?: Aggregator
  readonly implication?: 'material' | 'residuum'
}

/**
 * The semantics of a fuzzy theory with the defaults of Logic Tensor Networks: the product family with Reichenbach's
 * implication, the p-mean error (p = 2) for `forall` and the p-mean (p = 2) for `exists`.
 *
 * @param options - Any of the family, quantifiers and implication.
 * @returns The semantics.
 * @example
 * ```ts
 * import { fuzzySemantics } from 'mdml-engine/logic'
 * fuzzySemantics({ norm: 'goedel' }).norm.name // => 'goedel'
 * ```
 */
export function fuzzySemantics(options: FuzzySemanticsOptions = {}): FuzzySemantics {
  const norm = typeof options.norm === 'object' ? options.norm : normFamily(options.norm ?? 'product')
  return {
    norm,
    forall: options.forall ?? pMeanError(2),
    exists: options.exists ?? pMean(2),
    implication: options.implication ?? 'material',
  }
}

// ------------------------------------------------------------------------------------------------ propositional

/**
 * The satisfaction of a propositional formula over truth tensors: variables take their truth values from the
 * valuation (any shapes that broadcast), and the connectives are the semantics' t-norm family, so the result is a
 * truth tensor differentiable in the inputs. The use is a rule over a network's outputs: the satisfaction of
 * `a -> b` over a batch of predicted probabilities, as a loss term.
 *
 * @param formula - A propositional formula, or text in `parseFormula` syntax.
 * @param values - The truth value of each variable.
 * @param semantics - The semantics (default product).
 * @returns The truth value, with the broadcast shape of the inputs.
 * @throws MdmlError with code `value` for a variable the valuation lacks.
 * @example
 * ```ts
 * import { formulaSatisfaction } from 'mdml-engine/logic'
 * Number(formulaSatisfaction('a -> b', { a: 0.8, b: 0.5 }).data[0].toFixed(6)) // => 0.6
 * ```
 */
export function formulaSatisfaction(
  formula: Formula | string,
  values: Readonly<Record<string, TensorLike>>,
  semantics: FuzzySemantics = fuzzySemantics(),
): Value {
  const f = typeof formula === 'string' ? parseFormula(formula) : formula
  const { norm } = semantics
  const imp = (a: TensorLike, b: TensorLike): Value =>
    semantics.implication === 'residuum' ? norm.residuum(a, b) : norm.implies(a, b)
  const go = (g: Formula): Value => {
    switch (g.kind) {
      case 'var': {
        const v = values[g.name]
        if (v === undefined) throw new MdmlError('value', `formulaSatisfaction: no value for variable '${g.name}'`)
        return asTensor(v, 'float64')
      }
      case 'const':
        return asTensor(g.value ? 1 : 0, 'float64')
      case 'not':
        return norm.not(go(g.arg))
      case 'and':
        return g.args.map(go).reduce((a, b) => norm.and(a, b), asTensor(1, 'float64'))
      case 'or':
        return g.args.map(go).reduce((a, b) => norm.or(a, b), asTensor(0, 'float64'))
      case 'implies':
        return imp(go(g.left), go(g.right))
      case 'iff': {
        const a = go(g.left)
        const b = go(g.right)
        return norm.and(imp(a, b), imp(b, a))
      }
      case 'xor': {
        const a = go(g.left)
        const b = go(g.right)
        return norm.and(norm.or(a, b), norm.not(norm.and(a, b)))
      }
    }
  }
  return go(f)
}

// ------------------------------------------------------------------------------------------------ predicates

/**
 * A learnable predicate: how many parameters it has, their initial values, and its truth values on the cartesian
 * product of its arguments' feature rows (an output of one axis per argument, in order).
 */
export interface PredicateModel {
  readonly size: number
  readonly init: readonly number[]
  /** Truth values in [0, 1], shape `[n_1, ..., n_k]` for arguments of `n_i` rows. */
  truth(theta: Value, args: readonly Value[]): Value
}

/**
 * A predicate that is a table of truth logits over the constants of its argument domains: the arguments' features are
 * one-hot rows over the domain sizes given, and the truth is the sigmoid of the logit of the tuple.
 *
 * @param sizes - The domain size of each argument (empty for a proposition).
 * @param init - The initial logit of every entry (default 0).
 * @returns The predicate.
 * @example
 * ```ts
 * import { tablePredicate } from 'mdml-engine/logic'
 * tablePredicate([3, 3]).size // => 9
 * ```
 */
export function tablePredicate(sizes: readonly number[], init = 0): PredicateModel {
  const size = sizes.reduce((a, b) => a * b, 1)
  const cols = 'abcdefgh'
  const rows = 'ijklmnop'
  return {
    size,
    init: new Array<number>(size).fill(init),
    truth: (theta, args) => {
      if (args.length !== sizes.length) throw new MdmlError('value', 'tablePredicate: wrong number of arguments')
      const table = reshape(theta, sizes.length === 0 ? [] : [...sizes])
      if (sizes.length === 0) return sigmoid(table)
      const spec = `${args.map((_, i) => `${rows[i]}${cols[i]}`).join(',')},${cols.slice(0, args.length)}->${rows.slice(0, args.length)}`
      return sigmoid(einsum(spec, ...args, table))
    },
  }
}

/**
 * A predicate that is a small network on the features of its arguments: one hidden `tanh` layer on the concatenated
 * feature rows of the tuple, `sigmoid(w2 . tanh(W1 x + b1) + b2)`.
 *
 * @param inputDims - The feature width of each argument.
 * @param hidden - The hidden width.
 * @param k - The key the initial weights are drawn from.
 * @returns The predicate.
 * @throws MdmlError with code `value` for non-positive widths.
 * @example
 * ```ts
 * import { networkPredicate } from 'mdml-engine/logic'
 * import { key } from 'mdml-engine/random'
 * networkPredicate([2, 2], 3, key(0)).size // => 4 * 3 + 3 + 3 + 1
 * ```
 */
export function networkPredicate(inputDims: readonly number[], hidden: number, k: Key): PredicateModel {
  const d = inputDims.reduce((a, b) => a + b, 0)
  if (!(d > 0 && hidden > 0)) throw new MdmlError('value', 'networkPredicate: widths must be positive')
  const w1 = Array.from(normal(k, [d * hidden], { scale: 1 / Math.sqrt(d) }).data)
  const w2 = Array.from(normal(makeKey(0x5eed), [hidden], { scale: 1 / Math.sqrt(hidden) }).data)
  const size = d * hidden + hidden + hidden + 1
  return {
    size,
    init: [...w1, ...new Array<number>(hidden).fill(0), ...w2, 0],
    truth: (theta, args) => {
      if (args.length !== inputDims.length) throw new MdmlError('value', 'networkPredicate: wrong number of arguments')
      const counts = args.map((a) => a.shape[0] ?? 1)
      const wide = args.map((a, i) => {
        // [n_i, d_i] to [1, .., n_i, .., 1, d_i], broadcast over the other arguments.
        const shape = [...counts.map((_, j) => (j === i ? counts[i] : 1)), inputDims[i] as number]
        return broadcastTo(reshape(a, shape as number[]), [...counts, inputDims[i] as number])
      })
      const features = args.length === 1 ? (wide[0] as Value) : concat(wide, -1)
      const W1 = reshape(sliceAxis(theta, 0, 0, d * hidden), [d, hidden])
      const b1 = sliceAxis(theta, 0, d * hidden, d * hidden + hidden)
      const v2 = sliceAxis(theta, 0, d * hidden + hidden, d * hidden + 2 * hidden)
      const b2 = squeeze(sliceAxis(theta, 0, size - 1, size), 0)
      const h = tanh(add(matmul(features, W1), b1))
      return sigmoid(add(matmul(h, v2), b2))
    },
  }
}

// ------------------------------------------------------------------------------------------------ evaluation

interface Evaluated {
  readonly value: Value
  readonly vars: readonly string[]
}

/** What formulas are evaluated against: the variables' domains, the constants' features and the predicates' truth. */
interface Environment {
  readonly variables: ReadonlyMap<string, Value>
  readonly constants: ReadonlyMap<string, Value>
  readonly predicates: ReadonlyMap<string, (args: readonly Value[]) => Value>
}

/** Bring an evaluated formula to a common order of variables, with axes of length 1 for those it lacks. */
function align(e: Evaluated, order: readonly string[]): Value {
  const own = order.filter((v) => e.vars.includes(v))
  const perm = own.map((v) => e.vars.indexOf(v))
  let x = perm.every((p, i) => p === i) ? e.value : transpose(e.value, perm)
  const shape = order.map((v) => (e.vars.includes(v) ? (x.shape[own.indexOf(v)] as number) : 1))
  x = reshape(x, shape)
  return x
}

function combine(a: Evaluated, b: Evaluated, op: (x: Value, y: Value) => Value): Evaluated {
  const vars = [...a.vars, ...b.vars.filter((v) => !a.vars.includes(v))]
  return { value: op(align(a, vars), align(b, vars)), vars }
}

function evaluate(f: FolFormula, env: Environment, sem: FuzzySemantics): Evaluated {
  const { norm } = sem
  const imp = (a: Value, b: Value): Value => (sem.implication === 'residuum' ? norm.residuum(a, b) : norm.implies(a, b))
  switch (f.kind) {
    case 'const':
      return { value: asTensor(f.value ? 1 : 0, 'float64'), vars: [] }
    case 'atom': {
      const t = f.atom
      const name = t.kind === 'compound' ? t.functor : t.kind === 'atom' ? t.name : ''
      const terms = t.kind === 'compound' ? t.args : []
      const predicate = env.predicates.get(name)
      if (predicate === undefined) throw new MdmlError('value', `fuzzy theory: unknown predicate '${name}'`)
      const vars: string[] = []
      const constantAxes: number[] = []
      const args = terms.map((a, i) => {
        if (a.kind === 'var') {
          if (vars.includes(a.name)) throw new MdmlError('value', `fuzzy theory: variable ${a.name} repeats in ${name}`)
          const v = env.variables.get(a.name)
          if (v === undefined) throw new MdmlError('value', `fuzzy theory: unknown variable '${a.name}'`)
          vars.push(a.name)
          return v
        }
        if (a.kind === 'atom') {
          const c = env.constants.get(a.name)
          if (c === undefined) throw new MdmlError('value', `fuzzy theory: unknown constant '${a.name}'`)
          constantAxes.push(i)
          return c.shape.length === 1 ? reshape(c, [1, c.shape[0] as number]) : c
        }
        throw new MdmlError('value', 'fuzzy theory: only variables and constants are terms')
      })
      let value = predicate(args)
      if (constantAxes.length > 0) value = squeeze(value, constantAxes)
      return { value, vars }
    }
    case 'not': {
      const a = evaluate(f.arg, env, sem)
      return { value: norm.not(a.value), vars: a.vars }
    }
    case 'and':
    case 'or': {
      const parts = f.args.map((a) => evaluate(a, env, sem))
      const op = f.kind === 'and' ? norm.and : norm.or
      let acc: Evaluated = { value: asTensor(f.kind === 'and' ? 1 : 0, 'float64'), vars: [] }
      for (const p of parts) acc = combine(acc, p, op)
      return acc
    }
    case 'implies':
      return combine(evaluate(f.left, env, sem), evaluate(f.right, env, sem), imp)
    case 'iff': {
      const a = evaluate(f.left, env, sem)
      const b = evaluate(f.right, env, sem)
      return combine(combine(a, b, imp), combine(b, a, imp), norm.and)
    }
    case 'forall':
    case 'exists': {
      const body = evaluate(f.body, env, sem)
      const axis = body.vars.indexOf(f.variable)
      if (axis < 0) return body
      const agg = f.kind === 'forall' ? sem.forall : sem.exists
      return { value: agg(body.value, axis), vars: body.vars.filter((v) => v !== f.variable) }
    }
  }
}

/** The names of the variables free in a formula (those not bound by a quantifier). */
function freeVariables(f: FolFormula, bound: readonly string[] = []): string[] {
  switch (f.kind) {
    case 'const':
      return []
    case 'atom': {
      const t = f.atom
      return t.kind === 'compound'
        ? t.args.flatMap((a) => (a.kind === 'var' && !bound.includes(a.name) ? [a.name] : []))
        : []
    }
    case 'not':
      return freeVariables(f.arg, bound)
    case 'and':
    case 'or':
      return f.args.flatMap((a) => freeVariables(a, bound))
    case 'implies':
    case 'iff':
      return [...freeVariables(f.left, bound), ...freeVariables(f.right, bound)]
    case 'forall':
    case 'exists':
      return freeVariables(f.body, [...bound, f.variable])
  }
}

// ------------------------------------------------------------------------------------------------ the theory

/** An axiom of a fuzzy theory: a first-order formula that should hold, and how much it counts. */
export interface FuzzyAxiom {
  /** In `parseFol` syntax; free upper-case variables are universally quantified. */
  readonly formula: string
  /** The weight in the aggregate satisfaction (default 1). */
  readonly weight?: number
}

/** The configuration of a fuzzy theory. */
export interface FuzzyTheoryConfig {
  /** The feature rows `[n, d]` each variable ranges over (named as the formulas name them). */
  readonly variables: Readonly<Record<string, TensorLike>>
  /** The feature row of each constant. */
  readonly constants?: Readonly<Record<string, TensorLike>>
  /** The learnable predicates by name. */
  readonly predicates: Readonly<Record<string, PredicateModel>>
  readonly axioms: readonly FuzzyAxiom[]
  readonly semantics?: FuzzySemanticsOptions
  /**
   * How the axioms' satisfactions combine: their weighted `mean`, or the weighted `pMeanError` that one badly violated
   * axiom pulls down (default `{ pMeanError: 2 }`).
   */
  readonly aggregate?: 'mean' | { readonly pMeanError: number }
}

/** A fuzzy theory: axioms over learnable predicates, with the satisfaction as a function of the parameters. */
export interface FuzzyTheory {
  /** The predicates in parameter order. */
  readonly predicateNames: readonly string[]
  readonly size: number
  readonly x0: readonly number[]
  readonly semantics: FuzzySemantics
  /** The truth of each axiom and the aggregate satisfaction at the parameters. */
  satisfaction(theta: TensorLike): { readonly axioms: Value; readonly total: Value }
  /** The truth of any formula (a tensor over its free variables, or the scalar of a closed one). */
  truth(theta: TensorLike, formula: string): Value
  /** The theory's loss, `1 - satisfaction`, as a smooth problem for the solver slot. */
  problem(): SmoothProblem
}

/**
 * Build a fuzzy theory. The parameters are the predicates' parameters concatenated in the order of
 * {@link FuzzyTheory.predicateNames}; each axiom is evaluated by the semantics (formulas become tensors with one axis
 * per free variable, quantifiers aggregate those axes away), and the aggregate satisfaction is the loss's complement.
 *
 * @param config - The domains, constants, predicates, axioms and semantics.
 * @returns The theory.
 * @throws MdmlError with code `value` for an axiom that does not parse or names something undeclared (found on use).
 * @example
 * ```ts
 * import { fuzzyTheory, tablePredicate } from 'mdml-engine/logic'
 * const t = fuzzyTheory({
 *   variables: { x: [[1, 0], [0, 1]] },
 *   predicates: { p: tablePredicate([2]) },
 *   axioms: [{ formula: 'all x. p(x)' }],
 * })
 * t.size // => 2
 * ```
 */
export function fuzzyTheory(config: FuzzyTheoryConfig): FuzzyTheory {
  const semantics = fuzzySemantics(config.semantics)
  const predicateNames = Object.keys(config.predicates).sort()
  const offsets = new Map<string, number>()
  let size = 0
  for (const n of predicateNames) {
    offsets.set(n, size)
    size += (config.predicates[n] as PredicateModel).size
  }
  const x0 = predicateNames.flatMap((n) => [...(config.predicates[n] as PredicateModel).init])
  const variables = new Map(Object.entries(config.variables).map(([k, v]) => [k, asTensor(v, 'float64')]))
  const constants = new Map(Object.entries(config.constants ?? {}).map(([k, v]) => [k, asTensor(v, 'float64')]))
  const parsed = config.axioms.map((a) => parseFol(a.formula))
  const weights = config.axioms.map((a) => a.weight ?? 1)
  const aggregate = config.aggregate ?? { pMeanError: 2 }
  const environment = (theta: TensorLike): Environment => {
    const t = asTensor(theta, 'float64')
    if ((t.shape[0] ?? 0) !== size) throw new MdmlError('shape', `fuzzyTheory: expected ${size} parameters`)
    const predicates = new Map<string, (args: readonly Value[]) => Value>()
    for (const n of predicateNames) {
      const p = config.predicates[n] as PredicateModel
      const start = offsets.get(n) as number
      predicates.set(n, (args) => p.truth(sliceAxis(t, 0, start, start + p.size), args))
    }
    return { variables, constants, predicates }
  }
  const close = (f: FolFormula, env: Environment): Value => {
    // Free variables are universally quantified, outermost first.
    const free = [...new Set(freeVariables(f))]
    let e = evaluate(f, env, semantics)
    for (const v of free.reverse()) {
      const axis = e.vars.indexOf(v)
      if (axis >= 0) e = { value: semantics.forall(e.value, axis), vars: e.vars.filter((x) => x !== v) }
    }
    if (e.vars.length > 0) throw new MdmlError('value', 'fuzzy theory: a formula must be closed')
    return e.value
  }
  const satisfaction = (theta: TensorLike): { axioms: Value; total: Value } => {
    const env = environment(theta)
    const values = parsed.map((f) => reshape(close(f, env), [1]))
    const axioms = values.length === 1 ? (values[0] as Value) : concat(values, 0)
    const w = asTensor(weights, 'float64')
    const total = div(sum(mul(w, axioms)), sum(w))
    if (aggregate === 'mean') return { axioms, total }
    const p = aggregate.pMeanError
    const errors = pow(clip(one(axioms), EPS, 1), p)
    return { axioms, total: one(pow(div(sum(mul(w, errors)), sum(w)), 1 / p)) }
  }
  return {
    predicateNames,
    size,
    x0,
    semantics,
    satisfaction,
    truth: (theta, formula) => close(parseFol(formula), environment(theta)),
    problem: () => ({ objective: (x: Value) => one(satisfaction(x).total), x0: [...x0] }),
  }
}

/** The paths of {@link fuzzyTheoryAlgorithm}: the solver's, and the theory's. */
export type FuzzyTheoryPath = SolverPath | GradientPath | 'theory/satisfaction' | 'theory/axioms' | 'theory/parameters'

/** The log contract of {@link fuzzyTheoryAlgorithm}. */
export const fuzzyTheoryContract: Contract<unknown, FuzzyTheoryPath> = defineContract({
  ...solverContract,
  ...gradientContract,
  'theory/satisfaction': scalar({ description: 'the aggregate satisfaction of the axioms', direction: 'higher' }),
  'theory/axioms': tensorSource({ role: 'output', shape: ['axioms'], description: 'the truth of each axiom' }),
  'theory/parameters': tensorSource({
    role: 'param',
    shape: ['parameters'],
    description: 'the predicates’ parameters',
  }),
})

/**
 * Train a fuzzy theory as a traced algorithm over the solver slot: the loss `1 - satisfaction` minimised by any solver
 * of `smoothSolvers` (L-BFGS by default; the `gradient` mode with Adam is the usual stochastic-free alternative),
 * logging the aggregate satisfaction, each axiom's truth and the parameters beside the solver's channels.
 *
 * @param theory - The theory.
 * @param options - The solver (a value of `smoothSolvers.space`) and the key for the solver.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { fuzzyTheory, fuzzyTheoryAlgorithm, tablePredicate } from 'mdml-engine/logic'
 * const t = fuzzyTheory({ variables: { x: [[1, 0], [0, 1]] }, predicates: { p: tablePredicate([2]) }, axioms: [{ formula: 'all x. p(x)' }] })
 * fuzzyTheoryAlgorithm(t).name // => 'lbfgs'
 * ```
 */
export function fuzzyTheoryAlgorithm(
  theory: FuzzyTheory,
  options: { readonly solver?: SpaceValue; readonly key?: Key } = {},
): Algorithm<unknown, FuzzyTheoryPath> {
  const inner = loggedSolver<'theory/satisfaction' | 'theory/axioms' | 'theory/parameters'>(
    theory.problem(),
    options.solver ?? smoothSolverConfig({ kind: 'lbfgs', gradTol: 1e-8 }),
    options.key ?? makeKey(0),
    (x, ctx) => {
      ctx.log('theory/parameters', x)
      const s = theory.satisfaction(fromStorage(Float64Array.from(x), [x.length], 'float64'))
      ctx.log('theory/satisfaction', item(s.total))
      ctx.log('theory/axioms', s.axioms)
    },
  )
  return { ...inner, schema: contractSchema(fuzzyTheoryContract) }
}

/** A fitted theory. */
export interface FuzzyFit {
  readonly theta: Float64Array
  /** The aggregate satisfaction at the fit. */
  readonly satisfaction: number
  /** The truth of each axiom. */
  readonly axioms: readonly number[]
  readonly converged: boolean
  readonly steps: number
}

/**
 * Train a theory to satisfy its axioms and report the fit.
 *
 * @param theory - The theory.
 * @param options - The solver, the most steps (default 500) and the key.
 * @returns The parameters, the satisfaction and the truth of each axiom.
 * @example
 * ```ts
 * import { fuzzyTheory, tablePredicate, trainFuzzyTheory } from 'mdml-engine/logic'
 * const t = fuzzyTheory({ variables: { x: [[1, 0], [0, 1]] }, predicates: { p: tablePredicate([2]) }, axioms: [{ formula: 'all x. p(x)' }] })
 * trainFuzzyTheory(t).satisfaction > 0.99 // => true
 * ```
 */
export function trainFuzzyTheory(
  theory: FuzzyTheory,
  options: { readonly solver?: SpaceValue; readonly maxSteps?: number; readonly key?: Key } = {},
): FuzzyFit {
  const key = options.key ?? makeKey(0)
  const alg = fuzzyTheoryAlgorithm(theory, { ...(options.solver === undefined ? {} : { solver: options.solver }), key })
  const run = runAlgorithm(alg as unknown as Algorithm<unknown>, { key, maxSteps: options.maxSteps ?? 500 })
  const x = (run.state as { x: Float64Array }).x
  const s = theory.satisfaction(fromStorage(Float64Array.from(x), [x.length], 'float64'))
  return {
    theta: x,
    satisfaction: item(s.total),
    axioms: toFlat(s.axioms),
    converged: run.status === 'converged',
    steps: run.steps,
  }
}
