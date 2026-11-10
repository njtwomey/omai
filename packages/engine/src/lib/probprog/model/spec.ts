/**
 * The model specification: `ModelSpec`, a JSON-able description of a probabilistic model as a graph of nodes (data,
 * latent, observed, derived, generated, factor, param) with named dimensions (plates), constraints, and expressions
 * over a closed registry of operators. A spec is a value: it can be inspected, serialised, sent to a worker, analysed
 * without running anything and compiled (to a `sample`/`observe` program by `toProgram`, and later to a factor graph for
 * message passing). It holds no functions and no tensors; the only escape from that is the `custom` expression, which
 * names a function the receiving side must supply.
 *
 * Contracts other code relies on, in one place:
 * - Nodes are stored in declaration order, which is a topological order: a node refers only to nodes before it.
 * - **Dimensions are named.** A node declares `dims`, the names of its batch axes in order; a value is a tensor whose
 *   leading axes are those dimensions and whose trailing `eventRank` axes belong to the distribution (a Dirichlet
 *   draw's category axis, a multivariate normal's coordinates). Sizes are not in the spec: they come from fixed
 *   `DimSpec.size` or from the data bound to it.
 * - **Ids.** A node's id in drawings, analyses and factor graphs is its name (data nodes `data:<name>`), a factor's is
 *   `factor:<site>`, a plate's `plate:<a>×<b>` in the spec's dimension order, an element's `<site>[<i>]`; see `ids.ts`. The
 *   same ids name nodes in posteriors and on tapes, so a view can paint live inference state on `toStructure`'s picture.
 * - **Site names.** Every latent, observed, derived (recorded), generated, factor and param node is a site of the
 *   compiled program under its own name; data nodes are not sites. An observed node may share its name with the data
 *   node that supplies its value; no other two nodes share a name.
 * - **Expressions** are trees: constants, references to earlier nodes by name, operator applications (`op` names an
 *   entry of the operator registry in `operators.ts`) and `custom` calls. Distribution parameters are expressions; the
 *   `family` names an entry of the family table in `families.ts`, whose parameter names are those of the library's
 *   constructors (`normal`: `loc`, `scale`; `bernoulli`: `probs` or `logits`).
 * - **Factor graph hooks** (for `probprog/model/factors.ts`): a node's distribution `family` and an operator's `op` are
 *   the names a message-passing factor registry is keyed by; an `OperatorDef.factor` may name a different factor
 *   operator when the two differ. A node whose expression holds a `custom` call, or a family with no
 *   exponential-family form, is opaque to message passing. `analyseSpec` reports these.
 *
 * @index structure ModelSpec: a JSON-able graph of data, latent, observed and derived nodes with named dims.
 */

import type { Json } from '../../base/index.ts'

/** The schema tag of a serialised spec, bumped when the format changes incompatibly. */
export const MODEL_SPEC_SCHEMA = 'omai.model/1'

/** The name of the operator registry a spec was written against (`defaultOperators()` in `operators.ts`). */
export const OPERATOR_REGISTRY = 'omai.operators/1'

/**
 * A named dimension: a plate, a coordinate axis and a shape in one. Its size is fixed here or inferred from the data
 * bound to a node that has the dimension.
 */
export interface DimSpec {
  readonly name: string
  /** A fixed size; absent when the size comes from the data. */
  readonly size?: number
  /** Coordinate labels (school names, feature names): carried into the posterior's coordinates; their count is the size. */
  readonly labels?: readonly (string | number)[]
}

/**
 * An expression: what a node computes, and what the parameters of a distribution are.
 *
 * - `const`: a number, or nested arrays of numbers (a tensor), or booleans for masks. Its leading axes are the named
 *   `dims` when given (a vector over `feature`); the rest are event axes.
 * - `ref`: the value of an earlier node (data, latent, derived, param) by name.
 * - `op`: an operator of the registry applied to argument expressions, with optional JSON `params` (the dimension an
 *   operator reduces over, for instance).
 * - `custom`: an opaque function the receiving side supplies by name; it declares the dims and event rank of its
 *   result because nothing can be inferred from a function. Works with the sampling engines, not with message passing,
 *   and it is the one thing that does not survive a trip through JSON on its own.
 */
export type ExprSpec =
  | { readonly kind: 'const'; readonly value: Json; readonly dims?: readonly string[] }
  | { readonly kind: 'ref'; readonly name: string }
  | {
      readonly kind: 'op'
      readonly op: string
      readonly args: readonly ExprSpec[]
      readonly params?: { readonly [key: string]: Json }
    }
  | {
      readonly kind: 'custom'
      readonly name: string
      readonly args: readonly ExprSpec[]
      readonly dims: readonly string[]
      readonly eventRank: number
    }

/** A distribution expression: a family of the table in `families.ts` and its parameters as expressions. */
export interface DistSpec {
  readonly family: string
  readonly params: { readonly [name: string]: ExprSpec }
}

/**
 * A constraint on a latent node: the set its values live in, as a Stan-style declaration (`<lower=0>`,
 * `<lower=0, upper=1>`, `simplex`, `ordered`). It replaces the support the distribution reports, so the unconstrained
 * parametrisation and its Jacobian are the existing `bijectorForSupport`'s; the density stays the distribution's (as
 * in Stan, a prior placed on a constrained parameter is not renormalised). Bounds are numbers or nested arrays.
 */
export type ConstraintSpec =
  | { readonly kind: 'real' | 'positive' | 'nonNegative' | 'unitInterval' | 'simplex' | 'ordered' }
  | { readonly kind: 'correlationCholesky' | 'positiveDefinite' | 'lowerCholesky' | 'correlation' }
  | { readonly kind: 'greaterThan'; readonly low: Json }
  | { readonly kind: 'lessThan'; readonly high: Json }
  | { readonly kind: 'interval'; readonly low: Json; readonly high: Json }

/** The values a data node accepts, checked when data is bound. */
export type DataSupport = 'real' | 'positive' | 'nonNegative' | 'binary' | 'count' | 'index' | 'probability'

/** What a data node is for; documentation and drawing, and which nodes a prediction rebinds (`covariate`). */
export type DataRole = 'response' | 'covariate' | 'index' | 'weights' | 'offset' | 'exposure' | 'constant'

/** What an observed node does with a missing (NaN) entry of its data. */
export type MissingPolicy = 'impute' | 'ignore' | 'error'

/** Fields every node has. */
interface NodeBase {
  readonly name: string
  /** The batch axes, by dimension name, in order. */
  readonly dims: readonly string[]
  /** The number of trailing axes that belong to the value's event (0 for scalars per element). */
  readonly eventRank: number
  readonly doc?: string
}

/** Data bound at run time: covariates, responses, index codes. */
export interface DataNode extends NodeBase {
  readonly kind: 'data'
  readonly support?: DataSupport
  readonly role?: DataRole
  /** For an `index` node: the dimension its integer codes index (each code is checked against that dimension's size). */
  readonly indexes?: string
  /** Whether the data may be left unbound (a covariate only a prediction needs). */
  readonly optional?: boolean
}

/** A random variable to infer: a distribution, optionally a constraint beyond its support. */
export interface LatentNode extends NodeBase {
  readonly kind: 'latent'
  readonly dist: DistSpec
  readonly constraint?: ConstraintSpec
}

/** A random variable with a known value (the likelihood): the value is an expression, usually a data node. */
export interface ObservedNode extends NodeBase {
  readonly kind: 'observed'
  readonly dist: DistSpec
  readonly value: ExprSpec
  readonly missing?: MissingPolicy
}

/** A value computed from other nodes. Recorded (a `deterministic` site) unless `record` is false. */
export interface DerivedNode extends NodeBase {
  readonly kind: 'derived'
  readonly expr: ExprSpec
  readonly record: boolean
}

/** A forward-only quantity (Stan's generated quantities): drawn for predictive checks, never inferred. */
export interface GeneratedNode extends NodeBase {
  readonly kind: 'generated'
  readonly dist: DistSpec
}

/** A log-density term added to the joint (Stan's `target +=`): the expression's elements are summed. */
export interface FactorNode extends NodeBase {
  readonly kind: 'factor'
  readonly expr: ExprSpec
}

/** A non-random learnable parameter (for optimisation and variational guides), with an initial value. */
export interface ParamNode extends NodeBase {
  readonly kind: 'param'
  readonly init: Json
  readonly constraint?: ConstraintSpec
}

/** A node of a {@link ModelSpec}. Reserved for later work packages: `scan` (recurrences) and `gate` (mixtures). */
export type ModelNode = DataNode | LatentNode | ObservedNode | DerivedNode | GeneratedNode | FactorNode | ParamNode

/** The kinds of {@link ModelNode}. */
export type NodeKind = ModelNode['kind']

/**
 * A probabilistic model as data. Build one with `defineModel`; bind data with `bind`; compile with `toProgram`.
 *
 * @concept model-spec
 */
export interface ModelSpec {
  readonly schema: typeof MODEL_SPEC_SCHEMA
  readonly name: string
  /** The operator registry the expressions were written against. */
  readonly registry: typeof OPERATOR_REGISTRY
  readonly dims: readonly DimSpec[]
  /** In declaration (topological) order. */
  readonly nodes: readonly ModelNode[]
  /** Names of the `custom` functions the spec calls: what a receiving side must supply. */
  readonly customs: readonly string[]
}

/**
 * Whether a node is a site of the compiled program (everything but data).
 *
 * @param node - The node.
 * @returns True for every kind but `data`.
 * @example
 * ```ts
 * import { isSiteNode } from 'omai-engine/probprog/model'
 * isSiteNode({ kind: 'data', name: 'x', dims: [], eventRank: 0 }) // => false
 * ```
 */
export function isSiteNode(node: ModelNode): boolean {
  return node.kind !== 'data'
}

/**
 * The names an expression refers to, in order of first appearance.
 *
 * @param expr - The expression.
 * @returns The referenced node names, without repeats.
 * @example
 * ```ts
 * import { exprRefs } from 'omai-engine/probprog/model'
 * exprRefs({ kind: 'op', op: 'add', args: [{ kind: 'ref', name: 'a' }, { kind: 'ref', name: 'b' }] }) // => ['a', 'b']
 * ```
 */
export function exprRefs(expr: ExprSpec): string[] {
  const seen = new Set<string>()
  const walk = (e: ExprSpec): void => {
    if (e.kind === 'ref') seen.add(e.name)
    else if (e.kind === 'op' || e.kind === 'custom') for (const a of e.args) walk(a)
  }
  walk(expr)
  return [...seen]
}

/**
 * The names a node depends on: the references of its distribution parameters, expression and value.
 *
 * @param node - The node.
 * @returns The referenced names, without repeats.
 * @example
 * ```ts
 * import { defineModel, nodeRefs } from 'omai-engine/probprog/model'
 * import { normal } from 'omai-engine/probprog/model/dists'
 * const m = defineModel('m', (b) => { const mu = b.latent('mu', normal(0, 1)); b.latent('x', normal(mu, 1)) })
 * nodeRefs(m.spec.nodes.find((n) => n.name === 'x') ?? m.spec.nodes[0] as never) // => ['mu']
 * ```
 */
export function nodeRefs(node: ModelNode): string[] {
  const out = new Set<string>()
  const add = (e: ExprSpec): void => {
    for (const r of exprRefs(e)) out.add(r)
  }
  switch (node.kind) {
    case 'latent':
    case 'generated':
      for (const e of Object.values(node.dist.params)) add(e)
      break
    case 'observed':
      for (const e of Object.values(node.dist.params)) add(e)
      add(node.value)
      break
    case 'derived':
    case 'factor':
      add(node.expr)
      break
    default:
      break
  }
  return [...out]
}
