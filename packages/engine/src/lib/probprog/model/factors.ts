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
 * `Compiler` is the driver: it walks the spec in order and gives each node to the first mapping of an ordered table that takes it
 * (`factor-graph/_table.ts`: `latentMappings`, `observedNodeMappings`, `observedRowMappings`, `observedElementMappings`; each entry names
 * what it maps, the factor operators it can emit and why it sits where it does: normal-gamma before the family lookup, a value that moves to
 * the mean before the other observations). The order is part of the meaning, and a mapping that recognises a node only by trying it answers
 * `NOT_MINE`, which takes whatever it made out of the graph again (`factor-graph/_mapping.ts`). The mappings are functions of the compilation's
 * `Context` and a node, one file per concern in `factor-graph/_mappings/` (Gaussian, positive and discrete latents, matrix-valued nodes, derived
 * nodes, observed Gaussians, numeric likelihoods, conjugate observations, observations chosen by a discrete variable, plates), over the
 * evaluator of expressions (`factor-graph/_symbolic/`: the value algebra of `_cells.ts` mapped to the engine's affine, product, gate and apply
 * operators), the constants (`_known.ts`), the layout plan (`_plan.ts`) and the recognisers (`_idioms.ts`). Anything with no mapping is reported
 * with a reason, never silently dropped. `docs/probprog/compiler-mappings.md` lists every entry of the tables (written from them by
 * `make ep-compiler-doc`). The refusal, the issue and layout types, the reading back of a run (`factorGraphPosterior`) and the
 * verdict (`factorGraphVerdict`) live in `factor-graph/`.
 *
 * Ids follow `ids.ts`: a variable is its site's name, `<site>[<i>]` for an element of a site over dimensions, and
 * `<site>#<k>[<i>]` for an auxiliary variable; a factor is `factor:` and the id of the variable it defines (an observed
 * element's likelihood is `factor:y[3]`); `siteOfId` maps every id back to the spec node it belongs to, so a view can paint a
 * run's marginals on `toStructure(spec)` by site name. Results are read back by `factorGraphPosterior`, which builds
 * the library's `Posterior` keyed by site name, in the spec's shapes.
 *
 * @index function toFactorGraph: a bound ModelSpec as a message-passing factor graph, plates expanded.
 */

import {
  type FactorGraphSpec,
  type FactorOperator,
  defaultOperators as factorOperators,
} from '../../message-passing/index.ts'
import type { SpecAnalysis } from './analysis.ts'
import type { BoundModel } from './bind.ts'
import { eventToDims } from './event-dims.ts'
import { Context } from './factor-graph/_context.ts'
import { Refusal } from './factor-graph/_refusal.ts'
import { mapNode } from './factor-graph/_table.ts'
import type { FactorGraphCompilation } from './factor-graph/types.ts'
import { defaultOperators, type OperatorTable } from './operators.ts'
import { ScanUnrollError, unrollScans } from './scan.ts'
import type { ExprSpec, ModelNode } from './spec.ts'

class Compiler {
  /** The state of the compilation, which the evaluator and the mappings work on. */
  readonly cx: Context

  constructor(
    bound: BoundModel,
    modelOps: OperatorTable,
    ops: Readonly<Record<string, FactorOperator>>,
    vectorise: boolean,
    packScalars: boolean,
    denseScales: boolean = true,
  ) {
    this.cx = new Context(bound, modelOps, ops, { vectorise, packScalars, denseScales })
  }

  // ----- the nodes -----

  run(): void {
    for (const [position, node] of this.cx.bound.spec.nodes.entries()) {
      this.cx.current = node.name
      this.cx.position = position
      try {
        mapNode(this.cx, node)
      } catch (e) {
        if (!(e instanceof Refusal)) throw e
        this.cx.issues.push({ node: node.name, kind: e.kind, code: e.code, reason: e.message })
        this.cx.failed.add(node.name)
      }
    }
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
 * term, a family with no message family, a `custom` call, a random scale that is neither the precision idiom nor the exp of a Gaussian form, a square of a random
 * precision, a non-affine operator on a random value (except as a Bernoulli link or one of the unary functions of the
 * registry), a gate over components of different families, a computed random index, a constraint beyond
 * the support of a Gaussian (a simplex, a positive-definite matrix), missing data that is not ignored, a subsampling batch and a plated table over discrete nodes each become an
 * issue naming the node and the reason, and no graph is produced. (A product of two random precisions is mapped, as a `gamma-product`
 * variable, and a gate over known and random components is a lookup that the observation reads as a gate of branches.)
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
        issues: [{ node: e.node, kind: 'structure', code: 'structure', reason: e.message }],
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
  const cx = c.cx
  cx.g.prune(cx.sites)
  const graph: FactorGraphSpec = { variables: cx.g.variables, factors: cx.g.factors }
  const opaque = (options.analysis?.opaque ?? opaqueNodes(bound)).filter((name) => {
    const node = bound.spec.nodes.find((n) => n.name === name && n.kind !== 'data')
    if (node === undefined) return true
    const called = customNames(node)
    // A node whose custom calls were all evaluated (known arguments) or read as a matrix (affine in one random vector) is not opaque.
    return called.length === 0 || !called.every((n) => cx.handledCustoms.has(n))
  })
  for (const name of opaque)
    if (!cx.issues.some((i) => i.node === name))
      cx.issues.push({
        node: name,
        kind: 'opaque',
        code: 'opaque',
        reason: `'${name}' is opaque to message passing (a custom call, or a family with no exponential-family form)`,
      })
  const plates = bound.spec.dims
    .filter((d) => cx.sites.some((s) => s.dims.includes(d.name)))
    .map((d) => ({ dim: d.name, size: bound.sizes[d.name] as number }))
  return {
    name: bound.spec.name,
    graph: cx.issues.length === 0 ? graph : undefined,
    issues: cx.issues,
    opaque,
    sites: cx.sites,
    auxiliary: cx.g.auxiliary,
    plated: cx.plated,
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
