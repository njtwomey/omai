/**
 * The state of one compilation, shared by the evaluator of expressions and the mappings of nodes: the graph being built, the
 * issues and site layouts reported so far, what is known (`_known.ts`), the layout plan (`_plan.ts`), and the maps the walk fills
 * as it goes (the symbolic value of each node, the apply chains, the normal-gamma pairs it has activated, the Wishart and
 * correlation sites, the parameter cells of observed nodes). A mapping is a function of this context and a node; the driver
 * (`../factors.ts`) owns the order. Besides the state, only the small operations every mapping needs live here: sizes and shapes,
 * the cells of a value over a node's dimensions, "is this known?", evaluation of a known expression, and the layout record of a site.
 */

import type { FactorOperator, FamilyKind } from '../../../message-passing/index.ts'
import type { BoundModel } from '../bind.ts'
import { distributionFamilies } from '../dists/families.ts'
import { elementId } from '../ids.ts'
import type { NamedValue, OperatorTable } from '../operators.ts'
import type { ExprSpec, ModelNode } from '../spec.ts'
import { GraphBuilder } from './_builder.ts'
import { type Cell, logName, type Sym } from './_cells.ts'
import type { ChainBuild } from './_chains.ts'
import { type IdiomEnv, type LatentScope, latentScope } from './_idioms.ts'
import { type EvalEnv, evaluateKnown, isKnownExpr, type Known, knownConstants } from './_known.ts'
import { type LayoutPlan, layoutPlan, type NormalGammaPlan } from './_plan.ts'
import { dataNotBound, dependsOnFailed, randomWhereKnown } from './_reasons.ts'
import { cellAt, fromNamed, joinDims, overDims, shapeOf, sizeOf, toNamed, zipSyms } from './_symbolic/_dims.ts'
import type { ApplyStep } from './_symbolic/_functions.ts'
import type { FactorIssue, SiteLayout } from './types.ts'

/** How a compilation is asked to lay the model out. */
export interface CompileOptions {
  /** An observed node whose rows are one regression on a weight vector is one plated factor (`plates: 'vectorise'`, the default). */
  readonly vectorise: boolean
  /** Scalar normals that an observation reads together are one joint vector. */
  readonly packScalars: boolean
  /** A vector of coefficients with the precisions of one shared gamma is one Gaussian vector. */
  readonly denseScales: boolean
}

/** The state of one compilation: see the file's summary. */
export class Context {
  /** The graph under construction: variables, factors, auxiliary names, atoms. */
  readonly g: GraphBuilder
  readonly issues: FactorIssue[] = []
  readonly sites: SiteLayout[] = []
  readonly plated: string[] = []
  /** What is known before and while the walk maps the nodes: bound data and constant derived nodes (`_known.ts`). */
  readonly known: Known
  /** The custom functions the evaluation of each known derived node called (replayed into `handledCustoms` when the walk skips the node). */
  readonly customsCalled: ReadonlyMap<string, readonly string[]>
  /** The symbolic value of each node the walk has mapped. */
  readonly syms: Map<string, Sym> = new Map()
  /** The derived nodes that depend on a discrete variable, kept as expressions (a table over scalar nodes reads them). */
  readonly lazy: Map<string, ExprSpec> = new Map()
  /** The nodes whose mapping was refused. */
  readonly failed: Set<string> = new Set()
  readonly unbound: ReadonlySet<string>
  /** The layout decisions made from the whole spec before the walk (`_plan.ts`). */
  readonly plan: LayoutPlan
  /** The non-centred vectors the dense factor still absorbs (the plan's, less those the walk found it does not apply to), by name, with the name of `z`. */
  readonly nonCentred: Map<string, string>
  /** The scalar groups whose joint vector the walk has made. */
  readonly groupsMade: Set<string> = new Set()
  /** The Gaussian vectors assembled from two scalar variables that a joint pair site reads (`observedPair`), by the names of the two. */
  readonly stacked: Map<string, string> = new Map()
  /** The chains of an unrolled scan being collected, by the scan's name, and the ones that ended (`_chains.ts`). */
  readonly chains: Map<string, ChainBuild> = new Map()
  readonly chainsDone: ChainBuild[] = []
  /** Whether the joint-structure layouts are on (`packScalars`): the scalar groups and the chains of a scan. */
  readonly packScalars: boolean
  /** The Gaussian log-scale variable behind a positive scale variable (the value variable `tau` is `exp` of it), by the value variable's name. */
  readonly logScale: Map<string, string> = new Map()
  /** The apply factors made so far: the input variable and the chain of steps behind each output, so arithmetic on an output collapses into one chain. */
  readonly applied: Map<string, { readonly input: string; readonly steps: readonly ApplyStep[] }> = new Map()
  /** The normal-gamma pairs the walk has activated (a Gaussian mean and the precision its variance shares), by the name of the mean's node. */
  readonly normalGammaByMean: Map<string, NormalGammaPlan> = new Map()
  /** The observed Gaussian nodes of an activated normal-gamma pair, with the known multiple of the precision each carries. */
  readonly normalGammaObserved: Map<string, { readonly plan: NormalGammaPlan; readonly multiple: number }> = new Map()
  /** The correlation latents (`(rho + 1) / 2` is the beta variable that carries each). */
  readonly correlations: Map<string, string> = new Map()
  /** The Wishart (or inverse-Wishart) matrix latents: the variable that carries the precision matrix, its order and whether the site is the covariance. */
  readonly wisharts: Map<string, { readonly variable: string; readonly d: number; readonly covariance: boolean }> =
    new Map()
  /** The cells of the parameters of observed nodes over their dimensions, by `node|key`, made once (`cellsOf`; the gated mean of a node stores its `loc` here too). */
  readonly cells: Map<string, Cell[]> = new Map()
  /** The custom functions the compiler has evaluated or read as a matrix, so a node that calls only these is not opaque. */
  readonly handledCustoms: Set<string> = new Set()
  /** The nodes an expression can refer to, by name (every node but the observed ones). */
  readonly nodes: Map<string, ModelNode> = new Map()
  /** What the recognisers of `_idioms.ts` ask of this compilation: the nodes, what is known, and which nodes are inverse-gamma variances. */
  readonly idiom: IdiomEnv = {
    nodes: this.nodes,
    isConst: (e) => this.isConst(e),
    valueOf: (e) => this.evalNamed(e),
    isRecipNode: (name) => {
      const cells = this.syms.get(name)?.cells
      return cells !== undefined && cells.length > 0 && cells.every((c) => c.kind === 'recip')
    },
    hasLogVariable: (name) => this.syms.has(logName(name)),
  }
  /** The scope of the analysis module's latent dependence over the spec. */
  readonly latents: LatentScope
  readonly families: ReturnType<typeof distributionFamilies> = distributionFamilies()
  /** The node being compiled: the site that auxiliary variables of its expressions are named after (`withSite` borrows it for another node). */
  current = ''
  /** The index in the spec of the node being compiled. */
  position = 0

  /** Whether an observed node's rows are compiled as one plated factor where they can be. */
  readonly vectorise: boolean

  constructor(
    readonly bound: BoundModel,
    readonly modelOps: OperatorTable,
    ops: Readonly<Record<string, FactorOperator>>,
    options: CompileOptions,
  ) {
    this.vectorise = options.vectorise
    this.packScalars = options.packScalars
    this.latents = latentScope(bound.spec)
    this.g = new GraphBuilder(ops, modelOps)
    this.unbound = new Set(bound.unbound)
    const pre = knownConstants(bound, modelOps)
    this.known = pre.known
    this.customsCalled = pre.customsCalled
    for (const n of bound.spec.nodes) if (n.kind !== 'observed') this.nodes.set(n.name, n)
    // The order of the plan is fixed (packed, non-centred, groups, normal-gamma, gplvm: each reads the ones before it, see `_plan.ts`).
    this.plan = layoutPlan(bound, this.known, {
      modelOps,
      denseScales: options.denseScales,
      packScalars: options.packScalars,
    })
    this.nonCentred = new Map(this.plan.nonCentred)
  }

  // ----- sizes and indexing -----

  size(dim: string): number {
    return sizeOf(this.bound.sizes, dim)
  }

  shapeOf(dims: readonly string[]): number[] {
    return shapeOf(this.bound.sizes, dims)
  }

  cellAt(sym: Sym, outDims: readonly string[], idx: readonly number[]): Cell {
    return cellAt(this.bound.sizes, sym, outDims, idx)
  }

  joinDims(syms: readonly Sym[]): string[] {
    return joinDims(syms)
  }

  zip(syms: readonly Sym[], f: (cells: readonly Cell[], index: number) => Cell): Sym {
    return zipSyms(this.bound.sizes, syms, f)
  }

  /** The cells of a symbolic value laid out over a node's dimensions (broadcasting what it lacks). */
  over(sym: Sym, dims: readonly string[]): Cell[] {
    return overDims(this.bound.sizes, sym, dims)
  }

  // ----- values -----

  fromNamed(nv: NamedValue): Sym {
    return fromNamed(nv)
  }

  toNamed(sym: Sym): NamedValue {
    return toNamed(this.bound.sizes, sym)
  }

  /** Whether an expression is known (no random part): bound data, constants and what is built from them. */
  isConst(e: ExprSpec): boolean {
    return isKnownExpr(e, this.known, this.bound.customs)
  }

  /** The value of an expression that has no random part, by the operator registry's own evaluation. */
  evalNamed(e: ExprSpec, override?: ReadonlyMap<string, NamedValue>): NamedValue {
    const env: EvalEnv = {
      modelOps: this.modelOps,
      customs: this.bound.customs,
      ref: (name) => {
        const o = override?.get(name) ?? this.known.get(name)
        if (o !== undefined) return o
        const lazy = this.lazy.get(name)
        if (lazy !== undefined) return evaluateKnown(lazy, env)
        if (this.unbound.has(name)) throw dataNotBound(name)
        if (this.failed.has(name)) throw dependsOnFailed(name)
        throw randomWhereKnown(name)
      },
      onCustom: (name) => this.handledCustoms.add(name),
    }
    return evaluateKnown(e, env)
  }

  // ----- sites -----

  /** Run `f` with the auxiliary variables it makes named after another node, then name them after the current one again. */
  withSite<T>(site: string, f: () => T): T {
    const before = this.current
    this.current = site
    try {
      return f()
    } finally {
      this.current = before
    }
  }

  /** The variable of element `i` of a site over `dims` (the site's own name when it has none). */
  elementName(site: string, dims: readonly string[], i: number): string {
    return dims.length === 0 ? site : elementId(site, i)
  }

  /** Record how a node's variables are laid out on the graph, for the reading back of a run. */
  layout(
    node: ModelNode,
    kind: SiteLayout['kind'],
    family: FamilyKind | undefined,
    layout: SiteLayout['layout'],
    variables: readonly (string | null)[],
    posterior: boolean,
    extra: Partial<Pick<SiteLayout, 'reciprocal' | 'part' | 'correlation'>> = {},
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

  /**
   * The end of a mapping that made one variable per element (`vars`, in row-major order of the node's dimensions): the cells are the
   * node's symbolic value, and the site is laid out as one scalar variable (no dimensions) or one per element.
   */
  finishLatent(
    node: ModelNode,
    family: FamilyKind,
    cells: readonly Cell[],
    vars: readonly (string | null)[],
    extra: Partial<Pick<SiteLayout, 'reciprocal' | 'part' | 'correlation'>> = {},
    kind: SiteLayout['kind'] = 'latent',
  ): void {
    this.syms.set(node.name, { dims: node.dims, cells })
    this.layout(node, kind, family, node.dims.length === 0 ? 'scalar' : 'elements', vars, true, extra)
  }
}
