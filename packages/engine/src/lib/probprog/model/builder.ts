/**
 * The model builder: `defineModel(name, (m) => { ... })` runs a function against a `ModelBuilder` whose verbs declare
 * the model in the shape of Stan's blocks and PyMC's named dimensions: `range` (a named dimension), `data`, `latent`
 * (with a constraint), `observe`, `derived`, `generated`, `factor`, `param`, `scan` (a recurrence), `custom`, and `include`
 * for sub-models. Each verb
 * checks what it can as it is called (names, references, dimensions, parameters) and the result is a `ModelDefinition`
 * holding a plain-JSON `ModelSpec`. The builder computes nothing: expressions are `Var` handles, distributions are
 * `DistExpr`s, and the spec is compiled later by `toProgram`.
 *
 * @index function defineModel: build a ModelSpec with named dimensions, data, latent, observed and derived nodes.
 */

import { type Json, OmaiError } from '../../base/index.ts'
import { validateSpec } from './check.ts'
import { substituteNode } from './compose.ts'
import { type CustomFn, type CustomFns, ModelDefinition } from './definition.ts'
import type { DistExpr } from './dists/builders.ts'
import {
  type ConstraintSpec,
  type DataRole,
  type DataSupport,
  type DimSpec,
  type ExprSpec,
  type MissingPolicy,
  MODEL_SPEC_SCHEMA,
  type ModelNode,
  OPERATOR_REGISTRY,
  prevName,
  type ScanEmission,
} from './spec.ts'
import { constant, type Operand, toVar, Var } from './var.ts'

/** A named dimension, as `range` returns it: usable wherever a dimension name is. */
export interface Dim {
  readonly name: string
}

/** A dimension by handle or by name. */
export type DimArg = Dim | string

/**
 * A constraint on a latent, Stan style: `{ lower: 0 }`, `{ lower: 0, upper: 1 }`, or one of the named constraints
 * (`{ kind: 'simplex' }`, ...). Bounds are numbers (or nested arrays for per-element bounds).
 */
export type ConstraintArg = ConstraintSpec | { readonly lower?: Json; readonly upper?: Json }

/** Options of {@link ModelBuilder.range}. */
export interface RangeOptions {
  /** A fixed size; left out, the size comes from the data bound to a node over this dimension. */
  readonly size?: number
  /** Coordinate labels (their count is the size). */
  readonly labels?: readonly (string | number)[]
}

/** Options of {@link ModelBuilder.data}. */
export interface DataOptions {
  /** The dimensions of the data's leading axes. */
  readonly dims?: readonly DimArg[]
  /** The number of trailing event axes (default 0). */
  readonly eventRank?: number
  readonly support?: DataSupport
  readonly role?: DataRole
  /** For integer codes: the dimension they index. */
  readonly indexes?: DimArg
  /** May be left unbound (a response a prediction does not have). */
  readonly optional?: boolean
  readonly doc?: string
}

/** Options of {@link ModelBuilder.latent} and {@link ModelBuilder.generated}. */
export interface LatentOptions {
  /** The dimensions of the variable (default: those its parameters carry); must contain them. */
  readonly dims?: readonly DimArg[]
  readonly constraint?: ConstraintArg
  readonly doc?: string
}

/** Options of {@link ModelBuilder.observe}. */
export interface ObserveOptions {
  readonly dims?: readonly DimArg[]
  /** What to do with a NaN entry of the data (default `impute`: the entry becomes a latent value of the model). */
  readonly missing?: MissingPolicy
  readonly doc?: string
}

/** What the `step` function of a scan receives. */
export interface ScanStepContext {
  /** The state at the previous step, as an expression (`<scan>@prev`). */
  readonly prev: Var
}

/** What the `emit` function of a scan receives: the current state, and the verb that declares an observation at each step. */
export interface ScanEmitContext {
  /** The state at the current step. */
  readonly state: Var
  /**
   * Declare an observation drawn at every step, as the site `<name>[t]`: its distribution may read the state, its value is
   * a node over the scanned dimension (read at the current step) or a constant.
   */
  observe(name: string, dist: DistExpr, value: Operand, options?: { readonly doc?: string }): void
}

/** Options of {@link ModelBuilder.scan}. */
export interface ScanOptions {
  /** The scanned dimension (time). */
  readonly over: DimArg
  /** The distribution of the state at the first step. */
  readonly init: DistExpr
  /** The distribution of the state at later steps, from the previous one. */
  readonly step: (s: ScanStepContext) => DistExpr
  /** The observations drawn at every step, from the current state. */
  readonly emit?: (s: ScanEmitContext) => void
  readonly doc?: string
}

const nameOf = (d: DimArg): string => (typeof d === 'string' ? d : d.name)

function constraintOf(c: ConstraintArg): ConstraintSpec {
  if ('kind' in c) return c
  const { lower, upper } = c
  if (lower !== undefined && upper !== undefined) return { kind: 'interval', low: lower, high: upper }
  if (lower !== undefined) return lower === 0 ? { kind: 'positive' } : { kind: 'greaterThan', low: lower }
  if (upper !== undefined) return { kind: 'lessThan', high: upper }
  return { kind: 'real' }
}

/**
 * The builder `defineModel` hands to its function. Its verbs return `Var` handles for the nodes that other nodes may use.
 *
 * @concept model-spec
 */
export class ModelBuilder {
  private readonly dimSpecs = new Map<string, DimSpec>()
  private readonly nodes: ModelNode[] = []
  private readonly values = new Map<string, ModelNode>()
  private readonly sites = new Set<string>()
  private readonly customFns: Record<string, CustomFn> = {}
  /** The names a scan's expressions may refer to besides the nodes before it (its state and previous state), while it is built. */
  private scanNames: ReadonlySet<string> = new Set()

  constructor(
    readonly name: string,
    inherited: CustomFns = {},
  ) {
    Object.assign(this.customFns, inherited)
  }

  /**
   * Declare a named dimension: a plate, a coordinate axis and a size in one.
   *
   * @param name - The dimension's name (`'row'`, `'school'`).
   * @param options - A fixed size and coordinate labels.
   * @returns The dimension.
   */
  range(name: string, options: RangeOptions = {}): Dim {
    if (name === '') throw new OmaiError('value', 'a dimension needs a name')
    if (this.dimSpecs.has(name)) throw new OmaiError('value', `the dimension '${name}' is declared twice`, { name })
    const size = options.size ?? options.labels?.length
    this.dimSpecs.set(name, {
      name,
      ...(size === undefined ? {} : { size }),
      ...(options.labels === undefined ? {} : { labels: options.labels }),
    })
    return { name }
  }

  private dimNames(dims: readonly DimArg[] | undefined): string[] | undefined {
    if (dims === undefined) return undefined
    const names = dims.map(nameOf)
    for (const n of names) if (!this.dimSpecs.has(n)) this.dimSpecs.set(n, { name: n })
    if (new Set(names).size !== names.length)
      throw new OmaiError('value', `dims repeat a dimension [${names.join(', ')}]`)
    return names
  }

  private claimValue(name: string): void {
    if (name === '') throw new OmaiError('value', 'a node needs a name')
    if (this.values.has(name)) throw new OmaiError('value', `the name '${name}' is already used by a node`, { name })
  }

  private claimSite(name: string): void {
    if (name === '') throw new OmaiError('value', 'a node needs a name')
    if (this.sites.has(name)) throw new OmaiError('value', `the site name '${name}' is already used`, { name })
    this.sites.add(name)
  }

  private refVar(node: ModelNode): Var {
    return new Var({ kind: 'ref', name: node.name }, node.dims, node.eventRank)
  }

  private check(v: Var, who: string): void {
    const missing = new Set<string>()
    const walk = (e: ExprSpec): void => {
      if (e.kind === 'ref' && !this.values.has(e.name) && !this.scanNames.has(e.name)) missing.add(e.name)
      else if (e.kind === 'op' || e.kind === 'custom') for (const a of e.args) walk(a)
    }
    walk(v.expr)
    if (missing.size > 0)
      throw new OmaiError(
        'value',
        `${who}: refers to ${[...missing].map((m) => `'${m}'`).join(', ')}, not a node of this model`,
        {
          names: [...missing],
        },
      )
  }

  private joint(dists: readonly (readonly string[])[], explicit: readonly string[] | undefined, who: string): string[] {
    const union: string[] = []
    for (const d of dists) for (const n of d) if (!union.includes(n)) union.push(n)
    if (explicit === undefined) return union
    for (const n of union)
      if (!explicit.includes(n))
        throw new OmaiError('shape', `${who}: declares dims [${explicit.join(', ')}] but its parameters carry '${n}'`, {
          dim: n,
        })
    return [...explicit]
  }

  private distChecks(d: DistExpr, who: string): void {
    for (const e of Object.values(d.spec.params)) this.check(new Var(e, [], 0), who)
  }

  /**
   * Declare data: a covariate, a response or index codes, bound when the model is bound.
   *
   * @param name - The node's name.
   * @param options - Dimensions, support, role and whether it may be left unbound.
   * @returns The node as an expression.
   */
  data(name: string, options: DataOptions = {}): Var {
    this.claimValue(name)
    const dims = this.dimNames(options.dims) ?? []
    const node: ModelNode = {
      kind: 'data',
      name,
      dims,
      eventRank: options.eventRank ?? 0,
      ...(options.support === undefined ? {} : { support: options.support }),
      ...(options.role === undefined ? {} : { role: options.role }),
      ...(options.indexes === undefined ? {} : { indexes: this.dimNames([options.indexes])?.[0] as string }),
      ...(options.optional === true ? { optional: true } : {}),
      ...(options.doc === undefined ? {} : { doc: options.doc }),
    }
    this.nodes.push(node)
    this.values.set(name, node)
    return this.refVar(node)
  }

  /**
   * Declare a latent variable: a random variable to infer, with the constraint its support (or `constraint`) gives.
   *
   * @param name - The node's (and site's) name.
   * @param dist - Its distribution; its parameters may be other nodes.
   * @param options - Dimensions (default: those of the parameters) and a constraint.
   * @returns The node as an expression.
   */
  latent(name: string, dist: DistExpr, options: LatentOptions = {}): Var {
    this.claimValue(name)
    this.claimSite(name)
    this.distChecks(dist, `latent '${name}'`)
    const dims = this.joint([dist.dims], this.dimNames(options.dims), `latent '${name}'`)
    const node: ModelNode = {
      kind: 'latent',
      name,
      dims,
      eventRank: dist.eventRank,
      dist: dist.spec,
      ...(options.constraint === undefined ? {} : { constraint: constraintOf(options.constraint) }),
      ...(options.doc === undefined ? {} : { doc: options.doc }),
    }
    this.nodes.push(node)
    this.values.set(name, node)
    return this.refVar(node)
  }

  /**
   * Declare an observed variable: the likelihood. The value is usually a data node; a number or nested array is a
   * constant observation.
   *
   * @param name - The site's name.
   * @param dist - The distribution of the observation.
   * @param value - The observed value.
   * @param options - Dimensions, and the policy for missing entries.
   */
  observe(name: string, dist: DistExpr, value: Operand, options: ObserveOptions = {}): void {
    this.claimSite(name)
    const v = toVar(value)
    this.distChecks(dist, `observed '${name}'`)
    this.check(v, `observed '${name}'`)
    if (v.eventRank !== dist.eventRank)
      throw new OmaiError(
        'shape',
        `observed '${name}': the value has ${v.eventRank} event axes, the distribution ${dist.eventRank}`,
      )
    const dims = this.joint([dist.dims, v.dims], this.dimNames(options.dims), `observed '${name}'`)
    this.nodes.push({
      kind: 'observed',
      name,
      dims,
      eventRank: dist.eventRank,
      dist: dist.spec,
      value: v.expr,
      ...(options.missing === undefined ? {} : { missing: options.missing }),
      ...(options.doc === undefined ? {} : { doc: options.doc }),
    })
  }

  /**
   * Declare a derived quantity (Stan's transformed parameters): an expression of other nodes, recorded in traces and
   * posteriors unless `record` is false.
   *
   * @param name - The node's name.
   * @param expr - The expression.
   * @param options - Whether to record it (default true).
   * @returns The node as an expression.
   */
  derived(name: string, expr: Operand, options: { readonly record?: boolean; readonly doc?: string } = {}): Var {
    this.claimValue(name)
    const record = options.record ?? true
    if (record) this.claimSite(name)
    const v = toVar(expr)
    this.check(v, `derived '${name}'`)
    const node: ModelNode = {
      kind: 'derived',
      name,
      dims: v.dims,
      eventRank: v.eventRank,
      expr: v.expr,
      record,
      ...(options.doc === undefined ? {} : { doc: options.doc }),
    }
    this.nodes.push(node)
    this.values.set(name, node)
    return this.refVar(node)
  }

  /**
   * Declare a generated quantity (Stan's generated quantities): drawn forward for predictive checks, never inferred.
   *
   * @param name - The node's name.
   * @param dist - Its distribution.
   * @param options - Dimensions.
   * @returns The node as an expression.
   */
  generated(
    name: string,
    dist: DistExpr,
    options: { readonly dims?: readonly DimArg[]; readonly doc?: string } = {},
  ): Var {
    this.claimValue(name)
    this.claimSite(name)
    this.distChecks(dist, `generated '${name}'`)
    const dims = this.joint([dist.dims], this.dimNames(options.dims), `generated '${name}'`)
    const node: ModelNode = {
      kind: 'generated',
      name,
      dims,
      eventRank: dist.eventRank,
      dist: dist.spec,
      ...(options.doc === undefined ? {} : { doc: options.doc }),
    }
    this.nodes.push(node)
    this.values.set(name, node)
    return this.refVar(node)
  }

  /**
   * Add a log-density term to the joint (Stan's `target +=`); its elements are summed.
   *
   * @param name - The site's name.
   * @param expr - The log-density term.
   */
  factor(name: string, expr: Operand): void {
    this.claimSite(name)
    const v = toVar(expr)
    this.check(v, `factor '${name}'`)
    this.nodes.push({ kind: 'factor', name, dims: v.dims, eventRank: v.eventRank, expr: v.expr })
  }

  /**
   * Declare a learnable, non-random parameter with an initial value (for optimisation and variational guides).
   *
   * @param name - The node's name.
   * @param init - The initial value (a number or nested array).
   * @param options - Dimensions of its leading axes and a constraint.
   * @returns The node as an expression.
   */
  param(
    name: string,
    init: number | readonly Operand[],
    options: { readonly dims?: readonly DimArg[]; readonly constraint?: ConstraintArg } = {},
  ): Var {
    this.claimValue(name)
    this.claimSite(name)
    const dims = this.dimNames(options.dims) ?? []
    const c = constant(init, dims)
    const node: ModelNode = {
      kind: 'param',
      name,
      dims,
      eventRank: c.eventRank,
      init: (c.expr as { readonly value: Json }).value,
      ...(options.constraint === undefined ? {} : { constraint: constraintOf(options.constraint) }),
    }
    this.nodes.push(node)
    this.values.set(name, node)
    return this.refVar(node)
  }

  /**
   * Declare a recurrence over a named dimension: a state whose first value is drawn from `init` and each later value from
   * `step(prev)`, with the observations `emit` declares drawn at every step from the current state. Time series,
   * state-space models and hidden Markov models are scans. The result is the stack of the states over the dimension (dims
   * `[over]`), which later nodes may use; inside the scan a node over the dimension is read at the current step and
   * every expression is a scalar (or the state's event axes): one scan is one series. In the compiled program the states
   * are the sites `<name>[0]`, `<name>[1]`, ... under one `markov` loop, the observations `<emission>[t]`.
   *
   * @param name - The scan's name (the states' prefix).
   * @param options - The dimension, the initial and step distributions and the emissions.
   * @returns The stacked states as an expression.
   * @throws OmaiError with code `shape` when an expression inside the scan keeps a dimension, `value` for a repeated name.
   * @example
   * ```ts
   * import { defineModel } from 'omai-engine/probprog/model'
   * import { normal } from 'omai-engine/probprog/model/dists'
   * const m = defineModel('walk', (b) => {
   *   const y = b.data('y', { dims: ['time'] })
   *   b.scan('x', {
   *     over: 'time',
   *     init: normal(0, 1),
   *     step: ({ prev }) => normal(prev, 0.5),
   *     emit: ({ state, observe }) => observe('y', normal(state, 0.3), y),
   *   })
   * })
   * m.spec.nodes.map((n) => n.kind) // => ['data', 'scan']
   * ```
   */
  scan(name: string, options: ScanOptions): Var {
    this.claimValue(name)
    this.claimSite(name)
    const over = this.dimNames([options.over])?.[0] as string
    const { init } = options
    this.distChecks(init, `scan '${name}' init`)
    const rank = init.eventRank
    this.scanNames = new Set([name, prevName(name)])
    try {
      const inside = (d: DistExpr, who: string): void => {
        const left = d.dims.filter((x) => x !== over)
        if (left.length > 0)
          throw new OmaiError(
            'shape',
            `${who}: has dimensions [${left.join(', ')}] left inside the scan (one scan is one series)`,
            {
              dims: left,
            },
          )
      }
      inside(init, `scan '${name}' init`)
      const step = options.step({ prev: new Var({ kind: 'ref', name: prevName(name) }, [], rank) })
      this.distChecks(step, `scan '${name}' step`)
      inside(step, `scan '${name}' step`)
      if (step.eventRank !== rank)
        throw new OmaiError('shape', `scan '${name}': the step has event rank ${step.eventRank}, the init ${rank}`)
      const emissions: ScanEmission[] = []
      options.emit?.({
        state: new Var({ kind: 'ref', name }, [], rank),
        observe: (emission, dist, value, o = {}) => {
          this.claimSite(emission)
          const v = toVar(value)
          this.distChecks(dist, `scan '${name}' emission '${emission}'`)
          this.check(v, `scan '${name}' emission '${emission}'`)
          inside(dist, `scan '${name}' emission '${emission}'`)
          if (v.dims.some((d) => d !== over))
            throw new OmaiError(
              'shape',
              `scan '${name}' emission '${emission}': the value has dimensions [${v.dims.join(', ')}]`,
            )
          if (v.eventRank !== dist.eventRank)
            throw new OmaiError(
              'shape',
              `scan '${name}' emission '${emission}': the value has ${v.eventRank} event axes, the distribution ${dist.eventRank}`,
            )
          emissions.push({
            name: emission,
            dist: dist.spec,
            value: v.expr,
            ...(o.doc === undefined ? {} : { doc: o.doc }),
          })
        },
      })
      const node: ModelNode = {
        kind: 'scan',
        name,
        dims: [over],
        eventRank: rank,
        over,
        init: init.spec,
        step: step.spec,
        emissions,
        ...(options.doc === undefined ? {} : { doc: options.doc }),
      }
      this.nodes.push(node)
      this.values.set(name, node)
      return this.refVar(node)
    } finally {
      this.scanNames = new Set()
    }
  }

  /**
   * An opaque function of other nodes: the escape hatch for anything the operator registry lacks. It works with the
   * sampling engines and the compiled program; message passing and static analysis treat it as opaque, and the spec
   * lists it in `customs` so a JSON receiver knows to supply it.
   *
   * @param name - The function's name (the key a receiver supplies it under).
   * @param fn - The function, from its arguments (with dimension names) to a value laid out as `dims` and event axes.
   * @param args - Its argument expressions.
   * @param shape - The dimensions and event rank of the result.
   * @returns The call as an expression.
   */
  custom(
    name: string,
    fn: CustomFn,
    args: readonly Operand[],
    shape: { readonly dims?: readonly DimArg[]; readonly eventRank?: number } = {},
  ): Var {
    if (name === '') throw new OmaiError('value', 'a custom function needs a name')
    const known = this.customFns[name]
    if (known !== undefined && known !== fn)
      throw new OmaiError('value', `the custom function '${name}' is already defined differently`, { name })
    const vs = args.map(toVar)
    for (const v of vs) this.check(v, `custom '${name}'`)
    const dims = this.dimNames(shape.dims) ?? []
    this.customFns[name] = fn
    return new Var(
      { kind: 'custom', name, args: vs.map((v) => v.expr), dims, eventRank: shape.eventRank ?? 0 },
      dims,
      shape.eventRank ?? 0,
    )
  }

  /**
   * A constant as an expression (see `constant`), for symmetry with the other verbs.
   *
   * @param value - A number, nested array or tensor.
   * @param dims - The names of its leading axes.
   * @returns The constant.
   */
  const(value: Exclude<Operand, Var>, dims: readonly DimArg[] = []): Var {
    return constant(value, this.dimNames(dims) ?? [])
  }

  /**
   * Instantiate a sub-model under a prefix: every node of `sub` is added as `prefix/name`, its dimensions are renamed
   * `prefix/dim` unless mapped to the dimensions of this model, and data nodes named in `bind` are replaced by the
   * expressions given (the dimensions of an expression stand for the data node's, in order). The other data nodes stay
   * data nodes, `prefix/name`.
   *
   * @param sub - The sub-model.
   * @param options - The prefix, expressions for some of its data nodes, and dimension mappings.
   * @returns The sub-model's latent, derived, generated and unbound data nodes, by their original names, as expressions.
   * @throws OmaiError with code `value` for a prefix clash and `shape` when a bound expression has the wrong number of
   *   dimensions.
   */
  include(
    sub: ModelDefinition,
    options: {
      readonly prefix: string
      readonly bind?: Readonly<Record<string, Operand>>
      readonly dims?: Readonly<Record<string, DimArg>>
    },
  ): Record<string, Var> {
    const { prefix } = options
    if (prefix === '') throw new OmaiError('value', 'include: a prefix is required')
    const dimMap = new Map<string, string>()
    for (const [from, to] of Object.entries(options.dims ?? {})) dimMap.set(from, this.dimNames([to])?.[0] as string)
    const bindings = new Map<string, Var>()
    for (const [name, operand] of Object.entries(options.bind ?? {})) {
      const node = sub.spec.nodes.find((n) => n.kind === 'data' && n.name === name)
      if (node === undefined)
        throw new OmaiError('value', `include: '${name}' is not a data node of ${sub.spec.name}`, { name })
      const v = toVar(operand)
      this.check(v, `include '${prefix}'`)
      if (v.dims.length !== node.dims.length)
        throw new OmaiError(
          'shape',
          `include: '${name}' has ${node.dims.length} dimensions, the expression ${v.dims.length}`,
        )
      node.dims.forEach((d, i) => {
        const outer = v.dims[i] as string
        const prior = dimMap.get(d)
        if (prior !== undefined && prior !== outer)
          throw new OmaiError('shape', `include: dimension '${d}' is bound to both '${prior}' and '${outer}'`)
        dimMap.set(d, outer)
      })
      bindings.set(name, v)
    }
    const renameDim = (d: string): string => {
      const m = dimMap.get(d)
      if (m !== undefined) return m
      const fresh = `${prefix}/${d}`
      dimMap.set(d, fresh)
      return fresh
    }
    for (const d of sub.spec.dims) {
      const to = renameDim(d.name)
      if (!this.dimSpecs.has(to)) this.dimSpecs.set(to, { ...d, name: to })
    }
    const rename = new Map<string, ExprSpec>()
    for (const n of sub.spec.nodes) {
      if (n.kind === 'data' && bindings.has(n.name)) rename.set(n.name, (bindings.get(n.name) as Var).expr)
      else if (n.kind !== 'observed' && n.kind !== 'factor') {
        rename.set(n.name, { kind: 'ref', name: `${prefix}/${n.name}` })
        if (n.kind === 'scan') rename.set(prevName(n.name), { kind: 'ref', name: prevName(`${prefix}/${n.name}`) })
      }
    }
    const out: Record<string, Var> = {}
    const mapExprDims = (e: ExprSpec): ExprSpec => {
      switch (e.kind) {
        case 'const':
          return e.dims === undefined ? e : { ...e, dims: e.dims.map(renameDim) }
        case 'ref':
          return e
        case 'op':
          return { ...e, args: e.args.map(mapExprDims) }
        case 'custom':
          return { ...e, dims: e.dims.map(renameDim), args: e.args.map(mapExprDims) }
      }
    }
    const mapNodeDims = (n: ModelNode): ModelNode => {
      const dims = n.dims.map(renameDim)
      const base = { ...n, dims, name: `${prefix}/${n.name}` }
      switch (base.kind) {
        case 'latent':
        case 'generated':
          return { ...base, dist: mapDist(base.dist) }
        case 'observed':
          return { ...base, dist: mapDist(base.dist), value: mapExprDims(base.value) }
        case 'derived':
        case 'factor':
          return { ...base, expr: mapExprDims(base.expr) }
        case 'scan':
          return {
            ...base,
            over: renameDim(base.over),
            init: mapDist(base.init),
            step: mapDist(base.step),
            emissions: base.emissions.map((m) => ({
              ...m,
              name: `${prefix}/${m.name}`,
              dist: mapDist(m.dist),
              value: mapExprDims(m.value),
            })),
          }
        default:
          return base
      }
    }
    const mapDist = <T extends { readonly params: { readonly [k: string]: ExprSpec } }>(d: T): T => ({
      ...d,
      params: Object.fromEntries(Object.entries(d.params).map(([k, e]) => [k, mapExprDims(e)])),
    })
    for (const original of sub.spec.nodes) {
      if (original.kind === 'data' && bindings.has(original.name)) continue
      const node = substituteNode(mapNodeDims(original), rename)
      // Rewriting dimensions inside substituted expressions is left to the outer model's own names: they are already outer.
      this.claimNodeName(node)
      this.nodes.push(node)
      if (node.kind !== 'observed' && node.kind !== 'factor') {
        this.values.set(node.name, node)
        out[original.name] = this.refVar(node)
      }
    }
    for (const name of sub.spec.customs) {
      const fn = sub.customs[name]
      if (fn !== undefined) this.customFns[name] = fn
    }
    return out
  }

  private claimNodeName(node: ModelNode): void {
    if (node.kind !== 'observed' && node.kind !== 'factor') this.claimValue(node.name)
    if (node.kind !== 'data' && !(node.kind === 'derived' && !node.record)) this.claimSite(node.name)
    if (node.kind === 'scan') for (const m of node.emissions) this.claimSite(m.name)
  }

  /** The finished definition, checked. */
  finish(): ModelDefinition {
    const used = new Set<string>()
    const walk = (e: ExprSpec): void => {
      if (e.kind === 'custom') used.add(e.name)
      if (e.kind === 'op' || e.kind === 'custom') for (const a of e.args) walk(a)
    }
    for (const n of this.nodes) {
      if (n.kind === 'latent' || n.kind === 'generated') for (const e of Object.values(n.dist.params)) walk(e)
      else if (n.kind === 'observed') {
        for (const e of Object.values(n.dist.params)) walk(e)
        walk(n.value)
      } else if (n.kind === 'derived' || n.kind === 'factor') walk(n.expr)
    }
    const customs = [...used].sort()
    const spec = {
      schema: MODEL_SPEC_SCHEMA,
      name: this.name,
      registry: OPERATOR_REGISTRY,
      dims: [...this.dimSpecs.values()],
      nodes: this.nodes,
      customs,
    } as const
    const def = new ModelDefinition(spec, Object.fromEntries(customs.map((c) => [c, this.customFns[c] as CustomFn])))
    validateSpec(def.spec)
    return def
  }
}

/**
 * Define a model: run `build` against a `ModelBuilder` and return the definition (a spec plus its custom functions).
 * The spec is checked on the way out.
 *
 * @param name - The model's name.
 * @param build - Declares the model through the builder's verbs.
 * @returns The model definition; its `spec` is plain JSON.
 * @throws OmaiError with code `value` or `shape` for a repeated name, a reference to a node that does not exist, a
 *   parameter a family does not have, or dimensions that do not fit.
 * @concept model-spec
 * @index function Build a model spec with named dimensions, data, latents, observations and derived nodes.
 * @example
 * ```ts
 * import { defineModel } from 'omai-engine/probprog/model'
 * import { normal } from 'omai-engine/probprog/model/dists'
 * const m = defineModel('normal-mean', (b) => {
 *   const n = b.range('row')
 *   const y = b.data('y', { dims: [n] })
 *   const mu = b.latent('mu', normal(0, 2))
 *   b.observe('y', normal(mu, 1.5), y)
 * })
 * m.spec.nodes.map((x) => [x.name, x.kind]) // => [['y', 'data'], ['mu', 'latent'], ['y', 'observed']]
 * ```
 */
export function defineModel(name: string, build: (m: ModelBuilder) => void): ModelDefinition {
  const builder = new ModelBuilder(name)
  build(builder)
  return builder.finish()
}
