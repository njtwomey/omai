/**
 * Compiling a bound model spec to a factor graph for message passing: `toFactorGraph` expands the spec's named
 * dimensions (plates) into the scalar and vector variables the engine of `omai-engine/message-passing` supports, maps
 * distribution families and operators to the engine's message families and factor operators, and reports exactly what it
 * cannot map and why. The engine has no plates, so the graph is an expansion: a plate of N rows is N copies of the
 * factors over it, with the data folded into the factors' constants. That is the documented first version; plated
 * operators (a factor over a whole plate, with messages as tensors) would remove the copies, the per-row auxiliary
 * variables and the dependence of the graph on the data values.
 *
 * What it maps. Latents: `normal` (a Gaussian variable per element, or one `gaussian-mv` vector variable when a
 * `matmul` reads it, so the posterior keeps its correlations), `gamma`, `exponential`, `beta`, `dirichlet`, and
 * `bernoulli` or `categorical` with known probabilities. Values: expressions that are affine in Gaussian latents (`add`,
 * `sub`, `neg`, `mul` and `div` by a known value, `sum`, `mean`, `at` by index data, `matmul` with data) become the
 * affine operators of the engine (`plus`, `difference`, `times-constant`, `sum`, `linear`, `dot-constant`). Likelihoods:
 * observed `normal` with a known scale (`gaussian`, or `linear-regression-row` for a regression row), `bernoulli` with
 * `probit(affine)` (`probit`), `sigmoid(affine)` or `logits` (`logistic`), `gt(affine, c)` (`greater-than`, a hard
 * comparison), or `probs` a beta latent (`bernoulli`); `binomial`, `poisson` (with an exposure) and `categorical` on a
 * beta, gamma or Dirichlet latent (the conjugate operators); and a Bernoulli network over scalar nodes, as `table`
 * factors (what belief propagation runs on). Anything else is reported with a reason, never silently dropped.
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
import { bernoulli, beta, type Distribution, dirichlet, gamma, multivariateNormal, normal } from '../../dists/index.ts'
import {
  bernoulliEta,
  betaEta,
  type CompiledGraph,
  compileGraph,
  dirichletPrior,
  type EngineSettings,
  evidenceOf,
  type FactorGraphSpec,
  type FactorOperator,
  type FactorSpec,
  type FamilyKind,
  defaultOperators as factorOperators,
  gammaPrior,
  gaussianMvPrior,
  gaussianPrior,
  klqpProjection,
  type MpState,
  marginalsOf,
  momentProjection,
  priorFactor,
  sumProductProjection,
  type VariableSpec,
} from '../../message-passing/index.ts'
import { asTensor, fromStorage, type NestedArray, toFlat, type Value } from '../../tensor/index.ts'
import { definePosterior, type Posterior } from '../posterior.ts'
import type { SpecAnalysis } from './analysis.ts'
import type { BoundModel } from './bind.ts'
import { distributionFamilies } from './dists/families.ts'
import { auxId, elementId, factorId, siteOfId } from './ids.ts'
import { defaultOperators, type NamedValue, type OperatorTable } from './operators.ts'
import { type DistSpec, type ExprSpec, type ModelNode, nodeRefs } from './spec.ts'

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
  if (c.kind !== 'aff')
    throw new Refusal('operator', `arithmetic on the ${c.family} variable '${c.variable}': only Gaussian values add up`)
  return c.aff
}

function mulAff(a: Aff, b: Aff): Aff {
  if (a.terms.size === 0) return scaleAff(b, a.offset)
  if (b.terms.size === 0) return scaleAff(a, b.offset)
  throw new Refusal(
    'operator',
    'a product of two random values has no Gaussian message (the `product` factor is not mapped)',
  )
}

const prod = (xs: readonly number[]): number => xs.reduce((a, b) => a * b, 1)

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
  gamma: 'gamma',
  exponential: 'gamma',
  beta: 'beta',
  dirichlet: 'dirichlet',
  bernoulli: 'bernoulli',
  categorical: 'discrete',
}

/** The factor operator a registry name maps to: the model operator's own `factor`, else its name. */
function factorName(table: OperatorTable, op: string): string {
  return table[op]?.factor ?? op
}

class Compiler {
  readonly variables: VariableSpec[] = []
  readonly factors: FactorSpec[] = []
  readonly issues: FactorIssue[] = []
  readonly sites: SiteLayout[] = []
  readonly auxiliary: string[] = []
  private readonly consts = new Map<string, NamedValue>()
  private readonly syms = new Map<string, Sym>()
  private readonly lazy = new Map<string, ExprSpec>()
  private readonly failed = new Set<string>()
  private readonly unbound: ReadonlySet<string>
  private readonly atoms = new Map<string, Atom>()
  private readonly auxCount = new Map<string, number>()
  private readonly packed = new Set<string>()
  private readonly components = new Map<string, string>()
  private readonly names = new Set<string>()
  private readonly nodes = new Map<string, ModelNode>()
  private readonly families = distributionFamilies()

  constructor(
    private readonly bound: BoundModel,
    private readonly modelOps: OperatorTable,
    private readonly ops: Readonly<Record<string, FactorOperator>>,
  ) {
    this.unbound = new Set(bound.unbound)
    for (const n of bound.spec.nodes) if (n.kind !== 'observed') this.nodes.set(n.name, n)
    this.findPacked()
  }

  // ----- the plan: which vector latents stay one vector variable -----

  /** A latent over one dimension that a `matmul` reads stays a vector variable, so its posterior keeps the correlations. */
  private findPacked(): void {
    const walk = (e: ExprSpec): void => {
      if (e.kind !== 'op' && e.kind !== 'custom') return
      if (e.kind === 'op' && e.op === 'matmul')
        for (const a of e.args) {
          const target = a.kind === 'ref' ? this.nodes.get(a.name) : undefined
          if (target?.kind === 'latent' && target.dims.length === 1 && target.dist.family === 'normal')
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

  private zip(syms: readonly Sym[], f: (cells: readonly Cell[]) => Cell): Sym {
    const dims = this.joinDims(syms)
    const shape = this.shapeOf(dims)
    const cells: Cell[] = []
    for (let i = 0; i < prod(shape); i++) {
      const idx = unflatten(i, shape)
      cells.push(f(syms.map((s) => this.cellAt(s, dims, idx))))
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
        return false
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
      case 'custom':
        throw new Refusal('opaque', `the custom function '${e.name}' is opaque to message passing`)
    }
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
        throw new Refusal('opaque', `the custom function '${e.name}' is opaque to message passing`)
      case 'const':
        return this.fromNamed(this.evalNamed(e))
      case 'op':
        return this.symOp(
          e.op,
          e.args.map((a) => this.evalSym(a)),
          e.params ?? {},
        )
    }
  }

  private symOp(op: string, args: readonly Sym[], params: { readonly [k: string]: Json }): Sym {
    const [a, b] = args as [Sym, Sym]
    switch (op) {
      case 'add':
        return this.zip([a, b], ([x, y]) => ({ kind: 'aff', aff: combine(affOf(x as Cell), affOf(y as Cell), 1) }))
      case 'sub':
        return this.zip([a, b], ([x, y]) => ({ kind: 'aff', aff: combine(affOf(x as Cell), affOf(y as Cell), -1) }))
      case 'neg':
        return { dims: a.dims, cells: a.cells.map((x) => ({ kind: 'aff', aff: scaleAff(affOf(x), -1) })) }
      case 'mul':
        return this.zip([a, b], ([x, y]) => ({ kind: 'aff', aff: mulAff(affOf(x as Cell), affOf(y as Cell)) }))
      case 'div':
        return this.zip([a, b], ([x, y]) => {
          if (!isConstCell(y as Cell))
            throw new Refusal('operator', 'division by a random value has no Gaussian message')
          const d = (y as Cell & { kind: 'aff' }).aff.offset
          return { kind: 'aff', aff: scaleAff(affOf(x as Cell), 1 / d) }
        })
      case 'sum':
      case 'mean':
        return this.reduce(a, typeof params['dim'] === 'string' ? params['dim'] : undefined, op === 'mean')
      case 'at':
        return this.gather(a, b)
      case 'matmul':
        return this.matmul(a, b)
      default:
        throw new Refusal(
          'operator',
          `operator '${op}' applied to a random value has no message rule (message passing handles affine maps of Gaussians; a link such as probit or sigmoid is mapped only as the likelihood of a Bernoulli observation)`,
        )
    }
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
      if (!isConstCell(code)) throw new Refusal('operator', 'at: the index must be data, not random')
      const c = code.aff.offset
      if (!Number.isInteger(c) || c < 0 || c >= this.size(first))
        throw new Refusal('data', `at: index ${c} is out of range for '${first}'`)
      let flat = c
      for (let k = 0; k < rest.length; k++) flat = flat * (restShape[k] as number) + (at[idx.dims.length + k] as number)
      cells.push(a.cells[flat] as Cell)
    }
    return { dims, cells }
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
        acc = combine(acc, mulAff(affOf(x), affOf(y)), 1)
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
    })
  }

  private latent(node: ModelNode & { readonly kind: 'latent' }): void {
    const fam = node.dist.family
    if (node.constraint !== undefined && node.constraint.kind !== 'real')
      throw new Refusal(
        'constraint',
        `'${node.name}' has the constraint '${node.constraint.kind}': a message family lives on the family's own support, so a constraint beyond it is not expressible`,
      )
    const kind = MESSAGE_FAMILY[fam]
    if (kind === undefined)
      throw new Refusal(
        'family',
        `'${node.name}' is ${fam}: the engine has message families gaussian, gamma, beta, bernoulli, discrete and dirichlet, and none for ${fam}`,
      )
    if (fam === 'normal') {
      this.latentNormal(node)
      return
    }
    if (fam === 'bernoulli' || fam === 'categorical') {
      this.latentDiscrete(node)
      return
    }
    if (fam === 'dirichlet') {
      this.latentDirichlet(node)
      return
    }
    const shape = this.shapeOf(node.dims)
    const n = prod(shape)
    const params =
      fam === 'gamma'
        ? [this.constParam(node, 'concentration', 'the shape'), this.constParam(node, 'rate', 'the rate')]
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
      cells.push({ kind: 'rv', variable: name, family: kind })
      vars.push(name)
    }
    this.syms.set(node.name, { dims: node.dims, cells })
    this.layout(node, 'latent', kind, node.dims.length === 0 ? 'scalar' : 'elements', vars, true)
  }

  private latentNormal(node: ModelNode & { readonly kind: 'latent' }): void {
    const shape = this.shapeOf(node.dims)
    const n = prod(shape)
    const loc = this.over(this.paramSym(node.dist, 'loc'), node.dims)
    const scale = this.constParam(node, 'scale', 'the scale')
    for (const s of scale)
      if (!(s > 0)) throw new Refusal('data', `the scale of '${node.name}' is ${s}: it must be positive`)
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
      if (isConstCell(l)) this.addFactor(gaussianPrior(name, l.aff.offset, variance), name)
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

  private latentDiscrete(node: ModelNode & { readonly kind: 'latent' }): void {
    if (node.dist.family === 'categorical')
      throw new Refusal(
        'structure',
        `'${node.name}' is categorical: a discrete latent is mapped only for Bernoulli, with a table over scalar nodes`,
      )
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
    if (parents.length === 0) {
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
    if (!node.record) {
      this.syms.set(node.name, { dims: node.dims, cells })
      return
    }
    const vars: (string | null)[] = []
    const out: Cell[] = cells.map((c, i) => {
      if (c.kind === 'rv') {
        vars.push(c.variable)
        return c
      }
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

  private observed(node: ModelNode & { readonly kind: 'observed' }): void {
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
    const vars: (string | null)[] = []
    const discreteParents = this.discreteParents(node)
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
        const scale = this.paramCell(node, 'scale', i)
        if (!isConstCell(scale))
          throw new Refusal(
            'operator',
            `the scale of '${node.name}' depends on a random value: a known scale is needed (a Gamma-distributed precision is the 'gaussian-gamma-precision' operator, which is not mapped yet)`,
          )
        if (!(scale.aff.offset > 0))
          throw new Refusal('data', `the scale of '${node.name}' is ${scale.aff.offset}: it must be positive`)
        const precision = 1 / scale.aff.offset ** 2
        const loc = this.paramCell(node, 'loc', i)
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
          this.addFactor(
            { operator: 'linear-regression-row', args: [name, first.vector], params: { a, precision } },
            name,
          )
        } else {
          const m = this.materialise(aff, node.name, index)
          this.addFactor({ operator: 'gaussian', args: [name, m as string], params: { precision } }, name)
        }
        return name
      }
      case 'bernoulli':
        return this.observedBernoulli(node, name, y, i, index)
      case 'binomial':
      case 'poisson':
      case 'categorical':
        return this.observedConjugate(node, fam, name, y, i)
      default:
        throw new Refusal(
          'family',
          `'${node.name}' is observed as ${fam}: the likelihoods mapped are normal, bernoulli, binomial, poisson and categorical`,
        )
    }
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
 * term, a family with no message family, a `custom` call, a random scale, a product of two random values, a non-affine
 * operator on a random value (except as a Bernoulli link), a constraint beyond the family's support, missing data that
 * is not ignored, a subsampling batch and a plated table over discrete nodes each become an issue naming the node and
 * the reason, and no graph is produced.
 *
 * @param bound - The bound model.
 * @param options - The factor operators to compile against (default the engine's `defaultOperators()`) and the model's operator table.
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
 * const c = toFactorGraph(m.bind({ X: [[1, 0.5], [1, -1]], y: [1, 0] }))
 * c.graph?.variables.map((v) => v.name) // => ['w', 'y#0[0]', 'y[0]', 'y#0[1]', 'y[1]']
 * c.graph?.factors.map((f) => f.id) // => ['factor:w', 'factor:y#0[0]', 'factor:y[0]', 'factor:y#0[1]', 'factor:y[1]']
 * ```
 */
export function toFactorGraph(
  bound: BoundModel,
  options: {
    readonly operators?: Readonly<Record<string, FactorOperator>>
    readonly modelOperators?: OperatorTable
    readonly analysis?: SpecAnalysis
  } = {},
): FactorGraphCompilation {
  const c = new Compiler(bound, options.modelOperators ?? defaultOperators(), options.operators ?? factorOperators())
  c.run()
  const graph: FactorGraphSpec = { variables: c.variables, factors: c.factors }
  const opaque = options.analysis?.opaque ?? opaqueNodes(bound)
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
    expanded: { plates, variables: graph.variables.length, factors: graph.factors.length },
  }
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
 * The factors of a compiled graph whose operator has no message rule for a method (an operator with only an EP rule cannot
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
 * const g = compileGraph({ variables: [{ name: 'x', family: { kind: 'gaussian' } }, { name: 'b', family: { kind: 'bernoulli' }, observed: 1 }], factors: [{ operator: 'probit', args: ['b', 'x'] }, { operator: 'prior', args: ['x'], params: { eta: [0, 1] } }] }, defaultOperators())
 * [operatorsWithout(g, 'ep'), operatorsWithout(g, 'vmp')] // => [[], ['probit']]
 * ```
 */
export function operatorsWithout(g: CompiledGraph, rule: MessageRule): string[] {
  const out = new Set<string>()
  for (const f of g.factors) {
    const op = f.operator
    const has = rule === 'ep' ? op.ep !== undefined || op.tilted !== undefined : op[rule] !== undefined
    if (!has) out.add(op.name)
  }
  return [...out]
}

/** The engine settings a method's projection uses to read the evidence or bound off a state. */
function settingsFor(rule: MessageRule): EngineSettings {
  return {
    projection: rule === 'ep' ? momentProjection : rule === 'vmp' ? klqpProjection : sumProductProjection,
    schedule: 'serial',
    granularity: 'sweep',
    damping: 0,
    improper: 'skip',
    tol: 0,
    maxSweeps: 0,
  }
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
      case 'gamma':
        out[s.site] = gamma(
          T(
            vs.map((x) => (x['improper'] === true ? 1 : num(x['shape']))),
            shape,
          ),
          T(
            vs.map((x) => (x['improper'] === true ? 1e-6 : num(x['rate']))),
            shape,
          ),
        )
        break
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
  const evidence = evidenceOf(g, settingsFor(rule), state)
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
  if (lacking.length > 0)
    return {
      ok: false,
      reason: `the operator${lacking.length > 1 ? 's' : ''} ${lacking.join(', ')} ${lacking.length > 1 ? 'have' : 'has'} no ${rule === 'ep' ? 'expectation propagation' : rule === 'vmp' ? 'variational' : 'sum-product'} message rule`,
      missing: [`${rule} message rules`],
    }
  return {
    ok: true,
    reason: `a factor graph of ${g.variables.length} variables and ${g.factors.length} factors`,
    missing: [],
  }
}

/** The compiled graph of a compilation against the engine's operators. @throws OmaiError when the compilation has issues. */
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
