/**
 * The graph under construction: the variables and factors a compilation has made, the names it gave them (auxiliary variables
 * `y#0[3]` count up per site, so their ids depend on the order of evaluation), the atoms that affine forms are made of, and
 * the operations on them that are about the graph and not about the model: declare a variable, add a factor (checking
 * the operator exists in the registry), turn an affine form into a variable (`materialise`), and drop the dead `apply` factors
 * of an arithmetic chain (`prune`).
 */

import { type Json, OmaiError } from '../../../base/index.ts'
import {
  compileGraph,
  type FactorOperator,
  type FactorSpec,
  type GateBranch,
  type OperatorName,
  operatorNames,
  ParameterError,
  type VariableSpec,
} from '../../../message-passing/index.ts'
import { auxId, elementId, factorId } from '../ids.ts'
import type { OperatorTable } from '../operators.ts'
import { type Aff, type Atom, affOf, type Cell, isConstCell } from './_cells.ts'
import { factorRejected, noOperator } from './_reasons.ts'
import type { SiteLayout } from './types.ts'

/** A factor the compiler adds: the engine's `FactorSpec` with the operator named by `OperatorName`, so a typo or a rename fails `tsc`. */
export type CompilerFactor = Omit<FactorSpec, 'operator'> & { readonly operator: OperatorName }

/**
 * The operators the value algebra emits itself, whatever mapping asks for a value: turning an affine form into a variable (`materialise`
 * and the components of a vector: `plus`, `difference`, `sum`, `times-constant`, `equal`, `linear`, `dot-constant`), the arithmetic of random
 * values (`product`, `gamma-product`, the extremum `max` and `min`, `select`) and the chains of unary functions (`apply`, `apply-product`).
 * A mapping's own `requires` leaves them out.
 */
export const VALUE_ALGEBRA_OPERATORS: readonly OperatorName[] = [
  'apply',
  'apply-product',
  'difference',
  'dot-constant',
  'equal',
  'gamma-product',
  'linear',
  'max',
  'min',
  'plus',
  'product',
  'select',
  'sum',
  'times-constant',
]

/** Whether a name is one of the engine's factor operators (`operatorNames`, the registry's names in order). */
export function isOperatorName(name: string): name is OperatorName {
  return (operatorNames as readonly string[]).includes(name)
}

/**
 * The factor operator a model operator maps to: the model operator's own `factor`, else its name (which must then be a factor operator
 * too: a refusal names it when it is not).
 */
export function factorName(table: OperatorTable, op: string): OperatorName {
  const named = table[op]?.factor ?? op
  if (!isOperatorName(named)) throw noOperator(named)
  return named
}

/** How much a graph under construction held when `checkpoint` was taken (what `restore` cuts back to). */
export interface BuilderCheckpoint {
  readonly variables: number
  readonly factors: number
  readonly auxiliary: number
  readonly auxNames: number
  readonly atoms: number
}

/** The variables, factors and names of a graph being built, against a registry of factor operators. */
export class GraphBuilder {
  readonly variables: VariableSpec[] = []
  readonly factors: FactorSpec[] = []
  readonly auxiliary: string[] = []
  /** The scalar random values the affine forms are made of, by key (see `atomKey`). */
  readonly atoms: Map<string, Atom> = new Map()
  private readonly auxCount = new Map<string, number>()
  /** The keys of `auxCount` in the order they were counted up, so `restore` can count them down again. */
  private readonly auxLog: string[] = []
  private readonly components = new Map<string, string>()
  private readonly names = new Set<string>()
  private readonly byName = new Map<string, VariableSpec>()

  constructor(
    readonly ops: Readonly<Record<string, FactorOperator>>,
    private readonly modelOps: OperatorTable,
  ) {}

  /** What the graph holds now, to `restore` it if a mapping that only tried a node finds the node is not its pattern. */
  checkpoint(): BuilderCheckpoint {
    return {
      variables: this.variables.length,
      factors: this.factors.length,
      auxiliary: this.auxiliary.length,
      auxNames: this.auxLog.length,
      atoms: this.atoms.size,
    }
  }

  /** Whether a variable, factor or auxiliary name was made since `checkpoint`. */
  changedSince(cp: BuilderCheckpoint): boolean {
    return (
      this.variables.length !== cp.variables ||
      this.factors.length !== cp.factors ||
      this.auxiliary.length !== cp.auxiliary
    )
  }

  /**
   * Cut the graph back to a checkpoint: the variables, factors and auxiliary names made since are removed (with the names they took and the
   * components and atoms they introduced), so what an attempt made leaves no residue. The counters of auxiliary names are counted down
   * too, so the ids made after a failed attempt are the ids they would have had if it had not been made.
   */
  restore(cp: BuilderCheckpoint): void {
    for (const v of this.variables.slice(cp.variables)) {
      this.names.delete(v.name)
      this.byName.delete(v.name)
      this.components.delete(v.name)
    }
    this.variables.length = cp.variables
    this.factors.length = cp.factors
    this.auxiliary.length = cp.auxiliary
    for (const key of this.auxLog.splice(cp.auxNames)) this.auxCount.set(key, (this.auxCount.get(key) as number) - 1)
    let k = 0
    for (const key of [...this.atoms.keys()]) if (k++ >= cp.atoms) this.atoms.delete(key)
  }

  need(name: OperatorName): FactorOperator {
    const op = this.ops[name]
    if (op === undefined) throw noOperator(name)
    return op
  }

  addVariable(v: VariableSpec): string {
    if (this.names.has(v.name)) throw new OmaiError('value', `factor graph: the variable '${v.name}' is made twice`)
    this.names.add(v.name)
    this.byName.set(v.name, v)
    this.variables.push(v)
    return v.name
  }

  /**
   * The number of states of a discrete variable made so far. A variable with no declared size (a Bernoulli variable, or a name that
   * is not a variable of the graph) reads as 2: the compiler only asks about variables it has made, and a binary one has no size.
   */
  sizeOf(name: string): number {
    return (this.variables.find((u) => u.name === name)?.family.size ?? 2) as number
  }

  /**
   * What a Gaussian port of a factor takes for the value of a cell: `{ const }` when the cell is a known number, else the variable
   * that carries its affine form (made by `materialise` when the form is not one variable already).
   */
  portOf(cell: Cell, site: string, index: number | undefined): string | { readonly const: number } {
    if (isConstCell(cell)) return { const: cell.aff.offset }
    return this.materialise(affOf(cell), site, index) as string
  }

  /**
   * Add a factor defined by `variable`. The factor is checked in the way the engine will check the finished graph (`compileGraph`: the
   * parameters against the operator's declared ones, the arity, the families the ports take, the operator's own `check`), on the factor and the
   * variables it names, so a mapping that builds a factor the engine would refuse is refused where it is built.
   */
  addFactor(f: CompilerFactor, variable: string): void {
    this.need(f.operator)
    const factor = { ...f, id: factorId(variable) }
    const reason = this.rejection(factor)
    if (reason !== undefined) throw factorRejected(factor.id, reason)
    this.factors.push(factor)
  }

  /** What the engine says against a factor made of the variables it names (`compileGraph` on that part), or `undefined` when it takes it. */
  private rejection(factor: FactorSpec): string | undefined {
    const named: VariableSpec[] = []
    for (const a of factor.args) {
      const v = typeof a === 'string' ? this.byName.get(a) : undefined
      if (v !== undefined && !named.includes(v)) named.push(v)
    }
    try {
      compileGraph({ variables: named, factors: [factor] }, this.ops)
      return undefined
    } catch (e) {
      if (e instanceof ParameterError) return e.message
      throw e
    }
  }

  /**
   * Add a factor that one of the engine's constructors made (`gaussianPrior`, `priorFactor`, `dirichletPrior`, ...: they return the engine's
   * `FactorSpec`, whose operator is a plain string): the operator is checked against the names of the registry when it is added.
   */
  addLibraryFactor(f: FactorSpec, variable: string): void {
    if (!isOperatorName(f.operator)) throw noOperator(f.operator)
    this.addFactor({ ...f, operator: f.operator }, variable)
  }

  /**
   * Drops the `apply` factors whose output nothing reads (the Gaussian approximation an arithmetic chain passed through before it was
   * collapsed into one factor from the chain's input): a flat output sends nothing back, so they only cost time.
   */
  prune(sites: readonly SiteLayout[]): void {
    const kept = new Set(sites.flatMap((s) => s.variables.filter((v): v is string => v !== null)))
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

  auxName(site: string, index: number | undefined): string {
    const key = `${site}|${index ?? ''}`
    const k = this.auxCount.get(key) ?? 0
    this.auxCount.set(key, k + 1)
    this.auxLog.push(key)
    const name = auxId(site, k, index)
    this.auxiliary.push(name)
    return name
  }

  /**
   * The name of an auxiliary factor that defines no variable of its own (a tie of a variable to a coordinate of a vector): the next
   * `<site>#<k>[<flat index>]`, the id the factor is added under, kept in step with `auxName` so the two never collide.
   */
  auxFactorName(site: string, index: number | undefined): string {
    const key = `${site}|${index ?? ''}`
    const k = this.auxCount.get(key) ?? 0
    this.auxCount.set(key, k + 1)
    this.auxLog.push(key)
    return auxId(site, k, index)
  }

  atomKey(a: Atom): string {
    return a.kind === 'var' ? `v:${a.name}` : `c:${a.vector}:${a.index}`
  }

  varAtom(name: string): Cell {
    const atom: Atom = { kind: 'var', name }
    const key = this.atomKey(atom)
    this.atoms.set(key, atom)
    return { kind: 'aff', aff: { terms: new Map([[key, 1]]), offset: 0 } }
  }

  compAtom(vector: string, index: number, size: number): Cell {
    const atom: Atom = { kind: 'comp', vector, index, size }
    const key = this.atomKey(atom)
    this.atoms.set(key, atom)
    return { kind: 'aff', aff: { terms: new Map([[key, 1]]), offset: 0 } }
  }

  /** The scalar variable that stands for coordinate `index` of a vector variable (made on first use, a unit `dot-constant`). */
  component(vector: string, index: number, size: number, vagueVariance?: number): string {
    const name = elementId(vector, index)
    if (!this.components.has(name)) {
      this.components.set(name, name)
      this.addVariable({ name, family: { kind: 'gaussian' } })
      const a = new Array<number>(size).fill(0)
      a[index] = 1
      // A vector with no prior of its own begins with a vague message in each coordinate (the factor's first update replaces it),
      // so the other factors on it have a proper cavity to start from.
      const vague = [...new Array<number>(size).fill(0), ...new Array<number>(size * size).fill(0)]
      if (vagueVariance !== undefined) vague[size + index * size + index] = 1 / vagueVariance
      this.addFactor(
        {
          operator: 'dot-constant',
          args: [name, vector],
          params: { a },
          ...(vagueVariance === undefined ? {} : { start: [[0, 1 / vagueVariance], vague] }),
        },
        name,
      )
    }
    return name
  }

  /**
   * A variable (or a number, when it is constant) for an affine form: the variable itself when the form is one atom, else
   * an auxiliary variable defined by the affine operators of the engine. With `as`, a variable of that name is made.
   */
  materialise(aff: Aff, site: string, index: number | undefined, as?: string): string | number {
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
}

/** The branch of a state that has no factor (`{}`: the constant 1 on the gate's variables). */
export type NoFactor = Readonly<Record<string, never>>

/**
 * A `gate` factor under assembly: the selector (a discrete or Bernoulli variable), the variables the branches share, and one
 * branch per state of the selector. The factor's arguments are the selector and then `variables`; a branch refers to a variable
 * by its position in `variables` (`slot`), so position 0 is the first one, usually the output the gate defines.
 */
export class GateBuilder {
  readonly variables: string[]
  readonly branches: (GateBranch | NoFactor)[] = []

  constructor(
    readonly selector: string,
    variables: readonly string[] = [],
  ) {
    this.variables = [...variables]
  }

  /** The position of `variable` among the gate's variables, added at the end when it is not there yet. */
  slot(variable: string): number {
    let at = this.variables.indexOf(variable)
    if (at < 0) {
      at = this.variables.length
      this.variables.push(variable)
    }
    return at
  }

  /** The branch of the next state of the selector. */
  add(branch: GateBranch | NoFactor): void {
    this.branches.push(branch)
  }

  /** The `gate` factor: arguments the selector and the variables, one branch per state. */
  factor(): CompilerFactor {
    return {
      operator: 'gate',
      args: [this.selector, ...this.variables],
      params: { arity: this.variables.length, branches: this.branches as unknown as Json },
    }
  }
}

/**
 * The gate of a discrete variable `name` whose state is chosen by `selector` among Dirichlet rows `rowVars`: branch `j` is the
 * conjugate `discrete` factor of `name` on row `j` (a transition matrix to learn, or the words of a topic model).
 */
export function discreteRowGate(selector: string, name: string, rowVars: readonly string[]): CompilerFactor {
  const gate = new GateBuilder(selector, [name, ...rowVars])
  for (const j of rowVars.keys()) gate.add({ operator: 'discrete', args: [0, 1 + j], params: {} })
  return gate.factor()
}

/** The `gaussian-gamma-precision` factor of a Gaussian `name` with a mean and the precision `scale tau + offset`. */
export function precisionFactor(
  name: string,
  mean: string | { readonly const: number },
  tau: { readonly variable: string; readonly scale: number; readonly offset: number },
): CompilerFactor {
  const params: Record<string, Json> = {}
  if (tau.scale !== 1) params['scale'] = tau.scale
  if (tau.offset !== 0) params['offset'] = tau.offset
  return {
    operator: 'gaussian-gamma-precision',
    args: [name, mean, tau.variable],
    ...(Object.keys(params).length === 0 ? {} : { params }),
  }
}

/**
 * The `gaussian-gamma-precision` branch of a gate, over argument positions (`args`: the value, the mean and the position of the
 * gamma variable) and the precision `scale tau + offset`. Unlike the factor of `precisionFactor`, a branch always names both parameters.
 */
export function precisionBranch(
  args: GateBranch['args'],
  tau: { readonly scale: number; readonly offset: number },
): GateBranch {
  return { operator: 'gaussian-gamma-precision', args, params: { scale: tau.scale, offset: tau.offset } }
}
