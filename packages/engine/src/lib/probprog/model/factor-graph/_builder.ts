/**
 * The graph under construction: the variables and factors a compilation has made, the names it gave them (auxiliary variables
 * `y#0[3]` count up per site, so their ids depend on the order of evaluation), the atoms that affine forms are made of, and
 * the operations on them that are about the graph and not about the model: declare a variable, add a factor (checking
 * the operator exists in the registry), turn an affine form into a variable (`materialise`), and drop the dead `apply` factors
 * of an arithmetic chain (`prune`).
 */

import { type Json, OmaiError } from '../../../base/index.ts'
import type { FactorOperator, FactorSpec, VariableSpec } from '../../../message-passing/index.ts'
import { auxId, elementId, factorId } from '../ids.ts'
import type { OperatorTable } from '../operators.ts'
import { type Aff, type Atom, affOf, type Cell, isConstCell } from './_cells.ts'
import { Refusal } from './_refusal.ts'
import type { SiteLayout } from './types.ts'

/** The factor operator a registry name maps to: the model operator's own `factor`, else its name. */
export function factorName(table: OperatorTable, op: string): string {
  return table[op]?.factor ?? op
}

/** The variables, factors and names of a graph being built, against a registry of factor operators. */
export class GraphBuilder {
  readonly variables: VariableSpec[] = []
  readonly factors: FactorSpec[] = []
  readonly auxiliary: string[] = []
  /** The scalar random values the affine forms are made of, by key (see `atomKey`). */
  readonly atoms: Map<string, Atom> = new Map()
  private readonly auxCount = new Map<string, number>()
  private readonly components = new Map<string, string>()
  private readonly names = new Set<string>()

  constructor(
    readonly ops: Readonly<Record<string, FactorOperator>>,
    private readonly modelOps: OperatorTable,
  ) {}

  need(name: string): FactorOperator {
    const op = this.ops[name]
    if (op === undefined) throw new Refusal('operator', `the message-passing registry has no factor operator '${name}'`)
    return op
  }

  addVariable(v: VariableSpec): string {
    if (this.names.has(v.name)) throw new OmaiError('value', `factor graph: the variable '${v.name}' is made twice`)
    this.names.add(v.name)
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

  addFactor(f: FactorSpec, variable: string): void {
    this.need(f.operator)
    this.factors.push({ ...f, id: factorId(variable) })
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
    const name = auxId(site, k, index)
    this.auxiliary.push(name)
    return name
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
  component(vector: string, index: number, size: number): string {
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

/** One branch of a `gate` factor: the factor operator of the state, its arguments and parameters (`{}` for a state with no factor). */
export type GateBranch = Record<string, Json>

/**
 * A `gate` factor under assembly: the selector (a discrete or Bernoulli variable), the variables the branches share, and one
 * branch per state of the selector. The factor's arguments are the selector and then `variables`; a branch refers to a variable
 * by its position in `variables` (`slot`), so position 0 is the first one, usually the output the gate defines.
 */
export class GateBuilder {
  readonly variables: string[]
  readonly branches: GateBranch[] = []

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
  add(branch: GateBranch): void {
    this.branches.push(branch)
  }

  /** The `gate` factor: arguments the selector and the variables, one branch per state. */
  factor(): FactorSpec {
    return {
      operator: 'gate',
      args: [this.selector, ...this.variables],
      params: { arity: this.variables.length, branches: this.branches },
    }
  }
}

/**
 * The gate of a discrete variable `name` whose state is chosen by `selector` among Dirichlet rows `rowVars`: branch `j` is the
 * conjugate `discrete` factor of `name` on row `j` (a transition matrix to learn, or the words of a topic model).
 */
export function discreteRowGate(selector: string, name: string, rowVars: readonly string[]): FactorSpec {
  const gate = new GateBuilder(selector, [name, ...rowVars])
  for (const j of rowVars.keys()) gate.add({ operator: 'discrete', args: [0, 1 + j], params: {} })
  return gate.factor()
}
