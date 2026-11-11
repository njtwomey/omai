/**
 * Factor graphs for the engine: the validation and compilation of a `FactorGraphSpec` (JSON: variables with their
 * families, factors by operator name) against a registry of operators into the indexed form the engine runs on, the
 * ids a run reports by (a variable is its name, a factor `f<i>`), and the `Structure` of the same graph in the form
 * `pgm/structure.ts` draws factor graphs, so a diagram of the model can be coloured by a run of it.
 *
 * @index module Compile and draw a factor graph: variables, factors, edges, ids.
 */

import { OmaiError } from '../../base/index.ts'
import { type Structure, type StructureEdge, type StructureNode, structure } from '../../scene/index.ts'
import { familyOf } from './families-vector.ts'
import {
  type ArgSpec,
  type FactorGraphSpec,
  type FactorOperator,
  type Family,
  type FamilyKind,
  type FamilySpec,
  type Observed,
  type Params,
  type Port,
  takesPower,
  type VariableSpec,
} from './types.ts'

/**
 * The id of a variable: its name.
 *
 * @param name - The variable's name.
 * @returns The id (the name itself).
 * @example
 * ```ts
 * import { variableId } from 'omai-engine/message-passing'
 * variableId('w') // => 'w'
 * ```
 */
export const variableId = (name: string): string => name

/**
 * The default id of the i-th factor of a graph: `f<i>`, the id that `toStructure` of `pgm` gives the factors of a factor graph (a factor with an `id` of its own keeps that).
 *
 * @param index - The factor's index in `FactorGraphSpec.factors`.
 * @returns The id.
 * @example
 * ```ts
 * import { factorId } from 'omai-engine/message-passing'
 * factorId(3) // => 'f3'
 * ```
 */
export const factorId = (index: number): string => `f${index}`

/** A variable of a compiled graph. */
export interface CompiledVariable {
  readonly name: string
  readonly family: Family
  readonly observed: Observed | undefined
  /** The edges that end at the variable (its factors' ports); none when it is observed. */
  readonly edges: readonly number[]
  /** The edge of the factor that defines the variable as a deterministic function of others (`FactorOperator.defines`), if one does. */
  readonly definer: number | undefined
}

/** One port of a compiled factor: the variable it connects to (or none for a constant), its family and its edge. */
export interface CompiledPort {
  readonly variable: number | undefined
  readonly family: Family
  readonly value: Observed | undefined
  /** The edge, or `-1` for an observed variable or a constant (nothing is sent there). */
  readonly edge: number
}

/** A factor of a compiled graph. */
export interface CompiledFactor {
  readonly id: string
  readonly index: number
  readonly operator: FactorOperator
  readonly params: Params
  readonly power: number
  readonly ports: readonly CompiledPort[]
  /** The message each port starts with (`null`: flat), from `FactorSpec.start`. */
  readonly start: readonly (readonly number[] | null)[]
}

/** A directed-capable connection of a factor's port to a variable: the unit messages travel on. */
export interface CompiledEdge {
  readonly factor: number
  readonly port: number
  readonly variable: number
}

/** A factor graph compiled against a registry of operators. */
export interface CompiledGraph {
  readonly spec: FactorGraphSpec
  readonly variables: readonly CompiledVariable[]
  readonly factors: readonly CompiledFactor[]
  readonly edges: readonly CompiledEdge[]
  /** The index of a variable by name. */
  readonly index: ReadonlyMap<string, number>
  /** Whether some factor defines a variable as a deterministic function of others (`FactorOperator.defines`). */
  readonly hasDefined: boolean
}

const portsOf = (op: FactorOperator, params: Params): readonly Port[] =>
  typeof op.ports === 'function' ? op.ports(params) : op.ports

/** The family a constant argument is read in: the first kind its port accepts (sizes of vector kinds are not needed for a constant). */
function constantFamily(port: Port, value?: Observed): Family {
  const kind: FamilyKind = port.accepts[0] ?? 'gaussian'
  // a constant vector has the size of its value
  const size = kind === 'gaussian-mv' && Array.isArray(value) ? value.length : 2
  const spec: FamilySpec =
    kind === 'discrete' || kind === 'dirichlet' || kind === 'gaussian-mv' ? { kind, size } : { kind }
  return familyOf(spec)
}

/**
 * Validate a factor graph and compile it against a registry of operators: every variable's family is built, every
 * factor's operator found, its arguments checked against its ports (the number, the families a port accepts, whether
 * a port may be observed or a constant, the operator's own `check`), and the edges numbered.
 *
 * @param spec - The graph.
 * @param operators - The operators by name (`defaultOperators()` and any of your own).
 * @returns The compiled graph.
 * @throws OmaiError with code `value` for a repeated name, an unknown operator or variable, a port that does not take that family or a constant, or a factor with nothing to send to.
 * @concept factor-graphs
 * @index function Validate and compile a factor graph against a registry of operators.
 * @example
 * ```ts
 * import { compileGraph, defaultOperators } from 'omai-engine/message-passing'
 * const g = compileGraph(
 *   { variables: [{ name: 'x', family: { kind: 'gaussian' } }], factors: [{ operator: 'prior', args: ['x'], params: { eta: [0, 1] } }] },
 *   defaultOperators(),
 * )
 * [g.variables.length, g.edges.length] // => [1, 1]
 * ```
 */
export function compileGraph(
  spec: FactorGraphSpec,
  operators: Readonly<Record<string, FactorOperator>>,
): CompiledGraph {
  const index = new Map<string, number>()
  const edgesOfVariable: number[][] = []
  const variables: VariableSpec[] = []
  for (const v of spec.variables) {
    if (index.has(v.name)) throw new OmaiError('value', `factor graph: the variable ${v.name} is declared twice`)
    index.set(v.name, variables.length)
    variables.push(v)
    edgesOfVariable.push([])
  }
  const families = variables.map((v) => familyOf(v.family))
  const edges: { factor: number; port: number; variable: number }[] = []
  const ids = new Set<string>()
  const factors: CompiledFactor[] = spec.factors.map((f, fi) => {
    const id = f.id ?? factorId(fi)
    if (ids.has(id)) throw new OmaiError('value', `factor graph: the factor id ${id} is used twice`)
    ids.add(id)
    const op = operators[f.operator]
    if (op === undefined) throw new OmaiError('value', `factor graph: ${id} uses the unknown operator ${f.operator}`)
    const params = f.params ?? {}
    const ports = portsOf(op, params)
    if (f.args.length !== ports.length) {
      throw new OmaiError(
        'value',
        `factor graph: ${id} (${op.name}) takes ${ports.length} arguments, got ${f.args.length}`,
      )
    }
    const power = f.power ?? 1
    if (power !== 1 && !takesPower(op, params))
      throw new OmaiError('value', `factor graph: ${id} (${op.name}) does not support a power`)
    const compiled: CompiledPort[] = f.args.map((a: ArgSpec, pi): CompiledPort => {
      const port = ports[pi] as Port
      if (typeof a === 'string') {
        const vi = index.get(a)
        if (vi === undefined) throw new OmaiError('value', `factor graph: ${id} refers to the unknown variable ${a}`)
        const family = families[vi] as Family
        if (!port.accepts.includes(family.kind)) {
          throw new OmaiError(
            'value',
            `factor graph: port ${port.name} of ${id} (${op.name}) does not take a ${family.kind} variable`,
          )
        }
        const observed = variables[vi]?.observed
        if (observed !== undefined) {
          if (port.constant !== true)
            throw new OmaiError(
              'value',
              `factor graph: port ${port.name} of ${id} (${op.name}) cannot take the observed variable ${a}`,
            )
          return { variable: vi, family, value: observed, edge: -1 }
        }
        if (port.observed === true)
          throw new OmaiError(
            'value',
            `factor graph: port ${port.name} of ${id} (${op.name}) must be observed, ${a} is not`,
          )
        const edge = edges.length
        edges.push({ factor: fi, port: pi, variable: vi })
        edgesOfVariable[vi]?.push(edge)
        return { variable: vi, family, value: undefined, edge }
      }
      if (port.constant !== true)
        throw new OmaiError('value', `factor graph: port ${port.name} of ${id} (${op.name}) cannot take a constant`)
      return { variable: undefined, family: constantFamily(port, a.const), value: a.const, edge: -1 }
    })
    if (compiled.every((p) => p.edge < 0)) {
      throw new OmaiError('value', `factor graph: ${id} (${op.name}) has only observed arguments, so it sends nothing`)
    }
    const reason = op.check?.(
      params,
      compiled.map((p) => p.family),
    )
    if (reason !== undefined) throw new OmaiError('value', `factor graph: ${id}: ${reason}`)
    const start = f.start ?? compiled.map(() => null)
    if (start.length !== compiled.length)
      throw new OmaiError(
        'value',
        `factor graph: ${id} has ${start.length} start messages for ${compiled.length} arguments`,
      )
    for (const [pi, m] of start.entries()) {
      const port = compiled[pi] as CompiledPort
      if (m !== null && port.edge >= 0 && m.length !== port.family.size)
        throw new OmaiError(
          'value',
          `factor graph: the start message of port ${pi} of ${id} has ${m.length} numbers, not ${port.family.size}`,
        )
    }
    return { id, index: fi, operator: op, params, power, ports: compiled, start }
  })
  const definer = new Map<number, number>()
  for (const f of factors) {
    const k = f.operator.defines
    const e = k === undefined ? undefined : f.ports[k]?.edge
    if (e === undefined || e < 0) continue
    const v = (edges[e] as { variable: number }).variable
    if (definer.has(v))
      throw new OmaiError('value', `factor graph: the variable ${variables[v]?.name} is defined by two factors`)
    definer.set(v, e)
  }
  return {
    spec,
    variables: variables.map((v, i) => ({
      name: v.name,
      family: families[i] as Family,
      observed: v.observed,
      edges: edgesOfVariable[i] ?? [],
      definer: definer.get(i),
    })),
    factors,
    edges,
    index,
    hasDefined: definer.size > 0,
  }
}

/**
 * The structure of a factor graph, in the form `toStructure` of `pgm` gives a factor graph: a node for every variable
 * (id its name, role `latent` or `observed`) and every factor (id its `id` or `f<i>`, role `factor`, its operator as the detail),
 * an undirected edge between a factor and each variable it touches. Drawn, it takes the colours and annotations of a
 * run keyed by those ids.
 *
 * @param spec - The graph.
 * @returns The structure (type `factor-graph`).
 * @concept factor-graphs
 * @example
 * ```ts
 * import { graphStructure } from 'omai-engine/message-passing'
 * const s = graphStructure({ variables: [{ name: 'x', family: { kind: 'gaussian' } }], factors: [{ operator: 'prior', args: ['x'], params: { eta: [0, 1] } }] })
 * s.nodes.map((n) => n.id) // => ['x', 'f0']
 * ```
 */
export function graphStructure(spec: FactorGraphSpec): Structure {
  const variableNodes: StructureNode[] = spec.variables.map((v) => ({
    id: variableId(v.name),
    label: v.name,
    role: v.observed === undefined ? 'latent' : 'observed',
    detail: v.family.kind,
  }))
  const factorNodes: StructureNode[] = spec.factors.map((f, i) => ({
    id: f.id ?? factorId(i),
    label: f.name ?? f.operator,
    role: 'factor',
    detail: f.operator,
  }))
  const edges: StructureEdge[] = spec.factors.flatMap((f, i) =>
    f.args.flatMap((a): StructureEdge[] =>
      typeof a === 'string'
        ? [{ from: f.id ?? factorId(i), to: variableId(a), directed: false, kind: 'dependency' }]
        : [],
    ),
  )
  return structure('factor-graph', [...variableNodes, ...factorNodes], edges, { title: 'Factor graph' })
}
