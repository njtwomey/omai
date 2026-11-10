/**
 * Checking a spec from the outside: the shape of an expression from its parts (so a spec read from JSON, or rewritten by
 * `include` and `condition`, is verified the way the builder verifies one as it goes), and `validateSpec`, which walks a
 * spec's nodes in order and checks the schema tags, the names, the references, the operators and families, the
 * parameters of every distribution and the declared dimensions of every node against what its expressions imply.
 *
 * @index function validateSpec: check a ModelSpec's names, references, operators, families and dimensions.
 */

import { OmaiError } from '../../base/index.ts'
import { distributionFamilies, type FamilyDef } from './dists/families.ts'
import { defaultOperators, joinShapes, type NamedShape, type OperatorTable, operatorShape } from './operators.ts'
import {
  type ExprSpec,
  MODEL_SPEC_SCHEMA,
  type ModelNode,
  type ModelSpec,
  OPERATOR_REGISTRY,
  prevName,
} from './spec.ts'

/** What `validateSpec` checks against: the operator registry and the families (the defaults when left out). */
export interface CheckOptions {
  readonly operators?: OperatorTable
  readonly families?: Readonly<Record<string, FamilyDef>>
}

const fail = (message: string, details: Readonly<Record<string, unknown>> = {}): never => {
  throw new OmaiError('value', message, details)
}

function nestedRank(v: unknown): number {
  let r = 0
  let x = v
  while (Array.isArray(x)) {
    r += 1
    x = x[0] ?? 0
  }
  return r
}

/**
 * The dimensions and event rank of an expression, from the shapes of the nodes it refers to.
 *
 * @param expr - The expression.
 * @param scope - The shape of every referable node so far, by name.
 * @param table - The operator registry (default the shipped one).
 * @returns The expression's shape.
 * @throws OmaiError with code `value` for an unknown reference or operator, and `shape` for dimensions that do not fit.
 * @example
 * ```ts
 * import { exprShape } from 'omai-engine/probprog/model'
 * exprShape({ kind: 'ref', name: 'x' }, new Map([['x', { dims: ['row'], eventRank: 0 }]])).dims // => ['row']
 * ```
 */
export function exprShape(
  expr: ExprSpec,
  scope: ReadonlyMap<string, NamedShape>,
  table: OperatorTable = defaultOperators(),
): NamedShape {
  switch (expr.kind) {
    case 'const': {
      const dims = expr.dims ?? []
      const rank = nestedRank(expr.value)
      if (rank < dims.length)
        throw new OmaiError('shape', `a constant of rank ${rank} has ${dims.length} named dimensions`)
      return { dims, eventRank: rank - dims.length }
    }
    case 'ref': {
      const s = scope.get(expr.name)
      if (s === undefined) return fail(`'${expr.name}' is not a node defined before its use`, { name: expr.name })
      return s
    }
    case 'op':
      return operatorShape(
        table,
        expr.op,
        expr.args.map((a) => exprShape(a, scope, table)),
        expr.params ?? {},
      )
    case 'custom':
      for (const a of expr.args) exprShape(a, scope, table)
      return { dims: expr.dims, eventRank: expr.eventRank }
  }
}

function sameSet(a: readonly string[], b: readonly string[]): boolean {
  return a.length === b.length && a.every((d) => b.includes(d))
}

/** The shape a node's value has, as seen by the nodes after it (undefined for sites nothing can refer to). */
function nodeShape(node: ModelNode): NamedShape | undefined {
  switch (node.kind) {
    case 'observed':
    case 'factor':
      return undefined
    default:
      return { dims: node.dims, eventRank: node.eventRank }
  }
}

/**
 * Check a spec end to end. A spec the builder made always passes; the check is for specs read from JSON, composed by
 * `include`, or edited by hand.
 *
 * @param spec - The spec.
 * @param options - The registry and families to check against.
 * @throws OmaiError with code `value` (or `shape`) naming the node and what is wrong: a wrong schema tag, a repeated
 *   name, a reference to a node not yet defined, an unknown operator or family, a distribution parameter that does not
 *   belong to its family, an undeclared dimension, a node whose declared dimensions do not contain what its expressions
 *   carry, a `custom` function missing from `spec.customs`.
 * @concept model-spec
 * @example
 * ```ts
 * import { defineModel, validateSpec } from 'omai-engine/probprog/model'
 * import { normal } from 'omai-engine/probprog/model/dists'
 * const m = defineModel('m', (b) => { b.latent('x', normal(0, 1)) })
 * validateSpec(m.spec) // => undefined
 * ```
 */
export function validateSpec(spec: ModelSpec, options: CheckOptions = {}): void {
  const table = options.operators ?? defaultOperators()
  const families = options.families ?? distributionFamilies()
  if (spec.schema !== MODEL_SPEC_SCHEMA)
    fail(`unsupported model schema '${String(spec.schema)}' (expected ${MODEL_SPEC_SCHEMA})`)
  if (spec.registry !== OPERATOR_REGISTRY)
    fail(`the spec was written for operator registry '${String(spec.registry)}', this is ${OPERATOR_REGISTRY}`)
  const dimNames = new Set<string>()
  for (const d of spec.dims) {
    if (dimNames.has(d.name)) fail(`the dimension '${d.name}' is declared twice`)
    dimNames.add(d.name)
    if (d.size !== undefined && (!Number.isInteger(d.size) || d.size < 0))
      fail(`dimension '${d.name}': the size must be a non-negative integer, got ${d.size}`)
    if (d.labels !== undefined && d.size !== undefined && d.labels.length !== d.size)
      fail(`dimension '${d.name}': ${d.labels.length} labels for size ${d.size}`)
  }
  const scope = new Map<string, NamedShape>()
  const sites = new Set<string>()
  const customs = new Set(spec.customs)
  const checkCustoms = (e: ExprSpec, where: string): void => {
    if (e.kind === 'custom') {
      if (!customs.has(e.name)) fail(`${where}: the custom function '${e.name}' is not listed in spec.customs`)
      for (const a of e.args) checkCustoms(a, where)
    } else if (e.kind === 'op') for (const a of e.args) checkCustoms(a, where)
  }
  for (const node of spec.nodes) {
    const where = `node '${node.name}'`
    if (node.name === '') fail('a node needs a name')
    for (const d of node.dims) if (!dimNames.has(d)) fail(`${where}: the dimension '${d}' is not declared`)
    if (new Set(node.dims).size !== node.dims.length) fail(`${where}: repeats a dimension [${node.dims.join(', ')}]`)
    if (node.kind !== 'data') {
      if (sites.has(node.name)) fail(`${where}: the site name is used twice`)
      sites.add(node.name)
    }
    const distOf = (
      dist: {
        readonly family: string
        readonly params: Readonly<Record<string, ExprSpec>>
      },
      at: ReadonlyMap<string, NamedShape> = scope,
    ): NamedShape => {
      const f = families[dist.family]
      if (f === undefined) return fail(`${where}: unknown distribution family '${dist.family}'`)
      const given = Object.keys(dist.params)
      for (const p of given) if (!f.params.includes(p)) fail(`${where}: '${p}' is not a parameter of ${f.name}`)
      for (const p of f.params) {
        if (given.includes(p)) continue
        if (f.defaults[p] !== undefined || f.oneOf.some((g) => g.includes(p))) continue
        fail(`${where}: the parameter '${p}' of ${f.name} is missing`)
      }
      for (const g of f.oneOf)
        if (g.filter((p) => given.includes(p)).length !== 1) fail(`${where}: give exactly one of ${g.join(', ')}`)
      const shapes: NamedShape[] = []
      for (const e of Object.values(dist.params)) {
        checkCustoms(e, where)
        shapes.push(exprShape(e, at, table))
      }
      return { dims: joinShapes(shapes).dims, eventRank: f.eventRank }
    }
    const within = (inner: readonly string[]): void => {
      for (const d of inner)
        if (!node.dims.includes(d))
          fail(`${where}: declares dims [${node.dims.join(', ')}] but its expressions carry '${d}'`)
    }
    switch (node.kind) {
      case 'data':
        if (scope.has(node.name)) fail(`${where}: the name is used twice`)
        break
      case 'latent':
      case 'generated': {
        if (scope.has(node.name)) fail(`${where}: the name is used twice`)
        const s = distOf(node.dist)
        within(s.dims)
        if (node.eventRank !== s.eventRank)
          fail(`${where}: event rank ${node.eventRank}, the family has ${s.eventRank}`)
        break
      }
      case 'observed': {
        const s = distOf(node.dist)
        checkCustoms(node.value, where)
        const v = exprShape(node.value, scope, table)
        within(s.dims)
        within(v.dims)
        if (v.eventRank !== s.eventRank)
          fail(`${where}: the value has event rank ${v.eventRank}, the distribution ${s.eventRank}`)
        break
      }
      case 'derived': {
        if (scope.has(node.name)) fail(`${where}: the name is used twice`)
        checkCustoms(node.expr, where)
        const s = exprShape(node.expr, scope, table)
        if (!sameSet(s.dims, node.dims) || s.eventRank !== node.eventRank)
          fail(`${where}: declares dims [${node.dims.join(', ')}] but its expression has [${s.dims.join(', ')}]`)
        break
      }
      case 'factor':
        checkCustoms(node.expr, where)
        exprShape(node.expr, scope, table)
        break
      case 'param':
        if (scope.has(node.name)) fail(`${where}: the name is used twice`)
        break
      case 'scan': {
        if (scope.has(node.name)) fail(`${where}: the name is used twice`)
        const over = node.over
        if (!dimNames.has(over)) fail(`${where}: the scanned dimension '${over}' is not declared`)
        if (node.dims.length !== 1 || node.dims[0] !== over)
          fail(`${where}: a scan's dims are [${over}], got [${node.dims.join(', ')}]`)
        // What the expressions inside see: every node over the scanned dimension read at the current step.
        const sliced = new Map<string, NamedShape>()
        for (const [name, shape] of scope)
          sliced.set(name, { dims: shape.dims.filter((d) => d !== over), eventRank: shape.eventRank })
        const withState = new Map(sliced).set(node.name, { dims: [], eventRank: node.eventRank })
        const inStep = new Map(sliced).set(prevName(node.name), { dims: [], eventRank: node.eventRank })
        const inEmit = new Map(withState)
        const scalar = (s: NamedShape, what: string, rank: number): void => {
          if (s.dims.length > 0)
            fail(
              `${where}: ${what} has dimensions [${s.dims.join(', ')}] left inside the scan (one scan is one series)`,
            )
          if (s.eventRank !== rank) fail(`${where}: ${what} has event rank ${s.eventRank}, expected ${rank}`)
        }
        scalar(distOf(node.init, sliced), 'the initial distribution', node.eventRank)
        scalar(distOf(node.step, inStep), 'the step distribution', node.eventRank)
        const names = new Set<string>()
        for (const m of node.emissions) {
          const at = `${where} emission '${m.name}'`
          if (m.name === '' || m.name === node.name || names.has(m.name) || sites.has(m.name))
            fail(`${at}: the site name is used twice`)
          names.add(m.name)
          sites.add(m.name)
          const s = distOf(m.dist, inEmit)
          checkCustoms(m.value, at)
          const v = exprShape(m.value, inEmit, table)
          scalar(s, 'its distribution', s.eventRank)
          scalar(v, 'its value', s.eventRank)
        }
        break
      }
    }
    const shape = nodeShape(node)
    if (shape !== undefined) scope.set(node.name, shape)
  }
}
