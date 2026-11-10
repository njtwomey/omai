/**
 * A model spec drawn as a graphical model in plate notation: `toStructure` turns a spec (or a bound model, which adds the
 * plate sizes) into the library's `Structure` record, so the existing structure views draw it with no new view and the
 * same record can be handed to an overlay. A node per data, latent, observed, derived, generated, factor and param
 * node with its role (observed shaded, derived as a double circle, factors as squares, params as dots, data as
 * constants), its distribution or expression written beside it (`~ Normal(0, 2)`, `= mu + tau * z`), directed
 * dependency edges, and the named dimensions as nested plates labelled with their sizes. Ids follow `ids.ts`: a site's
 * name is its node's id, so a posterior marginal, a tape event or a message names the node it belongs to and a view can
 * paint live inference state on the picture; `view: 'factor'` draws the bipartite factor graph with `factor:<site>`
 * nodes, the picture message passing runs on.
 *
 * @index function toStructure: draw a ModelSpec as plate notation (or its factor graph) with ids that match tapes.
 */

import { type Json, OmaiError } from '../../base/index.ts'
import {
  type NodeRole,
  type Structure,
  type StructureEdge,
  type StructureGroup,
  type StructureNode,
  structure,
} from '../../scene/index.ts'
import { analyseSpec } from './analysis.ts'
import type { ModelLike } from './bind.ts'
import { canonicalDims, factorId, nodeId, plateId } from './ids.ts'
import type { DataRole, DistSpec, ExprSpec, ModelNode, ModelSpec } from './spec.ts'

/** Options of {@link toStructure}. */
export interface StructureOptions {
  /**
   * `plate` (default): the Bayesian network in plate notation; `factor`: the factor graph, one `factor:<site>` square
   * per distribution, operator or term, joining the variables it relates.
   */
  readonly view?: 'plate' | 'factor'
  /**
   * Which data nodes to draw: `covariates` (default: all but the ones that only supply an observed node's value, which
   * the shaded observed node already stands for), `all`, or `none`.
   */
  readonly data?: 'covariates' | 'all' | 'none'
  /** Write Greek names and subscripts as TeX (`tau_1` as `$\tau_{1}$`), which the views typeset (default false). */
  readonly tex?: boolean
  /** Write distributions and expressions beside the nodes (default true). */
  readonly details?: boolean
}

const GREEK = new Set([
  'alpha',
  'beta',
  'gamma',
  'delta',
  'epsilon',
  'zeta',
  'eta',
  'theta',
  'iota',
  'kappa',
  'lambda',
  'mu',
  'nu',
  'xi',
  'pi',
  'rho',
  'sigma',
  'tau',
  'phi',
  'chi',
  'psi',
  'omega',
])

/**
 * A variable name as TeX: a Greek name becomes its letter, and a trailing `_sub` or digits a subscript.
 *
 * @param name - The name.
 * @returns A `$...$` string, or the name unchanged when there is nothing to typeset.
 * @example
 * ```ts
 * import { texName } from 'omai-engine/probprog/model'
 * [texName('tau'), texName('theta_j'), texName('y')] // => ['$\\tau$', '$\\theta_{j}$', 'y']
 * ```
 */
export function texName(name: string): string {
  const m = /^([A-Za-z]+?)(?:_([A-Za-z0-9]+)|([0-9]+))?$/.exec(name)
  if (m === null) return name
  const base = m[1] as string
  const sub = m[2] ?? m[3]
  const greek = GREEK.has(base.toLowerCase()) && base === base.toLowerCase()
  if (!greek && sub === undefined) return name
  return `$${greek ? `\\${base}` : base}${sub === undefined ? '' : `_{${sub}}`}$`
}

const INFIX: Readonly<Record<string, string>> = {
  add: '+',
  sub: '-',
  mul: '*',
  div: '/',
  pow: '^',
  gt: '>',
  lt: '<',
  eq: '=',
}

/**
 * An expression as text: infix arithmetic, function calls, `a[idx]` for a gather, `sum_dim(a)` for a reduction.
 *
 * @param e - The expression.
 * @returns The text.
 * @example
 * ```ts
 * import { exprText } from 'omai-engine/probprog/model'
 * exprText({ kind: 'op', op: 'add', args: [{ kind: 'ref', name: 'mu' }, { kind: 'const', value: 1 }] }) // => 'mu + 1'
 * ```
 */
export function exprText(e: ExprSpec): string {
  const inner = (x: ExprSpec): string => (x.kind === 'op' && x.op in INFIX ? `(${exprText(x)})` : exprText(x))
  switch (e.kind) {
    case 'const':
      return JSON.stringify(e.value)
    case 'ref':
      return e.name
    case 'custom':
      return `${e.name}(${e.args.map(exprText).join(', ')})`
    case 'op': {
      const sym = INFIX[e.op]
      const [a, b] = e.args
      if (sym !== undefined && a !== undefined && b !== undefined) return `${inner(a)} ${sym} ${inner(b)}`
      if (e.op === 'neg' && a !== undefined) return `-${inner(a)}`
      if (e.op === 'matmul' && a !== undefined && b !== undefined) return `${inner(a)} @ ${inner(b)}`
      if (e.op === 'at' && a !== undefined && b !== undefined) return `${inner(a)}[${exprText(b)}]`
      const dim = typeof e.params?.['dim'] === 'string' ? `_${e.params['dim']}` : ''
      return `${e.op}${dim}(${e.args.map(exprText).join(', ')})`
    }
  }
}

/**
 * A distribution as text, `Normal(0, 2)`, its parameters in the family's order.
 *
 * @param d - The distribution spec.
 * @returns The text.
 * @example
 * ```ts
 * import { distText } from 'omai-engine/probprog/model'
 * distText({ family: 'normal', params: { loc: { kind: 'const', value: 0 }, scale: { kind: 'const', value: 2 } } }) // => 'Normal(0, 2)'
 * ```
 */
export function distText(d: DistSpec): string {
  const name = d.family.charAt(0).toUpperCase() + d.family.slice(1)
  return `${name}(${Object.values(d.params).map(exprText).join(', ')})`
}

const CONSTANT_ROLES: ReadonlySet<DataRole | undefined> = new Set([
  'covariate',
  'index',
  'weights',
  'offset',
  'exposure',
  'constant',
  undefined,
])

const ROLE: Readonly<Record<ModelNode['kind'], NodeRole>> = {
  data: 'constant',
  latent: 'latent',
  observed: 'observed',
  derived: 'deterministic',
  generated: 'latent',
  factor: 'factor',
  param: 'parameter',
}

/**
 * Describe a model spec as plate notation for the structure views. Data nodes are constants (`data:<name>`), latent
 * nodes open circles, observed nodes shaded circles, derived nodes double circles, generated quantities dashed-edge
 * circles marked `generated`, param nodes dots, factor nodes squares; edges are dependencies (`data` kind from data
 * nodes); each dimension is a plate, nested in the spec's dimension order and counted with its size when known (from
 * `range` or a bound model). Latent, observed and generated nodes carry their distribution as `detail`, derived nodes
 * their expression; every node's `data` field holds `{ spec, site, dims, plate }` for overlays. The structure's type is
 * `graphical-model` (or `factor-graph` for `view: 'factor'`), which the renderer already has a visual language for.
 *
 * Ids: see `ids.ts`. A node over `[row, feature]` and one over `[feature, row]` sit in the same plate
 * `plate:row×feature` when `row` is declared first; plates nest by prefix (`plate:row` holds `plate:row×feature`), the
 * form `nestPlates` in `omai-render/structures` reads. Overlapping plates that are not nested (nodes over `[a, b]` and
 * over `[b, c]`) are drawn as two nested chains and therefore repeat `b`.
 *
 * @param model - A `ModelDefinition`, a `BoundModel` (which gives the sizes) or a `ModelSpec`.
 * @param options - The view, which data to show, TeX names and details.
 * @returns The structure.
 * @throws OmaiError with code `value` for an unknown view.
 * @concept structures
 * @example
 * ```ts
 * import { defineModel, toStructure } from 'omai-engine/probprog/model'
 * import { normal } from 'omai-engine/probprog/model/dists'
 * const m = defineModel('m', (b) => {
 *   const y = b.data('y', { dims: ['row'] })
 *   b.observe('y', normal(b.latent('mu', normal(0, 1)), 1), y)
 * })
 * const s = toStructure(m)
 * s.nodes.map((n) => [n.id, n.role, n.group ?? '']) // => [['mu', 'latent', ''], ['y', 'observed', 'plate:row']]
 * s.edges.map((e) => `${e.from}>${e.to}`) // => ['mu>y']
 * ```
 */
export function toStructure(model: ModelLike | ModelSpec, options: StructureOptions = {}): Structure {
  const spec: ModelSpec = 'spec' in model ? model.spec : model
  const view = options.view ?? 'plate'
  if (view !== 'plate' && view !== 'factor') throw new OmaiError('value', `toStructure: unknown view '${String(view)}'`)
  const a = analyseSpec(model)
  const sizeOf = new Map(a.dims.map((d) => [d.name, d.size]))
  const label = (name: string): string => (options.tex === true ? texName(name) : name)
  const groups = new Map<string, StructureGroup>()
  const groupOf = (dims: readonly string[]): string | undefined => {
    const ordered = canonicalDims(spec, dims)
    let parent: string | undefined
    let id: string | undefined
    for (let i = 0; i < ordered.length; i++) {
      const d = ordered[i] as string
      id = plateId(ordered.slice(0, i + 1)) as string
      if (!groups.has(id)) {
        const count = sizeOf.get(d)
        groups.set(id, {
          id,
          label: label(d),
          kind: 'plate',
          ...(parent === undefined ? {} : { parent }),
          count: count ?? d.toUpperCase(),
        })
      }
      parent = id
    }
    return id
  }
  // A data node that only supplies an observed node's value is the observation itself in the picture.
  const onlyValue = (name: string): boolean => {
    const users = spec.nodes
      .filter((n) => n.kind !== 'data')
      .filter((n) => a.edges.some((e) => e.from === `data:${name}` && e.to === n.name))
    return (
      users.length > 0 && users.every((u) => u.kind === 'observed' && u.value.kind === 'ref' && u.value.name === name)
    )
  }
  const shown = (n: ModelNode): boolean => {
    if (n.kind !== 'data') return true
    if (options.data === 'none') return false
    if (options.data === 'all') return true
    return !(n.role === 'response' || onlyValue(n.name)) && CONSTANT_ROLES.has(n.role)
  }
  const details = options.details !== false
  const visible = spec.nodes.filter(shown)
  const ids = new Set(visible.map(nodeId))
  const nodes: StructureNode[] = []
  const edges: StructureEdge[] = []
  const plateOfNode = (n: ModelNode): string | undefined => groupOf(n.dims)
  for (const n of visible) {
    const group = plateOfNode(n)
    const detail = !details
      ? undefined
      : n.kind === 'latent' || n.kind === 'observed' || n.kind === 'generated'
        ? `~ ${distText(n.dist)}`
        : n.kind === 'derived' || n.kind === 'factor'
          ? `= ${exprText(n.expr)}`
          : n.kind === 'data'
            ? (n.support ?? 'data')
            : undefined
    const badges: Record<string, string | number> = {}
    if (n.dims.length > 0) badges['dims'] = `[${n.dims.join(', ')}]`
    if (n.kind === 'data' && n.role !== undefined) badges['role'] = n.role
    if (n.kind === 'generated') badges['generated'] = 'yes'
    if (n.kind === 'latent' && n.constraint !== undefined) badges['constraint'] = n.constraint.kind
    const info: Record<string, Json> = { spec: n.kind, site: n.kind === 'data' ? null : n.name, dims: [...n.dims] }
    if (group !== undefined) info['plate'] = group
    nodes.push({
      id: nodeId(n),
      label: label(n.name),
      role: ROLE[n.kind],
      ...(group === undefined ? {} : { group }),
      ...(detail === undefined ? {} : { detail }),
      ...(Object.keys(badges).length === 0 ? {} : { badges }),
      data: info,
      ...(n.kind === 'generated' ? { look: { dash: [3, 3] } } : {}),
    })
  }
  const parentsOf = (n: ModelNode): string[] =>
    a.nodes.find((x) => x.id === nodeId(n))?.parents.filter((p) => ids.has(p)) ?? []
  if (view === 'plate') {
    for (const e of a.edges)
      if (ids.has(e.from) && ids.has(e.to))
        edges.push({ from: e.from, to: e.to, directed: true, kind: e.from.startsWith('data:') ? 'data' : 'dependency' })
  } else {
    const kept: StructureNode[] = []
    for (const n of visible) {
      const id = nodeId(n)
      const node = nodes.find((x) => x.id === id) as StructureNode
      if (n.kind === 'factor') {
        // A factor term is its own factor node.
        kept.push({ ...node, id: factorId(n.name), label: label(n.name) })
        for (const p of parentsOf(n)) edges.push({ from: p, to: factorId(n.name), directed: false, kind: 'other' })
        continue
      }
      kept.push(node)
      if (n.kind === 'data') continue
      const group = plateOfNode(n)
      kept.push({
        id: factorId(n.name),
        label: '',
        role: 'factor',
        ...(group === undefined ? {} : { group }),
        data: { spec: 'factor', site: n.name, of: n.kind },
      })
      for (const p of parentsOf(n)) edges.push({ from: p, to: factorId(n.name), directed: false, kind: 'other' })
      edges.push({ from: factorId(n.name), to: id, directed: false, kind: 'other' })
    }
    nodes.length = 0
    nodes.push(...kept)
  }
  return structure(view === 'plate' ? 'graphical-model' : 'factor-graph', nodes, edges, {
    groups: [...groups.values()],
    title: spec.name,
    direction: 'down',
  })
}
