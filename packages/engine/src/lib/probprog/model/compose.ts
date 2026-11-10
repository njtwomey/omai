/**
 * Models from models, as pure transformations of specs: `conditionSpec` turns latent nodes into observed ones with
 * given values (PyMC's `observe`), `doSpec` turns them into constants that cut their incoming dependencies (Pearl's
 * `do`, PyMC's `do`), and `substitute` rewrites references in an expression, which `include` in the builder uses to
 * instantiate a sub-model under a prefix with some of its data nodes bound to expressions of the outer model. Each
 * returns a new spec that passes `validateSpec`; none touches the original.
 *
 * @index function conditionSpec and doSpec: latent nodes turned into observations or interventions, as new specs.
 */

import { type Json, OmaiError } from '../../base/index.ts'
import type { ExprSpec, ModelNode, ModelSpec } from './spec.ts'

/** Values for nodes, by node name: numbers or nested arrays shaped `[dims..., event...]`. */
export type SiteConstants = Readonly<Record<string, Json>>

/**
 * An expression with the references in `map` replaced by their expressions.
 *
 * @param expr - The expression.
 * @param map - Replacement expressions by referenced name.
 * @returns The rewritten expression (the same object where nothing changed).
 * @example
 * ```ts
 * import { substitute } from 'omai-engine/probprog/model'
 * substitute({ kind: 'ref', name: 'a' }, new Map([['a', { kind: 'const', value: 1 }]])) // => { kind: 'const', value: 1 }
 * ```
 */
export function substitute(expr: ExprSpec, map: ReadonlyMap<string, ExprSpec>): ExprSpec {
  switch (expr.kind) {
    case 'ref':
      return map.get(expr.name) ?? expr
    case 'const':
      return expr
    case 'op':
      return { ...expr, args: expr.args.map((a) => substitute(a, map)) }
    case 'custom':
      return { ...expr, args: expr.args.map((a) => substitute(a, map)) }
  }
}

/** The node with `map` applied to every expression in it. */
export function substituteNode(node: ModelNode, map: ReadonlyMap<string, ExprSpec>): ModelNode {
  const dist = <T extends { readonly params: { readonly [k: string]: ExprSpec } }>(d: T): T => ({
    ...d,
    params: Object.fromEntries(Object.entries(d.params).map(([k, e]) => [k, substitute(e, map)])),
  })
  switch (node.kind) {
    case 'latent':
    case 'generated':
      return { ...node, dist: dist(node.dist) }
    case 'observed':
      return { ...node, dist: dist(node.dist), value: substitute(node.value, map) }
    case 'derived':
    case 'factor':
      return { ...node, expr: substitute(node.expr, map) }
    default:
      return node
  }
}

function rankOf(v: Json): number {
  let r = 0
  let x: Json = v
  while (Array.isArray(x)) {
    r += 1
    x = x[0] ?? 0
  }
  return r
}

function constantFor(node: ModelNode, value: Json, who: string): ExprSpec {
  const rank = rankOf(value)
  if (rank !== node.dims.length + node.eventRank)
    throw new OmaiError(
      'shape',
      `${who}: '${node.name}' has ${node.dims.length} named and ${node.eventRank} event axes, the value has ${rank}`,
      { name: node.name },
    )
  return node.dims.length === 0 ? { kind: 'const', value } : { kind: 'const', value, dims: node.dims }
}

function rewrite(
  spec: ModelSpec,
  values: SiteConstants,
  who: string,
  replace: (node: ModelNode & { readonly kind: 'latent' }, constant: ExprSpec) => ModelNode,
  inline: boolean,
): ModelSpec {
  const names = Object.keys(values)
  for (const n of names) {
    const node = spec.nodes.find((x) => x.name === n && x.kind !== 'data')
    if (node === undefined) throw new OmaiError('value', `${who}: the model has no site named '${n}'`, { name: n })
    if (node.kind !== 'latent')
      throw new OmaiError('value', `${who}: '${n}' is a ${node.kind} node; only latents can be set`, { name: n })
  }
  const inlined = new Map<string, ExprSpec>()
  const nodes: ModelNode[] = []
  for (const original of spec.nodes) {
    const node = substituteNode(original, inlined)
    const value = values[node.name]
    if (value !== undefined && node.kind === 'latent') {
      const constant = constantFor(node, value, who)
      if (inline) inlined.set(node.name, constant)
      nodes.push(replace(node, constant))
    } else nodes.push(node)
  }
  return { ...spec, nodes }
}

/**
 * The spec with latent nodes conditioned on values: each becomes an observed node with the same distribution and the
 * value as its (constant) data. References to it in later nodes read the value. Data bound to the new constants need
 * not be supplied, and their named dimensions take their sizes from the arrays.
 *
 * @param spec - The spec.
 * @param values - Values by latent node name, shaped `[dims..., event...]`.
 * @returns The conditioned spec.
 * @throws OmaiError with code `value` for a name that is not a latent node and `shape` for a value of the wrong rank.
 * @concept model-spec
 * @example
 * ```ts
 * import { conditionSpec, defineModel } from 'omai-engine/probprog/model'
 * import { normal } from 'omai-engine/probprog/model/dists'
 * const m = defineModel('m', (b) => { const mu = b.latent('mu', normal(0, 1)); b.latent('x', normal(mu, 1)) })
 * conditionSpec(m.spec, { mu: 0.5 }).nodes.map((n) => n.kind) // => ['observed', 'latent']
 * ```
 */
export function conditionSpec(spec: ModelSpec, values: SiteConstants): ModelSpec {
  return rewrite(
    spec,
    values,
    'condition',
    (node, constant) => ({
      kind: 'observed',
      name: node.name,
      dims: node.dims,
      eventRank: node.eventRank,
      dist: node.dist,
      value: constant,
      ...(node.doc === undefined ? {} : { doc: node.doc }),
    }),
    true,
  )
}

/**
 * The spec with latent nodes intervened on (Pearl's `do`): each becomes a recorded derived constant, so its incoming
 * dependencies are cut and its descendants see the value; unlike `conditionSpec` the node adds nothing to the density.
 *
 * @param spec - The spec.
 * @param values - Values by latent node name, shaped `[dims..., event...]`.
 * @returns The intervened spec.
 * @throws OmaiError with code `value` for a name that is not a latent node and `shape` for a value of the wrong rank.
 * @concept model-spec
 * @example
 * ```ts
 * import { defineModel, doSpec } from 'omai-engine/probprog/model'
 * import { normal } from 'omai-engine/probprog/model/dists'
 * const m = defineModel('m', (b) => { const mu = b.latent('mu', normal(0, 1)); b.latent('x', normal(mu, 1)) })
 * doSpec(m.spec, { mu: 0.5 }).nodes.map((n) => n.kind) // => ['derived', 'latent']
 * ```
 */
export function doSpec(spec: ModelSpec, values: SiteConstants): ModelSpec {
  return rewrite(
    spec,
    values,
    'do',
    (node, constant) => ({
      kind: 'derived',
      name: node.name,
      dims: node.dims,
      eventRank: node.eventRank,
      expr: constant,
      record: true,
      ...(node.doc === undefined ? {} : { doc: node.doc }),
    }),
    false,
  )
}
