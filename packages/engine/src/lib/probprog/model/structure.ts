/**
 * A model spec drawn as a graphical model: `toStructure` turns a spec (or a bound model, which adds the plate sizes) into
 * the library's `Structure` record, so the existing views draw it with no new view: a node per data, latent, observed,
 * derived, generated, factor and param node (latent and observed nodes carry their sites' names as ids), a directed edge
 * per dependency, and the named dimensions as plate boxes, nested in the order a node declares them. It is drawn from the
 * spec, so it sees the whole model and needs no run.
 *
 * @index function toStructure: draw a ModelSpec as the library's Structure record, dims as plates.
 */

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
import type { ModelNode, ModelSpec } from './spec.ts'

const ROLE: Readonly<Record<ModelNode['kind'], NodeRole>> = {
  data: 'input',
  latent: 'latent',
  observed: 'observed',
  derived: 'deterministic',
  generated: 'output',
  factor: 'factor',
  param: 'parameter',
}

/**
 * Describe a model spec as a graphical model for the structure views: nodes by kind (data as inputs, latents, observed,
 * derived as deterministic, generated as outputs, factors, params), dependency edges, and the dimensions as nested
 * plates labelled with their sizes when known (from `range` or a bound model).
 *
 * @param model - A `ModelDefinition`, a `BoundModel` (which gives the sizes) or a `ModelSpec`.
 * @returns The structure.
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
 * s.nodes.map((n) => [n.id, n.role, n.group ?? '']) // => [['data:y', 'input', 'plate:row'], ['mu', 'latent', ''], ['y', 'observed', 'plate:row']]
 * s.edges.map((e) => `${e.from}>${e.to}`) // => ['mu>y', 'data:y>y']
 * ```
 */
export function toStructure(model: ModelLike | ModelSpec): Structure {
  const spec: ModelSpec = 'spec' in model ? model.spec : model
  const a = analyseSpec(model)
  const sizeOf = new Map(a.dims.map((d) => [d.name, d.size]))
  const groups = new Map<string, StructureGroup>()
  const groupOf = (dims: readonly string[]): string | undefined => {
    let parent: string | undefined
    let id: string | undefined
    for (let i = 0; i < dims.length; i++) {
      const d = dims[i] as string
      id = `plate:${dims.slice(0, i + 1).join('/')}`
      if (!groups.has(id)) {
        const count = sizeOf.get(d)
        groups.set(id, {
          id,
          label: d,
          kind: 'plate',
          ...(parent === undefined ? {} : { parent }),
          ...(count === undefined ? {} : { count }),
        })
      }
      parent = id
    }
    return id
  }
  const nodes: StructureNode[] = spec.nodes.map((n, i) => {
    const info = a.nodes[i]
    const group = groupOf(n.dims)
    const badges: Record<string, string | number> = {}
    if (n.dims.length > 0) badges['dims'] = `[${n.dims.join(', ')}]`
    if (n.kind === 'data' && n.role !== undefined) badges['role'] = n.role
    if (n.kind === 'generated') badges['generated'] = 'yes'
    if (n.kind === 'latent' && n.constraint !== undefined) badges['constraint'] = n.constraint.kind
    const detail =
      info?.family ??
      (n.kind === 'data'
        ? (n.support ?? 'data')
        : n.kind === 'derived'
          ? `= ${info?.operators.join(', ') || 'expression'}`
          : undefined)
    return {
      id: info?.id ?? n.name,
      label: n.name,
      role: ROLE[n.kind],
      ...(group === undefined ? {} : { group }),
      ...(detail === undefined ? {} : { detail }),
      ...(Object.keys(badges).length === 0 ? {} : { badges }),
    }
  })
  const edges: StructureEdge[] = a.edges.map((e) => ({
    from: e.from,
    to: e.to,
    directed: true,
    kind: 'dependency' as const,
  }))
  return structure('model-spec', nodes, edges, {
    groups: [...groups.values()],
    title: spec.name,
    direction: 'down',
  })
}
