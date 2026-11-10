/**
 * The id convention of a model spec, one place for every consumer: the drawing (`toStructure`), the analysis
 * (`analyseSpec`), the message-passing factor graph (`probprog/model/factors.ts`, lane B's node ids) and anything that
 * overlays inference state on the picture, so that a tape event, a posterior marginal or a diagnostic names the same
 * thing the picture draws.
 *
 * - **Variable and value nodes**: the node's own name, which is also the name of its site in the compiled program, of its
 *   entry in a `Posterior` (`draws`, `mean`, `marginal`), and of its path suffix on a tape (`mcmc/site`, per-site ELBO
 *   terms, per-site diagnostics). Sites are latent, observed, derived (recorded), generated, factor and param nodes.
 * - **Data nodes** are not sites: `data:<name>`. (An observed node may share its name with its data node; the prefix
 *   keeps the two apart.)
 * - **Factors** of a message-passing factor graph: `factor:<site>` is the factor of site `<site>`: its distribution factor
 *   (latent, observed, generated), its deterministic factor (derived: the operator linking its parents) or the term itself
 *   (a spec `factor` node, which has no variable of its own). Messages are keyed `<from id>` to `<to id>`.
 * - **Plates**: `plate:<a>×<b>` for a node over dimensions a and b, taken in the spec's dimension order (not the node's
 *   own order, which only decides its axes), so that nodes over the same dimensions share one plate and plates nest by
 *   prefix (`plate:a` holds `plate:a×b`): the form `omai-render`'s `nestPlates` reads, as for a `pgm` plate model.
 * - **Elements** of a vectorised site, for overlays that go below the site: `<site>[<flat index>]`, the coordinate names
 *   `potentialEnergy` uses (`w[0]`, `w[1]`).
 * - **Edges** have no id of their own; they are named by their end ids, `<from id>><to id>`.
 *
 * @index function nodeId, factorId, plateId and elementId: the id convention of a model spec's picture and tapes.
 */

import type { ModelNode, ModelSpec } from './spec.ts'

/** The plate separator inside a plate id, as in the `pgm` plate models. */
const TIMES = '×'

/**
 * The id of a node: its name, or `data:<name>` for a data node.
 *
 * @param node - A node (or its kind and name).
 * @returns The id.
 * @example
 * ```ts
 * import { nodeId } from 'omai-engine/probprog/model'
 * [nodeId({ kind: 'latent', name: 'mu' }), nodeId({ kind: 'data', name: 'y' })] // => ['mu', 'data:y']
 * ```
 */
export function nodeId(node: Pick<ModelNode, 'kind' | 'name'>): string {
  return node.kind === 'data' ? `data:${node.name}` : node.name
}

/**
 * The id of the factor of a site in a factor graph.
 *
 * @param site - The site's name.
 * @returns `factor:<site>`.
 * @example
 * ```ts
 * import { factorId } from 'omai-engine/probprog/model'
 * factorId('y') // => 'factor:y'
 * ```
 */
export function factorId(site: string): string {
  return `factor:${site}`
}

/**
 * A node's dimensions in the spec's dimension order.
 *
 * @param spec - The spec.
 * @param dims - A node's dimensions.
 * @returns The same dimensions, ordered as `spec.dims` declares them.
 * @example
 * ```ts
 * import { canonicalDims, defineModel } from 'omai-engine/probprog/model'
 * const m = defineModel('m', (b) => { b.range('a'); b.range('b') })
 * canonicalDims(m.spec, ['b', 'a']) // => ['a', 'b']
 * ```
 */
export function canonicalDims(spec: Pick<ModelSpec, 'dims'>, dims: readonly string[]): string[] {
  const order = new Map(spec.dims.map((d, i) => [d.name, i]))
  return [...dims].sort((a, b) => (order.get(a) ?? 0) - (order.get(b) ?? 0))
}

/**
 * The id of the plate a set of dimensions forms (dimensions in the order given: use {@link canonicalDims} first).
 *
 * @param dims - The dimensions, outermost first.
 * @returns `plate:<a>×<b>`, or undefined for no dimensions.
 * @example
 * ```ts
 * import { plateId } from 'omai-engine/probprog/model'
 * plateId(['row', 'feature']) // => 'plate:row×feature'
 * ```
 */
export function plateId(dims: readonly string[]): string | undefined {
  return dims.length === 0 ? undefined : `plate:${dims.join(TIMES)}`
}

/**
 * The id of one element of a vectorised site (the flat index, as `potentialEnergy` names its coordinates).
 *
 * @param site - The site's name.
 * @param index - The flat element index.
 * @returns `<site>[<index>]`.
 * @example
 * ```ts
 * import { elementId } from 'omai-engine/probprog/model'
 * elementId('w', 1) // => 'w[1]'
 * ```
 */
export function elementId(site: string, index: number): string {
  return `${site}[${index}]`
}
