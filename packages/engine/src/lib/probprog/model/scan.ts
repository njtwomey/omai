/**
 * What the rest of the model layer needs to know about a `scan` node (the recurrence of `spec.ts`) beyond compiling it:
 * the emissions as the observed nodes they are (`emissionNodes`, and `expandScans`, the view of a spec the static
 * analyses read, in which every scan is followed by its emissions as ordinary observed nodes over the scanned
 * dimension), the linear-Gaussian test of a scan (`scanLinearGaussian`: a Gaussian initial state, a Gaussian step
 * whose mean is affine in the previous state, Gaussian emissions whose mean is affine in the state, scales that do not
 * depend on the states), and `unrollScans`, the bound model with every scan written out as the scalar latent and
 * observed nodes of its steps (`<scan>[t]`, `<emission>[t]`), which is what the factor-graph compiler reads. The unrolled
 * model has the log joint of the program the scan compiles to, with the same site names.
 *
 * @index function Unroll every scan of a bound spec into one node per step.
 */

import { OmaiError } from '../../base/index.ts'
import { type Tensor, take, toArray } from '../../tensor/index.ts'
import { isAffineIn, type LatentMemo, latentsIn, type NodesByName } from './analysis-exprs.ts'
import { type BoundModel, bind } from './bind.ts'
import { validateSpec } from './check.ts'
import { substitute } from './compose.ts'
import { distributionFamilies } from './dists/families.ts'
import { elementId } from './ids.ts'
import {
  type ExprSpec,
  type LatentNode,
  type ModelNode,
  type ModelSpec,
  nodeRefs,
  type ObservedNode,
  prevName,
  type ScanNode,
} from './spec.ts'

/**
 * The emissions of a scan as observed nodes over the scanned dimension, the form the analyses and drawings read.
 *
 * @param scan - The scan.
 * @returns One observed node per emission, in order.
 * @example
 * ```ts
 * import { defineModel, emissionNodes } from 'omai-engine/probprog/model'
 * import { normal } from 'omai-engine/probprog/model/dists'
 * const m = defineModel('walk', (b) => {
 *   const y = b.data('y', { dims: ['time'] })
 *   b.scan('x', { over: 'time', init: normal(0, 1), step: ({ prev }) => normal(prev, 1), emit: ({ state, observe }) => observe('y', normal(state, 1), y) })
 * })
 * const scan = m.spec.nodes[1]
 * scan?.kind === 'scan' ? emissionNodes(scan).map((n) => [n.name, n.dims]) : [] // => [['y', ['time']]]
 * ```
 */
export function emissionNodes(scan: ScanNode): ObservedNode[] {
  const families = distributionFamilies()
  return scan.emissions.map((m) => ({
    kind: 'observed',
    name: m.name,
    dims: scan.dims,
    eventRank: families[m.dist.family]?.eventRank ?? 0,
    dist: m.dist,
    value: m.value,
    ...(m.doc === undefined ? {} : { doc: m.doc }),
  }))
}

/**
 * A spec as the static analyses read it: every scan is followed by its emissions as observed nodes over the scanned
 * dimension (each refers to the scan's stack of states by the scan's name), and nothing else changes.
 *
 * @param spec - The spec.
 * @returns The same spec when it has no scan, else the expanded one.
 * @example
 * ```ts
 * import { expandScans, defineModel } from 'omai-engine/probprog/model'
 * import { normal } from 'omai-engine/probprog/model/dists'
 * const m = defineModel('walk', (b) => {
 *   const y = b.data('y', { dims: ['time'] })
 *   b.scan('x', { over: 'time', init: normal(0, 1), step: ({ prev }) => normal(prev, 1), emit: ({ state, observe }) => observe('y', normal(state, 1), y) })
 * })
 * expandScans(m.spec).nodes.map((n) => n.kind) // => ['data', 'scan', 'observed']
 * ```
 */
export function expandScans(spec: ModelSpec): ModelSpec {
  if (!spec.nodes.some((n) => n.kind === 'scan')) return spec
  return {
    ...spec,
    nodes: spec.nodes.flatMap((n): ModelNode[] => (n.kind === 'scan' ? [n, ...emissionNodes(n)] : [n])),
  }
}

/** The pseudo-latents a scan's own state and previous state are for the affine tests: Gaussian, free. */
function stateLatent(name: string, scan: ScanNode): LatentNode {
  return { kind: 'latent', name, dims: [], eventRank: scan.eventRank, dist: scan.init }
}

const GAUSSIAN = new Set(['normal', 'multivariateNormal'])

/**
 * Whether a scan is linear-Gaussian: a Gaussian first state, a Gaussian step whose mean is affine in the previous state,
 * Gaussian emissions whose mean is affine in the state, and scales (and every other parameter) free of the states. With
 * `strict` the latents outside the scan count as variables too (the whole-model test: the means must be affine jointly in
 * them and the scales free of all of them); without, they are constants the scan is conditioned on (the block test).
 *
 * @param scan - The scan.
 * @param byName - The nodes its expressions can refer to.
 * @param memo - The memo of derived nodes' latents.
 * @param strict - Whether the outer latents are variables.
 * @returns Whether it holds, and why or why not.
 */
export function scanLinearGaussian(
  scan: ScanNode,
  byName: NodesByName,
  memo: LatentMemo,
  strict: boolean,
): { readonly holds: boolean; readonly reason: string } {
  const prev = prevName(scan.name)
  const names = new Map(byName)
  names.set(scan.name, stateLatent(scan.name, scan))
  names.set(prev, stateLatent(prev, scan))
  const states = new Set([scan.name, prev])
  const vars = strict ? undefined : states
  const no = (reason: string): { holds: false; reason: string } => ({ holds: false, reason })
  const free = (e: ExprSpec): boolean => {
    const used = latentsIn(e, names, memo)
    for (const l of used) if (strict || states.has(l)) return false
    return true
  }
  const check = (
    what: string,
    dist: { readonly family: string; readonly params: { readonly [k: string]: ExprSpec } },
  ): string | undefined => {
    if (!GAUSSIAN.has(dist.family)) return `${what} is ${dist.family}, not Gaussian`
    for (const [p, e] of Object.entries(dist.params)) {
      if (p === 'loc') {
        if (!isAffineIn(e, names, memo, vars)) return `the mean of ${what} is not affine in the states`
      } else if (!free(e)) return `the ${p} of ${what} depends on a state or a latent`
    }
    return undefined
  }
  const problems = [
    check('the initial state', scan.init),
    check('the step', scan.step),
    ...scan.emissions.map((m) => check(`emission '${m.name}'`, m.dist)),
  ].filter((p): p is string => p !== undefined)
  const first = problems[0]
  return first === undefined
    ? {
        holds: true,
        reason: `a Gaussian chain: ${scan.emissions.length} Gaussian emission(s) with means affine in the state, linear steps`,
      }
    : no(first)
}

/** Why a scan cannot be unrolled; the factor-graph compiler reports it as an issue on the scan. */
export class ScanUnrollError extends OmaiError {
  constructor(
    readonly node: string,
    reason: string,
  ) {
    super('value', reason, { node })
  }
}

function constantOf(t: Tensor, step: number, axis: number): ExprSpec {
  const slice = take(t, step, axis)
  return { kind: 'const', value: toArray(slice) as never }
}

/**
 * The bound model with every scan written out as the nodes of its steps: the state at step `t` is a scalar latent named
 * `<scan>[t]` (its distribution the initial one at step 0, the step distribution after, with the previous state
 * and the current one replaced by those nodes), and each emission an observed node `<emission>[t]` with its value read from the
 * bound data. The log joint is the compiled scan's, site for site, so any analysis or compiler that knows plain nodes (the
 * message-passing factor graph, the linear-Gaussian blocks of the steps) sees the chain. Only a scan whose series are
 * data nodes can be unrolled (the values are folded in as constants), and nothing after the scan may read its stacked
 * states.
 *
 * @param bound - The bound model.
 * @returns The bound model without scans (the same one when it has none).
 * @throws ScanUnrollError (an `OmaiError` with code `value`) when a scan reads a series that is not a data node, a later node
 *   reads the stacked states, or the scanned dimension has no size.
 * @concept model-spec
 * @example
 * ```ts
 * import { defineModel, unrollScans } from 'omai-engine/probprog/model'
 * import { normal } from 'omai-engine/probprog/model/dists'
 * const m = defineModel('walk', (b) => {
 *   const y = b.data('y', { dims: ['time'] })
 *   b.scan('x', { over: 'time', init: normal(0, 1), step: ({ prev }) => normal(prev, 1), emit: ({ state, observe }) => observe('y', normal(state, 1), y) })
 * })
 * unrollScans(m.bind({ y: [1, 2, 3] })).spec.nodes.map((n) => n.name) // => ['y', 'x[0]', 'y[0]', 'x[1]', 'y[1]', 'x[2]', 'y[2]']
 * ```
 */
export function unrollScans(bound: BoundModel): BoundModel {
  const { spec } = bound
  if (!spec.nodes.some((n) => n.kind === 'scan')) return bound
  const nodes: ModelNode[] = []
  const stacked = new Set<string>()
  const byName = new Map<string, ModelNode>()
  for (const n of spec.nodes) {
    for (const r of nodeRefs(n))
      if (stacked.has(r))
        throw new ScanUnrollError(
          n.name,
          `'${n.name}' reads the stacked states of the scan '${r}', which an unrolled scan has no node for`,
        )
    if (n.kind !== 'scan') {
      nodes.push(n)
      if (n.kind !== 'observed' && n.kind !== 'factor') byName.set(n.name, n)
      continue
    }
    const length = bound.sizes[n.over]
    if (length === undefined)
      throw new ScanUnrollError(n.name, `the dimension '${n.over}' of the scan '${n.name}' has no size`)
    stacked.add(n.name)
    const outer = [...byName.values()].filter((o) => o.dims.includes(n.over))
    for (let t = 0; t < length; t++) {
      const map = new Map<string, ExprSpec>()
      for (const o of outer) {
        const data = o.kind === 'data' ? bound.data[o.name] : undefined
        if (data === undefined)
          throw new ScanUnrollError(
            n.name,
            `the scan '${n.name}' reads '${o.name}', a ${o.kind} node over '${n.over}'; only data series can be unrolled`,
          )
        map.set(o.name, constantOf(data, t, o.dims.indexOf(n.over)))
      }
      map.set(n.name, { kind: 'ref', name: elementId(n.name, t) })
      if (t > 0) map.set(prevName(n.name), { kind: 'ref', name: elementId(n.name, t - 1) })
      const dist = (d: { readonly family: string; readonly params: { readonly [k: string]: ExprSpec } }): typeof d => ({
        family: d.family,
        params: Object.fromEntries(Object.entries(d.params).map(([k, e]) => [k, substitute(e, map)])),
      })
      nodes.push({
        kind: 'latent',
        name: elementId(n.name, t),
        dims: [],
        eventRank: n.eventRank,
        dist: dist(t === 0 ? n.init : n.step),
      })
      for (const m of emissionNodes(n))
        nodes.push({ ...m, name: elementId(m.name, t), dims: [], dist: dist(m.dist), value: substitute(m.value, map) })
    }
  }
  const unrolled: ModelSpec = { ...spec, nodes }
  validateSpec(unrolled)
  return bind({ spec: unrolled, customs: bound.customs }, bound.data, bound.options)
}
