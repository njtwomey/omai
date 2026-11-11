/**
 * A vector written as one event (`f ~ MultivariateNormal(0, K)` with no named dimension, observed through `y ~
 * MultivariateNormal(f, sigma^2 I)` with the data `y` an event too) rewritten as the vector over a named dimension that the
 * factor-graph compiler reads (`f` over `event#12`, `y ~ Normal(f, sigma)` per element): the same model, with a dimension
 * that stands for the event axis. The rewrite is for the pairs the compiler can map, a multivariate normal latent with a known
 * mean and covariance and an observation of it through a diagonal known covariance; anything else keeps its event form and is
 * refused by the compiler as before.
 */

import { type BoundModel, bind } from './bind.ts'
import type { DistSpec, ExprSpec, ModelNode } from './spec.ts'

/** The dimension that stands for the event axis of a vector of `n` elements. */
const eventDim = (n: number): string => `event#${n}`

/** The number of elements of an event vector read from a constant expression (a vector, or a matrix for its rows), or undefined. */
function constLength(e: ExprSpec | undefined, axis: number): number | undefined {
  if (e === undefined || e.kind !== 'const' || e.dims !== undefined) return undefined
  let v: unknown = e.value
  for (let i = 0; i < axis; i++) v = Array.isArray(v) ? v[0] : undefined
  return Array.isArray(v) ? v.length : undefined
}

/**
 * The model with its event vectors rewritten as vectors over a named dimension, where the compiler can map them.
 *
 * @param bound - A bound spec.
 * @returns The same model when nothing is rewritten, else a bound spec in which each rewritten node has the dimension `event#<n>` (of its size) in place of its event axis.
 * @example
 * ```ts
 * import { eventToDims } from 'omai-engine/probprog/model'
 * import { defineModel } from 'omai-engine/probprog/model'
 * import { multivariateNormal } from 'omai-engine/probprog/model/dists'
 * const m = defineModel('m', (b) => { b.latent('f', multivariateNormal([0, 0], { covariance: [[1, 0], [0, 1]] })) })
 * eventToDims(m.bind({})).spec.nodes.map((n) => n.dims) // => [['event#2']]
 * ```
 */
export function eventToDims(bound: BoundModel): BoundModel {
  const nodes = bound.spec.nodes
  const isMvn = (d: DistSpec): boolean => d.family === 'multivariateNormal'
  // The vectors that are latents with a known mean (or covariance) of a known length.
  const latentSize = new Map<string, number>()
  for (const n of nodes) {
    if (n.kind !== 'latent' || !isMvn(n.dist) || n.dims.length !== 0 || n.eventRank !== 1) continue
    const size = constLength(n.dist.params['loc'], 0) ?? constLength(n.dist.params['covariance'], 0)
    if (size !== undefined) latentSize.set(n.name, size)
  }
  // The observations of an event vector through a diagonal known covariance, and the data they read.
  const observed = new Map<string, { readonly size: number; readonly data: string; readonly scale: number[] }>()
  for (const n of nodes) {
    if (n.kind !== 'observed' || !isMvn(n.dist) || n.dims.length !== 0 || n.eventRank !== 1) continue
    const cov = n.dist.params['covariance']
    if (n.value.kind !== 'ref' || cov === undefined || cov.kind !== 'const' || cov.dims !== undefined) continue
    const rows = cov.value as unknown
    if (!Array.isArray(rows)) continue
    const size = rows.length
    const diagonal = rows.every(
      (r, i) =>
        Array.isArray(r) && r.length === size && r.every((v, j) => typeof v === 'number' && (i === j || v === 0)),
    )
    const loc = n.dist.params['loc']
    const locOk =
      loc !== undefined && ((loc.kind === 'ref' && latentSize.get(loc.name) === size) || constLength(loc, 0) === size)
    const data = bound.data[n.value.name]
    if (!diagonal || !locOk || data === undefined || data.shape.length !== 1 || data.shape[0] !== size) continue
    observed.set(n.name, {
      size,
      data: n.value.name,
      scale: (rows as number[][]).map((r, i) => Math.sqrt(r[i] as number)),
    })
  }
  if (latentSize.size === 0 && observed.size === 0) return bound
  const dataNames = new Set([...observed.values()].map((o) => o.data))
  const sizes = new Set<number>([...latentSize.values(), ...[...observed.values()].map((o) => o.size)])
  const withDims = (e: ExprSpec | undefined, dim: string): ExprSpec | undefined =>
    e !== undefined && e.kind === 'const' && e.dims === undefined && Array.isArray(e.value)
      ? { kind: 'const', value: e.value, dims: [dim] }
      : e
  const rewritten: ModelNode[] = nodes.map((n): ModelNode => {
    if (n.kind === 'latent' && latentSize.has(n.name)) {
      const dim = eventDim(latentSize.get(n.name) as number)
      const loc = withDims(n.dist.params['loc'], dim)
      // As the dense Gaussian prior is written elsewhere (a multivariate normal over a named dimension), the family's event rank stays.
      return {
        ...n,
        dims: [dim],
        dist: { ...n.dist, params: { ...n.dist.params, ...(loc === undefined ? {} : { loc }) } },
      }
    }
    if (n.kind === 'data' && dataNames.has(n.name) && n.eventRank === 1 && n.dims.length === 0) {
      const size = [...observed.values()].find((o) => o.data === n.name)?.size as number
      return { ...n, dims: [eventDim(size)], eventRank: 0 }
    }
    const o = n.kind === 'observed' ? observed.get(n.name) : undefined
    if (n.kind === 'observed' && o !== undefined) {
      const dim = eventDim(o.size)
      const loc = withDims(n.dist.params['loc'], dim) as ExprSpec
      return {
        ...n,
        dims: [dim],
        eventRank: 0,
        dist: {
          family: 'normal',
          params: { loc, scale: { kind: 'const', value: o.scale, dims: [dim] } },
        },
      }
    }
    return n
  })
  const declared = new Set(bound.spec.dims.map((d) => d.name))
  const dims = [
    ...bound.spec.dims,
    ...[...sizes].filter((n) => !declared.has(eventDim(n))).map((n) => ({ name: eventDim(n), size: n })),
  ]
  return bind({ spec: { ...bound.spec, dims, nodes: rewritten }, customs: bound.customs }, bound.data, bound.options)
}
