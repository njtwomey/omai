/**
 * Relabelling the components of an EP fixed point: the permutations of a set of categories, the same permutation applied to the
 * discrete messages that start a run and to the records of every site that is indexed by the components (a vector of rates, a
 * probability vector over the labels, a matrix with a row and a column per component), and the canonical form of a fixed point,
 * its components sorted by one site's mean. A mixture or a regime chain written with exchangeable components has one fixed point
 * per labelling of the same reading of the data; sorting them makes those copies one mode, and summing their evidence is the
 * mode's share of the posterior mass.
 */

import {
  arange,
  argsort,
  matmul,
  mul,
  reshape,
  square,
  sub,
  take,
  tensor,
  toFlat,
  type Value,
} from '../../tensor/index.ts'
import { nodeExprs, nodesByName } from '../model/analysis-exprs.ts'
import { exprShape } from '../model/check.ts'
import type { NamedShape } from '../model/operators.ts'
import { type ExprSpec, exprRefs, type ModelSpec } from '../model/spec.ts'
import type { SiteRecord } from './_site-mixture.ts'

/**
 * Which axes of a site are indexed by the components: positions in the record's shape. `undefined` for a site means the axes are not
 * known and every axis of length `k` is taken to be one (the only reading there is without the model; wrong for a site with a
 * second axis of the same length that is not per component).
 */
export type ComponentAxes = (site: string) => readonly number[] | undefined

const DISCRETE = new Set(['categorical', 'bernoulli'])

/**
 * The component axes of the sites of a model, from how it uses its discrete latents: the dimension that a discrete latent looks
 * up (`mu.at(z)`: the first dimension of the table; `ops.pick(table, z)`: the last dimension of the table expression, as in
 * `pick(mul(x, slope), z)`) is a component dimension, so the axes of any node with that dimension are component axes; and the
 * trailing (event) axes of a node that a discrete latent's distribution reads (the weights of `categorical({ probs: weights })`)
 * are the categories, so component axes too. A model with no discrete latent gives `undefined`. A model with one and no lookup of
 * these kinds (a mixture written with `where`: the components are separate nodes) gives `undefined` for every site that is not
 * the class probabilities: its axes are not known, which is not the same as having none.
 *
 * @param spec - The model.
 * @returns The component axes of a site by name, or undefined when the model names none.
 */
export function componentAxesOf(spec: ModelSpec): ComponentAxes | undefined {
  const byName = nodesByName(spec)
  const discrete = new Set<string>()
  const categories = new Set<string>()
  for (const n of spec.nodes)
    if (n.kind === 'latent' && DISCRETE.has(n.dist.family)) {
      discrete.add(n.name)
      for (const e of Object.values(n.dist.params)) for (const r of exprRefs(e)) categories.add(r)
    }
  const scope = new Map<string, NamedShape>()
  for (const n of spec.nodes)
    if (n.kind !== 'observed' && n.kind !== 'factor') scope.set(n.name, { dims: n.dims, eventRank: n.eventRank })
  const dims = new Set<string>()
  const lookedUp = (e: ExprSpec, which: 'first' | 'last'): string | undefined => {
    try {
      const shape = exprShape(e, scope)
      return which === 'first' ? shape.dims[0] : shape.dims[shape.dims.length - 1]
    } catch {
      return undefined
    }
  }
  const walk = (e: ExprSpec): void => {
    if (e.kind !== 'op' && e.kind !== 'custom') return
    for (const a of e.args) walk(a)
    if (e.kind === 'op' && (e.op === 'at' || e.op === 'pick')) {
      const [table, index] = e.args
      if (table !== undefined && index?.kind === 'ref' && discrete.has(index.name)) {
        const dim = lookedUp(table, e.op === 'at' ? 'first' : 'last')
        if (dim !== undefined) dims.add(dim)
      }
    }
  }
  for (const n of spec.nodes) for (const e of nodeExprs(n)) walk(e)
  if (dims.size === 0 && categories.size === 0) return undefined
  return (site) => {
    const node = byName.get(site)
    if (node === undefined) return undefined
    // no lookup was found: the model says nothing about which axes of the other sites are components
    if (dims.size === 0 && !categories.has(site)) return undefined
    const axes: number[] = []
    for (const [i, d] of node.dims.entries()) if (dims.has(d)) axes.push(i)
    // the event axes of a node that is the class probabilities of a discrete latent
    if (categories.has(site))
      for (let i = node.dims.length; i < node.dims.length + Math.max(node.eventRank, 1); i++) axes.push(i)
    return axes
  }
}

/** The `k!` permutations of `0..k-1` in lexicographic order (the first is the identity). */
export function permutations(k: number): number[][] {
  if (k <= 1) return [[0]]
  const out: number[][] = []
  const rec = (prefix: number[], rest: number[]): void => {
    if (rest.length === 0) {
      out.push(prefix)
      return
    }
    for (const [i, v] of rest.entries()) rec([...prefix, v], [...rest.slice(0, i), ...rest.slice(i + 1)])
  }
  rec([], toFlat(arange(k)))
  return out
}

/** The values of `xs` (an axis of length `k`) moved so that the entry at `q` goes to position `perm[q]`. */
export function moveAxis<T>(xs: readonly T[], perm: readonly number[]): T[] {
  const out = xs.slice()
  for (const [q, to] of perm.entries()) out[to] = xs[q] as T
  return out
}

/** The same move applied to the component axes of a row-major array of the given shape (`axes`; absent: every axis of length `k`). */
function moveAxes(
  values: readonly number[],
  shape: readonly number[],
  k: number,
  perm: readonly number[],
  axes?: readonly number[],
): number[] {
  // Entry `q` goes to `perm[q]`, so position `j` takes the entry whose destination is `j`: the inverse permutation.
  const from = argsort(tensor(perm))
  let cur: Value = reshape(tensor(values), shape)
  for (const [axis, len] of shape.entries())
    if (len === k && (axes === undefined || axes.includes(axis))) cur = take(cur, from, axis)
  return toFlat(cur)
}

/**
 * A site's record with the components relabelled: entry `q` of every component axis of length `k` moves to `perm[q]`. A covariance
 * is moved along both of its axes, a categorical site's probabilities along their category axis (when they are over `k` categories).
 *
 * @param r - The record.
 * @param perm - The permutation: entry `q` goes to `perm[q]`.
 * @param k - The number of components.
 * @param axes - The axes of the record's shape that are indexed by the components (`componentAxesOf`); absent: every axis of
 * length `k`, which moves a second axis of the same length that is not per component (a `k` by `k` site of a model with `k`
 * dimensions of data).
 * @returns The relabelled record.
 */
export function relabelRecord(r: SiteRecord, perm: readonly number[], k: number, axes?: readonly number[]): SiteRecord {
  const moved = (v: readonly number[]): number[] => moveAxes(v, r.shape, k, perm, axes)
  // The covariance of a vector site of shape [k] is a k by k block: both of its axes move.
  const covariance =
    r.covariance !== undefined && r.mean.length === k && (axes === undefined || axes.includes(0))
      ? moveAxes(r.covariance, [k, k], k, perm)
      : r.covariance
  let probs: number[] | undefined
  // a categorical site over `k` categories: its probabilities are `rows` rows of `k` (another category count is no component axis)
  const rows = r.shape.reduce((a, b) => a * b, 1)
  if (r.probs !== undefined && r.probs.length === rows * k) {
    probs = toFlat(take(reshape(tensor(r.probs), [rows, k]), argsort(tensor(perm)), 1))
  }
  if (probs !== undefined) {
    // The index of a categorical site is a label: its mean and variance are those of the relabelled probabilities.
    const labels = arange(k)
    const p = reshape(tensor(probs), [rows, k])
    const m = matmul(p, labels)
    const mean = toFlat(m)
    const variance = toFlat(sub(matmul(p, square(labels)), mul(m, m)))
    return { ...r, mean, variance, probs }
  }
  return {
    ...r,
    mean: moved(r.mean),
    variance: moved(r.variance),
    ...(covariance === undefined ? {} : { covariance: covariance as readonly number[] }),
  }
}

/**
 * The site that sorts the components: the first (in the order of `sites`) whose shape is the vector of `k` components and whose
 * belief is a continuous one (a vector of rates or locations), or undefined when there is none. With the model's `axesOf`, a vector
 * whose axis is not a component axis (a location of a two-dimensional observation, when there are two components) is not a key.
 */
export function orderingSite(
  sites: Readonly<Record<string, SiteRecord>>,
  k: number,
  axesOf?: ComponentAxes,
): string | undefined {
  for (const [name, r] of Object.entries(sites))
    if (
      r.shape.length === 1 &&
      r.shape[0] === k &&
      r.kind !== 'categorical' &&
      r.kind !== 'moments' &&
      (axesOf?.(name)?.includes(0) ?? true)
    )
      return name
  return undefined
}

/** The permutation that sorts the components by the mean of `key`, ascending (entry `q` goes to its rank). */
export function sortingPermutation(r: SiteRecord, k: number): number[] {
  // The rank of each component by its mean (ties by position): the inverse of the sorting order.
  const means = tensor(toFlat(arange(k)).map((q) => r.mean[q] ?? 0))
  return toFlat(argsort(argsort(means)))
}

/** The record set of a fixed point in canonical form: every site relabelled so that the components are sorted by the mean of `key`. */
export function canonicalSites(
  sites: Readonly<Record<string, SiteRecord>>,
  key: string,
  k: number,
  axesOf?: ComponentAxes,
): Record<string, SiteRecord> {
  const keyed = sites[key]
  if (keyed === undefined) return { ...sites }
  const perm = sortingPermutation(keyed, k)
  const out: Record<string, SiteRecord> = {}
  for (const [name, r] of Object.entries(sites)) out[name] = relabelRecord(r, perm, k, axesOf?.(name))
  return out
}
