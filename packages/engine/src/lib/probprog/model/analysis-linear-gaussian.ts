/**
 * The linear-Gaussian sub-models of a spec: the largest sets of Gaussian latents that, given every other latent, are
 * jointly Gaussian with their observed children, because every node that depends on them is Gaussian with a mean affine
 * in them and a scale, and coefficients, that do not. Such a block can be integrated out exactly (its marginal likelihood
 * is a Gaussian density) and its posterior given the rest is a Gaussian, so it is what a collapsed, message-passing or
 * conditional-Gaussian step takes. The whole-model test of `analyseSpec` is the special case where nothing is left over;
 * a hierarchical regression is a block conditional on its scales. Each block is a record with its sites, the latents it
 * is conditioned on, the observations it explains and the size of the joint Gaussian.
 *
 * @index function linearGaussianBlocks: maximal Gaussian blocks that are exactly marginalisable given the other latents.
 */

import { isAffineIn, type LatentMemo, latentsIn, type NodesByName } from './analysis-exprs.ts'
import { canonicalDims } from './ids.ts'
import type { LatentNode, ModelNode, ModelSpec } from './spec.ts'

/** A set of Gaussian latents that is jointly Gaussian with its observations given the other latents. */
export interface LinearGaussianBlock {
  /** The latents in the block. */
  readonly sites: readonly string[]
  /** The latents it is conditioned on: they set its scales and coefficients, and are not part of the Gaussian. */
  readonly given: readonly string[]
  /** The observed nodes the block explains. */
  readonly observed: readonly string[]
  /** The derived nodes that carry the block's latents to its observed nodes. */
  readonly derived: readonly string[]
  /** The number of scalar variables in the joint Gaussian, when every dimension's size is known and no site is a vector event. */
  readonly dimension?: number
  /** Whether it is the whole model (nothing is conditioned on): then `analyseSpec` reports it as linear-Gaussian. */
  readonly whole: boolean
  /** What makes it exactly marginalisable, in a sentence. */
  readonly reason: string
}

/** What the analysis needs of the spec: its value nodes, the memo of derived latents and the sizes of dimensions. */
export interface LinearGaussianEnv {
  readonly spec: ModelSpec
  readonly byName: NodesByName
  readonly memo: LatentMemo
  readonly sizeOf: (dim: string) => number | undefined
}

const GAUSSIAN = new Set(['normal', 'multivariateNormal'])

/** Why a latent cannot be in a Gaussian block by itself: not a free Gaussian. */
function candidate(n: ModelNode): n is LatentNode {
  if (n.kind !== 'latent' || !GAUSSIAN.has(n.dist.family)) return false
  return n.constraint === undefined || n.constraint.kind === 'real'
}

/**
 * The linear-Gaussian blocks of a spec, from the largest set of Gaussian latents that stays valid when the latents that
 * break it (a product of two block latents, a non-Gaussian child, a scale that depends on one, an arbitrary factor term)
 * are left to the rest, then split into the parts that do not interact given the rest.
 *
 * Where two block latents multiply (the mean `a * b`), the one declared later is left out. Only blocks that explain an
 * observed node are reported.
 *
 * @param env - The spec, its value nodes by name, the memo of derived nodes' latents and the sizes of dimensions.
 * @returns The blocks, in the order of their first latent.
 */
export function linearGaussianBlocks(env: LinearGaussianEnv): LinearGaussianBlock[] {
  const { spec, byName, memo } = env
  const order = new Map(spec.nodes.map((n, i) => [n.name, i]))
  const block = new Set(spec.nodes.filter(candidate).map((n) => n.name))
  const within = (e: Parameters<typeof latentsIn>[0]): string[] =>
    [...latentsIn(e, byName, memo)]
      .filter((l) => block.has(l))
      .sort((a, b) => (order.get(a) ?? 0) - (order.get(b) ?? 0))
  for (let changed = true; changed; ) {
    changed = false
    const drop = (names: readonly string[]): void => {
      for (const l of names)
        if (block.delete(l)) {
          changed = true
        }
    }
    for (const n of spec.nodes) {
      if (n.kind === 'factor') {
        drop(within(n.expr))
        continue
      }
      if (n.kind !== 'latent' && n.kind !== 'observed') continue
      const params = Object.entries(n.dist.params)
      const touched = new Set(params.flatMap(([, e]) => within(e)))
      if (n.kind === 'observed') for (const l of within(n.value)) drop([l])
      if (touched.size === 0) continue
      if (!GAUSSIAN.has(n.dist.family)) {
        drop([...touched])
        continue
      }
      for (const [p, e] of params) {
        const vars = within(e)
        if (vars.length === 0) continue
        if (p !== 'loc') {
          drop(vars)
          continue
        }
        // The largest set of the expression's block latents it is affine in, earlier declarations first.
        const keep = new Set<string>()
        for (const v of vars) if (isAffineIn(e, byName, memo, new Set([...keep, v]))) keep.add(v)
        drop(vars.filter((v) => !keep.has(v)))
      }
    }
  }
  if (block.size === 0) return []
  // Split into the parts that share no node: latents tied by a prior or by a common child.
  const parent = new Map<string, string>([...block].map((l) => [l, l]))
  const find = (l: string): string => {
    let r = l
    while (parent.get(r) !== r) r = parent.get(r) as string
    return r
  }
  const union = (a: string, b: string): void => {
    parent.set(find(a), find(b))
  }
  for (const n of spec.nodes) {
    if (n.kind !== 'latent' && n.kind !== 'observed') continue
    const vars = [...new Set(Object.values(n.dist.params).flatMap((e) => within(e)))]
    if (n.kind === 'latent' && block.has(n.name)) vars.push(n.name)
    for (const v of vars.slice(1)) union(vars[0] as string, v)
  }
  const groups = new Map<string, string[]>()
  for (const l of block) groups.set(find(l), [...(groups.get(find(l)) ?? []), l])
  const out: LinearGaussianBlock[] = []
  const latentCount = spec.nodes.filter((n) => n.kind === 'latent').length
  for (const members of groups.values()) {
    const sites = members.sort((a, b) => (order.get(a) ?? 0) - (order.get(b) ?? 0))
    const set = new Set(sites)
    const observed: string[] = []
    const given = new Set<string>()
    const derived = new Set<string>()
    for (const n of spec.nodes) {
      if (n.kind !== 'latent' && n.kind !== 'observed') continue
      const inBlock = n.kind === 'latent' && set.has(n.name)
      const touches = Object.values(n.dist.params).some((e) => [...latentsIn(e, byName, memo)].some((l) => set.has(l)))
      if (!inBlock && !touches) continue
      if (n.kind === 'observed') observed.push(n.name)
      for (const e of Object.values(n.dist.params))
        for (const l of latentsIn(e, byName, memo)) if (!set.has(l)) given.add(l)
    }
    for (const d of spec.nodes) if (d.kind === 'derived' && refersTo(d, byName, memo, set)) derived.add(d.name)
    if (observed.length === 0) continue
    let dimension: number | undefined = 0
    for (const s of sites) {
      const node = spec.nodes.find((n) => n.name === s && n.kind === 'latent')
      if (node === undefined || node.eventRank > 0) {
        dimension = undefined
        break
      }
      let k = 1
      for (const d of canonicalDims(spec, node.dims)) {
        const size = env.sizeOf(d)
        if (size === undefined) {
          dimension = undefined
          break
        }
        k *= size
      }
      if (dimension === undefined) break
      dimension += k
    }
    const whole = given.size === 0 && sites.length === latentCount
    out.push({
      sites,
      given: [...given].sort((a, b) => (order.get(a) ?? 0) - (order.get(b) ?? 0)),
      observed,
      derived: [...derived],
      ...(dimension === undefined ? {} : { dimension }),
      whole,
      reason:
        given.size === 0
          ? `every node that touches ${sites.join(', ')} is Gaussian with a mean affine in them: jointly Gaussian, exactly marginalisable`
          : `given ${[...given].join(', ')}, every node that touches ${sites.join(', ')} is Gaussian with a mean affine in them and a scale and coefficients free of them: jointly Gaussian, exactly marginalisable`,
    })
  }
  return out.sort((a, b) => (order.get(a.sites[0] as string) ?? 0) - (order.get(b.sites[0] as string) ?? 0))
}

function refersTo(d: ModelNode, byName: NodesByName, memo: LatentMemo, set: ReadonlySet<string>): boolean {
  return d.kind === 'derived' && [...latentsIn(d.expr, byName, memo)].some((l) => set.has(l))
}
