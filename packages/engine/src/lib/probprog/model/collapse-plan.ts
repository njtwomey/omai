/**
 * Collapse planning as a tool the user invokes: `collapsePlan` reads a model's structure and says which conjugate latent
 * sites the `collapse` handler can integrate out, what each costs and gains (the unconstrained coordinates a sampler no
 * longer moves in, the sites it conditions on, the plates summed over) and which conjugate-looking sites it cannot take and
 * why; `applyCollapse` gives the collapsed program through the existing handler. Nothing here chooses to collapse: a
 * person reads the plan, may edit its `sites`, and applies it, then runs whatever method they like on the collapsed
 * program and draws the removed sites back with `recover`.
 *
 * @index function collapsePlan: which sites the collapse handler can integrate out, and why not the rest.
 */

import { OmaiError } from '../../base/index.ts'
import type { Key } from '../../random/index.ts'
import { type Collapsed, collapse } from '../collapse.ts'
import { analyseSpec } from './analysis.ts'
import type { ConjugacyRecord } from './analysis-conjugacy.ts'
import type { BoundModel, ModelLike } from './bind.ts'
import { toProgram } from './compile.ts'
import { canonicalDims } from './ids.ts'
import type { ModelSpec } from './spec.ts'

/** One site the handler can integrate out, with what that costs and gains. */
export interface CollapseEntry {
  readonly site: string
  /** The observed nodes it is integrated into. */
  readonly children: readonly string[]
  /** The conjugate pair (`normal-normal loc`) and its closed-form posterior. */
  readonly pair: string
  readonly posterior: string
  /** The latent sites the marginal likelihood is a function of (they stay in the program). */
  readonly given: readonly string[]
  /** The dimensions of the site, and how many unconstrained coordinates it removes, when the sizes are known. */
  readonly dims: readonly string[]
  readonly removes?: number
  /** The dimensions summed over, as a plate sum. */
  readonly sumsOver: readonly string[]
}

/** A site that looks conjugate and that the handler cannot take, with the reason. */
export interface CollapseRefusal {
  readonly site: string
  readonly reason: string
}

/**
 * What the collapse handler can do for a model.
 *
 * @concept collapsed-inference
 */
export interface CollapsePlan {
  /** The sites to hand to the handler, in program order (edit the list to collapse fewer). */
  readonly sites: readonly string[]
  readonly entries: readonly CollapseEntry[]
  readonly refused: readonly CollapseRefusal[]
  /** The continuous latent sites that stay for a sampler. */
  readonly remaining: readonly string[]
  /** The unconstrained coordinates before and after, when every size is known and no latent has an event axis. */
  readonly dimensionBefore?: number
  readonly dimensionAfter?: number
  /** The plan as text. */
  readonly text: string
}

function elementsOf(spec: ModelSpec, name: string, sizeOf: (dim: string) => number | undefined): number | undefined {
  const node = spec.nodes.find((n) => n.name === name && n.kind === 'latent')
  if (node === undefined || node.eventRank > 0) return undefined
  let k = 1
  for (const d of canonicalDims(spec, node.dims)) {
    const size = sizeOf(d)
    if (size === undefined) return undefined
    k *= size
  }
  return k
}

/**
 * Which conjugate latents the `collapse` handler can integrate out of a model, what that gains, and why it cannot take
 * the others. A site is taken when its observed children all use it as one conjugate pair's parameter (a plate sum is
 * fine: the posterior sums over the plate), the children's other parameters do not depend on it, nothing else reads it
 * (no latent child, factor term or recorded derived node), the children are not subsampled or masked, and no other
 * collapsed site is one of its `given` latents (collapsing two sites that depend on each other would condition one on the
 * other's placeholder). Where two sites exclude each other the earlier one in program order is taken. Sites that are
 * conjugate only through an affine map, a lookup by index, a known multiple or a latent child are listed in `refused` with
 * the reason: the closed form exists (see `analyseSpec(model).conjugacy`) but the handler does not apply it.
 *
 * @param model - A bound model (its sizes, batch and missing data are read), or a definition or spec.
 * @returns The plan.
 * @concept collapsed-inference
 * @index function collapsePlan: which sites the collapse handler can integrate out, with cost and gain; user-invoked.
 * @example
 * ```ts
 * import { collapsePlan, defineModel } from 'omai-engine/probprog/model'
 * import { halfNormal, normal } from 'omai-engine/probprog/model/dists'
 * const m = defineModel('schools', (b) => {
 *   const school = b.range('school', { size: 8 })
 *   const mu = b.latent('mu', normal(0, 5))
 *   const tau = b.latent('tau', halfNormal(5))
 *   const theta = b.latent('theta', normal(mu, tau), { dims: [school] })
 *   b.observe('y', normal(theta, 10), b.data('y', { dims: [school] }))
 * })
 * const plan = collapsePlan(m)
 * [plan.sites, plan.remaining, plan.dimensionBefore, plan.dimensionAfter] // => [['theta'], ['mu', 'tau'], 10, 2]
 * ```
 */
export function collapsePlan(model: ModelLike | ModelSpec): CollapsePlan {
  const a = analyseSpec(model)
  const spec: ModelSpec = 'spec' in model ? model.spec : model
  const bound = 'sizes' in model ? (model as BoundModel) : undefined
  const sizeOf = (dim: string): number | undefined => a.dims.find((d) => d.name === dim)?.size
  const chosen: ConjugacyRecord[] = []
  const refused: CollapseRefusal[] = []
  for (const r of a.conjugacy) {
    if (r.kind === 'joint-prior') {
      refused.push({ site: r.sites.join(', '), reason: r.collapse.reason })
      continue
    }
    const site = r.sites[0] as string
    let reason: string | undefined = r.collapse.applies ? undefined : r.collapse.reason
    if (reason === undefined && bound !== undefined) {
      if (
        bound.batch !== undefined &&
        r.links.some((l) =>
          canonicalDims(spec, spec.nodes.find((n) => n.name === l.node)?.dims ?? []).includes(bound.batch?.dim ?? ''),
        )
      )
        reason = `its children are subsampled over ${bound.batch.dim}: the handler does not take subsampled sites`
      const masked = r.links.find(
        (l) => (bound.missing[l.node]?.count ?? 0) > 0 && bound.missing[l.node]?.policy !== 'impute',
      )
      if (masked !== undefined)
        reason = `'${masked.node}' has missing values that are masked: the handler does not take masked sites`
    }
    if (reason === undefined) {
      const clash = chosen.find((c) => r.given.includes(c.sites[0] as string) || c.given.includes(site))
      if (clash !== undefined)
        reason = `it and ${clash.sites[0]} depend on each other (one is a known parameter or hyperparameter of the other): collapsing both would condition each on the other's placeholder`
    }
    if (reason === undefined) chosen.push(r)
    else refused.push({ site, reason })
  }
  const entries: CollapseEntry[] = chosen.map((r) => {
    const site = r.sites[0] as string
    const node = spec.nodes.find((n) => n.name === site)
    const removes = elementsOf(spec, site, sizeOf)
    return {
      site,
      children: r.links.map((l) => l.node),
      pair: r.pairs.join(', '),
      posterior: r.posterior,
      given: r.given,
      dims: node === undefined ? [] : canonicalDims(spec, node.dims),
      ...(removes === undefined ? {} : { removes }),
      sumsOver: r.sumsOver,
    }
  })
  const collapsed = new Set(chosen.map((r) => r.sites[0] as string))
  const remaining = a.continuous.filter((s) => !collapsed.has(s))
  const sizes = a.continuous.map((s) => elementsOf(spec, s, sizeOf))
  const before = sizes.every((k) => k !== undefined) ? (sizes as number[]).reduce((x, y) => x + y, 0) : undefined
  const removed = entries.map((e) => e.removes)
  const after =
    before !== undefined && removed.every((k) => k !== undefined)
      ? before - (removed as number[]).reduce((x, y) => x + y, 0)
      : undefined
  const lines: string[] =
    entries.length === 0
      ? [`Collapse plan for ${a.name}: no site can be integrated out by the handler.`]
      : [
          `Collapse plan for ${a.name}: integrate out ${entries.map((e) => `\`${e.site}\``).join(', ')}${before === undefined ? '' : `, leaving ${after} of ${before} unconstrained coordinates (${remaining.map((s) => `\`${s}\``).join(', ') || 'none'}) for a sampler`}.`,
          ...entries.map(
            (e) =>
              `  - ${e.site}: ${e.pair} with ${e.children.join(', ')}${e.sumsOver.length > 0 ? `, summed over ${e.sumsOver.join(', ')}` : ''}${e.given.length > 0 ? `; the marginal likelihood depends on ${e.given.join(', ')}` : ''}${e.removes === undefined ? '' : `; removes ${e.removes} coordinate${e.removes === 1 ? '' : 's'}`}. Posterior (recover draws it): ${e.posterior}.`,
          ),
        ]
  for (const r of refused) lines.push(`  - not ${r.site}: ${r.reason}.`)
  return {
    sites: entries.map((e) => e.site),
    entries,
    refused,
    remaining,
    ...(before === undefined ? {} : { dimensionBefore: before }),
    ...(after === undefined ? {} : { dimensionAfter: after }),
    text: lines.join('\n'),
  }
}

/**
 * The collapsed program for a plan: the bound model compiled to its program and the handler `collapse` applied to the
 * plan's sites. Run any method on `.model` (a sampler on the remaining sites, a Laplace fit, importance sampling) and draw
 * the removed sites back from their closed-form posteriors with `.recover(values, undefined, key)`.
 *
 * @param bound - The bound model the plan was made for.
 * @param plan - The plan (edit `sites` to collapse fewer).
 * @param options - The key of the prototype run the handler uses to find the structure.
 * @returns The collapsed program with `posteriors` and `recover`.
 * @throws OmaiError with code `value` when the plan has no site, or the handler refuses one (its message says why).
 * @concept collapsed-inference
 * @index function applyCollapse: the collapsed program of a collapsePlan, through the collapse handler.
 * @example
 * ```ts
 * import { applyCollapse, collapsePlan, defineModel } from 'omai-engine/probprog/model'
 * import { normal } from 'omai-engine/probprog/model/dists'
 * import { logJoint } from 'omai-engine/probprog'
 * import { item } from 'omai-engine/tensor'
 * const m = defineModel('m', (b) => {
 *   const mu = b.latent('mu', normal(0, 2))
 *   b.observe('y', normal(mu, 1), b.data('y', { dims: ['n'] }))
 * })
 * const bound = m.bind({ y: [1.2, 0.4, 2.1] })
 * const c = applyCollapse(bound, collapsePlan(bound))
 * Number.isFinite(item(logJoint(c.model, undefined).logJoint)) // => true
 * ```
 */
export function applyCollapse(
  bound: BoundModel,
  plan: CollapsePlan,
  options: { readonly key?: Key } = {},
): Collapsed<undefined> {
  if (plan.sites.length === 0)
    throw new OmaiError(
      'value',
      `applyCollapse: the plan has no site to collapse (${plan.refused.map((r) => `${r.site}: ${r.reason}`).join('; ') || 'no conjugate site'})`,
    )
  return collapse(toProgram(bound), undefined, {
    sites: plan.sites,
    ...(options.key === undefined ? {} : { key: options.key }),
  })
}
