/**
 * Gibbs sampling on probabilistic programs, which updates one site at a time from its conditional distribution given
 * the rest: exact conditionals for discrete sites by enumeration, conjugate updates where available, and collapsed
 * updates for Dirichlet and beta sites. A traced `gibbsAlgorithm`, the `gibbs` function and the engine for the planner.
 */

import { type Json, logSumExpOf, OmaiError } from '../../base/index.ts'
import { bijectors, CONJUGATES, type ConjugatePair, dirichlet } from '../../dists/index.ts'
import { categorical as drawCategorical, foldIn, type Key } from '../../random/index.ts'
import { numericTarget, sliceSampler } from '../../sampling/index.ts'
import {
  asTensor,
  broadcastTo,
  cast,
  concat,
  fromStorage,
  item,
  mul,
  primalOf,
  reshape,
  sum,
  type Tensor,
  type TensorLike,
  take,
  toFlat,
  type Value,
} from '../../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../../trace/index.ts'
import { enumeratedMarginals, traceFactors } from '../_contract.ts'
import { unravel } from '../_discrete.ts'
import { unexpanded } from '../_expand.ts'
import { analyse, type ConjugateStructure, type ProgramAnalysis, pairParameter } from '../analysis.ts'
import { logJoint } from '../density.ts'
import { condition, enumerate, type ProgramTrace, type SampleSite, seed, traceOf } from '../handlers.ts'
import { type Posterior, samplePosterior } from '../posterior.ts'
import type { Model } from '../primitives.ts'
import { conjugateData } from './conjugate.ts'
import { type EngineRunOptions, feasibility, type InferenceEngine } from './engine.ts'

/** How a collapsed site's children use it: its batch elements directly (broadcast over plates) or indexed by a site. */
interface Collapsed {
  readonly name: string
  /** The prior concentrations by group `[groups][K]` (beta priors become two categories: 0 and 1). */
  readonly alpha: readonly (readonly number[])[]
  readonly groupShape: readonly number[]
  readonly k: number
  readonly beta: boolean
  readonly children: readonly { readonly site: string; readonly gather: string | undefined }[]
}

/** A Gibbs block: how one latent site is updated. */
type Block =
  | { readonly kind: 'discrete'; readonly site: string }
  | { readonly kind: 'conjugate'; readonly site: string; readonly structure: ConjugateStructure }
  | {
      readonly kind: 'grouped'
      readonly site: string
      readonly pair: ConjugatePair
      readonly children: readonly { readonly site: string; readonly gather: string }[]
    }
  | { readonly kind: 'collapsed'; readonly site: string; readonly collapsed: Collapsed }
  | { readonly kind: 'slice'; readonly site: string }

/** The state of {@link gibbsAlgorithm}: the current value of every latent site. */
export interface GibbsState {
  readonly values: Readonly<Record<string, Value>>
  readonly logJoint: number
}

/** Options of {@link gibbsAlgorithm}. */
export interface GibbsOptions {
  /** Integrate Dirichlet and beta sites out of categorical and Bernoulli children (default true). */
  readonly collapse?: boolean
  /** Initial values by site (constrained); others start at prior draws. */
  readonly init?: Readonly<Record<string, TensorLike>>
  /** The slice width for continuous sites without a conjugate conditional (default 1, unconstrained scale). */
  readonly sliceWidth?: number
}

const product = (shape: readonly number[]): number => shape.reduce((a, b) => a * b, 1)

function siteOf(t: ProgramTrace, name: string): SampleSite | undefined {
  const s = t.sites.get(name)
  return s?.type === 'sample' ? s : undefined
}

function close(a: Value, b: Value): boolean {
  try {
    const x = toFlat(primalOf(a))
    const y = toFlat(primalOf(broadcastTo(b, a.shape)))
    return x.length === y.length && x.every((v, i) => Math.abs(v - (y[i] ?? Number.NaN)) <= 1e-10 * (1 + Math.abs(v)))
  } catch {
    return false
  }
}

/** The group (flat index into the collapsed site's batch) of each element of a child, for an identity mapping. */
function identityGroups(childBatch: readonly number[], groupShape: readonly number[]): Int32Array {
  const n = product(childBatch)
  const out = new Int32Array(n)
  const off = childBatch.length - groupShape.length
  for (let i = 0; i < n; i++) {
    const idx = unravel(i, childBatch)
    let g = 0
    for (let a = 0; a < groupShape.length; a++)
      g = g * (groupShape[a] ?? 1) + ((groupShape[a] ?? 1) === 1 ? 0 : (idx[off + a] ?? 0))
    out[i] = g
  }
  return out
}

/** Recognise how each latent site can be updated (conditionals at the prototype values). */
function plan<D = void>(model: Model<D>, data: D, a: ProgramAnalysis, collapse: boolean): Block[] {
  const proto = a.prototype
  const values: Record<string, Value> = {}
  for (const n of a.latent) {
    const s = siteOf(proto, n)
    if (s !== undefined) values[n] = s.value
  }
  const blocks: Block[] = []
  const collapsedNames = new Set<string>()
  // Collapsible Dirichlet and beta sites.
  if (collapse)
    for (const name of a.continuous) {
      const s = siteOf(proto, name)
      if (s === undefined) continue
      const fam = unexpanded(s.fn).name
      if (fam !== 'dirichlet' && fam !== 'beta') continue
      if (a.sites.find((x) => x.name === name)?.parents.some((p) => a.latent.includes(p)) === true) continue
      const kids = a.edges.filter((e) => e.from === name).map((e) => e.to)
      if (kids.length === 0) continue
      const children: { site: string; gather: string | undefined }[] = []
      let ok = true
      for (const c of kids) {
        const cs = siteOf(proto, c)
        const cf = cs === undefined ? '' : unexpanded(cs.fn).name
        if (cs === undefined || (fam === 'dirichlet' ? cf !== 'categorical' : cf !== 'bernoulli')) {
          ok = false
          break
        }
        const probs = unexpanded(cs.fn).params['probs']
        if (probs === undefined) {
          ok = false
          break
        }
        const eventless = s.fn.eventShape.length
        const thetaBatch = s.value.shape.slice(0, s.value.shape.length - eventless)
        if (close(probs, s.value) && thetaBatch.length <= cs.fn.batchShape.length) {
          children.push({ site: c, gather: undefined })
          continue
        }
        const parentsOfChild = a.sites.find((x) => x.name === c)?.parents ?? []
        const gather = a.discrete.sites.find((g) => {
          const gs = siteOf(proto, g)
          if (gs === undefined || g === c || !parentsOfChild.includes(g)) return false
          if (gs.value.shape.join() !== cs.value.shape.join()) return false
          return (s.value.shape[0] ?? -1) > 0 && close(probs, take(s.value, cast(gs.value, 'int32'), 0))
        })
        if (gather === undefined) {
          ok = false
          break
        }
        children.push({ site: c, gather })
      }
      if (!ok) continue
      const alphaParam = fam === 'dirichlet' ? unexpanded(s.fn).params['concentration'] : undefined
      const shape = fam === 'dirichlet' ? s.value.shape.slice(0, -1) : s.value.shape
      const k = fam === 'dirichlet' ? (s.value.shape[s.value.shape.length - 1] ?? 0) : 2
      const groups = product(shape)
      let alpha: number[][]
      if (fam === 'dirichlet' && alphaParam !== undefined) {
        const full = toFlat(primalOf(broadcastTo(alphaParam, [...shape, k])))
        alpha = Array.from({ length: groups }, (_, g) => full.slice(g * k, (g + 1) * k))
      } else {
        const base = unexpanded(s.fn)
        const av = toFlat(primalOf(broadcastTo(base.params['a'] ?? asTensor(1), shape)))
        const bv = toFlat(primalOf(broadcastTo(base.params['b'] ?? asTensor(1), shape)))
        alpha = Array.from({ length: groups }, (_, g) => [bv[g] ?? 1, av[g] ?? 1])
      }
      collapsedNames.add(name)
      blocks.push({
        kind: 'collapsed',
        site: name,
        collapsed: { name, alpha, groupShape: shape, k, beta: fam === 'beta', children },
      })
    }
  for (const name of a.latent) {
    if (collapsedNames.has(name)) continue
    const info = a.sites.find((x) => x.name === name)
    if (info?.discrete === true) {
      blocks.push({ kind: 'discrete', site: name })
      continue
    }
    const others: Record<string, Value> = {}
    for (const [k, v] of Object.entries(values)) if (k !== name) others[k] = v
    const cond = analyse(condition(model, others), data)
    const structure = cond.conjugate.find((c) => c.prior === name)
    if (structure !== undefined && cond.latent.length === 1) {
      blocks.push({ kind: 'conjugate', site: name, structure })
      continue
    }
    const grouped = groupedConjugacy(proto, name, a)
    if (grouped !== undefined) {
      blocks.push(grouped)
      continue
    }
    blocks.push({ kind: 'slice', site: name })
  }
  // Collapsed sites are drawn last, given their children.
  return [...blocks.filter((b) => b.kind !== 'collapsed'), ...blocks.filter((b) => b.kind === 'collapsed')]
}

/** A conjugate site whose children use one component each, chosen by a discrete site (mixture components). */
function groupedConjugacy(proto: ProgramTrace, name: string, a: ProgramAnalysis): Block | undefined {
  const s = siteOf(proto, name)
  if (s === undefined) return undefined
  const kids = a.edges.filter((e) => e.from === name).map((e) => e.to)
  if (kids.length === 0) return undefined
  const priorFamily = unexpanded(s.fn).name
  let pair: ConjugatePair | undefined
  const children: { site: string; gather: string }[] = []
  for (const c of kids) {
    const cs = siteOf(proto, c)
    if (cs === undefined || !cs.isObserved) return undefined
    const family = unexpanded(cs.fn).name
    const parentsOfChild = a.sites.find((x) => x.name === c)?.parents ?? []
    const found = CONJUGATES.filter((p) => p.likelihood === family && p.prior === priorFamily).find((p) => {
      const v = pairParameter(unexpanded(cs.fn), p)
      if (v === undefined) return false
      return a.discrete.sites.some((g) => {
        if (!parentsOfChild.includes(g)) return false
        const gs = siteOf(proto, g)
        return gs !== undefined && close(v, take(s.value, cast(gs.value, 'int32'), 0))
      })
    })
    if (found === undefined || (pair !== undefined && found !== pair)) return undefined
    pair = found
    const v = pairParameter(unexpanded(cs.fn), found) as Value
    const gather = a.discrete.sites.find((g) => {
      const gs = siteOf(proto, g)
      return parentsOfChild.includes(g) && gs !== undefined && close(v, take(s.value, cast(gs.value, 'int32'), 0))
    })
    if (gather === undefined) return undefined
    children.push({ site: c, gather })
  }
  return pair === undefined ? undefined : { kind: 'grouped', site: name, pair, children }
}

/** Counts `[groups][K]` of a collapsed site's children, and each child element's group and state. */
function collapsedCounts(
  c: Collapsed,
  t: ProgramTrace,
): { readonly counts: number[][]; readonly members: { readonly groups: Int32Array; readonly states: Int32Array }[] } {
  const counts = c.alpha.map(() => new Array<number>(c.k).fill(0))
  const members = c.children.map((child) => {
    const cs = siteOf(t, child.site)
    if (cs === undefined) throw new OmaiError('value', `gibbs: no site ${child.site}`)
    const states = Int32Array.from(toFlat(primalOf(cs.value)))
    const groups =
      child.gather === undefined
        ? identityGroups(cs.value.shape, c.groupShape)
        : Int32Array.from(toFlat(primalOf(siteOf(t, child.gather)?.value ?? asTensor([]))))
    for (let i = 0; i < states.length; i++) {
      const row = counts[groups[i] ?? 0]
      if (row !== undefined) row[states[i] ?? 0] = (row[states[i] ?? 0] ?? 0) + 1
    }
    return { groups, states }
  })
  return { counts, members }
}

function draw(key: Key, logits: readonly number[]): number {
  const z = logSumExpOf(logits)
  return (
    toFlat(
      drawCategorical(
        key,
        logits.map((l) => Math.exp(l - z)),
        [1],
      ),
    )[0] ?? 0
  )
}

/**
 * Gibbs sampling on a program as a traced algorithm (one sweep per step): each discrete site is drawn from its exact
 * conditional by enumeration (elementwise for plated sites), each continuous site with a conjugate conditional
 * (`dists` conjugate pairs, given the current values of everything else) from its closed-form conditional, mixture
 * components (a conjugate site indexed by a discrete site) group by group, and other continuous sites by a slice
 * step on the unconstrained scale. Dirichlet and beta sites whose children are categorical or Bernoulli are
 * integrated out (collapsed Gibbs, as for topic models): their children are updated element by element from the
 * Dirichlet-multinomial predictive, and the collapsed sites are drawn from their conditional after each sweep.
 *
 * @param model - The program.
 * @param data - Its data.
 * @param options - Collapsing, initial values and the slice width.
 * @returns The algorithm.
 * @concept gibbs-sampling
 * @example
 * ```ts
 * import { gibbsAlgorithm, observe, plate, sample } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const model = () => { const m = sample('m', normal(0, 1)); plate('n', 3, () => observe('y', normal(m, 1), [1, 2, 3])) }
 * runAlgorithm(gibbsAlgorithm(model, undefined), { seed: 0, maxSteps: 5 }).steps // => 5
 * ```
 */
export function gibbsAlgorithm<D = void>(model: Model<D>, data: D, options: GibbsOptions = {}): Algorithm<GibbsState> {
  let blocks: Block[] = []
  const collapseOn = options.collapse ?? true
  const width = options.sliceWidth ?? 1
  const runAt = (values: Readonly<Record<string, Value>>, key?: Key): ProgramTrace =>
    traceOf(key === undefined ? condition(model, values) : seed(condition(model, values), key), data)
  return {
    name: 'gibbs',
    config: { collapse: collapseOn, sliceWidth: width } as Json,
    schema: channels({ metrics: { logJoint: { description: 'log density of the current state' } } }),
    init: (key) => {
      const a = analyse(model, data, { key: foldIn(key, 0) })
      if (a.discrete.unbounded.length > 0)
        throw new OmaiError('value', `gibbs: unbounded discrete sites (${a.discrete.unbounded.join(', ')})`)
      blocks = plan(model, data, a, collapseOn)
      const values: Record<string, Value> = {}
      for (const n of a.latent) {
        const s = siteOf(a.prototype, n)
        if (s !== undefined) values[n] = s.value
      }
      for (const [k, v] of Object.entries(options.init ?? {})) values[k] = primalOf(asTensor(v))
      return { values, logJoint: item(primalOf(logJoint(model, data, values).logJoint)) }
    },
    step: (state, ctx) => {
      let values: Record<string, Value> = { ...state.values }
      const collapsed = blocks.flatMap((b) => (b.kind === 'collapsed' ? [b.collapsed] : []))
      blocks.forEach((b, bi) => {
        const key = foldIn(ctx.key, bi)
        if (b.kind === 'discrete')
          values = { ...values, [b.site]: updateDiscrete(model, data, values, b.site, collapsed, key) }
        else if (b.kind === 'conjugate') {
          const others = { ...values }
          delete others[b.site]
          const t = runAt(others, foldIn(key, 1))
          const prior = siteOf(t, b.site)
          if (prior === undefined) return
          const { data: x, fixed } = conjugateData(t, b.structure)
          const post = b.structure.pair.posterior(prior.fn, x, fixed)
          values = { ...values, [b.site]: primalOf(post.sample(key, [])) }
        } else if (b.kind === 'grouped') {
          const t = runAt(values)
          const prior = siteOf(t, b.site)
          if (prior === undefined) return
          const k = prior.value.shape[0] ?? 0
          const parts: Value[] = []
          for (let j = 0; j < k; j++) {
            const rows: Value[] = []
            let fixed: Record<string, Value> = {}
            for (const ch of b.children) {
              const cs = siteOf(t, ch.site)
              const g = siteOf(t, ch.gather)
              if (cs === undefined || g === undefined) continue
              const gv = toFlat(primalOf(g.value))
              const yv = toFlat(primalOf(cs.value))
              const pick = gv.flatMap((z, i) => (z === j ? [yv[i] ?? 0] : []))
              rows.push(fromStorage(Float64Array.from(pick), [pick.length], 'float64'))
              const f: Record<string, Value> = {}
              for (const name of b.pair.fixed) {
                const pv = unexpanded(cs.fn).params[name]
                if (pv !== undefined) f[name] = primalOf(pv)
              }
              fixed = f
            }
            const post = b.pair.posterior(prior.fn, concat(rows), fixed)
            // Component j's posterior uses its own data; parameters shared by the components broadcast.
            const drawn = primalOf(broadcastTo(post.sample(foldIn(key, j), []), prior.value.shape))
            parts.push(reshape(take(drawn, [j], 0), prior.value.shape.slice(1)))
          }
          values = {
            ...values,
            [b.site]: primalOf(reshape(concat(parts.map((p) => reshape(p, [1, ...p.shape]))), prior.value.shape)),
          }
        } else if (b.kind === 'slice') {
          const t = runAt(values)
          const s = siteOf(t, b.site)
          if (s === undefined) return
          const bij = bijectors.bijectorForSupport(s.fn.support)
          const u0 = primalOf(bij.inverse(values[b.site] as Value))
          const shape = u0.shape
          const dim = Math.max(1, product(shape))
          const others = { ...values }
          const target = numericTarget((u) => {
            const uu = fromStorage(Float64Array.from(u), shape, 'float64')
            const x = bij.forward(uu)
            const lj = item(primalOf(logJoint(model, data, { ...others, [b.site]: x }).logJoint))
            return lj + item(primalOf(sum(bij.forwardLogDetJacobian(uu))))
          }, dim)
          const kernel = sliceSampler(target, { width })
          const st = kernel.step(key, kernel.init(toFlat(u0)))
          values = {
            ...values,
            [b.site]: primalOf(bij.forward(fromStorage(Float64Array.from(st.state.position), shape, 'float64'))),
          }
        } else {
          // A collapsed site: draw it from its Dirichlet (or beta) conditional given the children.
          const c = b.collapsed
          const t = runAt(values)
          const { counts } = collapsedCounts(c, t)
          const conc = c.alpha.map((row, g) => row.map((x, j) => x + (counts[g]?.[j] ?? 0)))
          const d = dirichlet(fromStorage(Float64Array.from(conc.flat()), [...c.groupShape, c.k], 'float64'))
          const p = primalOf(d.sample(key, []))
          values = {
            ...values,
            [c.name]: c.beta ? primalOf(reshape(take(reshape(p, [-1, 2]), [1], 1), c.groupShape)) : p,
          }
        }
      })
      const lj = item(primalOf(logJoint(model, data, values).logJoint))
      return { values, logJoint: lj }
    },
    done: (s) => (Number.isNaN(s.logJoint) ? 'failed' : undefined),
    observe: (s) => ({ metrics: Number.isFinite(s.logJoint) ? { logJoint: s.logJoint } : {} }),
  }
}

/** Draw a discrete site from its conditional, with collapsed predictive terms element by element where needed. */
function updateDiscrete<D = void>(
  model: Model<D>,
  data: D,
  values: Readonly<Record<string, Value>>,
  site: string,
  collapsed: readonly Collapsed[],
  key: Key,
): Value {
  const others: Record<string, Value> = { ...values }
  delete others[site]
  const r = logJoint(enumerate(model, { sites: [site] }), data, others)
  const t = r.trace
  const own = siteOf(t, site)
  if (own === undefined) throw new OmaiError('value', `gibbs: no site ${site}`)
  const depth =
    t.sites.size === 0
      ? 0
      : [...t.sites.values()].reduce((m, s) => s.plates.reduce((x, f) => Math.max(x, -f.dim), m), 0)
  const all = traceFactors(t, depth)
  const relevant = collapsed.filter((c) => c.children.some((ch) => ch.site === site || ch.gather === site))
  const keep = all.factors.filter((f) => f.site === undefined || !relevantSite(relevant, f.site))
  // The site's own factor may be collapsed away: a zero factor over its dims carries its marginal instead.
  const own0 = all.factors.find((f) => f.site === site)
  if (own0 !== undefined && !keep.includes(own0)) keep.push({ ...own0, value: mul(own0.value, 0) })
  const { marginals } = enumeratedMarginals(keep, all.sumVars, all.scales)
  const support = own.fn.support
  const low = support.kind === 'integerInterval' ? support.low : 0
  const K = support.kind === 'boolean' ? 2 : support.kind === 'integerInterval' ? support.high - support.low + 1 : 0
  const current = values[site]
  const shape = current?.shape ?? []
  const n = product(shape)
  const mg = marginals.get(site)
  const baseProbs = mg === undefined ? undefined : toFlat(primalOf(mg.probs))
  const base = (i: number, k: number): number => {
    if (baseProbs === undefined) return 0
    const v = baseProbs.length === K ? baseProbs[k] : baseProbs[i * K + k]
    return Math.log(Math.max(v ?? 0, 0))
  }
  if (relevant.length === 0) {
    const out = new Float64Array(n)
    for (let i = 0; i < n; i++)
      out[i] =
        low +
        draw(
          foldIn(key, i),
          Array.from({ length: K }, (_, k) => base(i, k)),
        )
    return fromStorage(out, shape, 'float64')
  }
  // Collapsed: sequential updates with Dirichlet-multinomial predictive terms.
  const t0 = traceOf(condition(model, values), data)
  const states = Int32Array.from(toFlat(primalOf(current ?? asTensor(new Array(n).fill(low)))), (v) => v - low)
  const info = relevant.map((c) => ({ c, ...collapsedCounts(c, t0) }))
  for (let i = 0; i < n; i++) {
    const logits = Array.from({ length: K }, (_, k) => base(i, k))
    for (const { c, counts, members } of info) {
      c.children.forEach((ch, j) => {
        const m = members[j]
        if (m === undefined) return
        if (ch.site === site) {
          // This site is a child: remove its count, then the predictive of each state.
          const g = m.groups[i] ?? 0
          const row = counts[g] as number[]
          row[m.states[i] ?? 0] = (row[m.states[i] ?? 0] ?? 0) - 1
          const a = c.alpha[g] ?? []
          const total = row.reduce((s, v) => s + v, 0) + a.reduce((s, v) => s + v, 0)
          for (let k = 0; k < K; k++) logits[k] = (logits[k] ?? 0) + Math.log(((a[k] ?? 0) + (row[k] ?? 0)) / total)
        } else if (ch.gather === site) {
          // This site picks the group of a child element: remove it from its group, then each group's predictive.
          const old = m.groups[i] ?? 0
          const y = m.states[i] ?? 0
          const rowOld = counts[old] as number[]
          rowOld[y] = (rowOld[y] ?? 0) - 1
          for (let k = 0; k < K; k++) {
            const row = counts[k] ?? []
            const a = c.alpha[k] ?? []
            const total = row.reduce((s, v) => s + v, 0) + a.reduce((s, v) => s + v, 0)
            logits[k] = (logits[k] ?? 0) + Math.log(((a[y] ?? 0) + (row[y] ?? 0)) / total)
          }
        }
      })
    }
    const k = draw(foldIn(key, i), logits)
    states[i] = k
    for (const { c, counts, members } of info) {
      c.children.forEach((ch, j) => {
        const m = members[j]
        if (m === undefined) return
        if (ch.site === site) {
          const row = counts[m.groups[i] ?? 0] as number[]
          row[k] = (row[k] ?? 0) + 1
          m.states[i] = k
        } else if (ch.gather === site) {
          const row = counts[k] as number[]
          const y = m.states[i] ?? 0
          row[y] = (row[y] ?? 0) + 1
          m.groups[i] = k
        }
      })
    }
  }
  return fromStorage(
    Float64Array.from(states, (v) => v + low),
    shape,
    'float64',
  )
}

function relevantSite(relevant: readonly Collapsed[], site: string): boolean {
  return relevant.some((c) => c.name === site || c.children.some((ch) => ch.site === site))
}

/** Options of {@link gibbs}. */
export interface GibbsRunOptions extends GibbsOptions {
  readonly numDraws?: number
  readonly numWarmup?: number
  readonly thin?: number
}

/**
 * Run Gibbs sampling and return the draws after warm-up as a posterior (see {@link gibbsAlgorithm}).
 *
 * @param model - The program.
 * @param data - Its data.
 * @param options - Draws, warm-up, thinning, Gibbs options, key and tracing.
 * @returns The posterior of draws; `diagnostics.blocks` describes how each site was updated.
 * @concept gibbs-sampling
 * @example
 * ```ts
 * import { gibbs, observe, plate, sample } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * import { item } from 'omai-engine/tensor'
 * const model = () => { const m = sample('m', normal(0, 1)); plate('n', 3, () => observe('y', normal(m, 1), [1, 2, 3])) }
 * Math.abs(item(gibbs(model, undefined, { key: key(0), numDraws: 2000, numWarmup: 50 }).mean('m')) - 1.5) < 0.05 // => true
 * ```
 */
export function gibbs<D = void>(model: Model<D>, data: D, options: GibbsRunOptions & EngineRunOptions): Posterior {
  const numDraws = options.numDraws ?? 1000
  const numWarmup = options.numWarmup ?? 200
  const thin = options.thin ?? 1
  const alg = gibbsAlgorithm(model, data, options)
  const kept: Record<string, Value>[] = []
  let sweeps = 0
  const recording: Algorithm<GibbsState> = {
    ...alg,
    step: (s, ctx) => {
      const next = alg.step(s, ctx)
      sweeps += 1
      if (sweeps > numWarmup && (sweeps - numWarmup) % thin === 0) kept.push(next.values)
      return next
    },
  }
  const r = runAlgorithm(recording, {
    key: options.key,
    maxSteps: numWarmup + numDraws * thin,
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
    ...(options.parent === undefined ? {} : { parent: options.parent }),
  })
  return gibbsPosterior(kept, r.status, { sweeps: r.steps, kept: kept.length }, Object.keys(r.state.values))
}

/**
 * The posterior of Gibbs draws: the kept sweeps' values stacked by site into `[draw, ...shape]`, as one chain.
 *
 * @param kept - The values of every latent site at each kept sweep.
 * @param status - How the run ended.
 * @param diagnostics - What to report.
 * @param sites - The sites to stack (default the sites of the first kept sweep).
 * @returns The posterior of draws.
 * @concept gibbs-sampling
 * @example
 * ```ts
 * import { gibbsPosterior } from 'omai-engine/probprog'
 * import { item, tensor } from 'omai-engine/tensor'
 * item(gibbsPosterior([{ m: tensor(1) }, { m: tensor(3) }], 'completed', {}).mean('m')) // => 2
 * ```
 */
export function gibbsPosterior(
  kept: readonly Readonly<Record<string, Value>>[],
  status: Posterior['status'],
  diagnostics: Readonly<Record<string, Json>>,
  sites?: readonly string[],
): Posterior {
  const names = sites ?? Object.keys(kept[0] ?? {})
  const draws: Record<string, Tensor> = {}
  for (const name of names) {
    const first = kept[0]?.[name]
    const shape = first?.shape ?? []
    const rows = kept.flatMap((v) => toFlat(primalOf(v[name] ?? asTensor(0))))
    draws[name] = fromStorage(Float64Array.from(rows), [kept.length, ...shape], 'float64')
  }
  return samplePosterior(draws, { engine: 'gibbs', status, chainShape: [1, kept.length], diagnostics })
}

/**
 * The Gibbs engine: conditional updates for programs whose latent sites are discrete, conjugate given the rest, mixture
 * components, or collapsible Dirichlet/beta sites (cost: sweeps x latent elements x states).
 *
 * @param options - Draws, warm-up and Gibbs options.
 * @returns The engine.
 * @concept inference-engines
 * @example
 * ```ts
 * import { gibbsEngine } from 'omai-engine/probprog'
 * gibbsEngine().name // => 'gibbs'
 * ```
 */
export function gibbsEngine(options: GibbsRunOptions = {}): InferenceEngine {
  return {
    name: 'gibbs',
    form: 'samples',
    description: 'Gibbs sampling with discrete, conjugate and collapsed conditionals',
    applies: (a: ProgramAnalysis) => {
      if (!a.hasLikelihood) return feasibility(false, 'mcmc', 0, 'observed sites have no likelihood')
      if (a.latent.length === 0) return feasibility(false, 'mcmc', 0, 'no latent sites')
      if (a.discrete.unbounded.length > 0)
        return feasibility(false, 'mcmc', 0, `unbounded discrete sites (${a.discrete.unbounded.join(', ')})`)
      const conjugate = new Set(a.conjugate.map((c) => c.prior))
      const plainContinuous = a.continuous.filter((n) => !conjugate.has(n))
      const elements = a.sites.filter((s) => s.role === 'latent').reduce((s, x) => s + x.elements, 0)
      const k = Object.values(a.discrete.cardinalities).reduce((m, c) => Math.max(m, c), 1)
      const sweeps = (options.numWarmup ?? 200) + (options.numDraws ?? 1000)
      // Continuous sites without a recognised conjugate structure may still have conjugate conditionals (checked at
      // run time) or fall back to slice steps; charge for the fallback.
      const penalty = plainContinuous.length > 0 && a.discrete.sites.length === 0 ? 4 : 1
      return feasibility(
        a.discrete.sites.length > 0 || a.conjugate.length > 0,
        'mcmc',
        sweeps * Math.max(1, elements) * k * Math.max(1, a.observations) * penalty,
        a.discrete.sites.length > 0 || a.conjugate.length > 0
          ? 'discrete or conjugate conditionals'
          : 'no discrete or conjugate conditionals (use NUTS)',
      )
    },
    run: <D>(model: Model<D>, data: D, _a: ProgramAnalysis, run: EngineRunOptions) =>
      gibbs(model, data, { ...options, ...run }),
  }
}
