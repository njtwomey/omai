/**
 * Structured and hierarchical guides, after Pyro's `AutoStructured` and `AutoGaussian` and TFP's structured surrogate
 * posteriors: a Gaussian over the flat unconstrained latent vector that follows the model's graph. Sites are grouped
 * into blocks (by default each site is one, a plated site being independent across its elements); a block has a full
 * covariance or a diagonal one; and a block's mean is a learned linear function of the blocks before it that it depends
 * on in the model (its parents, or its whole Markov blanket, or every earlier block). With no dependencies it is mean-field
 * across blocks with full covariance within them; with every dependency it is the full-rank guide; in between it keeps
 * the correlations the model's structure puts in the posterior (a hierarchy's scales with its group effects, explaining
 * away between causes of one effect) and not the rest, so it has far fewer parameters than the full-rank guide.
 */

import { OmaiError } from '../base/index.ts'
import { type Distribution, multivariateNormal, type Support } from '../dists/index.ts'
import { cholesky, solve, solveTriangular } from '../numerics/linalg/index.ts'
import {
  add,
  asTensor,
  diagonal,
  eye,
  fromStorage,
  matmul,
  maximum,
  mul,
  primalOf,
  sqrt,
  sub,
  type Tensor,
  type TensorLike,
  toFlat,
  transpose,
  type Value,
  zeros,
} from '../tensor/index.ts'
import { isolated } from './_stack.ts'
import { analyse, latentDependencies } from './analysis.ts'
import { type AutoGuide, type AutoGuideOptions, cachedGuideSites, type GuideSite, jointGuide } from './guides.ts'
import { type Model, param } from './primitives.ts'

/** What a block's mean may depend on: nothing, its parents, its parents and co-parents, or every earlier block. */
export type StructuredDependencies = 'none' | 'parents' | 'markov' | 'full'

/** Options of {@link autoStructured}. */
export interface AutoStructuredOptions extends AutoGuideOptions {
  /**
   * What a block depends on among the blocks before it: `none` (mean-field across blocks), `parents` (the model's
   * latent parents of its sites), `markov` (default: the parents, and the sites that share a child with them, observed
   * children included: the neighbours in the moral graph of the model, which is the structure of a Gaussian posterior's
   * precision) or `full` (every earlier block).
   */
  readonly dependencies?: StructuredDependencies
  /**
   * The covariance inside a block: `diag`, `full`, or per site. The default is `full` for a site with no plate
   * (a vector of coefficients) and `diag` for a site under a plate (independent elements given what it depends on).
   */
  readonly within?: 'diag' | 'full' | Readonly<Record<string, 'diag' | 'full'>>
  /** Sites to put in one block, each list adjacent in the model's order; the other sites are blocks of their own. */
  readonly blocks?: readonly (readonly string[])[]
}

/** The structure of a structured guide: its blocks, their covariance and what each depends on. */
export interface GuideStructure {
  readonly blocks: readonly {
    readonly sites: readonly string[]
    readonly within: 'diag' | 'full'
    /** Indices of the earlier blocks this block's mean is a linear function of. */
    readonly parents: readonly number[]
    /** The block's number of unconstrained elements. */
    readonly size: number
  }[]
  /** The flat unconstrained dimension. */
  readonly size: number
  /** Learnable numbers in the dependency and covariance factors (against D (D + 1) / 2 + D for the full-rank guide). */
  readonly parameters: number
}

interface Masks {
  readonly structure: GuideStructure
  /** D x D, 1 where an element's mean depends on an earlier element. */
  readonly dep: Tensor
  /** D x D, 1 where the covariance factor may be non-zero (lower triangle inside a block). */
  readonly lam: Tensor
  /** Each block's span in the flat vector. */
  readonly spans: readonly { readonly start: number; readonly end: number }[]
}

function masksOf(
  sites: readonly GuideSite[],
  analysis: ReturnType<typeof analyse>,
  options: AutoStructuredOptions,
): Masks {
  const dependencies = options.dependencies ?? 'markov'
  const byName = new Map(sites.map((s) => [s.name, s]))
  // Blocks: given lists of adjacent sites, then singletons, in the order of the sites.
  const groups = options.blocks ?? []
  const blockOf = new Map<string, number>()
  const members: string[][] = []
  const order = sites.map((s) => s.name)
  for (const g of groups) {
    const idx = g.map((n) => order.indexOf(n))
    if (idx.some((i) => i < 0))
      throw new OmaiError('value', `autoStructured: blocks name a site the guide does not cover (${g.join(', ')})`)
    const sorted = [...idx].sort((a, b) => a - b)
    sorted.forEach((i, k) => {
      if (k > 0 && i !== (sorted[k - 1] as number) + 1)
        throw new OmaiError(
          'value',
          `autoStructured: a block's sites must be adjacent in the model's order (${g.join(', ')})`,
        )
    })
  }
  for (const name of order) {
    if (blockOf.has(name)) continue
    const group = groups.find((g) => g.includes(name))
    const names = group === undefined ? [name] : order.filter((n) => group.includes(n))
    for (const n of names) blockOf.set(n, members.length)
    members.push(names)
  }
  // Blocks must be in the flat order too (a block is a run of adjacent sites).
  const withinOf = (names: readonly string[]): 'diag' | 'full' => {
    const w = options.within
    if (typeof w === 'string') return w
    const first = names[0] as string
    if (names.length > 1) return w?.[first] ?? 'full'
    const s = byName.get(first) as GuideSite
    return w?.[first] ?? (s.plates.length > 0 ? 'diag' : 'full')
  }
  // The model's latent edges: parent to child, and co-parents of a common child (the moral graph).
  const deps = latentDependencies(analysis.sites)
  const isGuide = new Set(order)
  const parents = new Map<string, Set<string>>()
  const neighbours = new Map<string, Set<string>>()
  for (const n of order) {
    parents.set(n, new Set())
    neighbours.set(n, new Set())
  }
  for (const s of analysis.sites) {
    if (s.role === 'param' || s.role === 'deterministic') continue
    const ps = [...(deps.get(s.name) ?? [])].filter((p) => isGuide.has(p) && p !== s.name)
    if (isGuide.has(s.name))
      for (const p of ps) {
        parents.get(s.name)?.add(p)
        neighbours.get(s.name)?.add(p)
        neighbours.get(p)?.add(s.name)
      }
    for (let i = 0; i < ps.length; i++)
      for (let j = i + 1; j < ps.length; j++) {
        neighbours.get(ps[i] as string)?.add(ps[j] as string)
        neighbours.get(ps[j] as string)?.add(ps[i] as string)
      }
  }
  const blocks = members.map((names, b) => {
    const wanted = new Set<number>()
    if (dependencies === 'full') for (let c = 0; c < b; c++) wanted.add(c)
    else if (dependencies !== 'none')
      for (const n of names)
        for (const p of (dependencies === 'parents' ? parents : neighbours).get(n) ?? []) {
          const c = blockOf.get(p) as number
          if (c < b) wanted.add(c)
        }
    const size = names.reduce((a, n) => a + (byName.get(n)?.size ?? 0), 0)
    return { sites: names, within: withinOf(names), parents: [...wanted].sort((a, b2) => a - b2), size }
  })
  const D = sites.reduce((a, s) => a + s.size, 0)
  const dep = new Float64Array(D * D)
  const lam = new Float64Array(D * D)
  const spans = blocks.map((b) => {
    const first = byName.get(b.sites[0] as string) as GuideSite
    return { start: first.offset, end: first.offset + b.size }
  })
  let parameters = 0
  blocks.forEach((b, bi) => {
    const { start, end } = spans[bi] as { start: number; end: number }
    for (let i = start; i < end; i++) {
      for (let j = start; j <= i; j++) if (b.within === 'full' || i === j) lam[i * D + j] = 1
      for (const c of b.parents) {
        const sp = spans[c] as { start: number; end: number }
        for (let j = sp.start; j < sp.end; j++) dep[i * D + j] = 1
      }
    }
    parameters += b.within === 'full' ? (b.size * (b.size + 1)) / 2 : b.size
    parameters += b.parents.reduce(
      (a, c) =>
        a +
        b.size *
          ((spans[c] as { start: number; end: number }).end - (spans[c] as { start: number; end: number }).start),
      0,
    )
  })
  return {
    structure: { blocks, size: D, parameters },
    dep: fromStorage(dep, [D, D], 'float64'),
    lam: fromStorage(lam, [D, D], 'float64'),
    spans,
  }
}

/** The covariance factor L = (I - B)^-1 Lambda of the sequential linear-Gaussian conditionals, from the parameters. */
function factor(dep: Value, scaleTril: Value, masks: Masks): Value {
  const D = masks.structure.size
  const unit = sub(eye(D), mul(dep, masks.dep))
  return solveTriangular(unit, mul(scaleTril, masks.lam), { lower: true, unitDiagonal: true })
}

const asValue = (v: TensorLike | undefined, name: string): Value => {
  if (v === undefined) throw new OmaiError('value', `autoStructured: no parameter ${name}`)
  return primalOf(asTensor(v))
}

/**
 * The structure of a structured guide on a model: its blocks, what each depends on and how many numbers it learns.
 *
 * @param model - The model.
 * @param data - Its data.
 * @param options - The guide's options.
 * @returns The structure.
 * @throws OmaiError with code `value` for sites in a subsampled plate, or a block that names a missing site or sites that
 *   are not adjacent.
 * @concept variational-inference
 * @example
 * ```ts
 * import { guideStructure, observe, plate, sample } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * const model = () => {
 *   const mu = sample('mu', normal(0, 2))
 *   plate('j', 3, () => { const th = sample('theta', normal(mu, 1)); observe('y', normal(th, 1), [0.1, 0.4, -0.2]) })
 * }
 * guideStructure(model, undefined).blocks.map((b) => [b.sites[0], b.within, b.parents]) // => [['mu', 'full', []], ['theta', 'diag', [0]]]
 * ```
 */
export function guideStructure<D = void>(
  model: Model<D>,
  data: D,
  options: AutoStructuredOptions = {},
): GuideStructure {
  const sites = cachedGuideSites(model)(data)
  const sub0 = sites.find((s) => s.subsampled)
  if (sub0 !== undefined)
    throw new OmaiError('value', `autoStructured: site ${sub0.name} is in a subsampled plate; use autoNormal`)
  return isolated(() => masksOf(sites, analyse(model, data), options)).structure
}

/**
 * A structured Gaussian guide that follows the model's graph (the ideas of Pyro's `AutoStructured` and `AutoGaussian`;
 * NumPyro has no counterpart). The flat unconstrained latent vector is drawn block by block, in the model's order:
 * a block's mean is `loc` plus a learned linear map of the blocks it depends on, its noise has a full or diagonal
 * covariance, and every site is pushed onto its support. `dependencies` `none` is mean-field across blocks (with full
 * covariance within a block), `full` is the full-rank guide, and the default `markov` gives each block the blocks that
 * are its neighbours in the model's moral graph, which is exact for a Gaussian posterior whose graph is a tree, a chain,
 * a hierarchy or an explaining-away pair, and an approximation otherwise. Parameters: `auto_loc`, `auto_dep` (the linear
 * maps, masked by the structure) and `auto_scale_tril` (masked too).
 *
 * @param model - The model.
 * @param options - Dependencies, covariance inside blocks, blocks, initialisation and initial scale.
 * @returns The guide; `guide.structure(data)` is its {@link GuideStructure}.
 * @concept variational-inference
 * @index function Structured Gaussian guide following the model's graph: parents, Markov blanket or full.
 * @example
 * ```ts
 * import { autoStructured, observe, sample, seed, traceOf } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { key } from 'omai-engine/random'
 * import { add } from 'omai-engine/tensor'
 * const model = () => { const a = sample('a', normal(0, 1)); const b = sample('b', normal(0, 1)); observe('y', normal(add(a, b), 0.5), 1) }
 * const t = traceOf(seed(autoStructured(model), key(0)), undefined)
 * t.sites.get('auto_dep')?.value.shape // => [2, 2]
 * ```
 */
export function autoStructured<D = void>(
  model: Model<D>,
  options: AutoStructuredOptions = {},
): AutoGuide<D> & { structure(data: D): GuideStructure } {
  const sitesOf = cachedGuideSites(model)
  const s0 = options.initScale ?? 0.1
  let lastData: D | undefined
  let masks: Masks | undefined
  const masksFor = (data: D): Masks => {
    if (masks !== undefined && lastData === data) return masks
    const sites = sitesOf(data)
    const bad = sites.find((s) => s.subsampled)
    if (bad !== undefined)
      throw new OmaiError('value', `autoStructured: site ${bad.name} is in a subsampled plate; use autoNormal`)
    masks = isolated(() => masksOf(sites, analyse(model, data), options))
    lastData = data
    return masks
  }
  let current: Masks | undefined
  const inner = jointGuide(
    model,
    'autoStructured',
    {
      latent: (D, _sites, init): Distribution => {
        const m = current as Masks
        const loc = param('auto_loc', init)
        const dep = param('auto_dep', zeros([D, D]))
        const L = param('auto_scale_tril', mul(eye(D), s0), { support: { kind: 'lowerCholesky' } as Support })
        return multivariateNormal(loc, { scaleTril: factor(dep, L, m) })
      },
      fitted: (params, _D, data) => {
        const m = masksFor(data as D)
        return multivariateNormal(asValue(params['auto_loc'], 'auto_loc'), {
          scaleTril: factor(
            asValue(params['auto_dep'], 'auto_dep'),
            asValue(params['auto_scale_tril'], 'auto_scale_tril'),
            m,
          ),
        })
      },
      centre: (params) => asValue(params['auto_loc'], 'auto_loc'),
    },
    options,
  )
  const program = (data: D): unknown => {
    current = masksFor(data)
    return inner(data)
  }
  return Object.assign(program, {
    kind: inner.kind,
    model,
    sites: inner.sites,
    median: inner.median,
    posterior: inner.posterior,
    structure: (data: D): GuideStructure => masksFor(data).structure,
  })
}

/**
 * The parameters of a structured guide that reproduce a Gaussian as closely as its structure allows: the mean as `auto_loc`,
 * and for each block the regression of its coordinates on those of the blocks it depends on (`auto_dep`) with the
 * conditional covariance's factor (`auto_scale_tril`). When the Gaussian's own conditional independences are the guide's (a
 * hierarchy's, a chain's), the guide is then exactly that Gaussian; otherwise it is the nearest the structure gives
 * block by block. This is how a Laplace fit initialises the guide.
 *
 * @param model - The model.
 * @param data - Its data.
 * @param gaussian - The mean (flat, unconstrained) and the covariance.
 * @param options - The guide's options.
 * @returns The constrained parameter values by name.
 * @concept variational-inference
 * @example
 * ```ts
 * import { observe, sample, structuredInitialParams } from 'omai-engine/probprog'
 * import { normal } from 'omai-engine/dists'
 * import { tensor, toArray } from 'omai-engine/tensor'
 * const model = () => { const a = sample('a', normal(0, 1)); sample('b', normal(a, 1)) }
 * const p = structuredInitialParams(model, undefined, { mean: [0, 0], covariance: tensor([[1, 1], [1, 2]]) })
 * toArray(p['auto_dep'] as never) // => [[0, 0], [1, 0]]
 * ```
 */
export function structuredInitialParams<D = void>(
  model: Model<D>,
  data: D,
  gaussian: { readonly mean: ArrayLike<number>; readonly covariance: TensorLike },
  options: AutoStructuredOptions = {},
): Record<string, Tensor> {
  const sites = cachedGuideSites(model)(data)
  const m = isolated(() => masksOf(sites, analyse(model, data), options))
  const D = m.structure.size
  const cov = toFlat(primalOf(asTensor(gaussian.covariance)))
  const at = (i: number, j: number): number => cov[i * D + j] ?? 0
  const dep = new Float64Array(D * D)
  const scale = new Float64Array(D * D)
  const matrix = (rows: readonly number[], cols: readonly number[]): Tensor =>
    fromStorage(
      Float64Array.from(rows.flatMap((i) => cols.map((j) => at(i, j)))),
      [rows.length, cols.length],
      'float64',
    )
  m.structure.blocks.forEach((b, bi) => {
    const span = m.spans[bi] as { start: number; end: number }
    const rows = Array.from({ length: span.end - span.start }, (_, k) => span.start + k)
    const parents = b.parents.flatMap((c) => {
      const sp = m.spans[c] as { start: number; end: number }
      return Array.from({ length: sp.end - sp.start }, (_, k) => sp.start + k)
    })
    let conditional: Value = matrix(rows, rows)
    if (parents.length > 0) {
      const sbp = matrix(rows, parents)
      const spp = matrix(parents, parents)
      // B = S_bp S_pp^-1, through the symmetric solve S_pp B^T = S_pb.
      const bmat = transpose(solve(spp, transpose(sbp)))
      const flat = toFlat(primalOf(bmat))
      for (const [r, i] of rows.entries())
        for (const [c, j] of parents.entries()) dep[i * D + j] = flat[r * parents.length + c] ?? 0
      conditional = sub(conditional, matmul(bmat, transpose(sbp)))
    }
    const sym = mul(add(conditional, transpose(conditional)), 0.5)
    const n = rows.length
    const factorOfBlock = toFlat(
      primalOf(b.within === 'full' ? cholesky(sym) : mul(eye(n), sqrt(maximum(diagonal(sym), 1e-12)))),
    )
    for (const [r, i] of rows.entries())
      for (const [c, j] of rows.entries()) scale[i * D + j] = factorOfBlock[r * n + c] ?? 0
  })
  return {
    auto_loc: fromStorage(Float64Array.from(gaussian.mean), [D], 'float64'),
    auto_dep: fromStorage(dep, [D, D], 'float64'),
    auto_scale_tril: fromStorage(scale, [D, D], 'float64'),
  }
}
