/**
 * Static analysis of a model spec, with nothing run: every node with its kind, dimensions, family and parents, the
 * dependency edges, the dimensions (plates) and the nodes over each, the conjugate prior and likelihood pairs among its
 * latents and observations (through direct references and `at` lookups, with the conditions that make a pair usable),
 * a conservative linear-Gaussian check, the nodes that are opaque to message passing (a `custom` call, a family with no
 * exponential-family form), and a list of features that an inference method's `check` reads. Unlike `analyse` of a
 * program, it sees every branch of the model, depends on no data and no run, and is a value that can be printed or sent.
 *
 * @index function analyseSpec: static dependency, plate, conjugacy and feature analysis of a ModelSpec.
 */

import { CONJUGATES, type ConjugatePair } from '../../dists/index.ts'
import type { ModelLike } from './bind.ts'
import { distributionFamilies, type FamilyDef } from './dists/families.ts'
import { type ExprSpec, type ModelNode, type ModelSpec, nodeRefs } from './spec.ts'

/** One node as the analysis sees it. */
export interface SpecNodeInfo {
  /** A unique id: the node's name, or `data:<name>` for a data node (an observed node may share its data's name). */
  readonly id: string
  readonly name: string
  readonly kind: ModelNode['kind']
  readonly dims: readonly string[]
  readonly eventRank: number
  /** The distribution family of a latent, observed or generated node. */
  readonly family?: string
  readonly discrete?: boolean
  readonly exponentialFamily?: boolean
  /** The ids of the nodes this one's distribution, value or expression refers to. */
  readonly parents: readonly string[]
  readonly children: readonly string[]
  /** The operators its expressions apply. */
  readonly operators: readonly string[]
  /** Whether message passing cannot see through it (a `custom` call, or a family with no exponential-family form). */
  readonly opaque: boolean
}

/** A dimension and what uses it. */
export interface SpecDimInfo {
  readonly name: string
  /** The size, when it is fixed in the spec or known from the bound model. */
  readonly size?: number
  readonly nodes: readonly string[]
}

/** A latent node with a conjugate prior for an observed node's parameter. */
export interface SpecConjugate {
  readonly prior: string
  readonly priorFamily: string
  readonly likelihood: string
  readonly likelihoodFamily: string
  /** The likelihood parameter the prior is on (`rate`, `probs`, `loc`, `variance`). */
  readonly parameter: string
  /** `direct`: the parameter is the latent; `index`: it is a lookup of the latent by an index (`at`), a group structure. */
  readonly via: 'direct' | 'index'
  /** The likelihood's other parameters (the pair's `fixed` ones) do not depend on a latent. */
  readonly known: boolean
  /** The prior's own parameters do not depend on a latent. */
  readonly hyperparametersKnown: boolean
  /** Nothing but conjugate observed nodes depends on the latent. */
  readonly exclusive: boolean
  readonly pair: ConjugatePair
}

/** Whether every latent and observation is Gaussian with a mean that is affine in the latents, and why not. */
export interface SpecLinearGaussian {
  readonly holds: boolean
  readonly reason: string
}

/**
 * What a spec reveals without running (see {@link analyseSpec}).
 *
 * @concept model-analysis
 */
export interface SpecAnalysis {
  readonly name: string
  readonly nodes: readonly SpecNodeInfo[]
  readonly data: readonly string[]
  readonly latent: readonly string[]
  readonly observed: readonly string[]
  readonly derived: readonly string[]
  readonly generated: readonly string[]
  readonly factors: readonly string[]
  readonly params: readonly string[]
  readonly dims: readonly SpecDimInfo[]
  /** Dependency edges between node ids, parent to child, over every node (data included). */
  readonly edges: readonly { readonly from: string; readonly to: string }[]
  /** Latents with a continuous family, and with a finite discrete family (which can be enumerated). */
  readonly continuous: readonly string[]
  readonly finiteDiscrete: readonly string[]
  readonly unboundedDiscrete: readonly string[]
  readonly conjugate: readonly SpecConjugate[]
  readonly linearGaussian: SpecLinearGaussian
  /** Nodes message passing cannot see through. */
  readonly opaque: readonly string[]
  readonly hasLikelihood: boolean
  /** The features methods' `check` reads, as short phrases (`continuous latents`, `conjugate sites`, ...). */
  readonly features: readonly string[]
}

function exprOps(e: ExprSpec, out: Set<string>): void {
  if (e.kind === 'op') {
    out.add(e.op)
    for (const a of e.args) exprOps(a, out)
  } else if (e.kind === 'custom') for (const a of e.args) exprOps(a, out)
}

function exprHasCustom(e: ExprSpec): boolean {
  if (e.kind === 'custom') return true
  return e.kind === 'op' && e.args.some(exprHasCustom)
}

function nodeExprs(n: ModelNode): ExprSpec[] {
  switch (n.kind) {
    case 'latent':
    case 'generated':
      return Object.values(n.dist.params)
    case 'observed':
      return [...Object.values(n.dist.params), n.value]
    case 'derived':
    case 'factor':
      return [n.expr]
    default:
      return []
  }
}

function distOf(
  n: ModelNode,
): { readonly family: string; readonly params: { readonly [k: string]: ExprSpec } } | undefined {
  return n.kind === 'latent' || n.kind === 'generated' || n.kind === 'observed' ? n.dist : undefined
}

/** The latents an expression depends on, through derived nodes too. */
function latentsIn(e: ExprSpec, byName: ReadonlyMap<string, ModelNode>, memo: Map<string, Set<string>>): Set<string> {
  const out = new Set<string>()
  const walk = (x: ExprSpec): void => {
    if (x.kind === 'ref') {
      const node = byName.get(x.name)
      if (node?.kind === 'latent') out.add(x.name)
      else if (node?.kind === 'derived') {
        let m = memo.get(x.name)
        if (m === undefined) {
          m = latentsIn(node.expr, byName, memo)
          memo.set(x.name, m)
        }
        for (const l of m) out.add(l)
      }
    } else if (x.kind === 'op' || x.kind === 'custom') for (const a of x.args) walk(a)
  }
  walk(e)
  return out
}

/** Whether an expression is affine in the latents (constants and data aside), and whether it uses any. */
function affine(e: ExprSpec, byName: ReadonlyMap<string, ModelNode>, memo: Map<string, Set<string>>): boolean {
  const uses = (x: ExprSpec): boolean => latentsIn(x, byName, memo).size > 0
  switch (e.kind) {
    case 'const':
      return true
    case 'ref': {
      const node = byName.get(e.name)
      return node?.kind === 'derived' ? affine(node.expr, byName, memo) : true
    }
    case 'custom':
      return !e.args.some(uses)
    case 'op': {
      const [a, b] = e.args
      switch (e.op) {
        case 'add':
        case 'sub':
          return e.args.every((x) => affine(x, byName, memo))
        case 'neg':
        case 'sum':
        case 'mean':
          return e.args.every((x) => affine(x, byName, memo))
        case 'mul':
        case 'matmul':
          return (
            a !== undefined &&
            b !== undefined &&
            affine(a, byName, memo) &&
            affine(b, byName, memo) &&
            !(uses(a) && uses(b))
          )
        case 'div':
          return a !== undefined && b !== undefined && affine(a, byName, memo) && !uses(b)
        case 'at':
          return a !== undefined && b !== undefined && affine(a, byName, memo) && !uses(b)
        default:
          return !e.args.some(uses)
      }
    }
  }
}

function linearGaussian(spec: ModelSpec, byName: ReadonlyMap<string, ModelNode>): SpecLinearGaussian {
  const memo = new Map<string, Set<string>>()
  const gaussian = new Set(['normal', 'multivariateNormal'])
  let any = false
  for (const n of spec.nodes) {
    if (n.kind === 'factor') return { holds: false, reason: `factor '${n.name}' adds an arbitrary density term` }
    const d = distOf(n)
    if (d === undefined || n.kind === 'generated') continue
    any = true
    if (!gaussian.has(d.family)) return { holds: false, reason: `'${n.name}' is ${d.family}, not Gaussian` }
    for (const [p, e] of Object.entries(d.params)) {
      const isLocation = p === 'loc'
      if (isLocation ? !affine(e, byName, memo) : latentsIn(e, byName, memo).size > 0)
        return {
          holds: false,
          reason: isLocation
            ? `the mean of '${n.name}' is not affine in the latents`
            : `the ${p} of '${n.name}' depends on a latent`,
        }
    }
  }
  return any
    ? { holds: true, reason: 'Gaussian nodes with means affine in the latents and known scales' }
    : { holds: false, reason: 'no random nodes' }
}

/** Whether `e` is the reference `name`, or a lookup `at(name, index)`; the form of use that keeps a pair conjugate. */
function useOf(e: ExprSpec, name: string): 'direct' | 'index' | undefined {
  if (e.kind === 'ref' && e.name === name) return 'direct'
  if (e.kind === 'op' && e.op === 'at' && e.args[0]?.kind === 'ref' && e.args[0].name === name) return 'index'
  return undefined
}

function conjugates(
  spec: ModelSpec,
  byName: ReadonlyMap<string, ModelNode>,
  children: ReadonlyMap<string, readonly ModelNode[]>,
): SpecConjugate[] {
  const memo = new Map<string, Set<string>>()
  const out: SpecConjugate[] = []
  const hasLatent = (e: ExprSpec): boolean => latentsIn(e, byName, memo).size > 0
  for (const prior of spec.nodes) {
    if (prior.kind !== 'latent') continue
    const kids = children.get(prior.name) ?? []
    for (const kid of kids) {
      if (kid.kind !== 'observed') continue
      for (const [param, expr] of Object.entries(kid.dist.params)) {
        let via = useOf(expr, prior.name)
        let parameter = param
        // A normal with a variance prior: the scale is the square root of the latent.
        if (
          via === undefined &&
          kid.dist.family === 'normal' &&
          param === 'scale' &&
          expr.kind === 'op' &&
          expr.op === 'sqrt'
        ) {
          const inner = expr.args[0]
          via = inner === undefined ? undefined : useOf(inner, prior.name)
          parameter = 'variance'
        }
        if (via === undefined) continue
        const pair = CONJUGATES.find(
          (c) => c.likelihood === kid.dist.family && c.parameter === parameter && c.prior === prior.dist.family,
        )
        if (pair === undefined) continue
        const known = pair.fixed.every((f) => {
          const e = kid.dist.params[f]
          return e === undefined || !hasLatent(e)
        })
        const hyper = Object.values(prior.dist.params).every((e) => !hasLatent(e))
        const exclusive = kids.every(
          (c) =>
            c.kind === 'observed' &&
            Object.values(c.dist.params).some(
              (e) => useOf(e, prior.name) !== undefined || (e.kind === 'op' && e.op === 'sqrt'),
            ),
        )
        out.push({
          prior: prior.name,
          priorFamily: prior.dist.family,
          likelihood: kid.name,
          likelihoodFamily: kid.dist.family,
          parameter,
          via,
          known,
          hyperparametersKnown: hyper,
          exclusive,
          pair,
        })
      }
    }
  }
  return out
}

/**
 * Analyse a model spec statically: nodes with their parents and children, dimensions, conjugate pairs, the
 * linear-Gaussian check, the opaque nodes and the feature list. Accepts a definition, a bound model or a bare spec.
 *
 * Conjugacy is reported for the pairs of `CONJUGATES` (beta-Bernoulli and binomial, gamma-Poisson, normal-normal with
 * known scale, inverse gamma on a variance, Dirichlet-categorical, ...) between a latent and an observed node whose
 * parameter is the latent itself or an `at` lookup of it. `known`, `hyperparametersKnown` and `exclusive` say whether
 * the pair is usable as a closed form for the latent on its own; a planner reads them.
 *
 * @param model - A `ModelDefinition`, a `BoundModel` or a `ModelSpec`.
 * @returns The analysis.
 * @concept model-analysis
 * @index function Static analysis of a ModelSpec: structure, plates, conjugacy and features.
 * @example
 * ```ts
 * import { analyseSpec, defineModel } from 'omai-engine/probprog/model'
 * import { beta, bernoulli } from 'omai-engine/probprog/model/dists'
 * const m = defineModel('coin', (b) => {
 *   const y = b.data('y', { dims: ['flip'], support: 'binary' })
 *   const p = b.latent('p', beta(1, 1))
 *   b.observe('y', bernoulli({ probs: p }), y)
 * })
 * const a = analyseSpec(m)
 * [a.conjugate[0]?.prior, a.conjugate[0]?.pair.likelihood, a.features.includes('conjugate sites')] // => ['p', 'bernoulli', true]
 * ```
 */
export function analyseSpec(model: ModelLike | ModelSpec): SpecAnalysis {
  const spec: ModelSpec = 'spec' in model ? model.spec : model
  const sizes = 'sizes' in model ? (model as { readonly sizes: Readonly<Record<string, number>> }).sizes : {}
  const families: Readonly<Record<string, FamilyDef>> = distributionFamilies()
  // References resolve to value nodes (observed nodes and factors cannot be referred to).
  const byName = new Map<string, ModelNode>()
  for (const n of spec.nodes) if (n.kind !== 'observed' && n.kind !== 'factor') byName.set(n.name, n)
  const idOf = (n: ModelNode): string => (n.kind === 'data' ? `data:${n.name}` : n.name)
  const kidNodes = new Map<string, ModelNode[]>()
  const kidIds = new Map<string, string[]>()
  const edges: { from: string; to: string }[] = []
  for (const n of spec.nodes)
    for (const r of nodeRefs(n)) {
      const parent = byName.get(r)
      if (parent === undefined) continue
      kidNodes.set(r, [...(kidNodes.get(r) ?? []), n])
      kidIds.set(idOf(parent), [...(kidIds.get(idOf(parent)) ?? []), idOf(n)])
      edges.push({ from: idOf(parent), to: idOf(n) })
    }
  const nodes: SpecNodeInfo[] = spec.nodes.map((n) => {
    const d = distOf(n)
    const fam = d === undefined ? undefined : families[d.family]
    const ops = new Set<string>()
    for (const e of nodeExprs(n)) exprOps(e, ops)
    const custom = nodeExprs(n).some(exprHasCustom)
    return {
      id: idOf(n),
      name: n.name,
      kind: n.kind,
      dims: n.dims,
      eventRank: n.eventRank,
      ...(d === undefined ? {} : { family: d.family }),
      ...(fam === undefined ? {} : { discrete: fam.discrete, exponentialFamily: fam.exponentialFamily }),
      parents: nodeRefs(n).flatMap((r) => {
        const p = byName.get(r)
        return p === undefined ? [] : [idOf(p)]
      }),
      children: kidIds.get(idOf(n)) ?? [],
      operators: [...ops],
      opaque: custom || (fam !== undefined && !fam.exponentialFamily),
    }
  })
  const names = (kind: ModelNode['kind']): string[] => spec.nodes.filter((n) => n.kind === kind).map((n) => n.name)
  const latentInfo = nodes.filter((n) => n.kind === 'latent')
  const family = (n: SpecNodeInfo): FamilyDef | undefined => (n.family === undefined ? undefined : families[n.family])
  const continuous = latentInfo.filter((n) => family(n)?.discrete !== true).map((n) => n.name)
  const finiteDiscrete = latentInfo.filter((n) => family(n)?.finite === true).map((n) => n.name)
  const unboundedDiscrete = latentInfo
    .filter((n) => family(n)?.discrete === true && family(n)?.finite !== true)
    .map((n) => n.name)
  const conjugate = conjugates(spec, byName, kidNodes)
  const lg = linearGaussian(spec, byName)
  const opaque = nodes.filter((n) => n.opaque).map((n) => n.name)
  const dims: SpecDimInfo[] = spec.dims.map((d) => ({
    name: d.name,
    ...(sizes[d.name] !== undefined ? { size: sizes[d.name] as number } : d.size === undefined ? {} : { size: d.size }),
    nodes: spec.nodes.filter((n) => n.dims.includes(d.name)).map(idOf),
  }))
  const features: string[] = []
  if (continuous.length > 0) features.push('continuous latents')
  if (finiteDiscrete.length > 0) features.push('finite discrete latents')
  if (unboundedDiscrete.length > 0) features.push('unbounded discrete latents')
  if (names('observed').length > 0) features.push('likelihood')
  if (conjugate.length > 0) features.push('conjugate sites')
  if (lg.holds) features.push('linear-Gaussian')
  if (opaque.length === 0) features.push('exponential-family factors')
  else features.push('opaque nodes')
  if (names('generated').length > 0) features.push('generated quantities')
  if (names('factor').length > 0) features.push('factor terms')
  if (dims.some((d) => d.nodes.length > 0)) features.push('plates')
  if (spec.nodes.some((n) => n.kind === 'latent' && nodeRefs(n).some((r) => byName.get(r)?.kind === 'latent')))
    features.push('hierarchical')
  return {
    name: spec.name,
    nodes,
    data: names('data'),
    latent: names('latent'),
    observed: names('observed'),
    derived: names('derived'),
    generated: names('generated'),
    factors: names('factor'),
    params: names('param'),
    dims,
    edges,
    continuous,
    finiteDiscrete,
    unboundedDiscrete,
    conjugate,
    linearGaussian: lg,
    opaque,
    hasLikelihood: names('observed').length > 0,
    features,
  }
}
