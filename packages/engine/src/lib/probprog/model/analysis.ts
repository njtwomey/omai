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
import { type ConjugacyRecord, conjugacyRecords } from './analysis-conjugacy.ts'
import {
  distOf,
  exprHasCustom,
  exprOps,
  isAffineIn,
  type LatentMemo,
  latentsIn,
  nodeExprs,
  nodesByName,
  useOf,
} from './analysis-exprs.ts'
import { type LinearGaussianBlock, linearGaussianBlocks } from './analysis-linear-gaussian.ts'
import type { ModelLike } from './bind.ts'
import { distributionFamilies, type FamilyDef } from './dists/families.ts'
import { canonicalDims, factorId, nodeId, plateId } from './ids.ts'
import { expandScans, scanLinearGaussian } from './scan.ts'
import { type ExprSpec, type ModelNode, type ModelSpec, nodeRefs } from './spec.ts'
import { buildStructureReport, type StructureReport } from './structure-report.ts'

/** One node as the analysis sees it. */
export interface SpecNodeInfo {
  /** A unique id: the node's name, or `data:<name>` for a data node (an observed node may share its data's name). */
  readonly id: string
  readonly name: string
  readonly kind: ModelNode['kind']
  readonly dims: readonly string[]
  /** The id of the plate the node sits in (its dimensions in the spec's order), absent for a node over none. */
  readonly plate?: string
  /** The id of the node's factor in a factor graph (`factor:<name>`), for every kind but data. */
  readonly factor?: string
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

/** A plate: the dimensions it is made of (outermost first), the sizes where known, its parent plate and its nodes. */
export interface SpecPlateInfo {
  readonly id: string
  readonly dims: readonly string[]
  readonly parent?: string
  readonly sizes: readonly (number | undefined)[]
  /** The ids of the nodes whose dimensions are exactly this plate's. */
  readonly nodes: readonly string[]
}

/** A recurrence (`scan` node) as the analysis sees it: what its states are, what it reads and whether it is a Gaussian chain. */
export interface SpecScanInfo {
  readonly name: string
  readonly over: string
  /** The number of steps, when known (fixed in the spec or from the bound model). */
  readonly steps?: number
  /** The family of the step distribution (the initial one may differ in its parameters, not in kind). */
  readonly family: string
  readonly initFamily: string
  readonly eventRank: number
  /** The states are finite discrete (a hidden Markov chain), continuous, or unbounded discrete. */
  readonly stateKind: 'finite-discrete' | 'continuous' | 'unbounded-discrete'
  /** The observed nodes (emissions) drawn at every step. */
  readonly emissions: readonly string[]
  /** The latents (and other scans) its distributions read, through derived nodes too: its parameters. */
  readonly parameters: readonly string[]
  /** Gaussian chain given its parameters (the latents it reads are constants), and why or why not. */
  readonly linearGaussian: SpecLinearGaussian
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
  /** The plates the nodes sit in, nested by prefix (`plate:a` holds `plate:a×b`), outer ones first. */
  readonly plates: readonly SpecPlateInfo[]
  /** Dependency edges between node ids, parent to child, over every node (data included). */
  readonly edges: readonly { readonly from: string; readonly to: string }[]
  /** Latents with a continuous family, and with a finite discrete family (which can be enumerated). */
  readonly continuous: readonly string[]
  readonly finiteDiscrete: readonly string[]
  readonly unboundedDiscrete: readonly string[]
  readonly conjugate: readonly SpecConjugate[]
  /**
   * The conjugacy of the latents with what depends on them, beyond the direct and `at` pairs of `conjugate`: through plates
   * (the dimensions summed over), affine maps of Gaussian latents, known multiples of gamma rates, latent children (a
   * hierarchy) and joint normal and inverse-gamma priors, each with its closed-form posterior, the latents it is
   * conditional on, its conditions and whether the `collapse` handler can take it.
   */
  readonly conjugacy: readonly ConjugacyRecord[]
  readonly linearGaussian: SpecLinearGaussian
  /** The recurrences of the spec (its `scan` nodes), which count as one latent each in `latent` and the discrete and continuous lists. */
  readonly scans: readonly SpecScanInfo[]
  /** The maximal sets of Gaussian latents that are jointly Gaussian with their observations given the other latents. */
  readonly linearGaussianBlocks: readonly LinearGaussianBlock[]
  /** Nodes message passing cannot see through. */
  readonly opaque: readonly string[]
  readonly hasLikelihood: boolean
  /** The features methods' `check` reads, as short phrases (`continuous latents`, `conjugate sites`, ...). */
  readonly features: readonly string[]
  /** The structure report: blankets, components, the discrete part, minibatch axes and suggestions (see {@link structureReport}). */
  structure(): StructureReport
  /** What the structure allows, as text: what was found, which method or handler could use it and why. Advice only. */
  explain(): string
}

function linearGaussian(spec: ModelSpec, byName: ReadonlyMap<string, ModelNode>): SpecLinearGaussian {
  const memo = new Map<string, Set<string>>()
  const gaussian = new Set(['normal', 'multivariateNormal'])
  let any = false
  for (const n of spec.nodes) {
    if (n.kind === 'factor') return { holds: false, reason: `factor '${n.name}' adds an arbitrary density term` }
    if (n.kind === 'scan') {
      const r = scanLinearGaussian(n, byName, memo, true)
      if (!r.holds) return { holds: false, reason: `the scan '${n.name}': ${r.reason}` }
      any = true
      continue
    }
    const d = distOf(n)
    if (d === undefined || n.kind === 'generated') continue
    any = true
    if (!gaussian.has(d.family)) return { holds: false, reason: `'${n.name}' is ${d.family}, not Gaussian` }
    for (const [p, e] of Object.entries(d.params)) {
      const isLocation = p === 'loc'
      if (isLocation ? !isAffineIn(e, byName, memo) : latentsIn(e, byName, memo).size > 0)
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
  const authored: ModelSpec = 'spec' in model ? model.spec : model
  // The analyses read the spec with each scan followed by its emissions as observed nodes over the scanned dimension.
  const spec = expandScans(authored)
  const sizes = 'sizes' in model ? (model as { readonly sizes: Readonly<Record<string, number>> }).sizes : {}
  const families: Readonly<Record<string, FamilyDef>> = distributionFamilies()
  // References resolve to value nodes (observed nodes and factors cannot be referred to).
  const byName = nodesByName(spec)
  const idOf = nodeId
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
    const d = n.kind === 'scan' ? n.step : distOf(n)
    const fam = d === undefined ? undefined : families[d.family]
    const ops = new Set<string>()
    for (const e of nodeExprs(n)) exprOps(e, ops)
    const custom = nodeExprs(n).some(exprHasCustom)
    const plate = plateId(canonicalDims(spec, n.dims))
    return {
      id: idOf(n),
      name: n.name,
      kind: n.kind,
      dims: n.dims,
      ...(plate === undefined ? {} : { plate }),
      ...(n.kind === 'data' ? {} : { factor: factorId(n.name) }),
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
  const latentInfo = nodes.filter((n) => n.kind === 'latent' || n.kind === 'scan')
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
  const plateMap = new Map<string, { dims: string[]; nodes: string[] }>()
  for (const n of spec.nodes) {
    const dimsOf = canonicalDims(spec, n.dims)
    for (let k = 1; k <= dimsOf.length; k++) {
      const id = plateId(dimsOf.slice(0, k)) as string
      const entry = plateMap.get(id) ?? { dims: dimsOf.slice(0, k), nodes: [] }
      if (k === dimsOf.length) entry.nodes.push(idOf(n))
      plateMap.set(id, entry)
    }
  }
  const sizeOfDim = new Map(dims.map((x) => [x.name, x.size]))
  const plates: SpecPlateInfo[] = [...plateMap.entries()]
    .sort((x, y) => x[1].dims.length - y[1].dims.length)
    .map(([id, e]) => ({
      id,
      dims: e.dims,
      ...(e.dims.length > 1 ? { parent: plateId(e.dims.slice(0, -1)) as string } : {}),
      sizes: e.dims.map((x) => sizeOfDim.get(x)),
      nodes: e.nodes,
    }))
  const memoScan: LatentMemo = new Map()
  const scans: SpecScanInfo[] = authored.nodes.flatMap((n): SpecScanInfo[] => {
    if (n.kind !== 'scan') return []
    const fam = families[n.step.family]
    const parameters = new Set<string>()
    for (const e of nodeExprs(n)) for (const l of latentsIn(e, byName, memoScan)) if (l !== n.name) parameters.add(l)
    const steps = dims.find((x) => x.name === n.over)?.size
    return [
      {
        name: n.name,
        over: n.over,
        ...(steps === undefined ? {} : { steps }),
        family: n.step.family,
        initFamily: n.init.family,
        eventRank: n.eventRank,
        stateKind:
          fam?.discrete !== true ? 'continuous' : fam.finite === true ? 'finite-discrete' : 'unbounded-discrete',
        emissions: n.emissions.map((m) => m.name),
        parameters: [...parameters],
        linearGaussian: scanLinearGaussian(n, byName, memoScan, false),
      },
    ]
  })
  const features: string[] = []
  if (scans.length > 0) features.push('recurrences')
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
  const sizeOf = (dim: string): number | undefined => dims.find((x) => x.name === dim)?.size
  const env = { spec, byName, memo: new Map<string, Set<string>>(), sizeOf }
  const base = {
    name: spec.name,
    nodes,
    data: names('data'),
    latent: latentInfo.map((n) => n.name),
    observed: names('observed'),
    derived: names('derived'),
    generated: names('generated'),
    factors: names('factor'),
    params: names('param'),
    dims,
    plates,
    edges,
    continuous,
    finiteDiscrete,
    unboundedDiscrete,
    conjugate,
    conjugacy: conjugacyRecords(env),
    linearGaussian: lg,
    scans,
    linearGaussianBlocks: linearGaussianBlocks(env),
    opaque,
    hasLikelihood: names('observed').length > 0,
    features,
  }
  const analysis = base as unknown as SpecAnalysis
  let report: StructureReport | undefined
  const structure = (): StructureReport => {
    report ??= buildStructureReport(spec, base, sizeOf)
    return report
  }
  // Methods, not data: they stay out of enumeration, JSON and equality, so an analysis is still a plain value.
  Object.defineProperty(analysis, 'structure', { value: structure, enumerable: false })
  Object.defineProperty(analysis, 'explain', { value: () => structure().explain(), enumerable: false })
  return analysis
}

/**
 * The structure report of a model: what its structure lets a method exploit (conjugate sites and the collapse handler
 * that takes them, Gaussian blocks, the discrete part's width, independent components, minibatch axes, funnel geometry), as
 * records and as plain suggestions that name the method or handler to use and the knob values that apply. It is advice for
 * a person to read: nothing is selected or applied. Same as `analyseSpec(model).structure()`.
 *
 * @param model - A `ModelDefinition`, a `BoundModel` (which gives the sizes of the dimensions) or a `ModelSpec`.
 * @returns The report, with `explain()` for the text.
 * @concept model-analysis
 * @index function structureReport: what a spec's structure allows, as records and suggestions naming the method to use.
 * @example
 * ```ts
 * import { defineModel, structureReport } from 'omai-engine/probprog/model'
 * import { beta, bernoulli } from 'omai-engine/probprog/model/dists'
 * const m = defineModel('coin', (b) => {
 *   const y = b.data('y', { dims: ['flip'], support: 'binary' })
 *   b.observe('y', bernoulli({ probs: b.latent('p', beta(1, 1)) }), y)
 * })
 * const r = structureReport(m)
 * [r.conjugateBlocks[0]?.pairs[0], r.suggestions[0]?.kind] // => ['beta-bernoulli probs', 'closed-form']
 * ```
 */
export function structureReport(model: ModelLike | ModelSpec): StructureReport {
  return analyseSpec(model).structure()
}
