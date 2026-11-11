/**
 * Compiling a bound model spec to a factor graph for message passing: `toFactorGraph` expands the spec's named
 * dimensions (plates) into the scalar and vector variables the engine of `omai-engine/message-passing` supports, maps
 * distribution families and operators to the engine's message families and factor operators, and reports exactly what it
 * cannot map and why. The engine has no plates of its own, so a plate is compiled one of two ways. An observed node whose
 * rows are i.i.d. draws of one regression on a weight vector (a Gaussian with one known scale, or a Bernoulli with a
 * probit or logistic link, of `X.matmul(w)` with `w` a vector latent, missing rows left out) is ONE plated factor
 * (`linear-regression-plate`, `probit-plate`, `logistic-plate`) holding the rows as its parameters and computing its
 * message over all of them in one call: the graph has no variable or factor per row, does not grow with the data, and
 * compiles and reads back in time that does not depend on the rows but for copying the design matrix once. Every other plate
 * is an expansion: N copies of the factors over it, with the data folded into the factors' constants (a hierarchy, rows
 * with their own variables or scales, scalar weights, discrete tables over a plate). `plates: 'expand'` forces the
 * expansion everywhere (the same posterior, a message per row, so finer anytime steps).
 *
 * The mappings are the methods of `Compiler`, one per model idiom: `latent` and `observed` dispatch on the distribution family
 * and on the shape of its parameters (a Gaussian with a known or random scale, a precision that is a gamma variable, a mean chosen
 * by a discrete variable, a summed-out mixture, conjugate pairs, matrices, discrete structure under Dirichlet rows), and `derived`
 * maps the arithmetic of the value algebra (`factor-graph/_cells.ts`) to the engine's affine, product, gate and apply operators.
 * Anything with no mapping is reported with a reason, never silently dropped. The value algebra, the refusal, the issue and
 * layout types, the reading back of a run (`factorGraphPosterior`) and the verdict (`factorGraphVerdict`) live in `factor-graph/`.
 *
 * Ids follow `ids.ts`: a variable is its site's name, `<site>[<i>]` for an element of a site over dimensions, and
 * `<site>#<k>[<i>]` for an auxiliary variable; a factor is `factor:` and the id of the variable it defines (an observed
 * element's likelihood is `factor:y[3]`); `siteOfId` maps every id back to the spec node it belongs to, so a view can paint a
 * run's marginals on `toStructure(spec)` by site name. Results are read back by `factorGraphPosterior`, which builds
 * the library's `Posterior` keyed by site name, in the spec's shapes.
 *
 * @index function toFactorGraph: a bound ModelSpec as a message-passing factor graph, plates expanded.
 */

import type { Json } from '../../base/index.ts'
import {
  betaEta,
  type FactorGraphSpec,
  type FactorOperator,
  type FactorSpec,
  type FamilyKind,
  defaultOperators as factorOperators,
  gammaPrior,
  gaussianMvPrior,
  gaussianPrior,
  priorFactor,
} from '../../message-passing/index.ts'
import type { SpecAnalysis } from './analysis.ts'
import type { BoundModel } from './bind.ts'
import { eventToDims } from './event-dims.ts'
import { type GateBranch, GateBuilder, precisionBranch, precisionFactor } from './factor-graph/_builder.ts'
import { affOf, type Cell, combine, isConstCell, logName, type Precision, prod } from './factor-graph/_cells.ts'
import { Context } from './factor-graph/_context.ts'
import { expArgument, precisionExpr, stripClip, unwrap } from './factor-graph/_idioms.ts'
import { derived } from './factor-graph/_mappings/derived.ts'
import { bernoulliTable, latentDirichlet, latentDiscrete } from './factor-graph/_mappings/latent-discrete.ts'
import {
  latentGammaRate,
  latentNormalGamma,
  latentRandomConcentration,
} from './factor-graph/_mappings/latent-positive.ts'
import {
  latentCorrelation,
  latentMultivariateNormal,
  latentMvnWishart,
  latentWishart,
  observedBivariateNormal,
  observedMvnWishart,
  wishartOf,
} from './factor-graph/_mappings/matrix.ts'
import {
  observedBernoulli,
  observedConjugate,
  observedFailures,
  observedGammaRate,
  observedMixtureWeight,
} from './factor-graph/_mappings/observed-conjugate.ts'
import {
  hiddenSwitch,
  observedSwitch,
  observedSwitchLogScale,
  switchedCells,
  valueMovedToMean,
} from './factor-graph/_mappings/observed-gates.ts'
import {
  observedGaussian,
  observedHeavy,
  observedLogScale,
  observedNormalGamma,
  observedRandomPrecision,
  observedRandomScale,
} from './factor-graph/_mappings/observed-gaussian.ts'
import {
  observedHurdle,
  observedMulticlass,
  observedNumeric,
  observedPair,
} from './factor-graph/_mappings/observed-numeric.ts'
import { plate } from './factor-graph/_mappings/plates.ts'
import {
  constParam,
  discreteParents,
  evalNamedValue,
  paramIsConst,
  paramSym,
  randomPrecisions,
  rvName,
} from './factor-graph/_params.ts'
import { Refusal } from './factor-graph/_refusal.ts'
import { evalSym } from './factor-graph/_symbolic/_affine.ts'
import { mulAff } from './factor-graph/_symbolic/_functions.ts'
import type { FactorGraphCompilation } from './factor-graph/types.ts'
import { defaultOperators, type OperatorTable } from './operators.ts'
import { ScanUnrollError, unrollScans } from './scan.ts'
import type { ExprSpec, ModelNode } from './spec.ts'

/** A factor-graph name for the family of a distribution family, for the families the compiler maps. */
const MESSAGE_FAMILY: Readonly<Record<string, FamilyKind>> = {
  normal: 'gaussian',
  normalSpikeSlab: 'gaussian',
  gamma: 'gamma',
  inverseGamma: 'gamma',
  exponential: 'gamma',
  beta: 'beta',
  dirichlet: 'dirichlet',
  bernoulli: 'bernoulli',
  categorical: 'discrete',
  multivariateNormal: 'gaussian-mv',
}

/**
 * The families of a prior with no message family that are projected onto a Gaussian once (`latentProjected`): a positive
 * scale (half-normal, half-Cauchy, log-normal, uniform or truncated normal on the positive half line) as a Gaussian on its
 * log, any other bounded or truncated value as a Gaussian on the value.
 */
const PROJECTED_PRIORS: ReadonlySet<string> = new Set([
  'halfNormal',
  'halfCauchy',
  'logNormal',
  'uniform',
  'truncatedNormal',
])

class Compiler {
  /** The state of the compilation, which the evaluator and the mappings work on. */
  readonly cx: Context

  constructor(
    bound: BoundModel,
    modelOps: OperatorTable,
    ops: Readonly<Record<string, FactorOperator>>,
    vectorise: boolean,
    packScalars: boolean,
    denseScales: boolean = true,
  ) {
    this.cx = new Context(bound, modelOps, ops, { vectorise, packScalars, denseScales })
  }

  // ----- sizes and indexing -----

  // ----- values -----

  // ----- the nodes -----

  run(): void {
    for (const [position, node] of this.cx.bound.spec.nodes.entries()) {
      this.cx.current = node.name
      this.cx.position = position
      try {
        this.node(node)
      } catch (e) {
        if (!(e instanceof Refusal)) throw e
        this.cx.issues.push({ node: node.name, kind: e.kind, reason: e.message })
        this.cx.failed.add(node.name)
      }
    }
  }

  private node(node: ModelNode): void {
    switch (node.kind) {
      case 'data':
        // Bound data is known from the start (`knownConstants`): nothing to map.
        return
      case 'latent':
        // A standard normal vector that a non-centred derived node absorbs is compiled by that node only if the dense factor does not apply.
        if ([...this.cx.nonCentred.values()].includes(node.name)) return
        this.latent(node)
        return
      case 'derived':
        derived(this.cx, node, (z) => this.latent(z))
        return
      case 'observed':
        this.observed(node)
        return
      case 'generated':
        return
      case 'param':
        throw new Refusal(
          'node',
          `'${node.name}' is a param (a learnable value, not a random variable): it has no factor`,
        )
      case 'factor':
        throw new Refusal(
          'node',
          `'${node.name}' is an arbitrary log-density term: no message operator exists for it (a numeric factor operator could be given)`,
        )
    }
  }

  private latent(node: ModelNode & { readonly kind: 'latent' }): void {
    const fam = node.dist.family
    const truncated = node.constraint !== undefined && node.constraint.kind !== 'real' && fam === 'normal'
    if (node.constraint !== undefined && node.constraint.kind !== 'real' && !truncated)
      throw new Refusal(
        'constraint',
        `'${node.name}' has the constraint '${node.constraint.kind}': a message family lives on the family's own support, so a constraint beyond it is not expressible`,
      )
    if (PROJECTED_PRIORS.has(fam)) {
      this.latentProjected(node)
      return
    }
    if (fam === 'wishart' || fam === 'inverseWishart') {
      latentWishart(this.cx, node)
      return
    }
    if (fam === 'correlation') {
      latentCorrelation(this.cx, node)
      return
    }
    if (fam === 'bivariateNormal')
      throw new Refusal(
        'structure',
        `'${node.name}' is a latent bivariateNormal: only observed vectors are mapped (write the vectors with a Wishart precision, or observe them)`,
      )
    if (fam === 'multivariateNormal' && wishartOf(this.cx, node) !== undefined) {
      latentMvnWishart(this.cx, node)
      return
    }
    const meanOf = this.cx.normalGammaByMean.get(node.name)
    if (meanOf !== undefined) {
      this.cx.layout(node, 'latent', 'normal-gamma', 'scalar', [meanOf.variable], true, { part: 'mean' })
      return
    }
    if ((fam === 'gamma' || fam === 'inverseGamma') && latentNormalGamma(this.cx, node)) return
    const kind = MESSAGE_FAMILY[fam]
    if (kind === undefined)
      throw new Refusal(
        'family',
        `'${node.name}' is ${fam}: the engine has message families gaussian, gamma, beta, bernoulli, discrete and dirichlet, and none for ${fam}`,
      )
    if (fam === 'normal') {
      this.latentNormal(node)
      if (truncated) this.truncate(node)
      return
    }
    if (fam === 'multivariateNormal') {
      latentMultivariateNormal(this.cx, node)
      return
    }
    if (fam === 'normalSpikeSlab') {
      this.latentSpikeSlab(node)
      return
    }
    if (fam === 'bernoulli' || fam === 'categorical') {
      latentDiscrete(this.cx, node)
      return
    }
    if (fam === 'dirichlet') {
      const concentration = node.dist.params['concentration']
      if (concentration !== undefined && !this.cx.isConst(concentration)) latentRandomConcentration(this.cx, node)
      else latentDirichlet(this.cx, node)
      return
    }
    if (fam === 'beta' && ['a', 'b'].some((k) => !this.cx.isConst(node.dist.params[k] as ExprSpec))) {
      latentRandomConcentration(this.cx, node)
      return
    }
    // An inverse gamma `x` with scale `b` is the precision `1 / x` ~ Gamma(a, rate b): the variable is that gamma, the site its reciprocal.
    const inverse = fam === 'inverseGamma'
    const rateName = inverse ? 'scale' : 'rate'
    const rateExpr = node.dist.params[rateName]
    if ((fam === 'gamma' || inverse) && rateExpr !== undefined && !this.cx.isConst(rateExpr)) {
      latentGammaRate(this.cx, node)
      return
    }
    const shape = this.cx.shapeOf(node.dims)
    const n = prod(shape)
    const params =
      fam === 'gamma' || inverse
        ? [constParam(this.cx, node, 'concentration', 'the shape'), constParam(this.cx, node, rateName, 'the rate')]
        : fam === 'exponential'
          ? [new Array<number>(n).fill(1), constParam(this.cx, node, 'rate', 'the rate')]
          : [constParam(this.cx, node, 'a', 'the shape'), constParam(this.cx, node, 'b', 'the shape')]
    const cells: Cell[] = []
    const vars: string[] = []
    for (let i = 0; i < n; i++) {
      const name = this.cx.g.addVariable({ name: this.cx.elementName(node.name, node.dims, i), family: { kind } })
      const p = [params[0]?.[i] as number, params[1]?.[i] as number]
      const f =
        fam === 'beta'
          ? priorFactor(name, betaEta(p[0] as number, p[1] as number))
          : gammaPrior(name, p[0] as number, p[1] as number)
      this.cx.g.addFactor(f, name)
      cells.push(inverse ? { kind: 'recip', variable: name, scale: 1 } : { kind: 'rv', variable: name, family: kind })
      vars.push(name)
    }
    this.cx.finishLatent(node, kind, cells, vars, inverse ? { reciprocal: true } : {})
  }

  // ----- matrix-valued latents: a Wishart (or inverse-Wishart) precision, and the Gaussian vectors that read it -----

  // ----- Gaussian latents: a transition from the previous state, projected and truncated priors, the normal and its groups -----

  /**
   * A prior with no message family (half-normal, half-Cauchy, log-normal, uniform, truncated normal), the exact numeric
   * `prior-site` on a Gaussian variable (a log-normal is the exact Gaussian on its log). A density on the positive half line is
   * carried on its log: a Gaussian variable `l = log tau` with the density of `l` as its site, and the value `tau` is the variable
   * `exp(l)` of an `apply` factor, whose Gaussian belief is the moment match of a log-normal. A density on a bounded or truncated
   * real range is the site on the value itself. The site's message is the moment match against the current cavity at every sweep,
   * so the graph holds the prior itself; the belief of the variable is still one Gaussian.
   */
  private latentProjected(node: ModelNode & { readonly kind: 'latent' }): void {
    const fam = node.dist.family
    const n = prod(this.cx.shapeOf(node.dims))
    const names: Readonly<Record<string, readonly string[]>> = {
      halfNormal: ['scale'],
      halfCauchy: ['scale'],
      logNormal: ['mu', 'sigma'],
      uniform: ['low', 'high'],
      truncatedNormal: ['loc', 'scale', 'low', 'high'],
    }
    const bases: Readonly<Record<string, string>> = {
      halfNormal: 'half-normal',
      halfCauchy: 'half-cauchy',
      uniform: 'uniform',
      truncatedNormal: 'truncated-normal',
    }
    const values = Object.fromEntries((names[fam] ?? []).map((p) => [p, constParam(this.cx, node, p, `the ${p}`)]))
    const cells: Cell[] = []
    const vars: string[] = []
    const logCells: Cell[] = []
    const site = node.dims.length === 0 ? undefined : (i: number): number => i
    for (let i = 0; i < n; i++) {
      const p = (k: string): number => (values[k] as number[])[i] as number
      const name = this.cx.g.addVariable({
        name: this.cx.elementName(node.name, node.dims, i),
        family: { kind: 'gaussian' },
      })
      const low = fam === 'uniform' || fam === 'truncatedNormal' ? p('low') : 0
      const high = fam === 'uniform' || fam === 'truncatedNormal' ? p('high') : Number.POSITIVE_INFINITY
      if (fam === 'logNormal' && !(p('sigma') > 0))
        throw new Refusal('data', `the sigma of '${node.name}' must be positive`)
      const positive = fam === 'halfNormal' || fam === 'halfCauchy' || fam === 'logNormal' || low >= 0
      // The exact density as a numeric site (`prior-site`): its message is the moment match against the current cavity at every
      // sweep, so the graph holds the prior itself. Infinite bounds are left out of the parameters (they are JSON).
      const priorFactor = (variable: string, log: boolean): FactorSpec => ({
        operator: 'prior-site',
        args: [variable],
        params: {
          y: 0,
          base: bases[fam] as string,
          log: log ? 1 : 0,
          ...(names[fam]?.includes('scale') ? { scale: p('scale') } : {}),
          ...(fam === 'truncatedNormal' ? { loc: p('loc') } : {}),
          ...(Number.isFinite(low) && (fam === 'uniform' || fam === 'truncatedNormal') ? { low } : {}),
          ...(Number.isFinite(high) && (fam === 'uniform' || fam === 'truncatedNormal') ? { high } : {}),
        },
      })
      if (positive) {
        const l = this.cx.g.auxName(node.name, site === undefined ? undefined : site(i))
        this.cx.g.addVariable({ name: l, family: { kind: 'gaussian' } })
        if (fam === 'logNormal') this.cx.g.addFactor(gaussianPrior(l, p('mu'), p('sigma') ** 2), l)
        else this.cx.g.addFactor(priorFactor(l, true), l)
        this.cx.g.need('apply')
        this.cx.g.addFactor({ operator: 'apply', args: [name, l], params: { fn: 'exp' } }, name)
        this.cx.logScale.set(name, l)
        logCells.push(this.cx.g.varAtom(l))
      } else this.cx.g.addFactor(priorFactor(name, false), name)
      cells.push(this.cx.g.varAtom(name))
      vars.push(name)
    }
    this.cx.syms.set(node.name, { dims: node.dims, cells })
    // A positive value carried on its log (`tau = exp(l)`): the log is addressable, so a log link reads it directly (a rate or a scale).
    if (logCells.length > 0) this.cx.syms.set(logName(node.name), { dims: node.dims, cells: logCells })
    this.cx.layout(node, 'latent', 'gaussian', node.dims.length === 0 ? 'scalar' : 'elements', vars, true)
  }

  /**
   * The constraint of a Gaussian latent as comparison factors on its variables: `positive`, `nonNegative`, `greaterThan`,
   * `lessThan` and `interval` with known scalar bounds truncate each element (a `greater-than` or `is-between` factor with an
   * observed true outcome, whose moment matching is the truncated Gaussian), and `ordered` says each element is above the
   * previous one (a `greater-than` on the difference of neighbours). The prior is not renormalised, as in Stan.
   */
  private truncate(node: ModelNode & { readonly kind: 'latent' }): void {
    const c = node.constraint
    if (c === undefined) return
    const cells = this.cx.syms.get(node.name)?.cells ?? []
    const flag = (
      x: string,
      operator: 'greater-than' | 'is-between',
      params: Record<string, Json>,
      i: number,
    ): void => {
      const name = this.cx.g.auxName(node.name, i)
      this.cx.g.need(operator)
      this.cx.g.addVariable({ name, family: { kind: 'bernoulli' }, observed: 1 })
      this.cx.g.addFactor({ operator, args: [name, x], params }, name)
    }
    const number = (v: Json): number => {
      if (typeof v !== 'number')
        throw new Refusal('constraint', `the bound of the constraint of '${node.name}' must be one known number`)
      return v
    }
    if (c.kind === 'ordered') {
      if (node.dims.length !== 1)
        throw new Refusal('constraint', `'${node.name}' is ordered over [${node.dims}]: one dimension is mapped`)
      for (let i = 0; i + 1 < cells.length; i++) {
        const gap = combine(affOf(cells[i + 1] as Cell), affOf(cells[i] as Cell), -1)
        flag(this.cx.g.materialise(gap, node.name, i) as string, 'greater-than', { threshold: 0 }, i)
      }
      return
    }
    const bounds =
      c.kind === 'positive' || c.kind === 'nonNegative'
        ? { lower: 0, upper: Number.POSITIVE_INFINITY }
        : c.kind === 'greaterThan'
          ? { lower: number(c.low), upper: Number.POSITIVE_INFINITY }
          : c.kind === 'lessThan'
            ? { lower: Number.NEGATIVE_INFINITY, upper: number(c.high) }
            : c.kind === 'interval'
              ? { lower: number(c.low), upper: number(c.high) }
              : undefined
    if (bounds === undefined)
      throw new Refusal(
        'constraint',
        `'${node.name}' has the constraint '${c.kind}': only ordered and bounds on a Gaussian are mapped`,
      )
    for (const [i, cell] of cells.entries()) {
      const x = this.cx.g.materialise(affOf(cell), node.name, node.dims.length === 0 ? undefined : i) as string
      if (bounds.upper === Number.POSITIVE_INFINITY) flag(x, 'greater-than', { threshold: bounds.lower }, i)
      else flag(x, 'is-between', { lower: Math.max(bounds.lower, -1e12), upper: bounds.upper }, i)
    }
  }

  private latentNormal(node: ModelNode & { readonly kind: 'latent' }): void {
    const shape = this.cx.shapeOf(node.dims)
    const n = prod(shape)
    const loc = this.cx.over(paramSym(this.cx, node.dist, 'loc'), node.dims)
    const scaleExpr = node.dist.params['scale']
    if (scaleExpr !== undefined && !this.cx.isConst(scaleExpr)) {
      if (this.cx.plan.packed.has(node.name)) this.latentSharedPrecision(node, loc)
      else if (precisionExpr(this.cx.idiom, scaleExpr) === undefined) this.latentRandomScale(node, loc, scaleExpr)
      else this.latentRandomPrecision(node, loc)
      return
    }
    const scale = constParam(this.cx, node, 'scale', 'the scale')
    for (const s of scale)
      if (!(s > 0)) throw new Refusal('data', `the scale of '${node.name}' is ${s}: it must be positive`)
    const group = this.cx.plan.groups.get(node.name)
    if (group !== undefined) {
      this.latentGrouped(node, group)
      return
    }
    if (this.cx.plan.packed.has(node.name)) {
      const mean = loc.map((c) => {
        if (!isConstCell(c))
          throw new Refusal(
            'operator',
            `'${node.name}' is read as a vector by a matmul, so its prior mean must be known`,
          )
        return c.aff.offset
      })
      const cov = new Array<number>(n * n).fill(0)
      for (let i = 0; i < n; i++) cov[i * n + i] = (scale[i] as number) ** 2
      this.cx.g.addVariable({ name: node.name, family: { kind: 'gaussian-mv', size: n } })
      this.cx.g.addFactor(gaussianMvPrior(node.name, mean, cov), node.name)
      this.cx.syms.set(node.name, {
        dims: node.dims,
        cells: Array.from({ length: n }, (_, j) => this.cx.g.compAtom(node.name, j, n)),
      })
      this.cx.layout(node, 'latent', 'gaussian-mv', 'packed', [node.name], true)
      return
    }
    const cells: Cell[] = []
    const vars: string[] = []
    for (let i = 0; i < n; i++) {
      const name = this.cx.g.addVariable({
        name: this.cx.elementName(node.name, node.dims, i),
        family: { kind: 'gaussian' },
      })
      const l = loc[i] as Cell
      const variance = (scale[i] as number) ** 2
      if (l.kind === 'lookup') {
        // A mean chosen by a discrete variable (a mixture of components): a gate over one Gaussian prior per state, over the
        // variable and the random means of some states.
        this.cx.g.need('gate')
        const gate = new GateBuilder(l.variable, [name])
        for (const [k, v] of l.values.entries()) {
          const item = l.items?.[k]
          let mean: number | { readonly const: number } = { const: v }
          if (item !== undefined && item.terms.size > 0)
            mean = gate.slot(this.cx.g.materialise(item, node.name, node.dims.length === 0 ? undefined : i) as string)
          gate.add({ operator: 'gaussian', args: [0, mean], params: { precision: 1 / variance } })
        }
        this.cx.g.addFactor(gate.factor(), name)
      } else if (isConstCell(l)) this.cx.g.addFactor(gaussianPrior(name, l.aff.offset, variance), name)
      else {
        const m = this.cx.g.materialise(affOf(l), node.name, node.dims.length === 0 ? undefined : i)
        this.cx.g.addFactor(
          { operator: 'gaussian', args: [name, m as string], params: { precision: 1 / variance } },
          name,
        )
      }
      cells.push(this.cx.g.varAtom(name))
      vars.push(name)
    }
    this.cx.finishLatent(node, 'gaussian', cells, vars)
  }

  /**
   * A scalar normal latent that belongs to a group (`findGroups`): the first member made creates the joint Gaussian vector of
   * the group with a diagonal prior and a scalar readout variable for each member (the marginal of its coordinate, which is
   * what the posterior reads), and the members' cells are coordinates of the vector, so a predictor over several of them is
   * one dot product on it.
   */
  private latentGrouped(
    node: ModelNode & { readonly kind: 'latent' },
    group: { readonly vector: string; readonly members: readonly string[] },
  ): void {
    if (!this.cx.groupsMade.has(group.vector)) {
      this.cx.groupsMade.add(group.vector)
      const size = group.members.length
      const mean: number[] = []
      const cov = new Array<number>(size * size).fill(0)
      group.members.forEach((m, k) => {
        const member = this.cx.nodes.get(m) as ModelNode & { readonly kind: 'latent' }
        const loc = constParam(this.cx, member, 'loc', 'the mean')[0] as number
        const sd = constParam(this.cx, member, 'scale', 'the scale')[0] as number
        if (!(sd > 0)) throw new Refusal('data', `the scale of '${m}' is ${sd}: it must be positive`)
        mean.push(loc)
        cov[k * size + k] = sd * sd
      })
      this.cx.g.addVariable({ name: group.vector, family: { kind: 'gaussian-mv', size } })
      this.cx.g.auxiliary.push(group.vector)
      this.cx.g.addFactor(gaussianMvPrior(group.vector, mean, cov), group.vector)
      group.members.forEach((m, k) => {
        this.cx.g.addVariable({ name: m, family: { kind: 'gaussian' } })
        const a = new Array<number>(size).fill(0)
        a[k] = 1
        this.cx.g.addFactor({ operator: 'dot-constant', args: [m, group.vector], params: { a } }, m)
        this.cx.syms.set(m, { dims: [], cells: [this.cx.g.compAtom(group.vector, k, size)] })
      })
    }
    this.cx.layout(node, 'latent', 'gaussian', 'scalar', [node.name], true)
  }

  /**
   * A spike and slab latent (`normalSpikeSlab`: a mixture of a wide and a narrow zero-mean normal on every element): each
   * element is a Gaussian variable (or a coordinate of one packed Gaussian vector, when a `matmul` reads it) with a
   * `mixture-site` (`mixture-row`) factor, whose EP message is exact. A scalar site speaks first through the engine's flat-cavity
   * policy (`needsDensity`); the row on a packed vector starts with a very wide Gaussian message (a hundred slab standard
   * deviations), so the other factors have a proper cavity to begin from, and its own first update replaces it.
   */
  private latentSpikeSlab(node: ModelNode & { readonly kind: 'latent' }): void {
    const n = prod(this.cx.shapeOf(node.dims))
    if (['inclusion', 'slab', 'spike'].some((p) => !paramIsConst(this.cx, node, p))) {
      this.latentSpikeSlabSwitch(node, n)
      return
    }
    const inclusion = constParam(this.cx, node, 'inclusion', 'the inclusion probability')
    const slab = constParam(this.cx, node, 'slab', 'the slab scale')
    const spike = constParam(this.cx, node, 'spike', 'the spike scale')
    for (let i = 0; i < n; i++)
      if (!(slab[i] !== undefined && spike[i] !== undefined && (slab[i] as number) > 0 && (spike[i] as number) > 0))
        throw new Refusal('data', `the scales of '${node.name}' must be positive`)
    const mixture = (i: number): Record<string, Json> => ({
      weights: [inclusion[i] as number, 1 - (inclusion[i] as number)],
      means: [0, 0],
      sds: [slab[i] as number, spike[i] as number],
    })
    const wide = (i: number): number => (100 * Math.max(slab[i] as number, spike[i] as number)) ** 2
    if (this.cx.plan.packed.has(node.name)) {
      this.cx.g.need('mixture-row')
      this.cx.g.addVariable({ name: node.name, family: { kind: 'gaussian-mv', size: n } })
      for (let i = 0; i < n; i++) {
        const a = new Array<number>(n).fill(0)
        a[i] = 1
        // The row starts with a vague message in its own coordinate, so the other factors have a proper cavity to begin from.
        const start = [...new Array<number>(n).fill(0), ...new Array<number>(n * n).fill(0)]
        start[n + i * n + i] = 1 / wide(i)
        this.cx.g.addFactor(
          { operator: 'mixture-row', args: [node.name], params: { a, ...mixture(i) }, start: [start] },
          this.cx.elementName(node.name, node.dims, i),
        )
      }
      this.cx.syms.set(node.name, {
        dims: node.dims,
        cells: Array.from({ length: n }, (_, j) => this.cx.g.compAtom(node.name, j, n)),
      })
      this.cx.layout(node, 'latent', 'gaussian-mv', 'packed', [node.name], true)
      return
    }
    const cells: Cell[] = []
    const vars: string[] = []
    for (let i = 0; i < n; i++) {
      const name = this.cx.g.addVariable({
        name: this.cx.elementName(node.name, node.dims, i),
        family: { kind: 'gaussian' },
      })
      this.cx.g.addFactor({ operator: 'mixture-site', args: [name], params: mixture(i) }, name)
      cells.push(this.cx.g.varAtom(name))
      vars.push(name)
    }
    this.cx.finishLatent(node, 'gaussian', cells, vars)
  }

  /**
   * A spike and slab latent with a random inclusion probability (a beta latent) or random slab or spike widths (the precision idiom
   * of a gamma latent), the switch summed out: each element is a Gaussian variable under the gate of its spike and slab Gaussians
   * over a hidden Bernoulli switch (`hiddenSwitch`), the same graph as the explicit-switch form with the switch left out of the
   * posterior. A vector that a `matmul` reads (one joint Gaussian) has one covariance, so it is refused.
   */
  private latentSpikeSlabSwitch(node: ModelNode & { readonly kind: 'latent' }, n: number): void {
    if (this.cx.plan.packed.has(node.name))
      throw new Refusal(
        'operator',
        `'${node.name}' is read as a vector by a matmul and has a random inclusion or scale: a joint Gaussian vector has one covariance, so they must be known`,
      )
    const inclusion = this.cx.over(paramSym(this.cx, node.dist, 'inclusion'), node.dims)
    // The precision of each element's slab and spike: a known one, or the gamma variable's multiple.
    const precisions = (param: 'slab' | 'spike'): ({ readonly sd: number } | Precision)[] =>
      paramIsConst(this.cx, node, param)
        ? constParam(this.cx, node, param, `the ${param} scale`).map((sd) => {
            if (!(sd > 0))
              throw new Refusal('data', `the ${param} scale of '${node.name}' is ${sd}: it must be positive`)
            return { sd }
          })
        : randomPrecisions(this.cx, node, param)
    const slab = precisions('slab')
    const spike = precisions('spike')
    const cells: Cell[] = []
    const vars: string[] = []
    for (let i = 0; i < n; i++) {
      const index = node.dims.length === 0 ? undefined : i
      const name = this.cx.g.addVariable({
        name: this.cx.elementName(node.name, node.dims, i),
        family: { kind: 'gaussian' },
      })
      const z = hiddenSwitch(this.cx, node, inclusion[i] as Cell, index)
      const gate = new GateBuilder(z, [name])
      const branch = (p: { readonly sd: number } | Precision): GateBranch => {
        if ('sd' in p) return { operator: 'gaussian', args: [0, { const: 0 }], params: { precision: 1 / p.sd ** 2 } }
        const at = gate.slot(p.variable)
        this.cx.g.need('gaussian-gamma-precision')
        return precisionBranch([0, { const: 0 }, at], p)
      }
      // State 0 of the switch is the spike, state 1 the slab (the probability of the Bernoulli is the inclusion).
      gate.add(branch(spike[i] as never))
      gate.add(branch(slab[i] as never))
      this.cx.g.need('gate')
      this.cx.g.addFactor(gate.factor(), name)
      cells.push(this.cx.g.varAtom(name))
      vars.push(name)
    }
    this.cx.finishLatent(node, 'gaussian', cells, vars)
  }

  /**
   * A Gaussian latent whose scale is a random positive value (a half-normal or half-Cauchy sd of a hierarchy, `exp` of a
   * Gaussian), written in its non-centred form: `x = loc + scale * z` with a standard normal `z`, the product of two Gaussian
   * variables (the `product` factor of the engine). A scale that is not an affine Gaussian value is refused with the
   * precision idiom named.
   */
  private latentRandomScale(
    node: ModelNode & { readonly kind: 'latent' },
    loc: readonly Cell[],
    scaleExpr: ExprSpec,
  ): void {
    let scales: Cell[]
    try {
      scales = this.cx.over(evalSym(this.cx, stripClip(this.cx.idiom, scaleExpr)), node.dims)
    } catch (e) {
      if (!(e instanceof Refusal)) throw e
      this.latentRandomPrecision(node, loc)
      return
    }
    if (scales.some((c) => c.kind !== 'aff')) {
      this.latentRandomPrecision(node, loc)
      return
    }
    this.cx.g.need('product')
    const cells: Cell[] = []
    const vars: string[] = []
    for (const [i, c] of scales.entries()) {
      const index = node.dims.length === 0 ? undefined : i
      const name = this.cx.elementName(node.name, node.dims, i)
      const z = this.cx.g.auxName(node.name, index)
      this.cx.g.addVariable({ name: z, family: { kind: 'gaussian' } })
      this.cx.g.addFactor(gaussianPrior(z, 0, 1), z)
      const zAtom = this.cx.g.atomKey({ kind: 'var', name: z })
      this.cx.g.atoms.set(zAtom, { kind: 'var', name: z })
      const scale = affOf(c)
      const spread = mulAff(this.cx, scale, { terms: new Map([[zAtom, 1]]), offset: 0 }, i)
      this.cx.g.materialise(combine(affOf(loc[i] as Cell), spread, 1), node.name, index, name)
      cells.push(this.cx.g.varAtom(name))
      vars.push(name)
    }
    this.cx.finishLatent(node, 'gaussian', cells, vars)
  }

  /**
   * A Gaussian vector (read as a vector by a matmul) whose coefficients have the precisions `s_j lambda + o_j` of one shared
   * gamma variable (a smoothing parameter): one `gaussian-mv` variable with one `gaussian-mv-precision-scale` factor, so the
   * posterior keeps the dense correlations of the coefficients and the multiplier is inferred from them.
   */
  private latentSharedPrecision(node: ModelNode & { readonly kind: 'latent' }, loc: readonly Cell[]): void {
    if (!loc.every((c) => isConstCell(c) && c.aff.offset === 0))
      throw new Refusal(
        'operator',
        `'${node.name}' has a shared random precision and a mean: the vector with a smoothing parameter must have a zero known mean`,
      )
    const taus = randomPrecisions(this.cx, node)
    const shared = taus[0]?.variable
    if (shared === undefined || taus.some((t) => t.variable !== shared))
      throw new Refusal(
        'operator',
        `the precisions of '${node.name}' must be multiples of one gamma variable plus constants (a vector with several random precisions is the ARD form, one variable per coefficient)`,
      )
    this.cx.g.need('gaussian-mv-precision-scale')
    const n = taus.length
    this.cx.g.addVariable({ name: node.name, family: { kind: 'gaussian-mv', size: n } })
    this.cx.g.addFactor(
      {
        operator: 'gaussian-mv-precision-scale',
        args: [node.name, shared],
        params: { scale: taus.map((t) => t.scale), offset: taus.map((t) => t.offset) },
      },
      node.name,
    )
    this.cx.syms.set(node.name, {
      dims: node.dims,
      cells: Array.from({ length: n }, (_, j) => this.cx.g.compAtom(node.name, j, n)),
    })
    this.cx.layout(node, 'latent', 'gaussian-mv', 'packed', [node.name], true)
  }

  /** A Gaussian latent with a gamma precision: a `gaussian-gamma-precision` factor per element (the mean-field hierarchy of Bishop 2006, 10.2). */
  private latentRandomPrecision(node: ModelNode & { readonly kind: 'latent' }, loc: readonly Cell[]): void {
    if (this.cx.plan.packed.has(node.name))
      throw new Refusal(
        'operator',
        `'${node.name}' is read as a vector by a matmul and has a random scale: a joint Gaussian vector has one covariance, so its scale must be known`,
      )
    const taus = randomPrecisions(this.cx, node)
    const cells: Cell[] = []
    const vars: string[] = []
    for (const [i, tau] of taus.entries()) {
      const name = this.cx.g.addVariable({
        name: this.cx.elementName(node.name, node.dims, i),
        family: { kind: 'gaussian' },
      })
      const l = loc[i] as Cell
      const mean = this.cx.g.portOf(l, node.name, node.dims.length === 0 ? undefined : i)
      this.cx.g.addFactor(precisionFactor(name, mean, tau), name)
      cells.push(this.cx.g.varAtom(name))
      vars.push(name)
    }
    this.cx.finishLatent(node, 'gaussian', cells, vars)
  }

  private observed(node: ModelNode & { readonly kind: 'observed' }): void {
    if (node.dist.family === 'multivariateNormal' && wishartOf(this.cx, node) !== undefined) {
      observedMvnWishart(this.cx, node)
      return
    }
    if (node.dist.family === 'bivariateNormal') {
      observedBivariateNormal(this.cx, node)
      return
    }
    const moved = valueMovedToMean(this.cx, node)
    if (moved !== undefined) {
      this.observed(moved)
      return
    }
    const fam = node.dist.family
    const shape = this.cx.shapeOf(node.dims)
    const n = prod(shape)
    const info = this.cx.bound.missing[node.name]
    if (info !== undefined && info.count > 0 && info.policy !== 'ignore')
      throw new Refusal(
        'data',
        `'${node.name}' has ${info.count} missing value(s) with the policy '${info.policy}': only 'ignore' is mapped (the entries drop out of the graph)`,
      )
    if (this.cx.bound.batch !== undefined && node.dims.includes(this.cx.bound.batch.dim))
      throw new Refusal(
        'structure',
        `'${node.name}' is subsampled by the batch option: a factor graph is built over all the rows`,
      )
    const values = this.cx
      .over(this.cx.fromNamed(evalNamedValue(this.cx, node)), node.dims)
      .map((c) => (c as Cell & { kind: 'aff' }).aff.offset)
    const normalGamma = this.cx.normalGammaObserved.get(node.name)
    if (normalGamma !== undefined) {
      observedNormalGamma(this.cx, node, normalGamma, values)
      return
    }
    const vars: (string | null)[] = []
    const parentsOf = discreteParents(this.cx, node)
    if (this.cx.vectorise && node.dims.length > 0 && parentsOf.length === 0 && plate(this.cx, node, fam, values)) return
    if (fam === 'bernoulli' && parentsOf.length > 0) {
      if (node.dims.length > 0)
        throw new Refusal(
          'structure',
          `'${node.name}' is a Bernoulli over [${node.dims}] with discrete parents: tables over plated nodes are not expanded yet`,
        )
      const name = this.cx.g.addVariable({
        name: node.name,
        family: { kind: 'bernoulli' },
        observed: values[0] as number,
      })
      const pv = parentsOf.map((p) => rvName(this.cx, p))
      this.cx.g.addFactor(
        {
          operator: 'table',
          args: [...pv, name],
          params: {
            states: new Array<number>(pv.length + 1).fill(2),
            values: bernoulliTable(this.cx, node, parentsOf),
          },
        },
        name,
      )
      this.cx.layout(node, 'observed', 'bernoulli', 'scalar', [name], false)
      return
    }
    for (let i = 0; i < n; i++) {
      const y = values[i] as number
      if (Number.isNaN(y)) {
        vars.push(null)
        continue
      }
      const name = this.cx.elementName(node.name, node.dims, i)
      vars.push(this.observedElement(node, fam, name, y, i))
    }
    this.cx.layout(node, 'observed', undefined, node.dims.length === 0 ? 'scalar' : 'observed', vars, false)
  }

  private observedElement(
    node: ModelNode & { readonly kind: 'observed' },
    fam: string,
    name: string,
    y: number,
    i: number,
  ): string | null {
    const index = node.dims.length === 0 ? undefined : i
    switch (fam) {
      case 'normal': {
        const scaleExpr = node.dist.params['scale']
        // A log scale (and a mean) chosen by one discrete variable, each state with random ones of its own: one gate of `gaussian-scale` branches.
        if (observedSwitchLogScale(this.cx, node, y, i, index)) return null
        // A mean or a scale chosen by a discrete variable (a lookup): a table or a gate over that variable.
        const switched = switchedCells(this.cx, node, i)
        if (switched !== undefined) return observedSwitch(this.cx, node, name, y, index, switched.loc, switched.scale)
        if (scaleExpr !== undefined && !this.cx.isConst(scaleExpr)) {
          const logScale =
            expArgument(unwrap(this.cx.nodes, scaleExpr)) !== undefined
              ? observedLogScale(this.cx, node, name, y, i, index)
              : undefined
          if (logScale !== undefined) return logScale
          if (precisionExpr(this.cx.idiom, scaleExpr) === undefined)
            return observedRandomScale(this.cx, node, name, y, i, index, scaleExpr)
          return observedRandomPrecision(this.cx, node, name, y, i, index)
        }
        return observedGaussian(this.cx, node, name, y, i, index, 'loc', 'scale')
      }
      case 'studentT':
      case 'laplace':
      case 'normalOutliers':
        return observedHeavy(this.cx, node, fam, y, i, index)
      case 'beta':
        return observedPair(this.cx, node, 'beta-shapes', y, i, index)
      case 'orderedProbit':
      case 'orderedLogistic':
      case 'asymmetricLaplace': {
        const numeric = observedNumeric(this.cx, node, fam, y, i, index)
        if (numeric === undefined)
          throw new Refusal(
            'operator',
            `the parameters of '${node.name}' (${fam}) are not in a form with a message rule`,
          )
        return numeric
      }
      case 'logNormal':
        // log y is Gaussian around the predictor: the exact Gaussian rule on the log of the outcome.
        if (!(y > 0)) throw new Refusal('data', `'${node.name}' is log-normal: the outcome ${y} must be positive`)
        return observedGaussian(this.cx, node, name, Math.log(y), i, index, 'mu', 'sigma')
      case 'bernoulli':
        return observedBernoulli(this.cx, node, name, y, i, index)
      case 'binomial':
      case 'poisson': {
        const numeric = observedNumeric(this.cx, node, fam, y, i, index)
        return numeric !== undefined ? numeric : observedConjugate(this.cx, node, fam, name, y, i)
      }
      case 'categorical':
        if (node.dist.params['logits'] !== undefined)
          return observedMulticlass(this.cx, node, y, i, index, 'logits', 'logit')
        return observedConjugate(this.cx, node, fam, name, y, i)
      case 'categoricalProbit':
        return observedMulticlass(this.cx, node, y, i, index, 'scores', 'probit')
      case 'zeroTruncatedPoisson': {
        const numeric = observedNumeric(this.cx, node, fam, y, i, index)
        if (numeric === undefined)
          throw new Refusal(
            'operator',
            `the rate of '${node.name}' must be \`exposure * exp(affine)\` for the zero-truncated Poisson`,
          )
        return numeric
      }
      case 'hurdlePoisson':
        return observedHurdle(this.cx, node, y, i, index)
      case 'normalPair':
        return observedMixtureWeight(this.cx, node, name, y, i)
      case 'exponential':
      case 'gamma':
        return observedGammaRate(this.cx, node, fam, name, y, i)
      case 'geometric':
        return observedFailures(this.cx, node, fam, name, y, i)
      case 'negativeBinomial':
      case 'weibull': {
        const numeric = observedNumeric(this.cx, node, fam, y, i, index)
        if (numeric === undefined && fam === 'negativeBinomial') return observedFailures(this.cx, node, fam, name, y, i)
        if (numeric === undefined)
          throw new Refusal(
            'operator',
            `the parameters of '${node.name}' (${fam}) are not in a form with a message rule (see the log-link idioms of its observedNumeric)`,
          )
        return numeric
      }
      default:
        throw new Refusal(
          'family',
          `'${node.name}' is observed as ${fam}: the likelihoods mapped are normal, logNormal, studentT, laplace, asymmetricLaplace, orderedProbit, orderedLogistic, categoricalProbit, normalOutliers, bernoulli, binomial, poisson, normalPair, exponential, gamma, geometric, negativeBinomial, weibull and categorical`,
        )
    }
  }
}

/**
 * Compile a bound model to a factor graph for message passing, by expanding its plates. The result carries the graph (or
 * `undefined` with the reasons in `issues` when something cannot be mapped), the layout of each site for reading results
 * back, the auxiliary variables, the opaque nodes `analyseSpec` found, and the size of the expansion. Observed data is
 * folded into the graph (the observed variables' values and the factors' constants), so the graph belongs to the bound
 * model: rebind and compile again for new data.
 *
 * Nodes that the engine has no counterpart for are never dropped silently. A `param` node, a `factor` (log-density)
 * term, a family with no message family, a `custom` call, a random scale that is not the precision idiom, a square of a random
 * precision, a non-affine operator on a random value (except as a Bernoulli link or one of the unary functions of the
 * registry), a gate over components of different families, a computed random index, a constraint beyond
 * the support of a Gaussian (a simplex, a positive-definite matrix), missing data that is not ignored, a subsampling batch and a plated table over discrete nodes each become an
 * issue naming the node and the reason, and no graph is produced. (A product of two random precisions is mapped, as a `gamma-product`
 * variable, and a gate over known and random components is a lookup that the observation reads as a gate of branches.)
 *
 * @param bound - The bound model.
 * @param options - The factor operators to compile against (default the engine's `defaultOperators()`), the model's operator table, and `plates`: `vectorise` (default) or `expand`.
 * @returns The compilation.
 * @concept model-spec
 * @index function Compile a bound ModelSpec to a message-passing factor graph, plates expanded.
 * @example
 * ```ts
 * import { defineModel, ops, toFactorGraph } from 'omai-engine/probprog/model'
 * import { bernoulli, normal } from 'omai-engine/probprog/model/dists'
 * const m = defineModel('probit', (b) => {
 *   const n = b.range('row'), f = b.range('feature')
 *   const X = b.data('X', { dims: [n, f] })
 *   const y = b.data('y', { dims: [n], support: 'binary' })
 *   const w = b.latent('w', normal(0, 2), { dims: [f] })
 *   b.observe('y', bernoulli({ probs: ops.probit(X.matmul(w)) }), y)
 * })
 * const bound = m.bind({ X: [[1, 0.5], [1, -1]], y: [1, 0] })
 * const plated = toFactorGraph(bound)
 * plated.graph?.factors.map((f) => f.id) // => ['factor:w', 'factor:y']
 * const rows = toFactorGraph(bound, { plates: 'expand' })
 * rows.graph?.variables.map((v) => v.name) // => ['w', 'y#0[0]', 'y[0]', 'y#0[1]', 'y[1]']
 * rows.graph?.factors.map((f) => f.id) // => ['factor:w', 'factor:y#0[0]', 'factor:y[0]', 'factor:y#0[1]', 'factor:y[1]']
 * ```
 */
export function toFactorGraph(
  bound: BoundModel,
  options: {
    readonly operators?: Readonly<Record<string, FactorOperator>>
    readonly modelOperators?: OperatorTable
    readonly analysis?: SpecAnalysis
    /** `vectorise` (default): an observed node whose rows are one regression on a weight vector is one plated factor; `expand`: a factor and a variable per row. */
    readonly plates?: 'vectorise' | 'expand'
    /**
     * Compile the scalar normal latents that an observed node reads together (the intercept and slope of `a + b x`) as the
     * coordinates of one joint Gaussian vector, so message passing keeps their correlation (default false: scalar variables,
     * whose loop of messages keeps the means and loses the correlation, so the marginal sds come out too small).
     */
    readonly packScalars?: boolean
    /**
     * A vector whose coefficients have the precisions of one shared gamma variable (a smoothing parameter): `dense` (default)
     * is one Gaussian vector with a `gaussian-mv-precision-scale` factor (the posterior keeps its correlations), `elements` a
     * scalar variable and a `gaussian-gamma-precision` factor per coefficient (the same prior, a mean-field-like posterior).
     */
    readonly sharedScale?: 'dense' | 'elements'
  } = {},
): FactorGraphCompilation {
  // A vector written as one event (a multivariate normal with no named dimension) is read as a vector over a dimension of its size.
  const eventless = eventToDims(bound)
  if (eventless !== bound) return toFactorGraph(eventless, options)
  if (bound.spec.nodes.some((n) => n.kind === 'scan')) {
    // A chain has no plate: the engine reads it written out, one variable and factor per step.
    let unrolled: BoundModel
    try {
      unrolled = unrollScans(bound)
    } catch (e) {
      if (!(e instanceof ScanUnrollError)) throw e
      return {
        name: bound.spec.name,
        graph: undefined,
        issues: [{ node: e.node, kind: 'structure', reason: e.message }],
        opaque: [],
        sites: [],
        auxiliary: [],
        plated: [],
        expanded: { plates: [], variables: 0, factors: 0 },
      }
    }
    return toFactorGraph(unrolled, options)
  }
  const c = new Compiler(
    bound,
    options.modelOperators ?? defaultOperators(),
    options.operators ?? factorOperators(),
    options.plates !== 'expand',
    options.packScalars === true,
    options.sharedScale !== 'elements',
  )
  c.run()
  const cx = c.cx
  cx.g.prune(cx.sites)
  const graph: FactorGraphSpec = { variables: cx.g.variables, factors: cx.g.factors }
  const opaque = (options.analysis?.opaque ?? opaqueNodes(bound)).filter((name) => {
    const node = bound.spec.nodes.find((n) => n.name === name && n.kind !== 'data')
    if (node === undefined) return true
    const called = customNames(node)
    // A node whose custom calls were all evaluated (known arguments) or read as a matrix (affine in one random vector) is not opaque.
    return called.length === 0 || !called.every((n) => cx.handledCustoms.has(n))
  })
  for (const name of opaque)
    if (!cx.issues.some((i) => i.node === name))
      cx.issues.push({
        node: name,
        kind: 'opaque',
        reason: `'${name}' is opaque to message passing (a custom call, or a family with no exponential-family form)`,
      })
  const plates = bound.spec.dims
    .filter((d) => cx.sites.some((s) => s.dims.includes(d.name)))
    .map((d) => ({ dim: d.name, size: bound.sizes[d.name] as number }))
  return {
    name: bound.spec.name,
    graph: cx.issues.length === 0 ? graph : undefined,
    issues: cx.issues,
    opaque,
    sites: cx.sites,
    auxiliary: cx.g.auxiliary,
    plated: cx.plated,
    expanded: { plates, variables: graph.variables.length, factors: graph.factors.length },
  }
}

/** The names of the custom functions a node calls. */
function customNames(node: ModelNode): string[] {
  const out = new Set<string>()
  const walk = (e: ExprSpec): void => {
    if (e.kind === 'custom') out.add(e.name)
    if (e.kind === 'op' || e.kind === 'custom') for (const a of e.args) walk(a)
  }
  if (node.kind === 'latent' || node.kind === 'observed' || node.kind === 'generated') {
    for (const e of Object.values(node.dist.params)) walk(e)
    if (node.kind === 'observed') walk(node.value)
  } else if (node.kind === 'derived' || node.kind === 'factor') walk(node.expr)
  return [...out]
}

/** The nodes with a `custom` call, found without the analysis (for a caller that gives none). */
function opaqueNodes(bound: BoundModel): string[] {
  const custom = (e: ExprSpec): boolean => e.kind === 'custom' || (e.kind === 'op' && e.args.some(custom))
  return bound.spec.nodes
    .filter((n) => {
      if (n.kind === 'latent' || n.kind === 'observed' || n.kind === 'generated')
        return Object.values(n.dist.params).some(custom) || (n.kind === 'observed' && custom(n.value))
      return n.kind === 'derived' || n.kind === 'factor' ? custom(n.expr) : false
    })
    .map((n) => n.name)
}
