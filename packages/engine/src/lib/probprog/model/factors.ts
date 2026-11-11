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

import {
  betaEta,
  type FactorGraphSpec,
  type FactorOperator,
  type FamilyKind,
  defaultOperators as factorOperators,
  gammaPrior,
  priorFactor,
} from '../../message-passing/index.ts'
import type { SpecAnalysis } from './analysis.ts'
import type { BoundModel } from './bind.ts'
import { eventToDims } from './event-dims.ts'
import { type Cell, prod } from './factor-graph/_cells.ts'
import { Context } from './factor-graph/_context.ts'
import { expArgument, precisionExpr, unwrap } from './factor-graph/_idioms.ts'
import { derived } from './factor-graph/_mappings/derived.ts'
import { bernoulliTable, latentDirichlet, latentDiscrete } from './factor-graph/_mappings/latent-discrete.ts'
import { latentNormal, latentProjected, latentSpikeSlab, truncate } from './factor-graph/_mappings/latent-gaussian.ts'
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
import { constParam, discreteParents, evalNamedValue, rvName } from './factor-graph/_params.ts'
import { Refusal } from './factor-graph/_refusal.ts'
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
      latentProjected(this.cx, node)
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
      latentNormal(this.cx, node)
      if (truncated) truncate(this.cx, node)
      return
    }
    if (fam === 'multivariateNormal') {
      latentMultivariateNormal(this.cx, node)
      return
    }
    if (fam === 'normalSpikeSlab') {
      latentSpikeSlab(this.cx, node)
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
