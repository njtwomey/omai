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
  bernoulliEta,
  betaEta,
  dirichletPrior,
  discretePrior,
  type FactorGraphSpec,
  type FactorOperator,
  type FactorSpec,
  type FamilyKind,
  defaultOperators as factorOperators,
  gammaPrior,
  gaussianMvPrior,
  gaussianPrior,
  likelihoods,
  normalGammaPrior,
  pairLikelihoods,
  priorFactor,
  wishartEta,
} from '../../message-passing/index.ts'
import { inv } from '../../numerics/linalg/index.ts'
import { asTensor, toFlat } from '../../tensor/index.ts'
import type { SpecAnalysis } from './analysis.ts'
import type { BoundModel } from './bind.ts'
import { eventToDims } from './event-dims.ts'
import {
  discreteRowGate,
  factorName,
  type GateBranch,
  GateBuilder,
  precisionBranch,
  precisionFactor,
} from './factor-graph/_builder.ts'
import {
  type Aff,
  type Atom,
  affOf,
  type Cell,
  combine,
  constAff,
  isConstCell,
  logName,
  lookup,
  type Precision,
  prod,
  scaledOf,
  unflatten,
} from './factor-graph/_cells.ts'
import { Context } from './factor-graph/_context.ts'
import {
  expArgument,
  logVariableOf,
  logVariableOfCell,
  mentionsDiscrete,
  precisionExpr,
  randomTransitionRow,
  scaledExp,
  stripClip,
  transitionRow,
  unwrap,
} from './factor-graph/_idioms.ts'
import {
  cellsOf,
  constAt,
  constParam,
  discreteParents,
  evalNamedValue,
  eventVector,
  paramCell,
  paramIsConst,
  paramSym,
  randomPrecisions,
  rvName,
} from './factor-graph/_params.ts'
import type { NormalGammaPlan } from './factor-graph/_plan.ts'
import { Refusal } from './factor-graph/_refusal.ts'
import { evalSym } from './factor-graph/_symbolic/_affine.ts'
import type { ApplyStep } from './factor-graph/_symbolic/_functions.ts'
import { chainOf, mulAff } from './factor-graph/_symbolic/_functions.ts'
import { gate } from './factor-graph/_symbolic/_gates.ts'
import type { FactorGraphCompilation } from './factor-graph/types.ts'
import { elementId } from './ids.ts'
import { defaultOperators, type NamedValue, type OperatorTable } from './operators.ts'
import { ScanUnrollError, unrollScans } from './scan.ts'
import type { DistSpec, ExprSpec, ModelNode } from './spec.ts'

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

/** One component of a random concentration, `weight * alpha + offset`, with the variable of `alpha` (the variable of its log when `log`). */
interface ConcentrationPart {
  readonly variable?: string
  readonly log?: boolean
  readonly weight: number
  readonly offset: number
}

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
        this.derived(node)
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
      this.latentWishart(node)
      return
    }
    if (fam === 'correlation') {
      this.latentCorrelation(node)
      return
    }
    if (fam === 'bivariateNormal')
      throw new Refusal(
        'structure',
        `'${node.name}' is a latent bivariateNormal: only observed vectors are mapped (write the vectors with a Wishart precision, or observe them)`,
      )
    if (fam === 'multivariateNormal' && this.wishartOf(node) !== undefined) {
      this.latentMvnWishart(node)
      return
    }
    const meanOf = this.cx.normalGammaByMean.get(node.name)
    if (meanOf !== undefined) {
      this.cx.layout(node, 'latent', 'normal-gamma', 'scalar', [meanOf.variable], true, { part: 'mean' })
      return
    }
    if ((fam === 'gamma' || fam === 'inverseGamma') && this.latentNormalGamma(node)) return
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
      this.latentMultivariateNormal(node)
      return
    }
    if (fam === 'normalSpikeSlab') {
      this.latentSpikeSlab(node)
      return
    }
    if (fam === 'bernoulli' || fam === 'categorical') {
      this.latentDiscrete(node)
      return
    }
    if (fam === 'dirichlet') {
      const concentration = node.dist.params['concentration']
      if (concentration !== undefined && !this.cx.isConst(concentration)) this.latentRandomConcentration(node)
      else this.latentDirichlet(node)
      return
    }
    if (fam === 'beta' && ['a', 'b'].some((k) => !this.cx.isConst(node.dist.params[k] as ExprSpec))) {
      this.latentRandomConcentration(node)
      return
    }
    // An inverse gamma `x` with scale `b` is the precision `1 / x` ~ Gamma(a, rate b): the variable is that gamma, the site its reciprocal.
    const inverse = fam === 'inverseGamma'
    const rateName = inverse ? 'scale' : 'rate'
    const rateExpr = node.dist.params[rateName]
    if ((fam === 'gamma' || inverse) && rateExpr !== undefined && !this.cx.isConst(rateExpr)) {
      this.latentGammaRate(node)
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

  /**
   * The normal-gamma pair of a gamma (or inverse-gamma) latent `T` (a precision or a variance), when the plan found one (a Gaussian
   * latent `M` with a known mean and the variance `1 / (kappa lambda)` and observed Gaussians `y ~ N(M, 1 / (c lambda))`, with
   * nothing else reading either of them; `_plan.ts` has the detection). The pair is one `normal-gamma` variable with a prior of both
   * and a `gaussian-normal-gamma` factor per observation, so the joint posterior is exactly the conjugate normal-gamma (the classic
   * normal-inverse-gamma prior on a mean and a variance). Returns false (and changes nothing) when the model is not that shape: the
   * mean and precision then compile to a Gaussian with a gamma precision each.
   */
  private latentNormalGamma(node: ModelNode & { readonly kind: 'latent' }): boolean {
    const detection = this.cx.plan.normalGamma.get(node.name)
    if (detection === undefined) return false
    for (const name of detection.customs) this.cx.handledCustoms.add(name)
    if (detection.failure !== undefined) throw detection.failure
    if (detection.pair === undefined) return false
    const { plan, multiples } = detection.pair
    const inverse = plan.inverse
    const rateName = inverse ? 'scale' : 'rate'
    const mean = this.cx.nodes.get(plan.mean) as ModelNode & { readonly kind: 'latent' }
    this.cx.g.need('gaussian-normal-gamma')
    const a = constParam(this.cx, node, 'concentration', 'the shape')[0] as number
    const b = constParam(this.cx, node, rateName, 'the rate')[0] as number
    const m = constParam(this.cx, mean, 'loc', 'the mean')[0] as number
    this.cx.normalGammaByMean.set(mean.name, plan)
    for (const name of plan.observed)
      this.cx.normalGammaObserved.set(name, { plan, multiple: multiples.get(name) as number })
    this.cx.g.addVariable({ name: plan.variable, family: { kind: 'normal-gamma' } })
    this.cx.g.addFactor(normalGammaPrior(plan.variable, m, plan.kappa, a, b), plan.variable)
    this.cx.layout(node, 'latent', 'normal-gamma', 'scalar', [plan.variable], true, {
      part: 'precision',
      ...(inverse ? { reciprocal: true } : {}),
    })
    return true
  }

  /** The observed Gaussian rows of a normal-gamma pair: a `gaussian-normal-gamma` factor on the shared variable per row. */
  private observedNormalGamma(
    node: ModelNode & { readonly kind: 'observed' },
    ng: { readonly plan: NormalGammaPlan; readonly multiple: number },
    values: readonly number[],
  ): void {
    const vars: (string | null)[] = []
    values.forEach((y, i) => {
      if (Number.isNaN(y)) {
        vars.push(null)
        return
      }
      const name = this.cx.elementName(node.name, node.dims, i)
      this.cx.g.addVariable({ name, family: { kind: 'gaussian' }, observed: y })
      this.cx.g.addFactor(
        {
          operator: 'gaussian-normal-gamma',
          args: [name, ng.plan.variable],
          ...(ng.multiple === 1 ? {} : { params: { precision: ng.multiple } }),
        },
        name,
      )
      vars.push(name)
    })
    this.cx.layout(node, 'observed', undefined, node.dims.length === 0 ? 'scalar' : 'observed', vars, false)
  }

  /**
   * A multivariate normal latent with a known mean and covariance (a Gaussian process prior over the inputs of the data: the
   * covariance a constant Gram matrix, or given by its Cholesky factor `scaleTril`): one `gaussian-mv` variable with a
   * dense `gaussian-mv` prior, whose components the likelihood reads (`probit(f)`, `logits`, a Gaussian row), so the
   * correlations between the elements are kept and the engines see a dense Gaussian vector.
   */
  private latentMultivariateNormal(node: ModelNode & { readonly kind: 'latent' }): void {
    const shape = this.cx.shapeOf(node.dims)
    const n = prod(shape)
    if (node.dims.length !== 1)
      throw new Refusal('structure', `'${node.name}' is a multivariateNormal: one vector over one dimension is mapped`)
    const params = node.dist.params
    // A random mean that is a linear map of another Gaussian vector (`matvec(F, previous)`): one dense Gaussian step between the two.
    if (params['loc'] !== undefined && !this.cx.isConst(params['loc'])) {
      this.latentTransition(node, params['loc'])
      return
    }
    const known = (e: ExprSpec | undefined, what: string): number[] => {
      if (e === undefined || !this.cx.isConst(e))
        throw new Refusal(
          'operator',
          `the ${what} of '${node.name}' depends on a random value: a known matrix is mapped`,
        )
      return Array.from(toFlat(this.cx.evalNamed(e).value))
    }
    const loc = known(params['loc'], 'mean')
    const mean = loc.length === 1 ? new Array<number>(n).fill(loc[0] ?? 0) : loc
    let cov: number[]
    if (params['covariance'] !== undefined) cov = known(params['covariance'], 'covariance')
    else if (params['scaleTril'] !== undefined) {
      const L = known(params['scaleTril'], 'scale factor')
      cov = new Array<number>(n * n).fill(0)
      for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++) {
          let v = 0
          for (let k = 0; k <= Math.min(i, j); k++) v += (L[i * n + k] ?? 0) * (L[j * n + k] ?? 0)
          cov[i * n + j] = v
        }
    } else
      throw new Refusal(
        'structure',
        `'${node.name}' is a multivariateNormal with a precision: give a covariance or scaleTril`,
      )
    if (cov.length !== n * n) throw new Refusal('structure', `the covariance of '${node.name}' is not ${n} by ${n}`)
    this.cx.g.addVariable({ name: node.name, family: { kind: 'gaussian-mv', size: n } })
    this.cx.g.addFactor(gaussianMvPrior(node.name, mean, cov), node.name)
    this.cx.syms.set(node.name, {
      dims: node.dims,
      cells: Array.from({ length: n }, (_, j) => this.cx.g.compAtom(node.name, j, n)),
    })
    this.cx.layout(node, 'latent', 'gaussian-mv', 'packed', [node.name], true)
  }

  // ----- matrix-valued latents: a Wishart (or inverse-Wishart) precision, and the Gaussian vectors that read it -----

  /** The Wishart latent that the precision (or covariance) of a multivariate normal node names, or undefined for any other form. */
  private wishartOf(
    node: ModelNode & { readonly dist: DistSpec },
  ):
    | { readonly name: string; readonly variable: string; readonly d: number; readonly covariance: boolean }
    | undefined {
    for (const key of ['precision', 'covariance'] as const) {
      const e = node.dist.params[key]
      if (e === undefined) continue
      const u = unwrap(this.cx.nodes, e)
      if (u.kind !== 'ref') continue
      const w = this.cx.wisharts.get(u.name)
      if (w === undefined) continue
      if ((key === 'covariance') !== w.covariance)
        throw new Refusal(
          'structure',
          `'${node.name}' reads '${u.name}' as its ${key}, but '${u.name}' is ${w.covariance ? 'an inverse Wishart (a covariance): write { covariance: ... }' : 'a Wishart (a precision matrix): write { precision: ... }'}`,
        )
      return { name: u.name, ...w }
    }
    return undefined
  }

  /**
   * A Wishart latent (the precision matrix of a Gaussian) or an inverse-Wishart one (its covariance): one variable of the Wishart family
   * with the prior `W(df, scale)` (for the inverse Wishart `W(df, scale^-1)` on the precision, the site is its inverse). The matrix is
   * read only as the precision or covariance of a multivariate normal (`mvn-wishart`).
   */
  private latentWishart(node: ModelNode & { readonly kind: 'latent' }): void {
    const covariance = node.dist.family === 'inverseWishart'
    if (node.dims.length > 0)
      throw new Refusal(
        'structure',
        `'${node.name}' is a ${node.dist.family} over [${node.dims}]: one matrix is mapped`,
      )
    const dfExpr = node.dist.params['df']
    const scaleExpr = node.dist.params['scale']
    if (dfExpr === undefined || scaleExpr === undefined || !this.cx.isConst(dfExpr) || !this.cx.isConst(scaleExpr))
      throw new Refusal('operator', `the degrees of freedom and the scale of '${node.name}' must be known values`)
    const df = toFlat(this.cx.evalNamed(dfExpr).value)[0] as number
    const scale = this.cx.evalNamed(scaleExpr).value
    const d = scale.shape[0] as number
    if (scale.shape.length !== 2 || scale.shape[1] !== d)
      throw new Refusal('structure', `the scale of '${node.name}' must be a square matrix`)
    if (!(df > d - 1))
      throw new Refusal('data', `'${node.name}' has ${df} degrees of freedom: more than ${d - 1} are needed`)
    // W(df, S) on the precision; the inverse Wishart Psi on the covariance is W(df, Psi^-1) on the precision
    const matrix = covariance ? Array.from(toFlat(inv(scale))) : Array.from(toFlat(scale))
    const eta = wishartEta(df, matrix, d)
    const name = this.cx.g.addVariable({ name: node.name, family: { kind: 'wishart', size: d } })
    this.cx.g.addFactor({ operator: 'prior', args: [name], params: { eta } }, name)
    this.cx.wisharts.set(node.name, { variable: name, d, covariance })
    this.cx.layout(node, 'latent', 'wishart', 'scalar', [name], true, covariance ? { reciprocal: true } : {})
  }

  /**
   * The mean of a multivariate normal as the argument of `mvn-wishart`, per row: a known vector (or one per row), or a Gaussian vector
   * variable (a latent multivariate normal over a named dimension), shared by every row.
   */
  private vectorMean(
    node: ModelNode & { readonly dist: DistSpec },
    d: number,
    plateDims: readonly string[],
    vectorDim: string | undefined,
  ): { readonly at: (row: number) => string | { readonly const: readonly number[] }; readonly shared: boolean } {
    const e = node.dist.params['loc']
    if (e === undefined) throw new Refusal('structure', `'${node.name}': the mean is missing`)
    if (this.cx.isConst(e)) {
      const nv = this.cx.evalNamed(e)
      const kept = nv.dims.filter((x) => x !== vectorDim)
      const shape = this.cx.shapeOf(plateDims)
      if (nv.value.shape.length - kept.length !== 1 || nv.value.shape[nv.value.shape.length - 1] !== d)
        throw new Refusal('structure', `the mean of '${node.name}' must be a vector of ${d} (per row, when it varies)`)
      return {
        shared: kept.length === 0,
        at: (row) => ({
          const: eventVector(this.cx, { value: nv.value, dims: kept }, plateDims, unflatten(row, shape)),
        }),
      }
    }
    const sym = evalSym(this.cx, e)
    let vector: string | undefined
    const cells = sym.cells
    if (cells.length !== d)
      throw new Refusal('operator', `the mean of '${node.name}' must be one Gaussian vector of ${d} entries (or known)`)
    for (const [k, c] of cells.entries()) {
      const terms = c.kind === 'aff' ? [...c.aff.terms].filter(([, v]) => v !== 0) : []
      const atom = terms.length === 1 && terms[0]?.[1] === 1 ? this.cx.g.atoms.get(terms[0][0]) : undefined
      if (
        atom?.kind !== 'comp' ||
        atom.index !== k ||
        atom.size !== d ||
        (vector !== undefined && atom.vector !== vector)
      )
        throw new Refusal(
          'operator',
          `the mean of '${node.name}' must be a known vector or one whole Gaussian vector variable`,
        )
      vector = atom.vector
    }
    return { shared: true, at: () => vector as string }
  }

  /**
   * Gaussian vectors `x ~ N(loc, L^-1)` with `L` a Wishart latent (or `N(loc, Sigma)` with `Sigma` an inverse-Wishart one), one vector
   * per index of the plate dimensions before the last, which names the vector's axis (`dims: [group, coef]`; `dims: [coef]` is one
   * vector): a Gaussian vector variable and an `mvn-wishart` factor per vector. The mean is known or one Gaussian vector variable
   * (a population mean), so the vectors are the correlated random effects of a hierarchical model, and a likelihood reads their
   * coordinates as `theta.at(group)` summed against a design over the axis.
   */
  private latentMvnWishart(node: ModelNode & { readonly kind: 'latent' }): void {
    const w = this.wishartOf(node) as NonNullable<ReturnType<Compiler['wishartOf']>>
    if (node.dims.length === 0)
      throw new Refusal(
        'structure',
        `'${node.name}' is a multivariate normal with a Wishart precision: give the vector's axis a named dimension (dims: [..., axis])`,
      )
    const vectorDim = node.dims[node.dims.length - 1] as string
    const plateDims = node.dims.slice(0, -1)
    const d = this.cx.size(vectorDim)
    if (d !== w.d)
      throw new Refusal(
        'structure',
        `'${node.name}' has ${d} entries per vector, the matrix '${w.name}' is ${w.d} by ${w.d}`,
      )
    const shape = this.cx.shapeOf(plateDims)
    const rows = prod(shape)
    const mean = this.vectorMean(node, d, plateDims, vectorDim)
    const cells: Cell[] = []
    const vars: string[] = []
    for (let r = 0; r < rows; r++) {
      const name = this.cx.g.addVariable({
        name: this.cx.elementName(node.name, plateDims, r),
        family: { kind: 'gaussian-mv', size: d },
      })
      this.cx.g.addFactor({ operator: 'mvn-wishart', args: [name, mean.at(r), w.variable] }, name)
      for (let k = 0; k < d; k++) cells.push(this.cx.g.compAtom(name, k, d))
      vars.push(name)
    }
    this.cx.syms.set(node.name, { dims: node.dims, cells })
    this.cx.layout(node, 'latent', 'gaussian-mv', plateDims.length === 0 ? 'packed' : 'elements', vars, true)
  }

  /**
   * Observed vectors `y_i ~ N(loc, L^-1)` with `L` a Wishart latent (or `N(loc, Sigma)`, `Sigma` an inverse Wishart): a factor
   * `mvn-wishart` per row on the observed vector, the exact conjugate update of the matrix when the mean is known.
   */
  private observedMvnWishart(node: ModelNode & { readonly kind: 'observed' }): void {
    const w = this.wishartOf(node) as NonNullable<ReturnType<Compiler['wishartOf']>>
    const d = w.d
    const nv = evalNamedValue(this.cx, node)
    if (nv.value.shape[nv.value.shape.length - 1] !== d || nv.value.shape.length !== node.dims.length + 1)
      throw new Refusal('structure', `'${node.name}' is observed as vectors of ${d} entries over [${node.dims}]`)
    const shape = this.cx.shapeOf(node.dims)
    const mean = this.vectorMean(node, d, node.dims, undefined)
    if (this.plateVectors(node, nv, d, mean, 'mvn-wishart', [w.variable])) return
    const vars: (string | null)[] = []
    for (let i = 0; i < prod(shape); i++) {
      const y = eventVector(this.cx, nv, node.dims, unflatten(i, shape))
      if (y.some((v) => Number.isNaN(v))) {
        vars.push(null)
        continue
      }
      const name = this.cx.g.addVariable({
        name: this.cx.elementName(node.name, node.dims, i),
        family: { kind: 'gaussian-mv', size: d },
        observed: y,
      })
      this.cx.g.addFactor({ operator: 'mvn-wishart', args: [name, mean.at(i), w.variable] }, name)
      vars.push(name)
    }
    this.cx.layout(node, 'observed', undefined, node.dims.length === 0 ? 'scalar' : 'observed', vars, false)
  }

  /**
   * The rows of an observed vector node with a shared mean as ONE factor of `operator` (`mvn-wishart`, `mvn-sd-corr`): the sample mean of the
   * rows, their count and their scatter about it are all the likelihood depends on, so the graph does not grow with the data. False
   * when the rows are not plated (`plates: 'expand'`, a mean that varies by row, no plate dimension).
   */
  private plateVectors(
    node: ModelNode & { readonly kind: 'observed' },
    nv: NamedValue,
    d: number,
    mean: { readonly at: (row: number) => string | { readonly const: readonly number[] }; readonly shared: boolean },
    operator: string,
    tail: readonly (string | { readonly const: number })[],
  ): boolean {
    if (!this.cx.vectorise || node.dims.length === 0 || !mean.shared) return false
    const shape = this.cx.shapeOf(node.dims)
    const rows: number[][] = []
    for (let i = 0; i < prod(shape); i++) {
      const y = eventVector(this.cx, nv, node.dims, unflatten(i, shape))
      if (!y.some((v) => Number.isNaN(v))) rows.push(y)
    }
    if (rows.length === 0) return false
    const n = rows.length
    const ybar = Array.from({ length: d }, (_, k) => rows.reduce((acc, y) => acc + (y[k] as number), 0) / n)
    const scatter = new Array<number>(d * d).fill(0)
    for (const y of rows)
      for (let a = 0; a < d; a++)
        for (let b = 0; b < d; b++)
          scatter[a * d + b] =
            (scatter[a * d + b] as number) +
            ((y[a] as number) - (ybar[a] as number)) * ((y[b] as number) - (ybar[b] as number))
    this.addPlated(node, { operator, args: [{ const: ybar }, mean.at(0), ...tail], params: { count: n, scatter } })
    return true
  }

  /** A correlation latent: `rho = 2 u - 1` with `u ~ Beta(eta, eta)` (the LKJ prior of a 2 by 2 matrix), one beta variable with that prior. */
  private latentCorrelation(node: ModelNode & { readonly kind: 'latent' }): void {
    if (node.dims.length > 0)
      throw new Refusal('structure', `'${node.name}' is a correlation over [${node.dims}]: one correlation is mapped`)
    const e = node.dist.params['eta']
    const eta =
      e === undefined ? 1 : this.cx.isConst(e) ? (toFlat(this.cx.evalNamed(e).value)[0] as number) : Number.NaN
    if (!(eta > 0)) throw new Refusal('operator', `the concentration of '${node.name}' must be a known positive value`)
    const name = this.cx.g.addVariable({ name: node.name, family: { kind: 'beta' } })
    this.cx.g.addFactor({ operator: 'prior', args: [name], params: { eta: betaEta(eta, eta) } }, name)
    this.cx.correlations.set(node.name, name)
    this.cx.layout(node, 'latent', 'beta', 'scalar', [name], true, { correlation: true })
  }

  /** A standard deviation of a bivariate normal as the precision port of `mvn-sd-corr`: a known value, or the idiom `1 / sqrt(tau)` of a gamma latent. */
  private precisionPort(
    node: ModelNode & { readonly dist: DistSpec },
    param: string,
  ): string | { readonly const: number } {
    const e = node.dist.params[param]
    if (e === undefined) throw new Refusal('structure', `'${node.name}': ${param} is missing`)
    if (this.cx.isConst(e)) {
      const v = toFlat(this.cx.evalNamed(e).value)[0] as number
      if (!(v > 0)) throw new Refusal('data', `${param} of '${node.name}' must be positive`)
      return { const: 1 / (v * v) }
    }
    const tau = precisionExpr(this.cx.idiom, e)
    if (tau === undefined)
      throw new Refusal(
        'operator',
        `the ${param} of '${node.name}' depends on a random value that is not the precision idiom: write it as 1 / sqrt(tau) with tau a gamma latent`,
      )
    const cell = evalSym(this.cx, tau).cells[0] as Cell
    const scaled = scaledOf(cell)
    if (
      scaled === undefined ||
      scaled.scale !== 1 ||
      scaled.offset !== 0 ||
      cell.kind !== 'rv' ||
      cell.family !== 'gamma'
    )
      throw new Refusal(
        'operator',
        `the ${param} of '${node.name}' must be 1 / sqrt(tau) with tau a gamma latent (a known multiple or offset of it is not mapped)`,
      )
    return scaled.variable
  }

  /** The correlation of a bivariate normal as the beta port of `mvn-sd-corr`: a known value or a `correlation` latent. */
  private correlationPort(node: ModelNode & { readonly dist: DistSpec }): string | { readonly const: number } {
    const e = node.dist.params['rho']
    if (e === undefined) throw new Refusal('structure', `'${node.name}': rho is missing`)
    if (this.cx.isConst(e)) {
      const v = toFlat(this.cx.evalNamed(e).value)[0] as number
      if (!(v > -1 && v < 1)) throw new Refusal('data', `rho of '${node.name}' must be in (-1, 1)`)
      return { const: v }
    }
    const u = unwrap(this.cx.nodes, e)
    const variable = u.kind === 'ref' ? this.cx.correlations.get(u.name) : undefined
    if (variable === undefined)
      throw new Refusal('operator', `the rho of '${node.name}' must be known or a latent with the correlation prior`)
    return variable
  }

  /**
   * Observed vectors `y_i ~ bivariateNormal(loc, scale1, scale2, rho)` with the two standard deviations known or `1 / sqrt(tau)` of gamma latents and
   * the correlation known or a `correlation` latent: one `mvn-sd-corr` factor for the rows of a shared mean (a plate), else one per row.
   */
  private observedBivariateNormal(node: ModelNode & { readonly kind: 'observed' }): void {
    const nv = evalNamedValue(this.cx, node)
    if (nv.value.shape[nv.value.shape.length - 1] !== 2 || nv.value.shape.length !== node.dims.length + 1)
      throw new Refusal('structure', `'${node.name}' is observed as vectors of 2 entries over [${node.dims}]`)
    const vectorDim = node.dist.eventDims?.[0]
    const mean = this.vectorMean(node, 2, node.dims, vectorDim)
    const tail = [this.precisionPort(node, 'scale1'), this.precisionPort(node, 'scale2'), this.correlationPort(node)]
    if (this.plateVectors(node, nv, 2, mean, 'mvn-sd-corr', tail)) return
    const shape = this.cx.shapeOf(node.dims)
    const vars: (string | null)[] = []
    for (let i = 0; i < prod(shape); i++) {
      const y = eventVector(this.cx, nv, node.dims, unflatten(i, shape))
      if (y.some((v) => Number.isNaN(v))) {
        vars.push(null)
        continue
      }
      const name = this.cx.g.addVariable({
        name: this.cx.elementName(node.name, node.dims, i),
        family: { kind: 'gaussian-mv', size: 2 },
        observed: y,
      })
      this.cx.g.addFactor({ operator: 'mvn-sd-corr', args: [name, mean.at(i), ...tail] }, name)
      vars.push(name)
    }
    this.cx.layout(node, 'observed', undefined, node.dims.length === 0 ? 'scalar' : 'observed', vars, false)
  }

  // ----- Gaussian latents: a transition from the previous state, projected and truncated priors, the normal and its groups -----

  /**
   * A multivariate normal latent whose mean is `F w + c` for a known matrix and another Gaussian vector `w` (the step of a linear
   * dynamical system, `x_t ~ N(F x_(t-1), Q)`), with a known covariance: one `gaussian-vector` variable and a `gaussian-transition`
   * factor between it and `w`, so a chain of steps with linear Gaussian readings is the Kalman smoother.
   */
  private latentTransition(node: ModelNode & { readonly kind: 'latent' }, locExpr: ExprSpec): void {
    if (node.dims.length !== 1)
      throw new Refusal('structure', `'${node.name}' is a multivariateNormal: one vector over one dimension is mapped`)
    const n = this.cx.size(node.dims[0] as string)
    const params = node.dist.params
    const covarianceExpr = params['covariance']
    if (covarianceExpr === undefined || !this.cx.isConst(covarianceExpr))
      throw new Refusal('operator', `the covariance of '${node.name}' must be a known matrix when its mean is random`)
    const covariance = Array.from(toFlat(this.cx.evalNamed(covarianceExpr).value))
    if (covariance.length !== n * n)
      throw new Refusal('structure', `the covariance of '${node.name}' is not ${n} by ${n}`)
    const loc = this.cx.over(evalSym(this.cx, locExpr), node.dims)
    let vector: string | undefined
    let columns = 0
    const forms = loc.map((c) => {
      if (c.kind !== 'aff')
        throw new Refusal('operator', `the mean of '${node.name}' must be a linear map of another Gaussian vector`)
      const terms = [...c.aff.terms].filter(([, k]) => k !== 0)
      for (const [key] of terms) {
        const atom = this.cx.g.atoms.get(key)
        if (atom?.kind !== 'comp' || (vector !== undefined && atom.vector !== vector))
          throw new Refusal('operator', `the mean of '${node.name}' must be a linear map of one Gaussian vector`)
        vector = atom.vector
        columns = atom.size
      }
      return { aff: c.aff, terms }
    })
    if (vector === undefined)
      throw new Refusal('operator', `the mean of '${node.name}' does not depend on a Gaussian vector`)
    const matrix = new Array<number>(n * columns).fill(0)
    const offset = forms.map(({ aff, terms }, i) => {
      for (const [key, k] of terms)
        matrix[i * columns + (this.cx.g.atoms.get(key) as Atom & { kind: 'comp' }).index] = k
      return aff.offset
    })
    this.cx.g.addVariable({ name: node.name, family: { kind: 'gaussian-mv', size: n } })
    this.cx.g.addFactor(
      {
        operator: 'gaussian-transition',
        args: [node.name, vector],
        params: { matrix, covariance, ...(offset.some((v) => v !== 0) ? { offset } : {}) },
      },
      node.name,
    )
    this.cx.syms.set(node.name, {
      dims: node.dims,
      cells: Array.from({ length: n }, (_, j) => this.cx.g.compAtom(node.name, j, n)),
    })
    this.cx.layout(node, 'latent', 'gaussian-mv', 'packed', [node.name], true)
  }

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
      const z = this.hiddenSwitch(node, inclusion[i] as Cell, index)
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

  /**
   * A gamma latent whose rate is a known multiple of another gamma latent (`gamma(0.5, b)` with `b` gamma: the
   * Makalic-Schmidt half-Cauchy, or a Student-t's mixing precision): a `gamma-rate` factor per element, whose shape is known.
   */
  private latentGammaRate(node: ModelNode & { readonly kind: 'latent' }): void {
    const inverse = node.dist.family === 'inverseGamma'
    const shapes = constParam(this.cx, node, 'concentration', 'the shape')
    const rate = node.dist.params[inverse ? 'scale' : 'rate'] as ExprSpec
    const cells = this.cx.over(evalSym(this.cx, rate), node.dims)
    const out: Cell[] = []
    const vars: string[] = []
    for (const [i, c] of cells.entries()) {
      const r = scaledOf(c)
      if (r === undefined || r.offset !== 0 || !(r.scale > 0))
        throw new Refusal(
          'operator',
          `the rate of '${node.name}' must be a known multiple of a gamma latent (the message rule integrates the gamma's rate); it is not`,
        )
      const name = this.cx.g.addVariable({
        name: this.cx.elementName(node.name, node.dims, i),
        family: { kind: 'gamma' },
      })
      this.cx.g.need('gamma-rate')
      this.cx.g.addFactor(
        {
          operator: 'gamma-rate',
          args: [name, r.variable],
          params: { shape: shapes[i] as number, ...(r.scale === 1 ? {} : { scale: r.scale }) },
        },
        name,
      )
      out.push(inverse ? { kind: 'recip', variable: name, scale: 1 } : { kind: 'rv', variable: name, family: 'gamma' })
      vars.push(name)
    }
    this.cx.finishLatent(node, 'gamma', out, vars, inverse ? { reciprocal: true } : {})
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

  private latentDirichlet(node: ModelNode & { readonly kind: 'latent' }): void {
    const e = node.dist.params['concentration']
    if (e === undefined) throw new Refusal('structure', `'${node.name}': the concentration is missing`)
    let nv: NamedValue
    try {
      nv = this.cx.evalNamed(e)
    } catch (err) {
      if (err instanceof Refusal)
        throw new Refusal('operator', `the concentration of '${node.name}' depends on a random value: it must be known`)
      throw err
    }
    const shape = this.cx.shapeOf(node.dims)
    const cells: Cell[] = []
    const vars: string[] = []
    for (let i = 0; i < prod(shape); i++) {
      const alpha = eventVector(this.cx, nv, node.dims, unflatten(i, shape))
      const name = this.cx.g.addVariable({
        name: this.cx.elementName(node.name, node.dims, i),
        family: { kind: 'dirichlet', size: alpha.length },
      })
      this.cx.g.addFactor(dirichletPrior(name, alpha), name)
      cells.push({ kind: 'rv', variable: name, family: 'dirichlet' })
      vars.push(name)
    }
    this.cx.finishLatent(node, 'dirichlet', cells, vars)
  }

  /**
   * A beta or Dirichlet latent whose concentrations are known multiples of one random positive value plus known offsets
   * (`beta(alpha, alpha)`, `beta(alpha, 2)`, `dirichlet(alpha * ones)`: the strength of the prior learned from the data) with `alpha`
   * a gamma latent or a log-normal one (carried on its log): a `dirichlet-concentration` factor per element, which is the
   * prior of the variable (a numeric rule: a grid over `log alpha`).
   */
  private latentRandomConcentration(node: ModelNode & { readonly kind: 'latent' }): void {
    const isBeta = node.dist.family === 'beta'
    const shape = this.cx.shapeOf(node.dims)
    const n = prod(shape)
    // The concentration of each element: one cell per component (a beta has two parameters, a Dirichlet an event axis).
    let perElement: ConcentrationPart[][]
    if (isBeta) {
      const a = this.cx.over(evalSym(this.cx, node.dist.params['a'] as ExprSpec), node.dims)
      const b = this.cx.over(evalSym(this.cx, node.dist.params['b'] as ExprSpec), node.dims)
      perElement = Array.from({ length: n }, (_, i) => [
        this.concentrationOf(node, a[i] as Cell),
        this.concentrationOf(node, b[i] as Cell),
      ])
    } else {
      // `concentration = S * V + O` with `S` random (a gamma or log-normal latent, a known multiple of one plus a known value) and `V`, `O`
      // known vectors over the event axis: the cell of each component is `V_k S + O_k`.
      const parsed = this.concentrationForm(node.dist.params['concentration'] as ExprSpec)
      if (parsed === undefined)
        throw new Refusal(
          'operator',
          `the concentration of '${node.name}' is not a random value times a known vector plus a known vector`,
        )
      const scalar = this.cx.over(evalSym(this.cx, parsed.random), node.dims)
      perElement = scalar.map((s) => {
        const base = this.concentrationOf(node, s)
        return parsed.scale.map(
          (v, k): ConcentrationPart => ({
            ...base,
            weight: v * base.weight,
            offset: v * base.offset + (parsed.offset[k] as number),
          }),
        )
      })
    }
    const cells: Cell[] = []
    const vars: string[] = []
    this.cx.g.need('dirichlet-concentration')
    for (const [i, parts] of perElement.entries()) {
      const comps = parts
      const random = parts.filter((p) => p.variable !== undefined)
      const variable = random[0]?.variable
      const log = random[0]?.log === true
      // Two separate random shapes, each exactly a gamma or log-normal latent: the two-dimensional rule.
      const [pa, pb] = comps
      if (
        isBeta &&
        pa?.variable !== undefined &&
        pb?.variable !== undefined &&
        pa.variable !== pb.variable &&
        [pa, pb].every((q) => q.weight === 1 && q.offset === 0)
      ) {
        this.cx.g.need('beta-two-shapes')
        const name = this.cx.g.addVariable({
          name: this.cx.elementName(node.name, node.dims, i),
          family: { kind: 'beta' },
        })
        this.cx.g.addFactor({ operator: 'beta-two-shapes', args: [name, pa.variable, pb.variable] }, name)
        cells.push({ kind: 'rv', variable: name, family: 'beta' })
        vars.push(name)
        continue
      }
      if (variable === undefined || random.some((p) => p.variable !== variable || (p.log === true) !== log))
        throw new Refusal(
          'operator',
          `the concentration of '${node.name}' must be known multiples of one random positive value (a gamma or log-normal latent) plus known offsets`,
        )
      const family: FamilyKind = isBeta ? 'beta' : 'dirichlet'
      const name = this.cx.g.addVariable({
        name: this.cx.elementName(node.name, node.dims, i),
        family: isBeta ? { kind: 'beta' } : { kind: 'dirichlet', size: comps.length },
      })
      this.cx.g.addFactor(
        {
          operator: 'dirichlet-concentration',
          args: [name, variable],
          params: {
            weights: parts.map((p) => p.weight),
            offsets: parts.map((p) => p.offset),
            ...(log ? { log: true } : {}),
          },
        },
        name,
      )
      cells.push({ kind: 'rv', variable: name, family })
      vars.push(name)
    }
    this.cx.finishLatent(node, isBeta ? 'beta' : 'dirichlet', cells, vars)
  }

  /** `S * V + O` for known vectors `V` and `O` over an event axis and a random scalar expression `S`: the form of a Dirichlet's random concentration. */
  private concentrationForm(
    e: ExprSpec,
  ): { readonly random: ExprSpec; readonly scale: readonly number[]; readonly offset: readonly number[] } | undefined {
    const u = unwrap(this.cx.nodes, e)
    const vector = (x: ExprSpec): number[] | undefined => {
      if (!this.cx.isConst(x)) return undefined
      const nv = this.cx.evalNamed(x)
      return nv.value.shape.length > nv.dims.length ? toFlat(nv.value) : undefined
    }
    if (u.kind !== 'op') return undefined
    const [a, b] = u.args as [ExprSpec, ExprSpec]
    if (u.op === 'mul') {
      const va = vector(a)
      const vb = vector(b)
      if (va !== undefined && vb === undefined) return { random: b, scale: va, offset: va.map(() => 0) }
      if (vb !== undefined && va === undefined) return { random: a, scale: vb, offset: vb.map(() => 0) }
      return undefined
    }
    if (u.op === 'add') {
      const va = vector(a)
      const vb = vector(b)
      const [inner, o] = va !== undefined ? [b, va] : vb !== undefined ? [a, vb] : [undefined, undefined]
      if (inner === undefined || o === undefined) return undefined
      const base = this.concentrationForm(inner)
      return base === undefined ? undefined : { ...base, offset: base.offset.map((v, k) => v + (o[k] as number)) }
    }
    return undefined
  }

  /**
   * One component of a concentration as `weight * alpha + offset`: a known value, a gamma variable (or a known multiple of one
   * plus an offset), or a positive latent carried on its log (`alpha = exp(l)`, the variable returned is `l`, with `log`).
   */
  private concentrationOf(node: ModelNode, c: Cell): ConcentrationPart {
    if (isConstCell(c)) return { weight: 0, offset: c.aff.offset }
    const g = scaledOf(c)
    if (g !== undefined) return { variable: g.variable, weight: g.scale, offset: g.offset }
    if (c.kind === 'aff' && c.aff.terms.size === 1) {
      const [key, coefficient] = [...c.aff.terms][0] as [string, number]
      const l = logVariableOf(this.cx.g.atoms, this.cx.logScale, key)
      if (l !== undefined) return { variable: l, log: true, weight: coefficient, offset: c.aff.offset }
    }
    throw new Refusal(
      'operator',
      `a concentration of '${node.name}' is not a known multiple of a gamma or log-normal latent plus a known value`,
    )
  }

  /** The probability of 1 of a Bernoulli node as a function of its discrete parents, by enumeration, as a table over (parents..., node). */
  private bernoulliTable(node: ModelNode & { readonly dist: DistSpec }, parents: readonly string[]): number[] {
    const e = node.dist.params['probs'] ?? node.dist.params['logits']
    if (e === undefined) throw new Refusal('structure', `'${node.name}': the probability is missing`)
    const logits = node.dist.params['probs'] === undefined
    const values: number[] = []
    for (let a = 0; a < 2 ** parents.length; a++) {
      const override = new Map<string, NamedValue>()
      parents.forEach((p, k) => {
        const bit = (a >> (parents.length - 1 - k)) & 1
        override.set(p, { value: asTensor(bit, 'float64'), dims: [] })
      })
      const raw = toFlat(this.cx.evalNamed(e, override).value)[0] as number
      const p1 = logits ? 1 / (1 + Math.exp(-raw)) : raw
      if (!(p1 >= 0 && p1 <= 1)) throw new Refusal('data', `'${node.name}': a probability of ${p1} is outside [0, 1]`)
      values.push(1 - p1, p1)
    }
    return values
  }

  /**
   * A categorical latent: a `discrete` variable per element with a known probability vector (a `prior` factor), or with a
   * Dirichlet latent as its probabilities (the conjugate `discrete` factor, a mixture weight). Its role in a model is as
   * the gate of `at(components, z)` or the index of a conjugate observation.
   */
  private latentCategorical(node: ModelNode & { readonly kind: 'latent' }): void {
    const shape = this.cx.shapeOf(node.dims)
    const n = prod(shape)
    const probsExpr = node.dist.params['probs']
    const logitsExpr = node.dist.params['logits']
    const expr = probsExpr ?? logitsExpr
    if (expr === undefined) throw new Refusal('structure', `'${node.name}': give probs or logits`)
    const cells: Cell[] = []
    const vars: string[] = []
    const chain = logitsExpr === undefined && !this.cx.isConst(expr) ? transitionRow(this.cx.idiom, expr) : undefined
    if (chain !== undefined) {
      // The probabilities are a known row of a matrix chosen by another discrete variable (a Markov chain's transition): a table over the two.
      const parents = this.cx.over(evalSym(this.cx, chain.index), node.dims)
      const matrix = toFlat(this.cx.evalNamed(chain.matrix).value)
      for (const [i, p] of parents.entries()) {
        if (p.kind !== 'rv' || (p.family !== 'discrete' && p.family !== 'bernoulli'))
          throw new Refusal(
            'operator',
            `the row of '${node.name}' must be chosen by a discrete latent, not a computed value`,
          )
        const from = p.family === 'bernoulli' ? 2 : this.cx.g.sizeOf(p.variable)
        const to = matrix.length / from
        if (!Number.isInteger(to) || to < 1)
          throw new Refusal(
            'data',
            `the transition matrix of '${node.name}' has ${matrix.length} entries for ${from} states`,
          )
        const name = this.cx.g.addVariable({
          name: this.cx.elementName(node.name, node.dims, i),
          family: { kind: 'discrete', size: to },
        })
        this.cx.g.need('table')
        this.cx.g.addFactor(
          { operator: 'table', args: [p.variable, name], params: { states: [from, to], values: Array.from(matrix) } },
          name,
        )
        cells.push({ kind: 'rv', variable: name, family: 'discrete' })
        vars.push(name)
      }
    } else if (!this.cx.isConst(expr) && randomTransitionRow(this.cx.idiom, expr) !== undefined) {
      // The rows of the matrix are Dirichlet latents (a transition matrix to learn), chosen by another discrete variable: one
      // `gate` per element, whose state is the previous state and whose branch `j` is the conjugate `discrete` factor of this
      // state on row `j`. Each row then hears only the transitions out of its state, weighted by the belief that the chain was there.
      const chosen = randomTransitionRow(this.cx.idiom, expr) as { readonly matrix: ExprSpec; readonly index: ExprSpec }
      const matrix = evalSym(this.cx, chosen.matrix)
      const rows = matrix.cells
      const parents = this.cx.over(evalSym(this.cx, chosen.index), node.dims)
      if (matrix.dims.length !== 1 || rows.some((r) => r.kind !== 'rv' || r.family !== 'dirichlet'))
        throw new Refusal(
          'operator',
          `the rows of the matrix behind '${node.name}' must be one Dirichlet latent each (a latent over the dimension of the states)`,
        )
      const rowVars = rows.map((r) => (r as { readonly variable: string }).variable)
      const size = this.cx.g.sizeOf(rowVars[0] as string)
      for (const [i, p] of parents.entries()) {
        if (p.kind !== 'rv' || (p.family !== 'discrete' && p.family !== 'bernoulli'))
          throw new Refusal(
            'operator',
            `the row of '${node.name}' must be chosen by a discrete latent, not a computed value`,
          )
        const from = p.family === 'bernoulli' ? 2 : this.cx.g.sizeOf(p.variable)
        if (from !== rowVars.length)
          throw new Refusal('data', `'${node.name}' has ${from} previous states and its matrix ${rowVars.length} rows`)
        const name = this.cx.g.addVariable({
          name: this.cx.elementName(node.name, node.dims, i),
          family: { kind: 'discrete', size },
        })
        this.cx.g.need('gate')
        this.cx.g.need('discrete')
        this.cx.g.addFactor(discreteRowGate(p.variable, name, rowVars), name)
        cells.push({ kind: 'rv', variable: name, family: 'discrete' })
        vars.push(name)
      }
    } else if (!this.cx.isConst(expr)) {
      if (logitsExpr !== undefined)
        throw new Refusal(
          'operator',
          `the logits of '${node.name}' depend on a random value: only probs from a Dirichlet latent map`,
        )
      const over = this.cx.over(evalSym(this.cx, expr), node.dims)
      for (const [i, c] of over.entries()) {
        if (c.kind === 'sticks') {
          // The weights are the stick-breaking of beta breaks: a `stick-breaking` factor on the assignment and the breaks.
          const name = this.cx.g.addVariable({
            name: this.cx.elementName(node.name, node.dims, i),
            family: { kind: 'discrete', size: c.variables.length + 1 },
          })
          this.cx.g.need('stick-breaking')
          this.cx.g.addFactor(
            { operator: 'stick-breaking', args: [name, ...c.variables], params: { sticks: c.variables.length } },
            name,
          )
          cells.push({ kind: 'rv', variable: name, family: 'discrete' })
          vars.push(name)
          continue
        }
        if (c.kind !== 'rv' || c.family !== 'dirichlet')
          throw new Refusal(
            'operator',
            `the probabilities of '${node.name}' must be known or a Dirichlet latent for the conjugate 'discrete' operator`,
          )
        const size = this.cx.g.sizeOf(c.variable)
        const name = this.cx.g.addVariable({
          name: this.cx.elementName(node.name, node.dims, i),
          family: { kind: 'discrete', size },
        })
        this.cx.g.addFactor({ operator: 'discrete', args: [name, c.variable] }, name)
        cells.push({ kind: 'rv', variable: name, family: 'discrete' })
        vars.push(name)
      }
    } else {
      const nv = this.cx.evalNamed(expr)
      for (let i = 0; i < n; i++) {
        const raw = eventVector(this.cx, nv, node.dims, unflatten(i, shape))
        const probs =
          logitsExpr === undefined
            ? raw
            : (() => {
                const top = Math.max(...raw)
                const e = raw.map((v) => Math.exp(v - top))
                const t = e.reduce((a, b) => a + b, 0)
                return e.map((v) => v / t)
              })()
        const name = this.cx.g.addVariable({
          name: this.cx.elementName(node.name, node.dims, i),
          family: { kind: 'discrete', size: probs.length },
        })
        this.cx.g.addFactor(discretePrior(name, probs), name)
        cells.push({ kind: 'rv', variable: name, family: 'discrete' })
        vars.push(name)
      }
    }
    this.cx.finishLatent(node, 'discrete', cells, vars)
  }

  private latentDiscrete(node: ModelNode & { readonly kind: 'latent' }): void {
    if (node.dist.family === 'categorical') {
      this.latentCategorical(node)
      return
    }
    const parents = discreteParents(this.cx, node)
    if (node.dims.length > 0 && parents.length > 0)
      throw new Refusal(
        'structure',
        `'${node.name}' is a Bernoulli over [${node.dims}] with discrete parents: tables over plated nodes are not expanded yet`,
      )
    const shape = this.cx.shapeOf(node.dims)
    const n = prod(shape)
    const cells: Cell[] = []
    const vars: string[] = []
    const probsExpr = node.dist.params['probs']
    if (parents.length === 0 && probsExpr !== undefined && !this.cx.isConst(probsExpr)) {
      // A Bernoulli whose probability is a beta latent (an unknown mixing weight, a gate's prior): the conjugate `bernoulli` factor,
      // whose message to the probability is the moment-matched mixture of the two conjugate updates.
      this.cx.g.need('bernoulli')
      const over = this.cx.over(evalSym(this.cx, probsExpr), node.dims)
      for (const [i, c] of over.entries()) {
        if (c.kind !== 'rv' || c.family !== 'beta')
          throw new Refusal(
            'operator',
            `the probability of '${node.name}' must be known or a beta latent for the conjugate 'bernoulli' operator`,
          )
        const name = this.cx.g.addVariable({
          name: this.cx.elementName(node.name, node.dims, i),
          family: { kind: 'bernoulli' },
        })
        this.cx.g.addFactor({ operator: 'bernoulli', args: [name, c.variable] }, name)
        cells.push({ kind: 'rv', variable: name, family: 'bernoulli' })
        vars.push(name)
      }
    } else if (parents.length === 0) {
      const probs = constParam(
        this.cx,
        node,
        node.dist.params['probs'] === undefined ? 'logits' : 'probs',
        'the probability',
      )
      const p = node.dist.params['probs'] === undefined ? probs.map((l) => 1 / (1 + Math.exp(-l))) : probs
      for (let i = 0; i < n; i++) {
        const name = this.cx.g.addVariable({
          name: this.cx.elementName(node.name, node.dims, i),
          family: { kind: 'bernoulli' },
        })
        this.cx.g.addFactor(
          { operator: 'prior', args: [name], params: { eta: [...bernoulliEta(p[i] as number)] } },
          name,
        )
        cells.push({ kind: 'rv', variable: name, family: 'bernoulli' })
        vars.push(name)
      }
    } else {
      const name = this.cx.g.addVariable({ name: node.name, family: { kind: 'bernoulli' } })
      const pv = parents.map((p) => rvName(this.cx, p))
      this.cx.g.addFactor(
        {
          operator: 'table',
          args: [...pv, name],
          params: { states: new Array<number>(parents.length + 1).fill(2), values: this.bernoulliTable(node, parents) },
        },
        name,
      )
      cells.push({ kind: 'rv', variable: name, family: 'bernoulli' })
      vars.push(name)
    }
    this.cx.finishLatent(node, 'bernoulli', cells, vars)
  }

  private derived(node: ModelNode & { readonly kind: 'derived' }): void {
    const absorbed = this.cx.nonCentred.get(node.name)
    let scalars = false
    if (absorbed !== undefined) {
      if (this.denseNonCentred(node)) return
      scalars = true
      // The dense factor does not apply to this divisor: the standard normals are compiled as the scalar variables they would have been.
      const z = this.cx.nodes.get(absorbed) as ModelNode & { readonly kind: 'latent' }
      this.cx.nonCentred.delete(node.name)
      this.cx.withSite(z.name, () => this.latent(z))
    }
    if (this.cx.known.has(node.name)) {
      // A constant found before the walk: its value is known, and the custom functions its evaluation called count as evaluated.
      for (const name of this.cx.customsCalled.get(node.name) ?? []) this.cx.handledCustoms.add(name)
      this.cx.layout(
        node,
        'derived',
        undefined,
        'constant',
        new Array<null>(prod(this.cx.shapeOf(node.dims))).fill(null),
        false,
      )
      return
    }
    if (discreteParents(this.cx, node).length > 0) {
      this.cx.lazy.set(node.name, node.expr)
      return
    }
    const sym = evalSym(this.cx, node.expr)
    const shape = this.cx.shapeOf(node.dims)
    const cells = this.cx.over(sym, node.dims)
    if (cells.every(isConstCell)) {
      // Constant only after the symbolic evaluation (the expression mentions a random value, the cells do not): order-dependent.
      this.cx.known.declare(node.name, this.cx.toNamed({ dims: node.dims, cells }), this.cx.position)
      this.cx.known.semantic.push(node.name)
      this.cx.layout(node, 'derived', undefined, 'constant', new Array<null>(prod(shape)).fill(null), false)
      return
    }
    if (scalars && this.tiedToVector(node, cells)) return
    // Stick-breaking weights are not a random variable of the graph (the assignments read the breaks through one factor), so there is
    // nothing to record: the weights are read back from the breaks.
    if (!node.record || cells.some((c) => c.kind === 'sticks')) {
      this.cx.syms.set(node.name, { dims: node.dims, cells })
      return
    }
    const vars: (string | null)[] = []
    const out: Cell[] = cells.map((c, i) => {
      if (c.kind === 'rv') {
        vars.push(c.variable)
        return c
      }
      if (c.kind === 'sticks') throw new Refusal('structure', `'${node.name}' is a stick-breaking weight vector`)
      if (c.kind === 'scaled' || c.kind === 'lookup' || c.kind === 'recip')
        throw new Refusal(
          'structure',
          `'${node.name}' is a ${c.kind === 'scaled' ? 'scaled precision' : c.kind === 'recip' ? 'multiple of an inverse-gamma variance' : 'lookup by a discrete variable'}: it cannot be recorded as a value`,
        )
      const name = this.cx.elementName(node.name, node.dims, i)
      const v = this.cx.g.materialise(c.aff, node.name, node.dims.length === 0 ? undefined : i, name)
      if (typeof v === 'number') {
        vars.push(null)
        return c
      }
      vars.push(v)
      return this.cx.g.varAtom(v)
    })
    this.cx.finishLatent(node, 'gaussian', out, vars, {}, 'derived')
  }

  /**
   * `D = z / S(x)` as one Gaussian vector variable `D` and a `gaussian-mv-function-scale` factor with the variable `x` the divisor
   * is a chain of unary steps of (`1 / sqrt(exp(x) d_j + n_j)`: each coefficient's chain, a known divisor a constant). The standard
   * normals `z` never become variables: given `x` the vector is Gaussian with the diagonal covariance `1 / S_j(x)^2`, which the
   * factor integrates against the cavity of the whole vector. False (and nothing made) when the divisors are not chains of one variable.
   */
  private denseNonCentred(node: ModelNode & { readonly kind: 'derived' }): boolean {
    const divisor = node.expr.kind === 'op' ? node.expr.args[1] : undefined
    if (divisor === undefined) return false
    let cells: Cell[]
    try {
      cells = this.cx.over(evalSym(this.cx, divisor), node.dims)
    } catch (e) {
      if (e instanceof Refusal) return false
      throw e
    }
    let input: string | undefined
    const chains: ApplyStep[][] = []
    for (const cell of cells) {
      if (isConstCell(cell)) {
        const s = cell.aff.offset
        if (!(s > 0 && Number.isFinite(s))) return false
        chains.push([
          ['mul', 0],
          ['add', 1 / s],
        ])
        continue
      }
      if (cell.kind !== 'aff') return false
      const terms = [...cell.aff.terms].filter(([, c]) => c !== 0)
      const atom = terms.length === 1 ? this.cx.g.atoms.get(terms[0]?.[0] as string) : undefined
      if (atom?.kind !== 'var' || !this.cx.applied.has(atom.name)) return false
      const base = chainOf(this.cx, cell.aff)
      if (input !== undefined && input !== base.input) return false
      input = base.input
      chains.push([...base.steps, ['recip']])
    }
    if (input === undefined) return false
    const n = cells.length
    this.cx.g.need('gaussian-mv-function-scale')
    this.cx.g.addVariable({ name: node.name, family: { kind: 'gaussian-mv', size: n } })
    this.cx.g.addFactor(
      {
        operator: 'gaussian-mv-function-scale',
        args: [node.name, input],
        params: { steps: chains as unknown as Json },
      },
      node.name,
    )
    this.cx.syms.set(node.name, {
      dims: node.dims,
      cells: Array.from({ length: n }, (_, j) => this.cx.g.compAtom(node.name, j, n)),
    })
    if (node.record) this.cx.layout(node, 'derived', 'gaussian-mv', 'packed', [node.name], true)
    return true
  }

  /**
   * A non-centred vector whose coefficients each have a scale of their own (`z_j / sqrt(alpha_j)`, ARD): the products are scalar
   * variables (`apply-product`), each a coordinate of one Gaussian vector variable `D` (a `dot-constant` factor), and `D` is what the
   * rest of the model reads. The scalar factors are as before; the likelihood sees one vector, so its message to the coefficients is
   * dense and the posterior keeps their correlations. False (and nothing made) when a cell is not one variable.
   */
  private tiedToVector(node: ModelNode & { readonly kind: 'derived' }, cells: readonly Cell[]): boolean {
    const outs: string[] = []
    for (const cell of cells) {
      if (cell.kind !== 'aff' || cell.aff.offset !== 0 || cell.aff.terms.size !== 1) return false
      const [key, coefficient] = [...cell.aff.terms][0] as [string, number]
      const atom = this.cx.g.atoms.get(key)
      if (atom?.kind !== 'var' || coefficient !== 1 || outs.includes(atom.name)) return false
      outs.push(atom.name)
    }
    const n = outs.length
    this.cx.g.addVariable({ name: node.name, family: { kind: 'gaussian-mv', size: n } })
    for (const [j, out] of outs.entries()) {
      const a = new Array<number>(n).fill(0)
      a[j] = 1
      this.cx.g.addFactor({ operator: 'dot-constant', args: [out, node.name], params: { a } }, elementId(node.name, j))
    }
    this.cx.syms.set(node.name, {
      dims: node.dims,
      cells: Array.from({ length: n }, (_, j) => this.cx.g.compAtom(node.name, j, n)),
    })
    if (node.record) this.cx.layout(node, 'derived', 'gaussian-mv', 'packed', [node.name], true)
    return true
  }

  /**
   * A Gaussian observation of a value that is a known function of a discrete variable (a reading moved a period early or late by a
   * copy index): `N(v; m, s)` is `N(0; m - v, s)`, so the observed value moves across to the mean, which then is a gate over the
   * shifted forms. Undefined when the observed value is known, or the family is not the normal.
   */
  private valueMovedToMean(
    node: ModelNode & { readonly kind: 'observed' },
  ): (ModelNode & { readonly kind: 'observed' }) | undefined {
    const loc = node.dist.params['loc']
    if (node.dist.family !== 'normal' || loc === undefined || this.cx.isConst(node.value)) return undefined
    if (!mentionsDiscrete(this.cx.latents, node.value)) return undefined
    const fused = this.gatedMean(node, loc)
    if (fused !== undefined) return { ...node, value: { kind: 'const', value: 0 } }
    return {
      ...node,
      dist: { ...node.dist, params: { ...node.dist.params, loc: { kind: 'op', op: 'sub', args: [loc, node.value] } } },
      value: { kind: 'const', value: 0 },
    }
  }

  /**
   * The mean `at(table, z)` of an observation whose value is a known function of the same discrete variable `z`: one gate over the
   * table's entries each moved by what the value is in that state (`table[k] - value[k]`), so the observed value becomes zero and
   * the per-state shifts sit in the one gate (a gate of the shifted forms, not a gate of the mean followed by another of the
   * value). Stores the cells of the mean for the observation to read; false when the mean is not of that form.
   */
  private gatedMean(node: ModelNode & { readonly kind: 'observed' }, loc: ExprSpec): boolean {
    const e = unwrap(this.cx.nodes, loc)
    if (e.kind !== 'op' || e.op !== 'at') return false
    const [first, index] = e.args as [ExprSpec, ExprSpec]
    const table = evalSym(this.cx, first)
    if (table.dims.length !== 1) return false
    const values = this.cx.over(evalSym(this.cx, node.value), node.dims)
    const picks = this.cx.over(evalSym(this.cx, index), node.dims)
    const cells: Cell[] = []
    for (const [i, v] of values.entries()) {
      const pick = picks[i] as Cell
      if (v.kind !== 'lookup' || v.items !== undefined || pick.kind !== 'rv' || pick.variable !== v.variable)
        return false
      if (table.cells.length !== v.values.length || !table.cells.every((c) => c.kind === 'aff')) return false
      const shifted = table.cells.map(
        (c, k): Cell => ({
          kind: 'aff',
          aff: combine(affOf(c), constAff(v.values[k] as number), -1),
        }),
      )
      cells.push(gate(this.cx, 'plain', shifted, v.variable))
    }
    this.cx.cells.set(`${node.name}|loc`, cells)
    return true
  }

  private observed(node: ModelNode & { readonly kind: 'observed' }): void {
    if (node.dist.family === 'multivariateNormal' && this.wishartOf(node) !== undefined) {
      this.observedMvnWishart(node)
      return
    }
    if (node.dist.family === 'bivariateNormal') {
      this.observedBivariateNormal(node)
      return
    }
    const moved = this.valueMovedToMean(node)
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
      this.observedNormalGamma(node, normalGamma, values)
      return
    }
    const vars: (string | null)[] = []
    const parentsOf = discreteParents(this.cx, node)
    if (this.cx.vectorise && node.dims.length > 0 && parentsOf.length === 0 && this.plate(node, fam, values)) return
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
            values: this.bernoulliTable(node, parentsOf),
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
        if (this.observedSwitchLogScale(node, y, i, index)) return null
        // A mean or a scale chosen by a discrete variable (a lookup): a table or a gate over that variable.
        const switched = this.switchedCells(node, i)
        if (switched !== undefined) return this.observedSwitch(node, name, y, index, switched.loc, switched.scale)
        if (scaleExpr !== undefined && !this.cx.isConst(scaleExpr)) {
          const logScale =
            expArgument(unwrap(this.cx.nodes, scaleExpr)) !== undefined
              ? this.observedLogScale(node, name, y, i, index)
              : undefined
          if (logScale !== undefined) return logScale
          if (precisionExpr(this.cx.idiom, scaleExpr) === undefined)
            return this.observedRandomScale(node, name, y, i, index, scaleExpr)
          return this.observedRandomPrecision(node, name, y, i, index)
        }
        return this.observedGaussian(node, name, y, i, index, 'loc', 'scale')
      }
      case 'studentT':
      case 'laplace':
      case 'normalOutliers':
        return this.observedHeavy(node, fam, y, i, index)
      case 'beta':
        return this.observedPair(node, 'beta-shapes', y, i, index)
      case 'orderedProbit':
      case 'orderedLogistic':
      case 'asymmetricLaplace': {
        const numeric = this.observedNumeric(node, fam, y, i, index)
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
        return this.observedGaussian(node, name, Math.log(y), i, index, 'mu', 'sigma')
      case 'bernoulli':
        return this.observedBernoulli(node, name, y, i, index)
      case 'binomial':
      case 'poisson': {
        const numeric = this.observedNumeric(node, fam, y, i, index)
        return numeric !== undefined ? numeric : this.observedConjugate(node, fam, name, y, i)
      }
      case 'categorical':
        if (node.dist.params['logits'] !== undefined)
          return this.observedMulticlass(node, y, i, index, 'logits', 'logit')
        return this.observedConjugate(node, fam, name, y, i)
      case 'categoricalProbit':
        return this.observedMulticlass(node, y, i, index, 'scores', 'probit')
      case 'zeroTruncatedPoisson': {
        const numeric = this.observedNumeric(node, fam, y, i, index)
        if (numeric === undefined)
          throw new Refusal(
            'operator',
            `the rate of '${node.name}' must be \`exposure * exp(affine)\` for the zero-truncated Poisson`,
          )
        return numeric
      }
      case 'hurdlePoisson':
        return this.observedHurdle(node, y, i, index)
      case 'normalPair':
        return this.observedMixtureWeight(node, name, y, i)
      case 'exponential':
      case 'gamma':
        return this.observedGammaRate(node, fam, name, y, i)
      case 'geometric':
        return this.observedFailures(node, fam, name, y, i)
      case 'negativeBinomial':
      case 'weibull': {
        const numeric = this.observedNumeric(node, fam, y, i, index)
        if (numeric === undefined && fam === 'negativeBinomial') return this.observedFailures(node, fam, name, y, i)
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

  /**
   * A Poisson count whose rate is `k_z exp(L)`: `L` an affine form of Gaussian variables and `k_z` a known exposure chosen by a
   * discrete variable `z` (zero for a state that is a structural zero: zero inflation). The state of `z` selects which
   * likelihood holds, so it is a `gate` whose branches are the Poisson likelihood of `L` (a state with exposure zero is the
   * constant one, a count of zero) with a `table` that rules out the states in which a positive count has probability zero; when
   * `L` is known the gate is a table over `z`. The weights of `z` may be known or a Dirichlet latent, as for any discrete variable.
   */
  private observedPoissonGate(
    node: ModelNode & { readonly kind: 'observed' },
    y: number,
    i: number,
    index: number | undefined,
    inner: ExprSpec,
    exposure: ExprSpec,
  ): null {
    const cell = cellsOf(this.cx, node, 'exposure', () => evalSym(this.cx, exposure))[i] as Cell
    if (cell.kind !== 'lookup' || cell.items !== undefined)
      throw new Refusal(
        'operator',
        `the exposure of '${node.name}' is random: the poisson likelihood has a message rule for a known exposure, or a known value chosen by a discrete variable (a gate)`,
      )
    const z = cell.variable
    const exposures = cell.values
    if (!exposures.every((e) => e >= 0 && Number.isFinite(e)))
      throw new Refusal('data', `the exposure of '${node.name}' must be non-negative`)
    const why = likelihoods()['poisson-log']?.check?.(y, { likelihood: 'poisson-log', y, exposure: 1 })
    if (why !== undefined) throw new Refusal('data', `'${node.name}' (poisson-log): ${why}`)
    const predictor = cellsOf(this.cx, node, 'predictor', () => evalSym(this.cx, inner))[i] as Cell
    const id = this.cx.elementName(node.name, node.dims, i)
    const impossible = exposures.map((e) => e === 0 && y > 0)
    const logFactorial = Array.from({ length: y }, (_, k) => Math.log(k + 1)).reduce((a, b) => a + b, 0)
    if (isConstCell(predictor)) {
      // Nothing random but the state: the likelihood of the count under each state.
      const pmf = (e: number): number => {
        const rate = e * Math.exp(predictor.aff.offset)
        return rate === 0 ? (y === 0 ? 1 : 0) : Math.exp(-rate + y * Math.log(rate) - logFactorial)
      }
      this.cx.g.addFactor(
        { operator: 'table', args: [z], params: { states: [exposures.length], values: exposures.map(pmf) } },
        id,
      )
      return null
    }
    const v = this.cx.g.materialise(affOf(predictor), node.name, index) as string
    if (impossible.some(Boolean))
      this.cx.g.addFactor(
        {
          operator: 'table',
          args: [z],
          params: { states: [exposures.length], values: impossible.map((bad) => (bad ? 0 : 1)) },
        },
        `${id}|support`,
      )
    const gate = new GateBuilder(z, [v])
    for (const e of exposures)
      gate.add(
        e === 0
          ? {}
          : { operator: 'likelihood-site', args: [0], params: { likelihood: 'poisson-log', y, exposure: e } },
      )
    this.cx.g.addFactor(gate.factor(), id)
    return null
  }

  /**
   * The negative binomial in the NB2 idiom with a random dispersion `a`: `n = 1 / a`, `probs = 1 / (1 + a * mean)` with `mean`
   * `exp(L)` (times a known exposure) and `a` a positive latent (carried on its log by `latentProjected`): the
   * `negative-binomial-dispersion` factor on the predictor `L` and the log of `a`. `undefined` when the parameters are not that
   * idiom or the dispersion is known (the registry's `negative-binomial-log` takes that).
   */
  private negBinomialRandomDispersion(
    node: ModelNode & { readonly kind: 'observed' },
    y: number,
    i: number,
    index: number | undefined,
  ): string | null | undefined {
    const dist = node.dist.params
    const nExpr = dist['n'] === undefined ? undefined : unwrap(this.cx.nodes, dist['n'])
    const pExpr = dist['probs'] === undefined ? undefined : unwrap(this.cx.nodes, dist['probs'])
    if (nExpr?.kind !== 'op' || nExpr.op !== 'div' || pExpr?.kind !== 'op' || pExpr.op !== 'div') return undefined
    const isOne = (e: ExprSpec): boolean => this.cx.isConst(e) && constAt(this.cx, node, e, i) === 1
    const [one, a] = nExpr.args as [ExprSpec, ExprSpec]
    if (!isOne(one) || this.cx.isConst(a)) return undefined
    const [num, den] = pExpr.args as [ExprSpec, ExprSpec]
    const d = unwrap(this.cx.nodes, den)
    if (!isOne(num) || d.kind !== 'op' || d.op !== 'add') return undefined
    const [p, q] = d.args as [ExprSpec, ExprSpec]
    const [unit, product] = this.cx.isConst(p) ? [p, unwrap(this.cx.nodes, q)] : [q, unwrap(this.cx.nodes, p)]
    if (!isOne(unit) || product.kind !== 'op' || product.op !== 'mul') return undefined
    const same = (x: ExprSpec, z: ExprSpec): boolean =>
      JSON.stringify(unwrap(this.cx.nodes, x)) === JSON.stringify(unwrap(this.cx.nodes, z))
    const [m1, m2] = product.args as [ExprSpec, ExprSpec]
    const mean = same(m1, a) ? m2 : same(m2, a) ? m1 : undefined
    if (mean === undefined) return undefined
    const m = scaledExp(this.cx.idiom, mean)
    if (m === undefined) return undefined
    const aCell = cellsOf(this.cx, node, 'dispersion', () => evalSym(this.cx, a))[i] as Cell
    const logDispersion = logVariableOfCell(aCell, this.cx.g.atoms, this.cx.logScale)
    if (logDispersion === undefined)
      throw new Refusal(
        'operator',
        `the dispersion of '${node.name}' is random but not a positive latent (a half-normal, half-Cauchy, log-normal or uniform prior carried on its log): the negative-binomial-dispersion factor needs the log of it`,
      )
    const params: Record<string, Json> = { y, exposure: m.scale === undefined ? 1 : constAt(this.cx, node, m.scale, i) }
    if (!(Number.isInteger(y) && y >= 0))
      throw new Refusal('data', `'${node.name}' (negative binomial): the outcome must be a non-negative integer`)
    const cell = cellsOf(this.cx, node, 'predictor', () => evalSym(this.cx, m.inner))[i] as Cell
    if (isConstCell(cell)) return null
    const u = this.cx.g.materialise(affOf(cell), node.name, index) as string
    this.cx.g.addFactor(
      { operator: 'negative-binomial-dispersion', args: [u, logDispersion], params },
      this.cx.elementName(node.name, node.dims, i),
    )
    return null
  }

  /**
   * The likelihoods of the registry of `message-passing/operators/likelihoods.ts` for `observe y ~ D(g(affine))`: Poisson with
   * `rate = exposure * exp(affine)`, binomial with `probs = sigmoid(affine)` or `probit(affine)`, negative binomial with a log
   * link and a known dispersion (`n = r`, `probs = r / (r + mean)`, the mean `exposure * exp(affine)`), and a Weibull with
   * `scale = exp(affine)` and a known shape. Returns `undefined` when the parameters are not that shape (the caller tries the
   * conjugate rule or refuses), else the factor's outcome (always `null`: the outcome is a parameter of the factor).
   */
  private observedNumeric(
    node: ModelNode & { readonly kind: 'observed' },
    fam: string,
    y: number,
    i: number,
    index: number | undefined,
  ): string | null | undefined {
    const dist = node.dist.params
    const known = (p: string, what: string): number => {
      let c: Cell
      try {
        c = paramCell(this.cx, node, p, i)
      } catch (e) {
        // A parameter computed from a random value (1 / alpha with alpha a latent) is random.
        if (e instanceof Refusal && e.kind === 'operator')
          throw new Refusal(
            'operator',
            `the ${what} of '${node.name}' is random (${e.message}): the ${fam} likelihood has a message rule for a known ${what} only`,
          )
        throw e
      }
      if (!isConstCell(c))
        throw new Refusal(
          'operator',
          `the ${what} of '${node.name}' is random: the ${fam} likelihood has a message rule for a known ${what} only`,
        )
      return c.aff.offset
    }
    const emit = (likelihood: string, expr: ExprSpec, extra: Record<string, Json>): string | null => {
      const params: Record<string, Json> = { likelihood, y, ...extra }
      const why = likelihoods()[likelihood]?.check?.(y, params)
      if (why !== undefined) throw new Refusal('data', `'${node.name}' (${likelihood}): ${why}`)
      const cell = cellsOf(this.cx, node, 'predictor', () => evalSym(this.cx, expr))[i] as Cell
      return this.emitSite(node, cell, i, index, 'likelihood-row', 'likelihood-site', params)
    }
    if (fam === 'zeroTruncatedPoisson' || fam === 'hurdlePoisson') {
      const m = dist['rate'] === undefined ? undefined : scaledExp(this.cx.idiom, dist['rate'])
      if (m === undefined) return undefined
      return emit('poisson-log-truncated', m.inner, {
        exposure: m.scale === undefined ? 1 : constAt(this.cx, node, m.scale, i),
      })
    }
    if (fam === 'poisson') {
      const m = dist['rate'] === undefined ? undefined : scaledExp(this.cx.idiom, dist['rate'], true)
      if (m === undefined) return undefined
      if (m.scale !== undefined && !this.cx.isConst(m.scale))
        return this.observedPoissonGate(node, y, i, index, m.inner, m.scale)
      return emit('poisson-log', m.inner, { exposure: m.scale === undefined ? 1 : constAt(this.cx, node, m.scale, i) })
    }
    if (fam === 'binomial') {
      const probs = dist['probs'] === undefined ? undefined : unwrap(this.cx.nodes, dist['probs'])
      if (probs?.kind !== 'op' || (probs.op !== 'sigmoid' && probs.op !== 'probit')) return undefined
      return emit(probs.op === 'sigmoid' ? 'binomial-logit' : 'binomial-probit', probs.args[0] as ExprSpec, {
        n: known('n', 'number of trials'),
      })
    }
    if (fam === 'orderedProbit' || fam === 'orderedLogistic') {
      const cutsExpr = dist['cutpoints'] as ExprSpec
      if (!this.cx.isConst(cutsExpr))
        throw new Refusal(
          'operator',
          `the cut points of '${node.name}' are random: the ordinal likelihoods have a message rule for known cut points only`,
        )
      const cuts = Array.from(toFlat(this.cx.evalNamed(cutsExpr).value))
      return emit(fam === 'orderedProbit' ? 'ordinal-probit' : 'ordinal-logit', dist['predictor'] as ExprSpec, { cuts })
    }
    if (fam === 'asymmetricLaplace') {
      if (!isConstCell(paramCell(this.cx, node, 'scale', i)))
        return this.observedPair(node, 'asymmetric-laplace-scale', y, i, index)
      return emit('asymmetric-laplace', dist['loc'] as ExprSpec, {
        scale: known('scale', 'scale'),
        tau: known('tau', 'quantile level'),
      })
    }
    if (fam === 'weibull') {
      const m = dist['scale'] === undefined ? undefined : scaledExp(this.cx.idiom, dist['scale'])
      if (m === undefined || m.scale !== undefined) return undefined
      if (!isConstCell(paramCell(this.cx, node, 'concentration', i)))
        return this.observedPair(node, 'weibull-shape', y, i, index)
      return emit('weibull-log', m.inner, { shape: known('concentration', 'shape') })
    }
    if (fam === 'negativeBinomial') {
      const random = this.negBinomialRandomDispersion(node, y, i, index)
      if (random !== undefined) return random
      // The negative binomial with a log link: n = r, probs = r / (r + mean) = A / (A + K exp(L)) with mean = n K exp(L) / A.
      const r = known('n', 'dispersion (the number of successes n = 1 / dispersion)')
      const probs = dist['probs'] === undefined ? undefined : unwrap(this.cx.nodes, dist['probs'])
      if (probs?.kind !== 'op' || probs.op !== 'div') return undefined
      const [num, den] = probs.args as [ExprSpec, ExprSpec]
      const d = unwrap(this.cx.nodes, den)
      if (!this.cx.isConst(num) || d.kind !== 'op' || d.op !== 'add') return undefined
      const [p, q] = d.args as [ExprSpec, ExprSpec]
      const [one, rest] = this.cx.isConst(p) ? [p, q] : [q, p]
      if (!this.cx.isConst(one)) return undefined
      const A = constAt(this.cx, node, num, i)
      if (Math.abs(constAt(this.cx, node, one, i) - A) > 1e-12 * Math.abs(A)) return undefined
      const m = scaledExp(this.cx.idiom, rest)
      if (m === undefined) return undefined
      const K = m.scale === undefined ? 1 : constAt(this.cx, node, m.scale, i)
      return emit('negative-binomial-log', m.inner, { dispersion: 1 / r, exposure: (r * K) / A })
    }
    return undefined
  }

  /**
   * A categorical outcome whose class scores are affine forms of Gaussian variables over one extra dimension (the classes): a
   * `logits` expression `X W` with `W` over (feature, class) for the softmax, or `scores` for the multinomial probit. One
   * `multiclass-site` factor per observation on the `K` scores (each a Gaussian variable, or a known value). The outcome is a
   * parameter of the factor.
   */
  private observedMulticlass(
    node: ModelNode & { readonly kind: 'observed' },
    y: number,
    i: number,
    index: number | undefined,
    param: string,
    link: 'logit' | 'probit',
  ): string | null {
    const key = `${node.name}|${param}`
    let cells = this.cx.cells.get(key)
    if (cells === undefined) {
      const sym = evalSym(this.cx, node.dist.params[param] as ExprSpec)
      const extra = sym.dims.filter((dim) => (node.dist.eventDims ?? []).includes(dim))
      if (extra.length !== 1)
        throw new Refusal(
          'structure',
          `the ${param} of '${node.name}' must carry the class dimension declared by \`over\` (categorical(..., { over })), it has [${sym.dims}]`,
        )
      cells = this.cx.over(sym, [...node.dims, extra[0] as string])
      this.cx.cells.set(key, cells)
    }
    const classes = cells.length / Math.max(1, prod(this.cx.shapeOf(node.dims)))
    if (!(Number.isInteger(y) && y >= 0 && y < classes))
      throw new Refusal('data', `'${node.name}': the class ${y} is outside 0 to ${classes - 1}`)
    const args = Array.from({ length: classes }, (_, k) => {
      return this.cx.g.portOf(cells?.[i * classes + k] as Cell, node.name, index)
    })
    if (args.every((a) => typeof a !== 'string')) return null
    this.cx.g.addFactor(
      { operator: 'multiclass-site', args, params: { y, classes, link } },
      this.cx.elementName(node.name, node.dims, i),
    )
    return null
  }

  /**
   * A positive random parameter of an observed node as one port of a `likelihood-pair` factor, in the coordinate the rule reads
   * (the log of the value): a gamma variable (its own variable), a positive latent carried on its log (a log-normal, half-normal or
   * half-Cauchy prior: the log variable), or `exp(affine)` (a Gaussian variable made of the affine form). A known value is the
   * constant coordinate `log value`.
   */
  private positivePort(
    node: ModelNode & { readonly kind: 'observed' },
    key: string,
    i: number,
    index: number | undefined,
  ): string | { readonly const: number } {
    const expr = node.dist.params[key]
    const stripped = expr === undefined ? undefined : stripClip(this.cx.idiom, expr)
    const logArgument = stripped === undefined ? undefined : expArgument(stripped)
    if (logArgument !== undefined) {
      const cell = cellsOf(this.cx, node, `${key}:log`, () => evalSym(this.cx, logArgument))[i] as Cell
      if (cell.kind === 'aff') {
        const m = this.cx.g.materialise(cell.aff, node.name, index)
        return typeof m === 'string' ? m : { const: m }
      }
    }
    const c = paramCell(this.cx, node, key, i)
    if (isConstCell(c)) {
      if (!(c.aff.offset > 0))
        throw new Refusal('data', `the ${key} of '${node.name}' is ${c.aff.offset}: it must be positive`)
      return { const: Math.log(c.aff.offset) }
    }
    if (c.kind === 'rv' && c.family === 'gamma') return c.variable
    const logVar = logVariableOfCell(c, this.cx.g.atoms, this.cx.logScale)
    if (logVar !== undefined) return logVar
    throw new Refusal(
      'operator',
      `the ${key} of '${node.name}' is random but not a gamma or exponential latent, a positive latent carried on its log (a half-normal, half-Cauchy, log-normal or uniform prior) or exp of an affine value: the likelihood-pair factor needs one of those`,
    )
  }

  /**
   * A numeric likelihood over two random scalars (`likelihood-pair`): a Laplace with a random scale, a Student-t with random
   * degrees of freedom (known scale), a Weibull with a random shape (`scale = exp(affine)`) and a beta outcome with two random
   * shapes. The outcome is a parameter of the factor. A parameter that is known is a constant coordinate; nothing random at
   * all says nothing.
   */
  private observedPair(
    node: ModelNode & { readonly kind: 'observed' },
    likelihood: string,
    y: number,
    i: number,
    index: number | undefined,
  ): string | null {
    const extra: Record<string, Json> = {}
    const predictor = (cell: Cell): string | { readonly const: number } => this.cx.g.portOf(cell, node.name, index)
    let args: [string | { readonly const: number }, string | { readonly const: number }]
    if (likelihood === 'laplace-scale' || likelihood === 'asymmetric-laplace-scale') {
      if (likelihood === 'asymmetric-laplace-scale') {
        const tau = paramCell(this.cx, node, 'tau', i)
        if (!isConstCell(tau))
          throw new Refusal('operator', `the quantile level of '${node.name}' is random: it must be known`)
        extra['tau'] = tau.aff.offset
      }
      args = [predictor(paramCell(this.cx, node, 'loc', i)), this.positivePort(node, 'scale', i, index)]
    } else if (likelihood === 'student-t-df') {
      const scale = paramCell(this.cx, node, 'scale', i)
      if (!isConstCell(scale))
        throw new Refusal(
          'operator',
          `the degrees of freedom and the scale of '${node.name}' are both random: the likelihood-pair factor takes one random scale or shape beside the location`,
        )
      extra['scale'] = scale.aff.offset
      args = [predictor(paramCell(this.cx, node, 'loc', i)), this.positivePort(node, 'df', i, index)]
    } else if (likelihood === 'weibull-shape') {
      const m =
        node.dist.params['scale'] === undefined ? undefined : scaledExp(this.cx.idiom, node.dist.params['scale'])
      if (m === undefined || m.scale !== undefined)
        throw new Refusal(
          'operator',
          `the scale of '${node.name}' is not exp of an affine value: the Weibull with a random shape needs the log link`,
        )
      args = [
        predictor(cellsOf(this.cx, node, 'predictor', () => evalSym(this.cx, m.inner))[i] as Cell),
        this.positivePort(node, 'concentration', i, index),
      ]
    } else {
      args = [this.positivePort(node, 'a', i, index), this.positivePort(node, 'b', i, index)]
    }
    const params: Record<string, Json> = { likelihood, y, ...extra }
    const why = pairLikelihoods()[likelihood]?.check?.(y, params)
    if (why !== undefined) throw new Refusal('data', `'${node.name}' (${likelihood}): ${why}`)
    if (typeof args[0] !== 'string' && typeof args[1] !== 'string') return null
    this.cx.g.addFactor({ operator: 'likelihood-pair', args, params }, this.cx.elementName(node.name, node.dims, i))
    return null
  }

  /** An observed Gaussian with a known scale around an affine mean, named by the node's parameters (`loc`/`scale`, or a log-normal's `mu`/`sigma`). */
  private observedGaussian(
    node: ModelNode & { readonly kind: 'observed' },
    name: string,
    y: number,
    i: number,
    index: number | undefined,
    locName: string,
    scaleName: string,
  ): string | null {
    const scale = paramCell(this.cx, node, scaleName, i)
    if (!isConstCell(scale)) throw new Refusal('operator', `the scale of '${node.name}' is not known`)
    if (!(scale.aff.offset > 0))
      throw new Refusal('data', `the scale of '${node.name}' is ${scale.aff.offset}: it must be positive`)
    const precision = 1 / scale.aff.offset ** 2
    const loc = paramCell(this.cx, node, locName, i)
    if (isConstCell(loc)) return null
    const aff = affOf(loc)
    const terms = [...aff.terms].filter(([, c]) => c !== 0)
    const comps = terms.map(([k]) => this.cx.g.atoms.get(k) as Atom)
    const oneVector =
      aff.offset === 0 &&
      comps.length > 0 &&
      comps.every((a) => a.kind === 'comp' && a.vector === (comps[0] as Atom & { kind: 'comp' }).vector)
    this.cx.g.addVariable({ name, family: { kind: 'gaussian' }, observed: y })
    if (oneVector) {
      const first = comps[0] as Atom & { kind: 'comp' }
      const a = new Array<number>(first.size).fill(0)
      for (const [k, c] of terms) a[(this.cx.g.atoms.get(k) as Atom & { kind: 'comp' }).index] = c
      this.cx.g.addFactor(
        { operator: 'linear-regression-row', args: [name, first.vector], params: { a, precision } },
        name,
      )
    } else {
      const m = this.cx.g.materialise(aff, node.name, index)
      this.cx.g.addFactor({ operator: 'gaussian', args: [name, m as string], params: { precision } }, name)
    }
    return name
  }

  /**
   * A heavy-tailed likelihood of an affine predictor with a known scale: `studentT` (the degrees of freedom known),
   * `laplace`, or `normalOutliers` (a mixture of two normal noise levels). One row factor on the packed weights when the
   * predictor is a single dot product with one vector, else a site factor on an auxiliary predictor variable. The outcome
   * is a parameter of the factor, so the observed value has no variable of its own.
   */
  private observedHeavy(
    node: ModelNode & { readonly kind: 'observed' },
    fam: string,
    y: number,
    i: number,
    index: number | undefined,
  ): string | null {
    const known = (p: string): number => {
      const c = paramCell(this.cx, node, p, i)
      if (!isConstCell(c))
        throw new Refusal(
          'operator',
          `the ${p} of '${node.name}' is random: the ${fam} likelihood has a message rule for a known ${p} only`,
        )
      return c.aff.offset
    }
    if (fam === 'laplace' && !isConstCell(paramCell(this.cx, node, 'scale', i)))
      return this.observedPair(node, 'laplace-scale', y, i, index)
    if (fam === 'studentT' && !isConstCell(paramCell(this.cx, node, 'df', i)))
      return this.observedPair(node, 'student-t-df', y, i, index)
    if (fam === 'normalOutliers' && ['scale', 'outlier'].some((p) => !paramIsConst(this.cx, node, p)))
      return this.observedOutlierSwitch(node, y, i, index)
    const scale = known('scale')
    if (!(scale > 0)) throw new Refusal('data', `the scale of '${node.name}' is ${scale}: it must be positive`)
    let params: Record<string, Json>
    let row: string
    let site: string
    if (fam === 'studentT') {
      params = { y, scale, df: known('df') }
      row = 'student-t-row'
      site = 'student-t-site'
    } else if (fam === 'laplace') {
      params = { y, scale }
      row = 'laplace-row'
      site = 'laplace-site'
    } else {
      const outlier = known('outlier')
      params = { weights: [1 - outlier, outlier], means: [y, y], sds: [scale, scale * known('width')] }
      row = 'mixture-row'
      site = 'mixture-site'
    }
    return this.emitSite(node, paramCell(this.cx, node, 'loc', i), i, index, row, site, params)
  }

  /**
   * A Bernoulli variable no model node names, the switch that a summed-out mixture (`normalOutliers`, `normalSpikeSlab`) with a
   * random share or random precisions is read through: the mixture is the gate of its components over this switch, which is the
   * same graph as the explicit-switch form (`z ~ bernoulli(share)`, `where(z, ...)`) with the switch left out of the posterior.
   * A beta share is the conjugate `bernoulli` factor, a known one a prior on the switch.
   */
  private hiddenSwitch(node: ModelNode, probs: Cell, index: number | undefined): string {
    if (probs.kind === 'rv' && probs.family === 'beta') {
      const name = this.cx.g.addVariable({ name: this.cx.g.auxName(node.name, index), family: { kind: 'bernoulli' } })
      this.cx.g.addFactor({ operator: 'bernoulli', args: [name, probs.variable] }, name)
      return name
    }
    if (!isConstCell(probs))
      throw new Refusal(
        'operator',
        `the share of '${node.name}' must be known or a beta latent (the conjugate 'bernoulli' operator on the switch of a mixture)`,
      )
    const p = probs.aff.offset
    if (!(p > 0 && p < 1))
      throw new Refusal('data', `the share of '${node.name}' is ${p}: it must be strictly between 0 and 1`)
    const name = this.cx.g.addVariable({ name: this.cx.g.auxName(node.name, index), family: { kind: 'bernoulli' } })
    this.cx.g.addFactor({ operator: 'prior', args: [name], params: { eta: [...bernoulliEta(p)] } }, name)
    return name
  }

  /**
   * An observation `normalOutliers(loc, scale, share, width)` with a random scale (the precision idiom of a gamma latent) or a
   * random share (a beta latent), the switch summed out: the gate of an inlier and an outlier branch over a hidden Bernoulli
   * switch, read through the same route as an observation whose precision is chosen by a discrete variable (`observedSwitch`):
   * the inlier has the precision `tau`, the outlier `tau / width^2`, both `gaussian-gamma-precision` branches over the centre and
   * the gamma precision.
   */
  private observedOutlierSwitch(
    node: ModelNode & { readonly kind: 'observed' },
    y: number,
    i: number,
    index: number | undefined,
  ): string | null {
    const widthCell = paramCell(this.cx, node, 'width', i)
    if (!isConstCell(widthCell))
      throw new Refusal(
        'operator',
        `the width of '${node.name}' is random: the outlier mixture has a message rule for a known width`,
      )
    const width = widthCell.aff.offset
    if (!(width > 0)) throw new Refusal('data', `the width of '${node.name}' is ${width}: it must be positive`)
    const z = this.hiddenSwitch(node, paramCell(this.cx, node, 'outlier', i), index)
    const loc = paramCell(this.cx, node, 'loc', i)
    let scale: Cell
    if (paramIsConst(this.cx, node, 'scale')) {
      const sd = (paramCell(this.cx, node, 'scale', i) as Cell & { kind: 'aff' }).aff.offset
      scale = lookup(z, [sd, sd * width])
    } else {
      const scaleExpr = node.dist.params['scale'] as ExprSpec
      const precision = precisionExpr(this.cx.idiom, scaleExpr)
      const tau =
        precision === undefined
          ? undefined
          : scaledOf(cellsOf(this.cx, node, 'precisionProbe', () => evalSym(this.cx, precision))[i] as Cell)
      if (tau === undefined)
        throw new Refusal(
          'operator',
          `the scale of '${node.name}' must be the precision idiom 1 / sqrt(tau) of a gamma or exponential latent for the outlier mixture`,
        )
      const w2 = width * width
      scale = lookup(z, [0, 0], { precisions: [tau, { ...tau, scale: tau.scale / w2, offset: tau.offset / w2 }] })
    }
    return this.observedSwitch(node, this.cx.elementName(node.name, node.dims, i), y, index, loc, scale)
  }

  /**
   * The factor of a numeric likelihood of an affine predictor: one row factor on the packed weights when the predictor is a
   * single dot product with one vector, else a site factor on an auxiliary predictor variable. The outcome is a parameter of
   * the factor, so the observed value has no variable of its own. A predictor with no random part says nothing (no factor).
   */
  private emitSite(
    node: ModelNode & { readonly kind: 'observed' },
    predictor: Cell,
    i: number,
    index: number | undefined,
    row: string,
    site: string,
    params: Record<string, Json>,
  ): string | null {
    if (isConstCell(predictor)) return null
    const aff = affOf(predictor)
    const terms = [...aff.terms].filter(([, c]) => c !== 0)
    const comps = terms.map(([k]) => this.cx.g.atoms.get(k) as Atom)
    const oneVector =
      aff.offset === 0 &&
      comps.length > 0 &&
      comps.every((a) => a.kind === 'comp' && a.vector === (comps[0] as Atom & { kind: 'comp' }).vector)
    const id = this.cx.elementName(node.name, node.dims, i)
    if (oneVector) {
      const first = comps[0] as Atom & { kind: 'comp' }
      const a = new Array<number>(first.size).fill(0)
      for (const [k, c] of terms) a[(this.cx.g.atoms.get(k) as Atom & { kind: 'comp' }).index] = c
      this.cx.g.addFactor({ operator: row, args: [first.vector], params: { a, ...params } }, id)
    } else {
      const m = this.cx.g.materialise(aff, node.name, index)
      if (typeof m !== 'string') return null
      this.cx.g.addFactor({ operator: site, args: [m], params }, id)
    }
    return null
  }

  /**
   * The rows of an observed node as a design matrix on one packed weight vector, when every row is `a . w` (plus a known
   * offset where `offset` is allowed) of the same vector `w`: the vector's name, its size, and the row (`undefined` for a row
   * with no random part, which says nothing about `w`). Absent when the rows are not of that shape.
   */
  private designRows(
    cells: readonly Cell[],
    offset: boolean,
  ):
    | {
        readonly vector: string
        readonly size: number
        readonly rows: readonly ({ a: number[]; offset: number } | undefined)[]
      }
    | undefined {
    let vector: string | undefined
    let size = 0
    const rows: ({ a: number[]; offset: number } | undefined)[] = []
    for (const c of cells) {
      if (c.kind !== 'aff') return undefined
      const terms = [...c.aff.terms].filter(([, k]) => k !== 0)
      if (terms.length === 0) {
        rows.push(undefined)
        continue
      }
      if (c.aff.offset !== 0 && !offset) return undefined
      const atoms = terms.map(([k]) => this.cx.g.atoms.get(k) as Atom)
      for (const a of atoms) {
        if (a.kind !== 'comp') return undefined
        if (vector === undefined) {
          vector = a.vector
          size = a.size
        } else if (a.vector !== vector) return undefined
      }
      const a = new Array<number>(size).fill(0)
      for (const [k, coefficient] of terms) a[(this.cx.g.atoms.get(k) as Atom & { kind: 'comp' }).index] = coefficient
      rows.push({ a, offset: c.aff.offset })
    }
    return vector === undefined ? undefined : { vector, size, rows }
  }

  /**
   * An observed node over a plate as one plated factor on its weight vector, when its rows are i.i.d. draws of one
   * regression on a packed vector `w`: a Gaussian likelihood with one known scale (the exact `linear-regression-plate`, by
   * the sufficient statistics of the rows) or a Bernoulli with a probit or logistic link of `a . w` (`probit-plate`,
   * `logistic-plate`, the design matrix and outcomes as parameters). Anything else, or an engine registry without the
   * plate operators, is not handled here and falls to the expansion. Returns whether the node was compiled.
   */
  private plate(node: ModelNode & { readonly kind: 'observed' }, fam: string, values: readonly number[]): boolean {
    if (fam === 'normal') return this.plateNormal(node, values)
    if (fam === 'bernoulli') return this.plateBernoulli(node, values)
    return false
  }

  private addPlated(node: ModelNode & { readonly kind: 'observed' }, f: FactorSpec): void {
    this.cx.g.addFactor(f, node.name)
    this.cx.plated.push(node.name)
    this.cx.layout(
      node,
      'observed',
      undefined,
      'observed',
      new Array<null>(prod(this.cx.shapeOf(node.dims))).fill(null),
      false,
    )
  }

  private plateNormal(node: ModelNode & { readonly kind: 'observed' }, values: readonly number[]): boolean {
    if (!('linear-regression-plate' in this.cx.g.ops)) return false
    const scaleExpr = node.dist.params['scale']
    if (scaleExpr !== undefined && !this.cx.isConst(scaleExpr)) return false
    const scales = this.cx
      .over(paramSym(this.cx, node.dist, 'scale'), node.dims)
      .map((c) => (c as Cell & { kind: 'aff' }).aff.offset)
    const scale = scales[0] as number
    if (!(scale > 0) || scales.some((v) => v !== scale)) return false
    const locs = cellsOf(this.cx, node, 'loc', () => paramSym(this.cx, node.dist, 'loc'))
    const design = this.designRows(locs, true)
    if (design === undefined) return false
    const d = design.size
    const xtx = new Array<number>(d * d).fill(0)
    const xty = new Array<number>(d).fill(0)
    let yty = 0
    let n = 0
    for (const [i, row] of design.rows.entries()) {
      const y = (values[i] as number) - (row?.offset ?? 0)
      if (row === undefined || Number.isNaN(values[i])) continue
      n++
      yty += y * y
      for (let a = 0; a < d; a++) {
        xty[a] = (xty[a] as number) + (row.a[a] as number) * y
        for (let b = 0; b < d; b++)
          xtx[a * d + b] = (xtx[a * d + b] as number) + (row.a[a] as number) * (row.a[b] as number)
      }
    }
    if (n === 0) return false
    this.addPlated(node, {
      operator: 'linear-regression-plate',
      args: [design.vector],
      params: { xtx, xty, yty, n, precision: 1 / scale ** 2 },
    })
    return true
  }

  private plateBernoulli(node: ModelNode & { readonly kind: 'observed' }, values: readonly number[]): boolean {
    const probs = node.dist.params['probs']
    const logits = node.dist.params['logits']
    let link: 'probit' | 'sigmoid' | undefined
    let argument: ExprSpec | undefined
    if (logits !== undefined) {
      link = 'sigmoid'
      argument = logits
    } else if (probs !== undefined) {
      const e = unwrap(this.cx.nodes, probs)
      if (e.kind === 'op' && (e.op === 'probit' || e.op === 'sigmoid')) {
        link = e.op
        argument = e.args[0] as ExprSpec
      }
    }
    if (link === undefined || argument === undefined) return false
    const operator = link === 'probit' ? 'probit-plate' : 'logistic-plate'
    if (!(operator in this.cx.g.ops)) return false
    const arg = argument
    const cells = cellsOf(this.cx, node, `link:${link}`, () => evalSym(this.cx, arg))
    const design = this.designRows(cells, false)
    if (design === undefined) return false
    const x: number[] = []
    const y: number[] = []
    for (const [i, row] of design.rows.entries()) {
      if (row === undefined || Number.isNaN(values[i])) continue
      x.push(...row.a)
      y.push(values[i] as number)
    }
    if (y.length === 0) return false
    this.addPlated(node, { operator, args: [design.vector], params: { X: x, y } })
    return true
  }

  /** The location and scale cells of an observed normal element when one of them is a lookup by a discrete variable; absent otherwise. */
  private switchedCells(
    node: ModelNode & { readonly kind: 'observed' },
    i: number,
  ): { readonly loc: Cell; readonly scale: Cell } | undefined {
    const locExpr = node.dist.params['loc']
    const scaleExpr = node.dist.params['scale']
    const discrete = (e: ExprSpec | undefined): boolean =>
      e !== undefined && !this.cx.isConst(e) && mentionsDiscrete(this.cx.latents, e)
    if (!discrete(locExpr) && !discrete(scaleExpr)) return undefined
    // A precision chosen by a discrete variable (`1 / sqrt(where(z, tau_b, tau_a))`, `1 / sqrt(at(tau, z))`): the mean and the
    // precision are read as lookups of that variable, so the observation is one gate of `gaussian-gamma-precision` branches.
    const precision = scaleExpr === undefined ? undefined : precisionExpr(this.cx.idiom, scaleExpr)
    if (precision !== undefined && discrete(precision)) {
      const loc = paramCell(this.cx, node, 'loc', i, 'lookup')
      const lookedUp = cellsOf(this.cx, node, 'precisionLookup', () => evalSym(this.cx, precision, 'lookup'))[i] as Cell
      if (lookedUp.kind === 'lookup' && lookedUp.precisions !== undefined) return { loc, scale: lookedUp }
    }
    const loc = paramCell(this.cx, node, 'loc', i)
    // The precision idiom `1 / sqrt(tau)` where `tau` is a gamma variable, or one chosen among gamma variables by a discrete variable.
    const randomPrecision = (): boolean => {
      const p = scaleExpr === undefined ? undefined : precisionExpr(this.cx.idiom, scaleExpr)
      if (p === undefined) return false
      try {
        const c = cellsOf(this.cx, node, 'precisionProbe', () => evalSym(this.cx, p))[i] as Cell
        return c.kind === 'scaled' || (c.kind === 'rv' && c.family === 'gamma')
      } catch (e) {
        if (e instanceof Refusal) return false
        throw e
      }
    }
    // A mean that mentions a discrete variable but is a random Gaussian after its gates (a `select`), with a scale that does not
    // (a precision idiom `1 / sqrt(tau)`), is an ordinary observation: the scale is not evaluated as a lookup.
    if (loc.kind !== 'lookup' && (!discrete(scaleExpr) || randomPrecision())) return undefined
    const scale = paramCell(this.cx, node, 'scale', i)
    return loc.kind === 'lookup' || scale.kind === 'lookup' ? { loc, scale } : undefined
  }

  /**
   * An observed Gaussian whose location and (or) scale is a lookup by one discrete variable `z`: when the location is known in
   * every state, a table over `z` (the likelihood of the value under each state: a hidden Markov model's emission); when it is
   * random in some states, a `gate` of one Gaussian likelihood per state, over the random variables of the branches (the clutter
   * problem, a mixture with a known component, a change point).
   */
  private observedSwitch(
    node: ModelNode & { readonly kind: 'observed' },
    name: string,
    y: number,
    index: number | undefined,
    loc: Cell,
    scale: Cell,
  ): null {
    const looks = [loc, scale].filter((c): c is Cell & { kind: 'lookup' } => c.kind === 'lookup')
    const z = (looks[0] as Cell & { kind: 'lookup' }).variable
    if (looks.some((l) => l.variable !== z))
      throw new Refusal(
        'operator',
        `the location and the scale of '${node.name}' are chosen by different discrete variables`,
      )
    const states = (looks[0] as Cell & { kind: 'lookup' }).values.length
    if (looks.some((l) => l.values.length !== states))
      throw new Refusal('operator', `the lookups of '${node.name}' have different numbers of states`)
    const precisions = scale.kind === 'lookup' ? scale.precisions : undefined
    if (scale.kind === 'lookup' && scale.items !== undefined)
      throw new Refusal(
        'operator',
        `the scale of '${node.name}' is random in some states of a discrete variable: it must be known`,
      )
    if (scale.kind !== 'lookup' && !isConstCell(scale))
      throw new Refusal('operator', `the scale of '${node.name}' is not known`)
    // With a precision lookup, `values` are the known precisions of the states that have no gamma variable, not scales.
    const sdAt = (k: number): number =>
      precisions !== undefined
        ? 1 / Math.sqrt((scale as Cell & { kind: 'lookup' }).values[k] as number)
        : scale.kind === 'lookup'
          ? (scale.values[k] as number)
          : (scale as Cell & { kind: 'aff' }).aff.offset
    for (let k = 0; k < states; k++) {
      if (precisions?.[k] !== undefined) {
        const p = precisions[k] as Precision
        if (!(p.scale >= 0 && p.offset >= 0 && p.scale + p.offset > 0))
          throw new Refusal(
            'data',
            `the precision of '${node.name}' must be a non-negative multiple of a gamma variable`,
          )
        continue
      }
      if (!(sdAt(k) > 0)) throw new Refusal('data', `the scale of '${node.name}' is ${sdAt(k)}: it must be positive`)
    }
    const locAt = (k: number): Aff | number => {
      if (loc.kind === 'lookup') return loc.items?.[k] ?? (loc.values[k] as number)
      return affOf(loc)
    }
    const known =
      precisions === undefined &&
      Array.from({ length: states }, (_, k) => locAt(k)).every((m) => typeof m === 'number' || m.terms.size === 0)
    const pdf = (m: number, sd: number): number => Math.exp(-0.5 * ((y - m) / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI))
    const value = (m: Aff | number): number => (typeof m === 'number' ? m : m.offset)
    if (known) {
      this.cx.g.need('table')
      this.cx.g.addFactor(
        {
          operator: 'table',
          args: [z],
          params: { states: [states], values: Array.from({ length: states }, (_, k) => pdf(value(locAt(k)), sdAt(k))) },
        },
        name,
      )
      return null
    }
    this.cx.g.need('gate')
    const gate = new GateBuilder(z)
    for (let k = 0; k < states; k++) {
      const m = locAt(k)
      let mean: number | { readonly const: number }
      if (typeof m === 'number' || m.terms.size === 0) mean = { const: value(m) }
      else mean = gate.slot(this.cx.g.materialise(m, node.name, index) as string)
      const gamma = precisions?.[k]
      if (gamma !== undefined) {
        const at = gate.slot(gamma.variable)
        this.cx.g.need('gaussian-gamma-precision')
        gate.add(precisionBranch([{ const: y }, mean, at], gamma))
      } else
        gate.add({
          operator: 'gaussian',
          args: [{ const: y }, typeof mean === 'number' ? mean : mean],
          params: { precision: 1 / sdAt(k) ** 2 },
        })
    }
    this.cx.g.addFactor(gate.factor(), name)
    return null
  }

  /**
   * An observed Gaussian whose scale is `exp(s)` with `s` chosen by a discrete variable `z` among Gaussian forms of its own (a spread
   * per component of a mixture, `exp(at(logSd, z))`), whose mean may be chosen by the same `z` among random ones: ONE `gate` on `z`
   * whose branch `k` is the `gaussian-scale` factor of the observation on the mean and the log scale of state `k`, so the
   * component that explains the reading is the one whose mean and spread both move (a `select` of each would let EP lose which
   * component the pair came from, and a spread per component run off). False when the scale is not of that form (the other
   * routes are tried).
   */
  private observedSwitchLogScale(
    node: ModelNode & { readonly kind: 'observed' },
    y: number,
    i: number,
    index: number | undefined,
  ): boolean {
    const scaleExpr = node.dist.params['scale']
    if (scaleExpr === undefined || this.cx.isConst(scaleExpr)) return false
    const logArgument = expArgument(stripClip(this.cx.idiom, scaleExpr))
    if (logArgument === undefined || !mentionsDiscrete(this.cx.latents, logArgument)) return false
    let loc: Cell
    let logScale: Cell
    try {
      loc = paramCell(this.cx, node, 'loc', i, 'lookup')
      logScale = cellsOf(this.cx, node, 'logScaleLookup', () => evalSym(this.cx, logArgument, 'lookup'))[i] as Cell
    } catch (err) {
      if (err instanceof Refusal) return false
      throw err
    }
    if (logScale.kind !== 'lookup' || logScale.precisions !== undefined) return false
    const z = logScale.variable
    const states = logScale.values.length
    if (loc.kind === 'lookup' && (loc.variable !== z || loc.values.length !== states || loc.precisions !== undefined))
      throw new Refusal(
        'operator',
        `the mean and the spread of '${node.name}' are chosen by different discrete variables`,
      )
    if (loc.kind !== 'lookup' && loc.kind !== 'aff')
      throw new Refusal('operator', `the mean of '${node.name}' is not a Gaussian form or a lookup`)
    const locAt = (k: number): Aff | number => {
      if (loc.kind === 'lookup') return loc.items?.[k] ?? (loc.values[k] as number)
      return affOf(loc)
    }
    const gate = new GateBuilder(z)
    const argOf = (form: Aff | number | undefined, fallback: number): number | { readonly const: number } => {
      if (form === undefined || typeof form === 'number' || form.terms.size === 0)
        return { const: form === undefined ? fallback : typeof form === 'number' ? form : form.offset }
      return gate.slot(this.cx.g.materialise(form, node.name, index) as string)
    }
    this.cx.g.need('gate')
    this.cx.g.need('gaussian-scale')
    for (let k = 0; k < states; k++)
      gate.add({
        operator: 'gaussian-scale',
        args: [{ const: y }, argOf(locAt(k), 0), argOf(logScale.items?.[k], logScale.values[k] as number)],
        params: {},
      })
    this.cx.g.addFactor(gate.factor(), this.cx.elementName(node.name, node.dims, i))
    return true
  }

  /**
   * An observed Gaussian whose scale is `exp(s)` with `s` an affine form of Gaussian variables (a `gaussian-log-scale`
   * factor: the observation of a stochastic volatility model); the mean must be known.
   */
  private observedLogScale(
    node: ModelNode & { readonly kind: 'observed' },
    name: string,
    y: number,
    i: number,
    index: number | undefined,
  ): string | undefined {
    const inner = unwrap(this.cx.nodes, node.dist.params['scale'] as ExprSpec) as ExprSpec & { readonly kind: 'op' }
    const loc = paramCell(this.cx, node, 'loc', i)
    if (!isConstCell(loc)) return undefined
    let cell: Cell
    try {
      cell = cellsOf(this.cx, node, 'logScale', () => evalSym(this.cx, inner.args[0] as ExprSpec))[i] as Cell
    } catch (e) {
      // Not an affine form of Gaussian variables (a product of random values, a gamma latent): the precision idiom is tried next.
      if (e instanceof Refusal) return undefined
      throw e
    }
    if (isConstCell(cell) || cell.kind !== 'aff') return undefined
    const s = this.cx.g.materialise(affOf(cell), node.name, index) as string
    this.cx.g.addVariable({ name, family: { kind: 'gaussian' }, observed: y })
    this.cx.g.addFactor({ operator: 'gaussian-log-scale', args: [name, s], params: { loc: loc.aff.offset } }, name)
    return name
  }

  /**
   * An observed Gaussian whose scale is a positive latent (the `exp` of a Gaussian log-scale variable: a half-normal or
   * half-Cauchy sd projected by `latentProjected`, or `exp(affine)`) and whose mean may be random too: the `gaussian-scale`
   * factor, a one-dimensional quadrature over the log scale. Anything else is refused with the precision idiom named.
   */
  private observedRandomScale(
    node: ModelNode & { readonly kind: 'observed' },
    name: string,
    y: number,
    i: number,
    index: number | undefined,
    scaleExpr: ExprSpec,
  ): string {
    const stripped = stripClip(this.cx.idiom, scaleExpr)
    let logVar: string | undefined
    try {
      const logArgument = expArgument(stripped)
      if (logArgument !== undefined) {
        const cell = cellsOf(this.cx, node, 'logScale', () => evalSym(this.cx, logArgument))[i] as Cell
        if (cell.kind === 'aff' && !isConstCell(cell))
          logVar = this.cx.g.materialise(cell.aff, node.name, index) as string
      } else {
        const cell = cellsOf(this.cx, node, 'scaleValue', () => evalSym(this.cx, stripped))[i] as Cell
        logVar = logVariableOfCell(cell, this.cx.g.atoms, this.cx.logScale)
      }
    } catch (e) {
      if (!(e instanceof Refusal)) throw e
    }
    if (logVar === undefined) randomPrecisions(this.cx, node)
    const loc = paramCell(this.cx, node, 'loc', i)
    const mean = this.cx.g.portOf(loc, node.name, index)
    this.cx.g.need('gaussian-scale')
    this.cx.g.addVariable({ name, family: { kind: 'gaussian' }, observed: y })
    this.cx.g.addFactor({ operator: 'gaussian-scale', args: [name, mean, logVar as string] }, name)
    return name
  }

  /** An observed Gaussian with a gamma precision (`gaussian-gamma-precision`): the mean is a variable or a known value. */
  private observedRandomPrecision(
    node: ModelNode & { readonly kind: 'observed' },
    name: string,
    y: number,
    i: number,
    index: number | undefined,
  ): string {
    const tau = cellsOf(this.cx, node, 'precision', () => ({
      dims: node.dims,
      cells: randomPrecisions(this.cx, node).map((p): Cell => ({ kind: 'scaled', ...p })),
    }))[i] as Cell & { kind: 'scaled' }
    const loc = paramCell(this.cx, node, 'loc', i)
    const mean = this.cx.g.portOf(loc, node.name, index)
    this.cx.g.addVariable({ name, family: { kind: 'gaussian' }, observed: y })
    this.cx.g.addFactor(precisionFactor(name, mean, tau), name)
    return name
  }

  private observedBernoulli(
    node: ModelNode & { readonly kind: 'observed' },
    name: string,
    y: number,
    i: number,
    index: number | undefined,
  ): string | null {
    const probs = node.dist.params['probs']
    const logits = node.dist.params['logits']
    const link = (op: string, arg: ExprSpec, params: Record<string, Json> = {}): string | null => {
      const factor = factorName(this.cx.modelOps, op)
      this.cx.g.need(factor)
      const c = cellsOf(this.cx, node, `link:${op}`, () => evalSym(this.cx, arg))[i] as Cell
      if (isConstCell(c)) return null
      const x = this.cx.g.materialise(affOf(c), node.name, index) as string
      this.cx.g.addVariable({ name, family: { kind: 'bernoulli' }, observed: y })
      this.cx.g.addFactor({ operator: factor, args: [name, x], params }, name)
      return name
    }
    if (logits !== undefined) return link('sigmoid', logits)
    if (probs === undefined) throw new Refusal('structure', `'${node.name}': give probs or logits`)
    const e = unwrap(this.cx.nodes, probs)
    if (e.kind === 'op' && e.op === 'probit') return link('probit', e.args[0] as ExprSpec)
    if (e.kind === 'op' && e.op === 'sigmoid') return link('sigmoid', e.args[0] as ExprSpec)
    if (e.kind === 'op' && e.op === 'gt') {
      const threshold = this.cx.evalNamed(e.args[1] as ExprSpec)
      const cell = this.cx.over(this.cx.fromNamed(threshold), node.dims)[i] as Cell & { kind: 'aff' }
      return link('gt', e.args[0] as ExprSpec, { threshold: cell.aff.offset })
    }
    const p = paramCell(this.cx, node, 'probs', i)
    if (isConstCell(p)) return null
    if (p.kind === 'rv' && p.family === 'beta') {
      this.cx.g.addVariable({ name, family: { kind: 'bernoulli' }, observed: y })
      this.cx.g.addFactor({ operator: 'bernoulli', args: [name, p.variable] }, name)
      return name
    }
    throw new Refusal(
      'operator',
      `the probability of '${node.name}' is not a link of an affine value (probit, sigmoid, a comparison) or a beta latent: no message rule`,
    )
  }

  /**
   * An observed exponential or gamma value (a known shape) whose rate is a gamma latent or a known multiple of one: the conjugate
   * `gamma-observed` factor, a gamma message to the rate (a waiting time with a gamma prior on the rate).
   */
  private observedGammaRate(
    node: ModelNode & { readonly kind: 'observed' },
    fam: string,
    name: string,
    y: number,
    i: number,
  ): string | null {
    if (!(y > 0)) throw new Refusal('data', `'${node.name}' is ${fam}: the value ${y} must be positive`)
    const shape = fam === 'gamma' ? constAt(this.cx, node, node.dist.params['concentration'] as ExprSpec, i) : 1
    const target = paramCell(this.cx, node, 'rate', i)
    if (isConstCell(target)) return null
    const tau = scaledOf(target)
    if (tau === undefined || tau.offset !== 0 || !(tau.scale > 0))
      throw new Refusal(
        'operator',
        `the rate of '${node.name}' must be a gamma latent or a known multiple of one for the conjugate '${fam}' operator, it is not`,
      )
    this.cx.g.need('gamma-observed')
    this.cx.g.addVariable({ name, family: { kind: 'gaussian' }, observed: y })
    this.cx.g.addFactor(
      {
        operator: 'gamma-observed',
        args: [name, tau.variable],
        params: { shape, ...(tau.scale === 1 ? {} : { scale: tau.scale }) },
      },
      name,
    )
    return name
  }

  /**
   * An observed two-component mixture with known means and scale and a beta weight (`normalPair`, the switch summed out): the
   * `mixture-weight` factor on the weight, with the likelihood of the value under each component.
   */
  private observedMixtureWeight(
    node: ModelNode & { readonly kind: 'observed' },
    name: string,
    y: number,
    i: number,
  ): string | null {
    const target = paramCell(this.cx, node, 'weight', i)
    if (isConstCell(target)) return null
    if (target.kind !== 'rv' || target.family !== 'beta')
      throw new Refusal(
        'operator',
        `the weight of '${node.name}' must be a beta latent for the 'mixture-weight' operator`,
      )
    const [low, high, scale] = ['low', 'high', 'scale'].map((k) =>
      constAt(this.cx, node, node.dist.params[k] as ExprSpec, i),
    )
    const density = (m: number): number =>
      Math.exp(-0.5 * ((y - m) / (scale as number)) ** 2) / ((scale as number) * Math.sqrt(2 * Math.PI))
    this.cx.g.need('mixture-weight')
    this.cx.g.addFactor(
      {
        operator: 'mixture-weight',
        args: [target.variable],
        params: { l1: density(high as number), l0: density(low as number) },
      },
      name,
    )
    return null
  }

  /**
   * An observed geometric (trials up to the first success) or negative binomial (failures before the `n`-th success, `n` known)
   * count whose success probability is a beta latent: the conjugate `negative-binomial-beta` factor.
   */
  private observedFailures(
    node: ModelNode & { readonly kind: 'observed' },
    fam: string,
    name: string,
    y: number,
    i: number,
  ): string | null {
    const geometric = fam === 'geometric'
    const failures = geometric ? y - 1 : y
    if (!(failures >= 0 && Number.isInteger(failures)))
      throw new Refusal('data', `'${node.name}' is ${fam}: the count ${y} is not on its support`)
    const successes = geometric ? 1 : constAt(this.cx, node, node.dist.params['n'] as ExprSpec, i)
    const target = paramCell(this.cx, node, 'probs', i)
    if (isConstCell(target)) return null
    if (target.kind !== 'rv' || target.family !== 'beta')
      throw new Refusal(
        'operator',
        `the probs of '${node.name}' must be a beta latent for the conjugate '${fam}' operator, it is not`,
      )
    this.cx.g.need('negative-binomial-beta')
    this.cx.g.addVariable({ name, family: { kind: 'gaussian' }, observed: failures })
    this.cx.g.addFactor(
      { operator: 'negative-binomial-beta', args: [name, target.variable], params: { successes } },
      name,
    )
    return name
  }

  /**
   * An observed categorical whose probabilities are the row of a matrix of Dirichlet latents chosen by a discrete latent
   * (`categorical({ probs: phi.at(z) })`, the words of a topic model: `phi_k` the word distribution of topic `k`, `z` the topic of
   * the token). It is one `gate` on the index whose branch `k` is the conjugate `discrete` factor of the observed outcome on row
   * `k`: each row hears the tokens, weighted by the belief that the token chose it, and the index hears how well each row
   * explains the outcome. The same gate as an unknown transition row, with an observed outcome.
   */
  private observedRowGate(
    node: ModelNode & { readonly kind: 'observed' },
    name: string,
    y: number,
    i: number,
    expr: ExprSpec,
  ): string | null {
    const chosen = randomTransitionRow(this.cx.idiom, expr) as { readonly matrix: ExprSpec; readonly index: ExprSpec }
    const matrix = evalSym(this.cx, chosen.matrix)
    if (matrix.dims.length !== 1 || matrix.cells.some((r) => r.kind !== 'rv' || r.family !== 'dirichlet'))
      throw new Refusal(
        'operator',
        `the rows of the matrix behind '${node.name}' must be one Dirichlet latent each (a latent over the dimension of the states)`,
      )
    const rowVars = matrix.cells.map((r) => (r as { readonly variable: string }).variable)
    const size = this.cx.g.sizeOf(rowVars[0] as string)
    const parent = this.cx.over(evalSym(this.cx, chosen.index), node.dims)[i] as Cell
    if (parent.kind !== 'rv' || (parent.family !== 'discrete' && parent.family !== 'bernoulli'))
      throw new Refusal(
        'operator',
        `the row of '${node.name}' must be chosen by a discrete latent, not a computed value`,
      )
    const from = parent.family === 'bernoulli' ? 2 : this.cx.g.sizeOf(parent.variable)
    if (from !== rowVars.length)
      throw new Refusal('data', `'${node.name}' has ${from} choices of row and its matrix ${rowVars.length} rows`)
    if (!(Number.isInteger(y) && y >= 0 && y < size))
      throw new Refusal('data', `'${node.name}' observes ${y}, outside its ${size} categories`)
    this.cx.g.addVariable({ name, family: { kind: 'discrete', size }, observed: y })
    this.cx.g.need('gate')
    this.cx.g.need('discrete')
    this.cx.g.addFactor(discreteRowGate(parent.variable, name, rowVars), name)
    return name
  }

  /**
   * An observed hurdle count: zero with probability `zero`, else a zero-truncated Poisson of `exposure * exp(affine)`. The state
   * of the hurdle (a structural zero, or a positive count) is determined by the count and summed out exactly: a zero is an
   * observation of the Bernoulli `zero`, with a beta probability the conjugate `bernoulli` factor, and a positive count is the same
   * Bernoulli observed false and the zero-truncated Poisson site on the log rate (the gate of {point mass at zero, truncated
   * Poisson}, with its switch observed). The positive part and the zero part share no variable, so each is exact.
   */
  private observedHurdle(
    node: ModelNode & { readonly kind: 'observed' },
    y: number,
    i: number,
    index: number | undefined,
  ): string | null {
    if (!(Number.isInteger(y) && y >= 0))
      throw new Refusal('data', `'${node.name}' is a hurdle count: ${y} is not a count`)
    const zero = paramCell(this.cx, node, 'zero', i)
    if (!isConstCell(zero)) {
      if (zero.kind !== 'rv' || zero.family !== 'beta')
        throw new Refusal('operator', `the zero probability of '${node.name}' must be known or a beta latent`)
      const crossed = this.cx.g.auxName(node.name, index)
      this.cx.g.addVariable({ name: crossed, family: { kind: 'bernoulli' }, observed: y === 0 ? 1 : 0 })
      this.cx.g.addFactor({ operator: 'bernoulli', args: [crossed, zero.variable] }, crossed)
    }
    if (y === 0) return null
    const numeric = this.observedNumeric(node, 'hurdlePoisson', y, i, index)
    if (numeric === undefined)
      throw new Refusal(
        'operator',
        `the rate of '${node.name}' must be \`exposure * exp(affine)\` for the hurdle Poisson`,
      )
    return numeric
  }

  private observedConjugate(
    node: ModelNode & { readonly kind: 'observed' },
    fam: string,
    name: string,
    y: number,
    i: number,
  ): string | null {
    const param = fam === 'poisson' ? 'rate' : 'probs'
    const expr = node.dist.params[param]
    if (expr === undefined) throw new Refusal('structure', `'${node.name}': the ${param} is missing`)
    // A word read through a discrete index of Dirichlet rows (`phi.at(z)`, a topic model): a gate over the index.
    if (fam === 'categorical' && randomTransitionRow(this.cx.idiom, expr) !== undefined)
      return this.observedRowGate(node, name, y, i, expr)
    let exposure = 1
    let target: Cell
    const e = unwrap(this.cx.nodes, expr)
    if (fam === 'poisson' && e.kind === 'op' && e.op === 'mul') {
      const [a, b] = e.args as [ExprSpec, ExprSpec]
      const [k, r] = this.cx.isConst(a) ? [a, b] : [b, a]
      if (!this.cx.isConst(k))
        throw new Refusal('operator', `the rate of '${node.name}' is a product of two random values`)
      exposure = (this.cx.over(this.cx.fromNamed(this.cx.evalNamed(k)), node.dims)[i] as Cell & { kind: 'aff' }).aff
        .offset
      target = this.cx.over(evalSym(this.cx, r), node.dims)[i] as Cell
    } else target = paramCell(this.cx, node, param, i)
    if (isConstCell(target)) return null
    const want: FamilyKind = fam === 'poisson' ? 'gamma' : fam === 'binomial' ? 'beta' : 'dirichlet'
    if (target.kind !== 'rv' || target.family !== want)
      throw new Refusal(
        'operator',
        `the ${param} of '${node.name}' must be a ${want} latent for the conjugate '${fam}' operator, it is not`,
      )
    if (fam === 'poisson') {
      this.cx.g.addVariable({ name, family: { kind: 'gaussian' }, observed: y })
      this.cx.g.addFactor({ operator: 'poisson', args: [name, target.variable], params: { exposure } }, name)
    } else if (fam === 'binomial') {
      const trials = (
        this.cx.over(this.cx.fromNamed(this.cx.evalNamed(node.dist.params['n'] as ExprSpec)), node.dims)[i] as Cell & {
          kind: 'aff'
        }
      ).aff.offset
      this.cx.g.addVariable({ name, family: { kind: 'gaussian' }, observed: y })
      this.cx.g.addFactor({ operator: 'binomial', args: [name, target.variable], params: { n: trials } }, name)
    } else {
      const size = this.cx.g.sizeOf(target.variable)
      this.cx.g.addVariable({ name, family: { kind: 'discrete', size }, observed: y })
      this.cx.g.addFactor({ operator: 'discrete', args: [name, target.variable] }, name)
    }
    return name
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
