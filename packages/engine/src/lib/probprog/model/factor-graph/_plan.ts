/**
 * The layout plan of the model compiler: the decisions about how the latents of a model are laid out on the graph that are made
 * from the shape of the whole spec before any node is mapped, as one immutable value. Five detections run in a fixed order, each
 * reading only the ones before it: the vector latents that a `matmul` reads stay one Gaussian vector (`findPacked`, which also
 * records what a `matmul` reads), the non-centred vectors `z / S(x)` that one dense factor absorbs (`findNonCentred`, which reads
 * what `findPacked` recorded), the scalar normals an observation reads together that become one joint vector (`findGroups`,
 * which reads the packed vectors; only with `packScalars`) and the normal-gamma pairs (`findNormalGamma`, which reads the known
 * constants of `_known.ts` as they stand at each precision's position in the spec). The walk of the compiler reads the plan and
 * decides nothing about layout itself; what it changes while it maps (a non-centred vector whose dense factor does not apply,
 * a pair it has activated) is its own state, not the plan's.
 */

import type { BoundModel } from '../bind.ts'
import { auxId } from '../ids.ts'
import type { OperatorTable } from '../operators.ts'
import { type ExprSpec, exprRefs, type ModelNode, nodeRefs } from '../spec.ts'
import { findGplvm, type GplvmPlan } from './_gplvm.ts'
import {
  dependsOnLatent,
  type IdiomEnv,
  type LatentScope,
  latentScope,
  precisionMultiple,
  sharedGamma,
  unwrap,
} from './_idioms.ts'
import { evaluateKnown, isKnownExpr, type Known } from './_known.ts'
import { randomWhereKnown } from './_reasons.ts'

/** A Gaussian mean whose variance is a known multiple of the precision's reciprocal and the observations that read both: one normal-gamma variable. */
export interface NormalGammaPlan {
  /** The node of the precision (a gamma or inverse-gamma latent) and of the mean (a normal latent). */
  readonly precision: string
  readonly mean: string
  /** The variable (named as the mean's node) and whether the precision's node is an inverse gamma (a variance). */
  readonly variable: string
  readonly inverse: boolean
  /** The known multiple `kappa` of the mean's precision over the shared one: `mu | lambda ~ N(m, 1 / (kappa lambda))`. */
  readonly kappa: number
  readonly observed: readonly string[]
}

/** What the detection of a normal-gamma pair found at a precision's node. */
export interface NormalGammaDetection {
  /** The pair, with the known multiple of the precision each observed node carries; absent when the model is not that shape. */
  readonly pair: { readonly plan: NormalGammaPlan; readonly multiples: ReadonlyMap<string, number> } | undefined
  /** The custom functions that the detection evaluated (a custom function of known values in a scale counts as evaluated). */
  readonly customs: readonly string[]
  /** What evaluating a known value raised (a refusal of the registry): raised again where the walk maps the precision. */
  readonly failure?: unknown
}

/**
 * A group of normal latents compiled as the coordinates of one joint Gaussian vector. A group of scalars with known priors puts
 * its (diagonal) prior on the vector. A `tied` group has members whose priors are their own (a hierarchical mean, a vector of
 * county intercepts, a coefficient beside them): each member is made as it is without the group, and every element is tied to
 * its coordinate of the vector by an exact `dot-constant`, which only carries what the observations say jointly about them
 * (`sizes` is the number of elements of each member, in the order of `members`).
 */
export interface ScalarGroup {
  readonly vector: string
  readonly members: readonly string[]
  readonly tied?: true
  readonly sizes?: readonly number[]
}

/**
 * The most classes whose slopes (a vector over the class dimension) are packed for the `multiclass-row` factor: the rank of its integral
 * is the number of classes, and the product grid it runs on reaches six. Past that a softmax keeps one message per class.
 */
export const MULTICLASS_ROW_CLASSES = 6

/** The most coordinates of a tied group: its covariance is dense, and each row of a likelihood costs the square of it. */
export const TIED_GROUP_LIMIT = 200

/** The layout decisions of a model, found before its nodes are mapped. */
export interface LayoutPlan {
  /** The vector latents over one dimension that stay one Gaussian vector variable. */
  readonly packed: ReadonlySet<string>
  /** The nodes a `matmul` reads as a vector. */
  readonly matmulRead: ReadonlySet<string>
  /** The derived vectors `z / S(x)` that one dense factor absorbs, by name, with the name of `z`. */
  readonly nonCentred: ReadonlyMap<string, string>
  /** The scalar normals compiled as one joint vector (with `packScalars`), by member name. */
  readonly groups: ReadonlyMap<string, ScalarGroup>
  /** The normal-gamma detections, by the name of the precision's node (gamma and inverse-gamma latents over no dimension). */
  readonly normalGamma: ReadonlyMap<string, NormalGammaDetection>
  /** The latent variable model with random Fourier features, when the spec has that form (with `packScalars`; `_gplvm.ts`). */
  readonly gplvm: GplvmPlan | undefined
}

/** What the plan reads of the model besides its spec. */
export interface PlanOptions {
  /** The model operator registry (it evaluates the known values the detections read). */
  readonly modelOps: OperatorTable
  /** A vector of coefficients with the precisions of one shared gamma is one Gaussian vector (the default of `sharedScale`). */
  readonly denseScales: boolean
  /** Scalar normals that an observation reads together are one joint vector. */
  readonly packScalars: boolean
}

/**
 * The layout plan of a bound model, the five detections in their order.
 *
 * @param bound - The bound model (unrolled and with named dimensions only).
 * @param known - The constants of the model (`knownConstants`).
 * @param options - The registry and the layout switches.
 * @returns The plan.
 */
export function layoutPlan(bound: BoundModel, known: Known, options: PlanOptions): LayoutPlan {
  const nodes = new Map<string, ModelNode>()
  for (const n of bound.spec.nodes) if (n.kind !== 'observed') nodes.set(n.name, n)
  const scope = latentScope(bound.spec)
  const { packed, matmulRead } = findPacked(bound, nodes, scope, options.denseScales)
  const nonCentred = findNonCentred(bound, nodes, scope, matmulRead, options.denseScales)
  const groups = options.packScalars ? findGroups(bound, nodes, packed) : new Map<string, ScalarGroup>()
  const normalGamma = findNormalGamma(bound, nodes, known, options.modelOps)
  const gplvm = options.packScalars ? findGplvm(bound, nodes, known, options.modelOps) : undefined
  return { packed, matmulRead, nonCentred, groups, normalGamma, gplvm }
}

/** A latent over one dimension that a `matmul` reads stays a vector variable, so its posterior keeps the correlations. */
export function findPacked(
  bound: BoundModel,
  nodes: ReadonlyMap<string, ModelNode>,
  scope: LatentScope,
  denseScales: boolean,
): { readonly packed: Set<string>; readonly matmulRead: Set<string> } {
  const packed = new Set<string>()
  const matmulRead = new Set<string>()
  const walk = (e: ExprSpec): void => {
    if (e.kind !== 'op' && e.kind !== 'custom') return
    if (e.kind === 'op' && e.op === 'matmul') for (const a of e.args) if (a.kind === 'ref') matmulRead.add(a.name)
    // A custom function that is a matrix of a normal vector (read as one by the compiler) keeps that vector joint too.
    if (e.kind === 'op' ? e.op === 'matmul' : bound.customs[e.name] !== undefined)
      for (const a of e.args) {
        const target = a.kind === 'ref' ? nodes.get(a.name) : undefined
        // A normal with a random scale (ARD: a precision of its own for each coefficient) cannot be one joint Gaussian vector:
        // its elements stay separate variables, each with its own gaussian-gamma-precision factor.
        const scale = target?.kind === 'latent' ? target.dist.params['scale'] : undefined
        const randomScale =
          scale !== undefined &&
          dependsOnLatent(scope, scale) &&
          !(denseScales && sharedGamma(scope, scale) !== undefined)
        if (
          target?.kind === 'latent' &&
          target.dims.length === 1 &&
          ((target.dist.family === 'normal' && !randomScale) || target.dist.family === 'normalSpikeSlab')
        )
          packed.add(target.name)
      }
    for (const a of e.args) walk(a)
  }
  for (const n of bound.spec.nodes) {
    if (n.kind === 'latent' || n.kind === 'observed' || n.kind === 'generated')
      for (const e of Object.values(n.dist.params)) walk(e)
    if (n.kind === 'derived' || n.kind === 'factor') walk(n.expr)
  }
  return { packed, matmulRead }
}

/**
 * The non-centred vectors: a derived node `D = z / S` whose numerator `z` is a latent vector of independent standard normals
 * that nothing else reads, and whose divisor depends on a latent (the smoothing parameter of the non-centred penalised spline,
 * `z / sqrt(exp(x) d + n)`). Compiled as one Gaussian vector with a `gaussian-mv-function-scale` factor when the divisor is a
 * chain of unary steps from one variable (`denseNonCentred`), so the posterior keeps the correlations of the coefficients; `z` is
 * then not a variable of the graph (the factor integrates it out). Reads `matmulRead` of `findPacked`.
 */
export function findNonCentred(
  bound: BoundModel,
  nodes: ReadonlyMap<string, ModelNode>,
  scope: LatentScope,
  matmulRead: ReadonlySet<string>,
  denseScales: boolean,
): Map<string, string> {
  const nonCentred = new Map<string, string>()
  if (!denseScales) return nonCentred
  const readers = new Map<string, number>()
  for (const n of bound.spec.nodes) for (const r of nodeRefs(n)) readers.set(r, (readers.get(r) ?? 0) + 1)
  const is = (e: ExprSpec | undefined, value: number): boolean => e?.kind === 'const' && e.value === value
  for (const n of bound.spec.nodes) {
    if (n.kind !== 'derived' || n.expr.kind !== 'op' || n.expr.op !== 'div' || n.dims.length !== 1) continue
    if (!matmulRead.has(n.name)) continue
    const [numerator, divisor] = n.expr.args
    const z = numerator?.kind === 'ref' ? nodes.get(numerator.name) : undefined
    if (z?.kind !== 'latent' || z.dist.family !== 'normal' || z.dims.length !== 1 || z.dims[0] !== n.dims[0]) continue
    if (z.constraint !== undefined && z.constraint.kind !== 'real') continue
    if (!is(z.dist.params['loc'], 0) || !is(z.dist.params['scale'], 1) || readers.get(z.name) !== 1) continue
    if (divisor === undefined || !dependsOnLatent(scope, divisor)) continue
    nonCentred.set(n.name, z.name)
  }
  return nonCentred
}

/**
 * With `packScalars`: the normal latents that an observed node reads together, directly or through derived nodes, are grouped:
 * each group is one joint Gaussian vector, so a likelihood on `a + b x` sees the correlation of `a` and `b` instead of a loop of
 * scalar messages (which keeps the means and loses the correlations: marginal sds too small). The scalar normals with known
 * priors form the plain group (the prior is on the vector); when the group also has a normal that is a vector over one dimension
 * or has a prior of its own that depends on other latents (the county intercepts of a hierarchy and the floor effect beside
 * them), it is a `tied` group, the members made as they are and tied to the vector. Reads `packed` of `findPacked`.
 */
export function findGroups(
  bound: BoundModel,
  nodes: ReadonlyMap<string, ModelNode>,
  packed: ReadonlySet<string>,
): Map<string, ScalarGroup> {
  const groups = new Map<string, ScalarGroup>()
  const staticExpr = (e: ExprSpec): boolean =>
    e.kind === 'const' ||
    (e.kind === 'ref' && nodes.get(e.name)?.kind === 'data') ||
    (e.kind === 'op' && e.args.every(staticExpr))
  /** A normal whose prior is known: constants and data only. */
  const isStatic = (n: ModelNode & { readonly kind: 'latent' }): boolean =>
    Object.values(n.dist.params).every(staticExpr)
  const isStaticScalar = (n: ModelNode & { readonly kind: 'latent' }): boolean => n.dims.length === 0 && isStatic(n)
  const candidate = (n: ModelNode | undefined): n is ModelNode & { readonly kind: 'latent' } =>
    n?.kind === 'latent' &&
    n.dims.length <= 1 &&
    n.dist.family === 'normal' &&
    n.constraint === undefined &&
    !packed.has(n.name) &&
    (n.dims[0] === undefined || (bound.sizes[n.dims[0]] ?? 0) > 0)
  // The vectors that one observed node reads at two places (`skill[a] - skill[b]`: the players of a game).
  const crossing = new Set<string>()
  const parent = new Map<string, string>()
  const find = (x: string): string => {
    let r = x
    while ((parent.get(r) ?? r) !== r) r = parent.get(r) as string
    parent.set(x, r)
    return r
  }
  for (const n of bound.spec.nodes) {
    if (n.kind !== 'observed') continue
    const found: string[] = []
    const seen = new Set<string>()
    const reads = new Map<string, number>()
    // A name read by the observed node, directly or through derived nodes, and whether every read on the way is linear (a sum,
    // a difference, a known multiple, an index): only a linear read can use the coordinate of a vector.
    const visit = (name: string, linear: boolean): void => {
      const t = nodes.get(name)
      const asLinear = linear && !seen.has(`${name}|linear`)
      if (seen.has(name) && !asLinear) return
      seen.add(name)
      if (linear) seen.add(`${name}|linear`)
      if (candidate(t)) {
        if ((linear || isStaticScalar(t)) && !found.includes(name)) found.push(name)
      } else if (t?.kind === 'derived') walk(t.expr, linear)
    }
    const walk = (e: ExprSpec, linear: boolean): void => {
      if (e.kind === 'ref') {
        visit(e.name, linear)
        if (linear && candidate(nodes.get(e.name))) {
          reads.set(e.name, (reads.get(e.name) ?? 0) + 1)
          if ((reads.get(e.name) ?? 0) > 1) crossing.add(e.name)
        }
      } else if (e.kind === 'custom') for (const a of e.args) walk(a, false)
      else if (e.kind === 'op') {
        const open = e.args.filter((a) => !staticExpr(a))
        const keeps =
          e.op === 'add' || e.op === 'sub' || e.op === 'neg' || e.op === 'at' || e.op === 'sum' || e.op === 'mean'
            ? () => true
            : e.op === 'mul'
              ? () => open.length <= 1
              : e.op === 'div'
                ? (a: ExprSpec) => a === e.args[0]
                : () => false
        for (const a of e.args) walk(a, linear && keeps(a))
      }
    }
    // A link at the top of a parameter (`probit(x)`, `sigmoid(x)`, `exp(x)`) is the likelihood's, not a step of the predictor.
    const LINKS = ['probit', 'sigmoid', 'exp']
    for (const e of Object.values(n.dist.params))
      walk(e.kind === 'op' && LINKS.includes(e.op) && e.args.length === 1 ? (e.args[0] as ExprSpec) : e, true)
    walk(n.value, false)
    // A categorical over the classes reads a vector over the class dimension at every class (the slope of each class in the score of
    // that class): the vector is read at several places by one row, so it is one joint vector, as a vector read at two places is.
    const multiclass =
      (n.dist.family === 'categorical' && n.dist.params['logits'] !== undefined) ||
      n.dist.family === 'categoricalProbit'
    if (multiclass)
      for (const f of found) {
        const dim = nodes.get(f)?.dims[0]
        if (dim !== undefined && nodes.get(f)?.dims.length === 1 && (n.dist.eventDims ?? []).includes(dim))
          if ((bound.sizes[dim] ?? Number.POSITIVE_INFINITY) <= MULTICLASS_ROW_CLASSES) crossing.add(f)
      }
    for (const f of found) {
      if (!parent.has(f)) parent.set(f, f)
      if (found[0] !== undefined) parent.set(find(f), find(found[0]))
    }
  }
  const byRoot = new Map<string, string[]>()
  for (const name of parent.keys()) byRoot.set(find(name), [...(byRoot.get(find(name)) ?? []), name])
  const order = new Map(bound.spec.nodes.map((n, i) => [n.name, i]))
  const latent = (name: string): ModelNode & { readonly kind: 'latent' } =>
    nodes.get(name) as ModelNode & { readonly kind: 'latent' }
  for (const members of byRoot.values()) {
    // One member is a group only when it is a vector that a row reads at two places.
    if (members.length < 2 && !(latent(members[0] as string).dims.length === 1 && crossing.has(members[0] as string)))
      continue
    members.sort((a, b) => (order.get(a) as number) - (order.get(b) as number))
    const vector = auxId(members[0] as string, 8)
    const plain = members.every((m) => isStatic(latent(m)))
    const vectors = members.some((m) => latent(m).dims.length > 0)
    if (plain && !vectors) {
      for (const m of members) groups.set(m, { vector, members })
      continue
    }
    // A group with a vector needs a scalar beside it (something that every row shares) or a vector that a row reads twice, and a
    // joint covariance of a size to afford.
    const sizes = members.map((m) => {
      const dim = latent(m).dims[0]
      return dim === undefined ? 1 : (bound.sizes[dim] as number)
    })
    const total = sizes.reduce((a, b) => a + b, 0)
    const shared = members.some((m) => latent(m).dims.length === 0 || crossing.has(m))
    if (shared && total <= TIED_GROUP_LIMIT) {
      for (const m of members)
        groups.set(m, plain ? { vector, members, sizes } : { vector, members, tied: true, sizes })
      continue
    }
    // Too large, or nothing shared: the scalars with known priors are still the plain group.
    const statics = members.filter((m) => isStaticScalar(latent(m)))
    if (statics.length >= 2) for (const m of statics) groups.set(m, { vector, members: statics })
  }
  return groups
}

/**
 * The normal-gamma pair of each gamma (or inverse-gamma) latent `T` (a precision or a variance): a Gaussian latent `M` with a
 * known mean and the variance `1 / (kappa lambda)` (`mu ~ N(m, sigma2 / kappa)`) and observed Gaussians `y ~ N(M, 1 / (c lambda))`,
 * with nothing else reading either of them. The pair is one `normal-gamma` variable with a prior of both and a
 * `gaussian-normal-gamma` factor per observation, so the joint posterior is exactly the conjugate normal-gamma (the classic
 * normal-inverse-gamma prior on a mean and a variance). A precision whose model is not that shape has an entry with no pair: the
 * mean and precision then compile to a Gaussian with a gamma precision each.
 *
 * "Known" is read as of the precision's position in the spec: a constant declared after it does not count, as when the detection
 * ran in the walk at that node. A constant that only the symbolic evaluation of the walk finds known does not count either
 * (`Known.semantic`), because the walk has not run.
 */
export function findNormalGamma(
  bound: BoundModel,
  nodes: ReadonlyMap<string, ModelNode>,
  known: Known,
  modelOps: OperatorTable,
): Map<string, NormalGammaDetection> {
  const found = new Map<string, NormalGammaDetection>()
  for (const [at, node] of bound.spec.nodes.entries()) {
    if (node.kind !== 'latent' || node.dims.length !== 0) continue
    if (node.dist.family !== 'gamma' && node.dist.family !== 'inverseGamma') continue
    const customs: string[] = []
    const env: IdiomEnv = {
      nodes,
      isConst: (e) => isKnownExpr(e, known, bound.customs, at),
      valueOf: (e) =>
        evaluateKnown(e, {
          modelOps,
          customs: bound.customs,
          ref: (name) => {
            const v = known.get(name)
            if (v === undefined) throw randomWhereKnown(name)
            return v
          },
          onCustom: (name) => customs.push(name),
        }),
      // The precision idiom of an inverse-gamma variance is never a multiple of the gamma being detected: no pair either way.
      isRecipNode: () => false,
      // No node has been compiled yet: the detection reads no log variable.
      hasLogVariable: () => false,
    }
    try {
      found.set(node.name, { pair: detectPair(bound, node, env), customs })
    } catch (failure) {
      found.set(node.name, { pair: undefined, customs, failure })
    }
  }
  return found
}

function detectPair(
  bound: BoundModel,
  node: ModelNode & { readonly kind: 'latent' },
  env: IdiomEnv,
): NormalGammaDetection['pair'] {
  const inverse = node.dist.family === 'inverseGamma'
  const rateName = inverse ? 'scale' : 'rate'
  const concentration = node.dist.params['concentration']
  const rate = node.dist.params[rateName]
  if (concentration === undefined || rate === undefined) return undefined
  if (!env.isConst(concentration) || !env.isConst(rate)) return undefined
  const readers = bound.spec.nodes.filter((n) => n.name !== node.name && nodeRefs(n).includes(node.name))
  const means = readers.filter((n) => n.kind === 'latent')
  const mean = means[0]
  if (mean?.kind !== 'latent' || means.length !== 1) return undefined
  const loc = mean.dist.params['loc']
  const scale = mean.dist.params['scale']
  if (mean.dims.length !== 0 || mean.dist.family !== 'normal' || loc === undefined || scale === undefined)
    return undefined
  if (!env.isConst(loc) || mean.constraint !== undefined) return undefined
  const kappa = precisionMultiple(env, scale, node.name, inverse)
  if (kappa === undefined) return undefined
  const observed: (ModelNode & { readonly kind: 'observed' })[] = []
  const multiples = new Map<string, number>()
  for (const n of bound.spec.nodes) {
    if (n.name === node.name || n.name === mean.name) continue
    const refs = nodeRefs(n)
    if (!refs.includes(node.name) && !refs.includes(mean.name)) continue
    if (n.kind !== 'observed' || n.dist.family !== 'normal') return undefined
    const l = n.dist.params['loc']
    const sc = n.dist.params['scale']
    if (l === undefined || sc === undefined || unwrap(env.nodes, l).kind !== 'ref') return undefined
    if ((unwrap(env.nodes, l) as { readonly name: string }).name !== mean.name) return undefined
    if (exprRefs(n.value).some((r) => r === node.name || r === mean.name)) return undefined
    const c = precisionMultiple(env, sc, node.name, inverse)
    if (c === undefined) return undefined
    if (bound.batch !== undefined && n.dims.includes(bound.batch.dim)) return undefined
    observed.push(n)
    multiples.set(n.name, c)
  }
  if (observed.length === 0) return undefined
  const plan: NormalGammaPlan = {
    precision: node.name,
    mean: mean.name,
    variable: mean.name,
    inverse,
    kappa,
    observed: observed.map((n) => n.name),
  }
  return { plan, multiples }
}
