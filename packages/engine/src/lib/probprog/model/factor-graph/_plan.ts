/**
 * The layout plan of the model compiler: the decisions about how the latents of a model are laid out on the graph that are made
 * from the shape of the whole spec before any node is mapped, as one immutable value. Four detections run in a fixed order, each
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
import { Refusal } from './_refusal.ts'

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

/** A group of scalar normals compiled as the coordinates of one joint Gaussian vector. */
export interface ScalarGroup {
  readonly vector: string
  readonly members: readonly string[]
}

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
 * The layout plan of a bound model, the four detections in their order.
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
  return { packed, matmulRead, nonCentred, groups, normalGamma }
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
 * With `packScalars`: the scalar normal latents (known prior) that an observed node reads together, directly or through
 * derived nodes, are grouped: each group is one joint Gaussian vector, so a likelihood on `a + b x` sees the correlation of
 * `a` and `b` instead of a loop of scalar messages (which keeps the means and loses the correlations: marginal sds too small).
 * Reads `packed` of `findPacked`.
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
  const candidate = (n: ModelNode | undefined): n is ModelNode & { readonly kind: 'latent' } =>
    n?.kind === 'latent' &&
    n.dims.length === 0 &&
    n.dist.family === 'normal' &&
    !packed.has(n.name) &&
    Object.values(n.dist.params).every(staticExpr)
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
    const visit = (name: string): void => {
      if (seen.has(name)) return
      seen.add(name)
      const t = nodes.get(name)
      if (candidate(t)) found.push(name)
      else if (t?.kind === 'derived') for (const r of nodeRefs(t)) visit(r)
    }
    for (const r of nodeRefs(n)) visit(r)
    for (const f of found) {
      if (!parent.has(f)) parent.set(f, f)
      if (found[0] !== undefined) parent.set(find(f), find(found[0]))
    }
  }
  const byRoot = new Map<string, string[]>()
  for (const name of parent.keys()) byRoot.set(find(name), [...(byRoot.get(find(name)) ?? []), name])
  const order = new Map(bound.spec.nodes.map((n, i) => [n.name, i]))
  for (const members of byRoot.values()) {
    if (members.length < 2) continue
    members.sort((a, b) => (order.get(a) as number) - (order.get(b) as number))
    const vector = auxId(members[0] as string, 8)
    for (const m of members) groups.set(m, { vector, members })
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
            if (v === undefined) throw new Refusal('node', `'${name}' is random where a known value is needed`)
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
