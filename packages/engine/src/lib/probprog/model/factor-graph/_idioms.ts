/**
 * The recognisers of the model compiler: pure functions that read the shape of an expression (the precision idiom of a Gaussian
 * scale `1 / sqrt(tau)`, the known multiple of a gamma variable, an inverse-gamma variance, which latents an expression
 * depends on) and answer without making any graph structure. What they need to know about the model (which nodes there are,
 * which expressions are known and with what value) comes in as an `IdiomEnv`, so the constants pass, the layout plan and the
 * mappings ask the same questions the same way. A recogniser that does not match returns undefined or false; none refuses.
 */

import { toFlat } from '../../../tensor/index.ts'
import { type LatentMemo, latentsIn, type NodesByName, nodesByName, usesLatent } from '../analysis-exprs.ts'
import type { NamedValue } from '../operators.ts'
import type { ExprSpec, ModelNode, ModelSpec } from '../spec.ts'

/** What the recognisers ask of the model they read. */
export interface IdiomEnv {
  /** The nodes an expression can refer to, by name (every node but the observed ones). */
  readonly nodes: ReadonlyMap<string, ModelNode>
  /** Whether an expression is known (has no random part). */
  isConst(e: ExprSpec): boolean
  /** The value of a known expression. */
  valueOf(e: ExprSpec): NamedValue
  /** Whether a node is an inverse-gamma variance already compiled (its cells are all reciprocal gamma variables). */
  isRecipNode(name: string): boolean
}

/** The nodes of a spec an expression can refer to, and the memo that the analysis module's `latentsIn` keeps over them. */
export interface LatentScope {
  readonly byName: NodesByName
  readonly memo: LatentMemo
}

/** The scope of `usesLatent` and `latentsIn` over a spec. */
export function latentScope(spec: ModelSpec): LatentScope {
  return { byName: nodesByName(spec), memo: new Map() }
}

/**
 * Whether an expression refers (through derived nodes) to a latent: a random value, where data and constants are known. The
 * analysis module's `usesLatent`; it also counts a `scan`'s states as latents, which the compiler never sees (the scans are
 * unrolled into nodes before it runs), so the two definitions agree.
 */
export function dependsOnLatent(scope: LatentScope, e: ExprSpec): boolean {
  return usesLatent(e, scope.byName, scope.memo)
}

/**
 * The one scalar gamma (or exponential) latent that an expression depends on, through derived nodes, when it is the only
 * latent it reads: the shared smoothing parameter or ridge strength of a vector whose coefficients' precisions are
 * known multiples of it. Absent when there are several latents or the one is not a scalar gamma.
 */
export function sharedGamma(scope: LatentScope, e: ExprSpec): string | undefined {
  const found = latentsIn(e, scope.byName, scope.memo)
  if (found.size !== 1) return undefined
  const n = scope.byName.get([...found][0] as string)
  return n?.kind === 'latent' && n.dims.length === 0 && (n.dist.family === 'gamma' || n.dist.family === 'exponential')
    ? n.name
    : undefined
}

/** The expression a derived node stands for, followed through references to derived nodes (to see a link applied to it). */
export function unwrap(nodes: ReadonlyMap<string, ModelNode>, e: ExprSpec): ExprSpec {
  let cur = e
  for (let k = 0; k < 64 && cur.kind === 'ref'; k++) {
    const n = nodes.get(cur.name)
    if (n?.kind !== 'derived') break
    cur = n.expr
  }
  return cur
}

/** Whether an expression is an inverse-gamma variance or a known multiple of one (`sigma2`, `sigma2 / kappa`, `2 * sigma2`). */
export function isRecip(env: IdiomEnv, e: ExprSpec): boolean {
  const u = unwrap(env.nodes, e)
  if (u.kind === 'ref') return env.isRecipNode(u.name)
  if (u.kind === 'op' && (u.op === 'mul' || u.op === 'div')) {
    const [a, b] = u.args as [ExprSpec, ExprSpec]
    return (isRecip(env, a) && env.isConst(b)) || (u.op === 'mul' && env.isConst(a) && isRecip(env, b))
  }
  return false
}

/**
 * The expression a Gaussian's scale takes when it is the idiom of a random precision: `1 / sqrt(tau)`, `tau ** -0.5` or
 * `sqrt(1 / tau)`, followed through derived nodes. Absent when the scale is not that shape.
 */
export function precisionExpr(env: IdiomEnv, scale: ExprSpec): ExprSpec | undefined {
  const e = unwrap(env.nodes, scale)
  if (e.kind !== 'op') return undefined
  const isOne = (x: ExprSpec): boolean => {
    if (!env.isConst(x)) return false
    return toFlat(env.valueOf(x).value).every((v) => v === 1)
  }
  if (e.op === 'div') {
    const [num, den] = e.args as [ExprSpec, ExprSpec]
    const d = unwrap(env.nodes, den)
    if (isOne(num) && d.kind === 'op' && d.op === 'sqrt') return (d.args as [ExprSpec])[0]
  }
  if (e.op === 'pow') {
    const [base, power] = e.args as [ExprSpec, ExprSpec]
    if (env.isConst(power) && toFlat(env.valueOf(power).value).every((v) => v === -0.5)) return base
  }
  if (e.op === 'sqrt') {
    const inner = unwrap(env.nodes, (e.args as [ExprSpec])[0])
    if (inner.kind === 'op' && inner.op === 'div' && isOne(inner.args[0] as ExprSpec)) return inner.args[1]
    // The scale is the square root of an inverse-gamma variance (a known multiple of it): the precision is its reciprocal.
    if (isRecip(env, inner)) return { kind: 'op', op: 'div', args: [{ kind: 'const', value: 1 }, inner] }
  }
  return undefined
}

/**
 * The known multiple `c` of a value in a node: `tau`, `c * tau`, `tau / c` (and nested), 1 for the node itself. Absent when the
 * expression is anything else.
 */
export function multipleOf(env: IdiomEnv, e: ExprSpec, name: string): number | undefined {
  const u = unwrap(env.nodes, e)
  if (u.kind === 'ref') return u.name === name ? 1 : undefined
  if (u.kind !== 'op' || (u.op !== 'mul' && u.op !== 'div')) return undefined
  const [a, b] = u.args as [ExprSpec, ExprSpec]
  const known = (x: ExprSpec): number | undefined => {
    if (!env.isConst(x)) return undefined
    const v = toFlat(env.valueOf(x).value)
    return v.length === 1 && Number.isFinite(v[0]) ? (v[0] as number) : undefined
  }
  const ka = known(a)
  const kb = known(b)
  if (u.op === 'mul') {
    const m = ka !== undefined ? multipleOf(env, b, name) : kb !== undefined ? multipleOf(env, a, name) : undefined
    const k = ka ?? kb
    return m === undefined || k === undefined ? undefined : m * k
  }
  const m = kb !== undefined ? multipleOf(env, a, name) : undefined
  return m === undefined || kb === undefined ? undefined : m / kb
}

/**
 * The known multiple of the precision `tau` that a Gaussian's scale stands for, when it is the idiom of one: `1 / sqrt(c tau)`
 * (a gamma `tau`), or `sqrt(c sigma2)` with `sigma2` an inverse gamma (so the precision is `tau / c`). Absent otherwise.
 */
export function precisionMultiple(env: IdiomEnv, scale: ExprSpec, tau: string, inverse: boolean): number | undefined {
  const e = unwrap(env.nodes, scale)
  if (e.kind !== 'op') return undefined
  if (inverse) {
    const c = e.op === 'sqrt' ? multipleOf(env, e.args[0] as ExprSpec, tau) : undefined
    return c === undefined || !(c > 0) ? undefined : 1 / c
  }
  const p = precisionExpr(env, scale)
  const c = p === undefined ? undefined : multipleOf(env, p, tau)
  return c === undefined || !(c > 0) ? undefined : c
}
