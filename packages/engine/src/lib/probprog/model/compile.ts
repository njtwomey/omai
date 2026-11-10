/**
 * Compiling a bound model to the existing probabilistic-program layer: `toProgram` returns a `Model` that runs the spec's
 * nodes in order with `sample`, `observe`, `deterministic`, `factor`, `param` and `plate`, so every present engine, handler,
 * guide, density and criticism function works on a spec with no change. Named dimensions become plates (with the batch
 * dimension numbers assigned here, so no one counts them), constraints become the support the existing `bijectorForSupport`
 * maps (Jacobians included), expressions are evaluated by the registry on the library's tensor ops, missing observed
 * entries are masked or imputed as latent sites, and a batch dimension becomes a subsampled plate shared by every node over
 * it. There is no second evaluator: the log joint of a spec is `logJoint(toProgram(bound), ...)`.
 *
 * @index function toProgram: compile a bound model spec to the existing sample/observe program.
 */

import { OmaiError } from '../../base/index.ts'
import type { Distribution, Support } from '../../dists/index.ts'
import {
  asTensor,
  broadcastTo,
  isnan,
  logicalNot,
  type NestedArray,
  reshape,
  take,
  tensor,
  toFlat,
  type Value,
  where,
} from '../../tensor/index.ts'
import { mask as maskHandler } from '../handlers.ts'
import { deterministic, factor, type Model, observe, param, plate, sample } from '../primitives.ts'
import type { BoundModel } from './bind.ts'
import { distributionFamilies, type FamilyDef } from './dists/families.ts'
import { alignTo, defaultOperators, type NamedValue, type OperatorTable } from './operators.ts'
import type { ConstraintSpec, DistSpec, ExprSpec, ModelNode } from './spec.ts'
import { nodeRefs } from './spec.ts'

/** Options of {@link toProgram}. */
export interface CompileOptions {
  /** Include the `generated` nodes as unobserved sample sites (for predictive draws); by default they are left out. */
  readonly generated?: boolean
  /** The operator registry (default the shipped one). */
  readonly operators?: OperatorTable
}

/**
 * A constraint as the support the existing bijector machinery maps unconstrained reals onto.
 *
 * @param c - The constraint.
 * @returns The support.
 * @example
 * ```ts
 * import { constraintSupport } from 'omai-engine/probprog/model'
 * constraintSupport({ kind: 'positive' }).kind // => 'positive'
 * ```
 */
export function constraintSupport(c: ConstraintSpec): Support {
  const v = (x: unknown): Value => asTensor(x as NestedArray)
  switch (c.kind) {
    case 'greaterThan':
      return { kind: 'greaterThan', low: v(c.low) }
    case 'lessThan':
      return { kind: 'lessThan', high: v(c.high) }
    case 'interval':
      return { kind: 'interval', low: v(c.low), high: v(c.high) }
    default:
      return { kind: c.kind }
  }
}

interface Context {
  readonly bound: BoundModel
  readonly operators: OperatorTable
  readonly families: Readonly<Record<string, FamilyDef>>
  readonly env: Map<string, NamedValue>
  readonly batch: { readonly dim: string; readonly indices: Value } | undefined
}

const sizeOf = (ctx: Context, dim: string): number => ctx.bound.sizes[dim] as number

function evaluate(e: ExprSpec, ctx: Context): NamedValue {
  switch (e.kind) {
    case 'const':
      return { value: asTensor(e.value as NestedArray, 'float64'), dims: e.dims ?? [] }
    case 'ref': {
      const v = ctx.env.get(e.name)
      if (v === undefined)
        throw new OmaiError('value', `'${e.name}' has no value (an unbound data node?)`, { name: e.name })
      return v
    }
    case 'op': {
      const def = ctx.operators[e.op]
      if (def === undefined) throw new OmaiError('value', `unknown operator '${e.op}'`)
      return def.evaluate(
        e.args.map((a) => evaluate(a, ctx)),
        e.params ?? {},
      )
    }
    case 'custom': {
      const fn = ctx.bound.customs[e.name]
      if (fn === undefined)
        throw new OmaiError('value', `the custom function '${e.name}' was not supplied`, { name: e.name })
      const value = fn(e.args.map((a) => evaluate(a, ctx)))
      if (value.shape.length !== e.dims.length + e.eventRank)
        throw new OmaiError(
          'shape',
          `custom '${e.name}' returned shape [${value.shape.join(', ')}] for dims [${e.dims.join(', ')}] and ${e.eventRank} event axes`,
        )
      e.dims.forEach((d, i) => {
        const want = ctx.batch?.dim === d ? (ctx.batch.indices.shape[0] as number) : sizeOf(ctx, d)
        if (value.shape[i] !== want)
          throw new OmaiError(
            'shape',
            `custom '${e.name}': dimension '${d}' has size ${value.shape[i]}, expected ${want}`,
          )
      })
      return { value, dims: e.dims }
    }
  }
}

/** Run `fn` inside one plate per dimension, outermost first, with the batch dimensions assigned from the right. */
function inPlates<T>(ctx: Context, dims: readonly string[], fn: () => T): T {
  const go = (i: number): T => {
    if (i === dims.length) return fn()
    const d = dims[i] as string
    const dim = -(dims.length - i)
    const sub = ctx.batch?.dim === d ? { subsample: ctx.batch.indices } : {}
    return plate(d, sizeOf(ctx, d), () => go(i + 1), { dim, ...sub })
  }
  return go(0)
}

function distArgs(spec: DistSpec, ctx: Context, dims: readonly string[]): Record<string, Value> {
  const out: Record<string, Value> = {}
  for (const [name, e] of Object.entries(spec.params)) out[name] = aligned(evaluate(e, ctx), dims)
  return out
}

/** A parameter laid out for a node's dimensions (a scalar needs none). */
function aligned(x: NamedValue, dims: readonly string[]): Value {
  return x.dims.length === 0 ? x.value : alignTo(x, dims)
}

function buildDist(spec: DistSpec, ctx: Context, dims: readonly string[]): Distribution {
  const fam = ctx.families[spec.family]
  if (fam === undefined) throw new OmaiError('value', `unknown distribution family '${spec.family}'`)
  const args = distArgs(spec, ctx, dims)
  for (const p of fam.params) {
    const d = fam.defaults[p]
    if (args[p] === undefined && d !== undefined) args[p] = tensor(d)
  }
  return fam.make(args)
}

function withConstraint(d: Distribution, c: ConstraintSpec | undefined): Distribution {
  return c === undefined ? d : { ...d, support: constraintSupport(c) }
}

/** A value in the support of `d` to stand where an observation is missing (it is masked out of the density). */
function filler(d: Distribution, discrete: boolean): Value {
  try {
    return discrete ? d.mode() : d.mean()
  } catch {
    return tensor(0)
  }
}

function dependsOnUnbound(node: ModelNode, skipped: ReadonlySet<string>): string | undefined {
  for (const r of nodeRefs(node)) if (skipped.has(r)) return r
  return undefined
}

/**
 * Compile a bound model to a probabilistic program of the existing layer, so every engine, handler, guide and density
 * function runs on it unchanged. The nodes run in the spec's order; named dimensions become plates; a latent's
 * constraint becomes its support (so the unconstrained parametrisation and Jacobian are those of `bijectorForSupport`);
 * a derived node is a `deterministic` site (unless not recorded), a factor node a `factor`, a param node a `param`.
 *
 * Missing entries (NaN) of an observed node's data follow its `missing` policy: `ignore` masks them out of the
 * likelihood, `impute` also adds a latent site `<name>_missing` over them (scalar-event observations only, and not
 * together with a batch). With a `batch` option the first call draws one subset of the batch dimension and every node
 * over it uses that subset, rescaled by the plate. Observed nodes and generated or derived nodes that depend on an
 * unbound data node are left out, which is how a model without its response is run forward for prediction.
 *
 * @param bound - The bound model.
 * @param options - Whether to include generated quantities, and the operator registry.
 * @returns The program (a function of nothing; the data is bound in).
 * @throws OmaiError with code `value` when a latent depends on an unbound data node, and at run time for an operator or
 *   custom function the registry or the model lacks.
 * @concept model-spec
 * @index function Compile a bound ModelSpec to a sample/observe program every engine runs.
 * @example
 * ```ts
 * import { bind, defineModel, toProgram } from 'omai-engine/probprog/model'
 * import { normal } from 'omai-engine/probprog/model/dists'
 * import { logJoint } from 'omai-engine/probprog'
 * import { item } from 'omai-engine/tensor'
 * const m = defineModel('m', (b) => {
 *   const y = b.data('y', { dims: ['row'] })
 *   b.observe('y', normal(b.latent('mu', normal(0, 1)), 1), y)
 * })
 * // log N(0; 0, 1) + log N(1; 0, 1) = -log(2 pi) - 1/2
 * Math.abs(item(logJoint(toProgram(m.bind({ y: [1] })), undefined, { mu: 0 }).logJoint) + Math.log(2 * Math.PI) + 0.5) < 1e-14 // => true
 * ```
 */
export function toProgram(bound: BoundModel, options: CompileOptions = {}): Model {
  const operators = options.operators ?? defaultOperators()
  const families = distributionFamilies()
  return () => {
    const env = new Map<string, NamedValue>()
    const batch = bound.batch
    const base: Context = { bound, operators, families, env, batch: undefined }
    const indices =
      batch === undefined
        ? undefined
        : plate(batch.dim, sizeOf(base, batch.dim), (i) => i as Value, { subsampleSize: batch.size })
    const ctx: Context =
      batch === undefined || indices === undefined ? base : { ...base, batch: { dim: batch.dim, indices } }
    const skipped = new Set<string>(bound.unbound)
    for (const node of bound.spec.nodes) {
      switch (node.kind) {
        case 'data': {
          const t = bound.data[node.name]
          if (t === undefined) break
          const axis = ctx.batch === undefined ? -1 : node.dims.indexOf(ctx.batch.dim)
          env.set(node.name, {
            value: axis < 0 || ctx.batch === undefined ? t : take(t, ctx.batch.indices, axis),
            dims: node.dims,
          })
          break
        }
        case 'latent': {
          const dep = dependsOnUnbound(node, skipped)
          if (dep !== undefined)
            throw new OmaiError('value', `latent '${node.name}' depends on '${dep}', which is not bound`, {
              name: node.name,
            })
          const dist = withConstraint(buildDist(node.dist, ctx, node.dims), node.constraint)
          const value = inPlates(ctx, node.dims, () => sample(node.name, dist))
          env.set(node.name, { value, dims: node.dims })
          break
        }
        case 'observed': {
          if (dependsOnUnbound(node, skipped) !== undefined) break
          observed(node, ctx)
          break
        }
        case 'derived': {
          if (dependsOnUnbound(node, skipped) !== undefined) {
            skipped.add(node.name)
            break
          }
          const v = evaluate(node.expr, ctx)
          const value = node.record ? deterministic(node.name, v.value) : v.value
          env.set(node.name, { value, dims: v.dims })
          break
        }
        case 'generated': {
          if (options.generated !== true || dependsOnUnbound(node, skipped) !== undefined) {
            skipped.add(node.name)
            break
          }
          const dist = buildDist(node.dist, ctx, node.dims)
          env.set(node.name, { value: inPlates(ctx, node.dims, () => sample(node.name, dist)), dims: node.dims })
          break
        }
        case 'factor': {
          if (dependsOnUnbound(node, skipped) !== undefined) break
          const v = evaluate(node.expr, ctx)
          inPlates(ctx, node.dims, () => factor(node.name, aligned(v, node.dims)))
          break
        }
        case 'param': {
          const support = node.constraint === undefined ? undefined : constraintSupport(node.constraint)
          const value = param(node.name, asTensor(node.init as NestedArray, 'float64'), {
            eventDim: node.eventRank,
            ...(support === undefined ? {} : { support }),
          })
          env.set(node.name, { value, dims: node.dims })
          break
        }
      }
    }
    return undefined
  }
}

function observed(node: ModelNode & { readonly kind: 'observed' }, ctx: Context): void {
  const dist = buildDist(node.dist, ctx, node.dims)
  const raw = evaluate(node.value, ctx)
  const obs = alignTo(raw, node.dims)
  const info = ctx.bound.missing[node.name]
  const hasMissing = info !== undefined && info.count > 0 && node.value.kind === 'ref'
  if (!hasMissing) {
    inPlates(ctx, node.dims, () => observe(node.name, dist, obs))
    return
  }
  if (node.eventRank !== 0)
    throw new OmaiError('value', `observed '${node.name}': missing values need a scalar-event distribution`)
  const missing = isnan(obs)
  const fam = ctx.families[node.dist.family] as FamilyDef
  const filled = where(missing, filler(dist, fam.discrete), obs)
  const present = logicalNot(missing)
  maskHandler(() => inPlates(ctx, node.dims, () => observe(node.name, dist, filled)), present)(undefined)
  if (info.policy !== 'impute') return
  if (ctx.batch !== undefined)
    throw new OmaiError('value', `observed '${node.name}': impute cannot be combined with a batch`)
  const flat = toFlat(missing)
  const at: number[] = []
  flat.forEach((m, i) => {
    if (m !== 0) at.push(i)
  })
  const shape = obs.shape
  const subset = (v: Value): Value => {
    const eventShape = v.shape.slice(shape.length > v.shape.length ? 0 : shape.length)
    const full = broadcastTo(v, [...shape, ...eventShape])
    return take(reshape(full, [flat.length, ...eventShape]), at, 0)
  }
  const args: Record<string, Value> = {}
  for (const [name, v] of Object.entries(distArgs(node.dist, ctx, node.dims))) args[name] = subset(v)
  for (const p of fam.params) {
    const d = fam.defaults[p]
    if (args[p] === undefined && d !== undefined) args[p] = tensor(d)
  }
  const site = `${node.name}_missing`
  const imputed = fam.make(args)
  plate(`${site}_dim`, at.length, () => sample(site, imputed))
}
