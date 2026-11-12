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
  stack,
  take,
  tensor,
  toFlat,
  type Value,
  where,
} from '../../tensor/index.ts'
import { isFiniteDiscrete, mask as maskHandler } from '../handlers.ts'
import { deterministic, factor, type Model, markov, observe, param, plate, sample } from '../primitives.ts'
import type { BoundModel } from './bind.ts'
import { distributionFamilies, type FamilyDef } from './dists/families.ts'
import { alignTo, defaultOperators, type NamedValue, type OperatorTable } from './operators.ts'
import type { ConstraintSpec, DistSpec, ExprSpec, ModelNode, ScanNode } from './spec.ts'
import { exprRefs, nodeRefs, prevName } from './spec.ts'

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
  /** The batch dimension of each enumeration axis in the values of `env` (see {@link bindLatent}). */
  readonly enums: Map<string, number>
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
  // A dimension of the distribution's event (the classes of a categorical's logits) is laid out after the plate dimensions, as its last axis.
  const events = spec.eventDims ?? []
  for (const [name, e] of Object.entries(spec.params)) {
    const x = evaluate(e, ctx)
    out[name] = aligned(x, [...dims, ...events.filter((d) => x.dims.includes(d))], ctx, dims.length)
  }
  return out
}

/** The prefix of the name of an enumeration axis, which an enumerated latent carries as a dimension of its own. */
const ENUM = 'enum:'

/**
 * The value of a latent as the expressions see it. A discrete latent that the enumeration handler gave all its states
 * comes back with an axis of its states at a batch dimension left of every plate (`[K, 1, ..., 1]`), which the named
 * layout (dimensions first, event axes after) would read as event axes and misalign against a plated operand. So the axis
 * becomes a dimension named `enum:<site>` that broadcasts like any other, and {@link aligned} puts it back at its batch
 * dimension where a distribution or a factor takes the value.
 */
function bindLatent(node: ModelNode, value: Value, support: Support, ctx: Context): NamedValue {
  const rank = value.shape.length
  const own = node.dims.length
  const rest = value.shape.slice(1)
  const plain = rank > own && node.eventRank === 0 && isFiniteDiscrete(support)
  if (!plain || rest.length < own || rest.slice(0, rest.length - own).some((n) => n !== 1))
    return { value, dims: node.dims }
  const name = `${ENUM}${node.name}`
  ctx.enums.set(name, -rank)
  return {
    value: reshape(value, [value.shape[0] as number, ...rest.slice(rest.length - own)]),
    dims: [name, ...node.dims],
  }
}

/**
 * A value laid out for a node's dimensions (a scalar needs none). The first `batch` of `dims` are the node's plates, the
 * rest event dimensions; the enumeration axes of the value go to the batch dimensions they were given, left of the plates.
 */
function aligned(x: NamedValue, dims: readonly string[], ctx: Context, batch: number = dims.length): Value {
  const lead = x.dims
    .filter((d) => d.startsWith(ENUM))
    .map((name) => ({ name, dim: ctx.enums.get(name) as number }))
    .sort((a, b) => a.dim - b.dim)
  if (lead.length === 0) return x.dims.length === 0 ? x.value : alignTo(x, dims)
  const v = alignTo(x, [...lead.map((e) => e.name), ...dims])
  const rank = Math.max(batch, -(lead[0] as { dim: number }).dim)
  const shape: number[] = []
  for (let i = 0; i < rank; i++) {
    const dim = i - rank
    const at = lead.findIndex((e) => e.dim === dim)
    shape.push(dim >= -batch ? (v.shape[lead.length + batch + dim] as number) : at < 0 ? 1 : (v.shape[at] as number))
  }
  return reshape(v, [...shape, ...v.shape.slice(lead.length + batch)])
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

/** What the markov loop of a scan carries: the previous state, and every state so far (to stack them at the end). */
interface ScanCarry {
  readonly prev: Value | undefined
  readonly states: readonly Value[]
}

/**
 * Run a scan as one `markov` loop (history 1): at step `t` the state is the site `<name>[t]`, drawn from the initial or step
 * distribution, then each emission is observed as `<emission>[t]`. A node over the scanned dimension is read at the step
 * (the axis is taken away); the stack of the states is bound to the scan's name afterwards. An emission that reads an
 * unbound node is left out, as an observed node would be; where there is no stack of states to bind (a loop run by a handler
 * that takes over the steps, such as the particle filter, or states enumerated onto different axes) what reads the
 * stack is skipped.
 */
function runScan(node: ScanNode, ctx: Context, skipped: Set<string>): void {
  const over = node.over
  if (ctx.batch?.dim === over)
    throw new OmaiError('value', `scan '${node.name}': the scanned dimension '${over}' cannot be the batch dimension`)
  for (const e of [...Object.values(node.init.params), ...Object.values(node.step.params)])
    for (const r of exprRefs(e))
      if (skipped.has(r))
        throw new OmaiError('value', `scan '${node.name}' depends on '${r}', which is not bound`, { name: node.name })
  const emissions = node.emissions.filter(
    (m) => ![...Object.values(m.dist.params).flatMap(exprRefs), ...exprRefs(m.value)].some((r) => skipped.has(r)),
  )
  const length = sizeOf(ctx, over)
  const done = markov(
    `scan:${node.name}`,
    length,
    (t: number, carry: ScanCarry): ScanCarry => {
      const env = new Map<string, NamedValue>()
      for (const [name, v] of ctx.env) {
        const axis = v.dims.indexOf(over)
        env.set(name, axis < 0 ? v : { value: take(v.value, t, axis), dims: v.dims.filter((d) => d !== over) })
      }
      const at: Context = { ...ctx, env }
      if (carry.prev !== undefined) env.set(prevName(node.name), { value: carry.prev, dims: [] })
      const state = sample(`${node.name}[${t}]`, buildDist(t === 0 ? node.init : node.step, at, []))
      env.set(node.name, { value: state, dims: [] })
      for (const m of emissions) observe(`${m.name}[${t}]`, buildDist(m.dist, at, []), evaluate(m.value, at).value)
      return { prev: state, states: [...carry.states, state] }
    },
    { prev: undefined, states: [] } as ScanCarry,
  )
  // Enumeration gives each step's state its own batch axis, so the states may have no common shape to stack.
  const first = done.states[0]
  const stackable =
    length > 0 &&
    first !== undefined &&
    done.states.length === length &&
    done.states.every((v) => v.shape.length === first.shape.length && v.shape.every((n, i) => n === first.shape[i]))
  if (stackable) ctx.env.set(node.name, { value: stack(done.states, 0), dims: [over] })
  else skipped.add(node.name)
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
 * over it uses that subset, rescaled by the plate. A `scan` node is one `markov` loop (history 1): the state at step `t` is the
 * site `<name>[t]`, each emission the site `<emission>[t]`, and a node over the scanned dimension is read at the step. Observed nodes and generated or derived nodes that depend on an
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
    const base: Context = { bound, operators, families, env, enums: new Map<string, number>(), batch: undefined }
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
          env.set(node.name, bindLatent(node, value, dist.support, ctx))
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
          if (v.dims.some((d) => d.startsWith(ENUM))) {
            // An enumerated value is recorded as the sites see it, its states along their own batch dimension.
            const own = v.dims.filter((d) => !d.startsWith(ENUM))
            if (node.record) deterministic(node.name, aligned(v, own, ctx))
            env.set(node.name, v)
            break
          }
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
          inPlates(ctx, node.dims, () => factor(node.name, aligned(v, node.dims, ctx)))
          break
        }
        case 'scan':
          runScan(node, ctx, skipped)
          break
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
