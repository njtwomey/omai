import { WhiteboxError } from '../base/index.ts'
import {
  add,
  bind,
  isTraced,
  type MaybeZero,
  type Primitive,
  type Trace,
  type Traced,
  type Value,
  zeros,
} from '../tensor/index.ts'

/** A recorded application of a primitive inside a reverse-mode trace. */
export interface TapeNode {
  readonly primitive: Primitive<unknown>
  readonly params: unknown
  readonly inputs: readonly ReverseValue[]
  readonly outputs: readonly Value[]
  results: readonly ReverseValue[]
}

/**
 * A value inside a reverse-mode trace: its primal (a value of a lower level), and the node that produced it (`null`
 * for constants and leaves). Values record their own history; there is no global tape.
 */
export interface ReverseValue extends Traced {
  readonly primal: Value
  readonly node: TapeNode | null
  readonly leaf: boolean
}

/**
 * A reverse-mode trace at one level.
 *
 * @concept reverse-mode-autodiff
 */
export interface ReverseTrace extends Trace {
  readonly kind: 'reverse'
  leafOf(primal: Value): ReverseValue
}

/**
 * Create a reverse-mode trace. Use {@link vjp} or `grad` rather than this directly.
 *
 * @param level - The trace's level.
 * @returns The trace.
 * @example
 * ```ts
 * import { reverseTrace } from 'whitebox-engine/autodiff'
 * reverseTrace(1).kind // => 'reverse'
 * ```
 */
export function reverseTrace(level: number): ReverseTrace {
  const trace: ReverseTrace = {
    kind: 'reverse',
    level,
    lift: (x) => make(x, null, false),
    leafOf: (primal) => make(primal, null, true),
    process: <P>(primitive: Primitive<P>, inputs: readonly Traced[], params: P): readonly Value[] => {
      const rin = inputs as readonly ReverseValue[]
      const primals = rin.map((x) => x.primal)
      const outputs = bind(primitive, primals, params)
      const tracked = rin.some((x) => x.node !== null || x.leaf)
      if (!tracked || primitive.zeroDerivative === true) return outputs.map((o) => make(o, null, false))
      const node: TapeNode = { primitive: primitive as Primitive<unknown>, params, inputs: rin, outputs, results: [] }
      node.results = outputs.map((o) => make(o, node, false))
      return node.results
    },
  }
  function make(primal: Value, node: TapeNode | null, leaf: boolean): ReverseValue {
    return { kind: 'traced', dtype: primal.dtype, shape: primal.shape, trace, primal, node, leaf }
  }
  return trace
}

function topological(outputs: readonly ReverseValue[]): TapeNode[] {
  const order: TapeNode[] = []
  const seen = new Set<TapeNode>()
  const stack: { node: TapeNode; expanded: boolean }[] = []
  for (const o of outputs) if (o.node !== null) stack.push({ node: o.node, expanded: false })
  while (stack.length > 0) {
    const top = stack.pop() as { node: TapeNode; expanded: boolean }
    if (top.expanded) {
      order.push(top.node)
      continue
    }
    if (seen.has(top.node)) continue
    seen.add(top.node)
    stack.push({ node: top.node, expanded: true })
    for (const input of top.node.inputs) {
      if (input.node !== null && !seen.has(input.node)) stack.push({ node: input.node, expanded: false })
    }
  }
  return order
}

/**
 * The reverse sweep: propagate output cotangents to the leaves.
 *
 * @param outputs - Values of one reverse trace.
 * @param cotangents - One cotangent per output (`null` for zero).
 * @param leaves - The leaves to return cotangents for.
 * @returns One cotangent per leaf (zeros when the outputs do not depend on it).
 * @throws WhiteboxError with code `not-differentiable` when a cotangent reaches a primitive without derivative rules.
 * @concept reverse-mode-autodiff
 * @example
 * ```ts
 * import { reverseTrace, backward } from 'whitebox-engine/autodiff'
 * import { mul, tensor, toArray } from 'whitebox-engine/tensor'
 * const t = reverseTrace(1)
 * const x = t.leafOf(tensor(3))
 * const y = mul(x, x)
 * toArray(backward([y as never], [tensor(1)], [x])[0]) // => 6
 * ```
 */
export function backward(
  outputs: readonly ReverseValue[],
  cotangents: readonly MaybeZero[],
  leaves: readonly ReverseValue[],
): Value[] {
  const cts = new Map<ReverseValue, Value>()
  const accumulate = (x: ReverseValue, g: Value): void => {
    const prev = cts.get(x)
    cts.set(x, prev === undefined ? g : add(prev, g))
  }
  outputs.forEach((o, i) => {
    const g = cotangents[i] ?? null
    if (g !== null) accumulate(o, g)
  })
  const order = topological(outputs)
  for (let i = order.length - 1; i >= 0; i--) {
    const node = order[i] as TapeNode
    const gs = node.results.map((r) => cts.get(r) ?? null)
    if (gs.every((g) => g === null)) continue
    const rule = node.primitive.vjp
    if (rule === undefined) {
      throw new WhiteboxError('not-differentiable', `primitive ${node.primitive.name} has no derivative rule`, {
        primitive: node.primitive.name,
      })
    }
    const ins = rule(
      gs,
      node.inputs.map((x) => x.primal),
      node.params,
      node.outputs,
    )
    node.inputs.forEach((x, j) => {
      const g = ins[j] ?? null
      if (g !== null && (x.node !== null || x.leaf)) accumulate(x, g)
    })
  }
  return leaves.map((leaf) => cts.get(leaf) ?? zeros(leaf.shape, { dtype: leaf.dtype }))
}

/**
 * Whether a value belongs to a given reverse trace.
 *
 * @param x - Any value.
 * @param trace - The trace.
 * @returns `true` when `x` is a value of `trace`.
 * @example
 * ```ts
 * import { reverseTrace, isReverseValueOf } from 'whitebox-engine/autodiff'
 * import { tensor } from 'whitebox-engine/tensor'
 * const t = reverseTrace(1)
 * isReverseValueOf(t.leafOf(tensor(1)), t) // => true
 * ```
 */
export function isReverseValueOf(x: unknown, trace: ReverseTrace): x is ReverseValue {
  return isTraced(x) && x.trace === trace
}
