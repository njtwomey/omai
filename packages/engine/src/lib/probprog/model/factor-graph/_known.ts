/**
 * What the compiler knows before it maps a node: the constants of a model. Data nodes with bound values are known; a derived node
 * is known when its expression is built from known things only (constants, bound data, other known derived nodes, custom
 * functions that were supplied), and its value is then evaluated here by the operator registry's own evaluation, in the order of
 * the spec (which is topological: a node reads only the nodes before it). `knownConstants` is that pass, pure: it makes no
 * graph structure and no refusal.
 *
 * One kind of knowledge is still found by the walk and not by this pass, and it is order-dependent by nature: a derived node whose
 * expression mentions a random value but whose symbolic evaluation comes out constant (`0 * z`, `where(1, 3, z)`: the cells are
 * all known, the expression is not). The walk declares those into the same `Known` when it reaches the node, so a node that
 * reads one asks "is this known?" correctly only because the nodes before it were mapped; `Known.semantic` lists them.
 */

import { asTensor, type NestedArray } from '../../../tensor/index.ts'
import type { BoundModel } from '../bind.ts'
import type { NamedValue, OperatorTable } from '../operators.ts'
import type { ExprSpec } from '../spec.ts'
import { customNotSupplied, randomWhereKnown, unknownOperator } from './_reasons.ts'
import { Refusal } from './_refusal.ts'
import { fromNamed, overDims, type Sizes, toNamed } from './_symbolic/_dims.ts'

/** The known values of a model, by node name, with the position (the node's index in the spec) each was declared at. */
export class Known {
  private readonly values = new Map<string, { readonly value: NamedValue; readonly at: number }>()
  /** The derived nodes that only the symbolic evaluation of the walk found known (not the pass): see the file's summary. */
  readonly semantic: string[] = []

  /** Declare a node's value, known from position `at` on. */
  declare(name: string, value: NamedValue, at: number): void {
    this.values.set(name, { value, at })
  }

  /** Whether a node is known; with `before`, only when it is declared at a position before that one. */
  has(name: string, before: number = Number.POSITIVE_INFINITY): boolean {
    const v = this.values.get(name)
    return v !== undefined && v.at < before
  }

  /** The value of a known node, or undefined. */
  get(name: string): NamedValue | undefined {
    return this.values.get(name)?.value
  }
}

/** What evaluating a known expression needs: the registries, and what a reference to a node resolves to. */
export interface EvalEnv {
  readonly modelOps: OperatorTable
  readonly customs: BoundModel['customs']
  /** The value a reference stands for; a `Refusal` when it has none. */
  readonly ref: (name: string) => NamedValue
  /** Receives the name of each custom function that was called. */
  readonly onCustom?: (name: string) => void
}

/** The value of an expression that has no random part, by the operator registry's own evaluation. */
export function evaluateKnown(e: ExprSpec, env: EvalEnv): NamedValue {
  switch (e.kind) {
    case 'const':
      return { value: asTensor(e.value as NestedArray, 'float64'), dims: e.dims ?? [] }
    case 'ref':
      return env.ref(e.name)
    case 'op': {
      const def = env.modelOps[e.op]
      if (def === undefined) throw unknownOperator(e.op)
      return def.evaluate(
        e.args.map((a) => evaluateKnown(a, env)),
        e.params ?? {},
      )
    }
    case 'custom': {
      const fn = env.customs[e.name]
      if (fn === undefined) throw customNotSupplied(e.name)
      const value = fn(e.args.map((a) => evaluateKnown(a, env)))
      env.onCustom?.(e.name)
      return { value, dims: e.dims }
    }
  }
}

/**
 * Whether an expression is known: built from constants, known nodes (declared before position `before`, when given) and
 * supplied custom functions of known values (the Gram matrix of a kernel whose length scale is fixed).
 */
export function isKnownExpr(
  e: ExprSpec,
  known: Known,
  customs: BoundModel['customs'],
  before: number = Number.POSITIVE_INFINITY,
): boolean {
  switch (e.kind) {
    case 'const':
      return true
    case 'ref':
      return known.has(e.name, before)
    case 'op':
      return e.args.every((a) => isKnownExpr(a, known, customs, before))
    case 'custom':
      return customs[e.name] !== undefined && e.args.every((a) => isKnownExpr(a, known, customs, before))
  }
}

/** The result of the constants pass: the known values and, for each known derived node, the custom functions its evaluation called. */
export interface KnownConstants {
  readonly known: Known
  readonly customsCalled: ReadonlyMap<string, readonly string[]>
}

/**
 * The constants of a bound model, in the order of its spec: bound data, then each derived node whose expression is known, as the
 * value over the node's own dimensions (what the walk would compute for it). A derived node whose evaluation is refused (an
 * unknown operator, a value that does not fit its dimensions) is not known here; the walk meets the same refusal at the node.
 *
 * @param bound - The bound model.
 * @param modelOps - The model operator registry that evaluates known operators.
 * @returns The known values and the custom functions each derived value called.
 */
export function knownConstants(bound: BoundModel, modelOps: OperatorTable): KnownConstants {
  const known = new Known()
  const customsCalled = new Map<string, readonly string[]>()
  const sizes: Sizes = bound.sizes
  for (const [at, node] of bound.spec.nodes.entries()) {
    if (node.kind === 'data') {
      const t = bound.data[node.name]
      if (t !== undefined) known.declare(node.name, { value: t, dims: node.dims }, at)
    } else if (node.kind === 'derived' && isKnownExpr(node.expr, known, bound.customs, at)) {
      const called: string[] = []
      const env: EvalEnv = {
        modelOps,
        customs: bound.customs,
        ref: (name) => {
          const v = known.get(name)
          if (v === undefined) throw randomWhereKnown(name)
          return v
        },
        onCustom: (name) => called.push(name),
      }
      try {
        const cells = overDims(sizes, fromNamed(evaluateKnown(node.expr, env)), node.dims)
        known.declare(node.name, toNamed(sizes, { dims: node.dims, cells }), at)
        customsCalled.set(node.name, called)
      } catch (e) {
        if (!(e instanceof Refusal)) throw e
      }
    }
  }
  return { known, customsCalled }
}
