import { type Structure, type StructureEdge, type StructureNode, structure } from '../scene/index.ts'
import { formatShape, isTensor, sizeOf, treeFlatten, treeUnflatten, type Value } from '../tensor/index.ts'
import { withLevel } from './_level.ts'
import { isReverseValueOf, type ReverseValue, reverseTrace, type TapeNode } from './reverse.ts'

/** Options for {@link computationGraph}. */
export interface GraphOptions {
  /** Names for the input leaves, in flattening order (default: their pytree paths). */
  readonly inputNames?: readonly string[]
  /** Include scalar values as badges (default true). */
  readonly values?: boolean
}

function scalarOf(x: Value): number | undefined {
  return isTensor(x) && sizeOf(x.shape) === 1 ? (x.data[0] ?? undefined) : undefined
}

/**
 * Record the computation graph of a function at an input: every primitive applied to values derived from the input,
 * with shapes, operations and (for scalars) values, as a {@link Structure} that whitebox-render draws.
 *
 * @param f - The function.
 * @param input - Its input (a value or a pytree).
 * @param options - Input names and whether to show values.
 * @returns The structure (type `computation-graph`).
 * @concept computation-graph
 * @example
 * ```ts
 * import { computationGraph } from 'whitebox-engine/autodiff'
 * import { add, mul, tensor, type Value } from 'whitebox-engine/tensor'
 * const g = computationGraph((x: Value) => add(mul(x, x), 1), tensor(2))
 * g.nodes.map((n) => n.label) // => ['x', 'mul', '1', 'add']
 * ```
 */
export function computationGraph<P>(f: (input: P) => unknown, input: P, options: GraphOptions = {}): Structure {
  const { leaves, def } = treeFlatten(input)
  return withLevel((level) => {
    const trace = reverseTrace(level)
    const traced = leaves.map((l) => trace.leafOf(l))
    const result = f(treeUnflatten(def, traced) as P)
    const outputs = treeFlatten(result).leaves.filter((o): o is ReverseValue => isReverseValueOf(o, trace))
    const names = options.inputNames ?? (leaves.length === 1 ? ['x'] : leaves.map((_, i) => `x${i}`))
    const showValues = options.values ?? true

    const nodes: StructureNode[] = []
    const edges: StructureEdge[] = []
    const ids = new Map<ReverseValue | TapeNode, string>()
    const nextId = (): string => `n${nodes.length}`
    const badge = (v: Value): Record<string, string | number> => {
      const s = showValues ? scalarOf(v) : undefined
      return s === undefined ? { shape: formatShape(v.shape) } : { value: Number(s.toPrecision(6)) }
    }

    const visitValue = (x: ReverseValue): string => {
      const known = ids.get(x.node ?? x)
      if (known !== undefined) return known
      if (x.node !== null) return visitNode(x.node)
      const id = nextId()
      const leafIndex = traced.indexOf(x)
      const primal = x.primal
      const label = leafIndex >= 0 ? (names[leafIndex] ?? `x${leafIndex}`) : String(scalarOf(primal) ?? 'const')
      nodes.push({ id, label, role: leafIndex >= 0 ? 'input' : 'constant', badges: badge(primal) })
      ids.set(x, id)
      return id
    }

    const visitNode = (node: TapeNode): string => {
      const known = ids.get(node)
      if (known !== undefined) return known
      const inputIds = node.inputs.map(visitValue)
      const id = nextId()
      const detail = node.primitive.describe?.(node.params)
      nodes.push({
        id,
        label: node.primitive.name,
        role: 'operation',
        ...(detail === undefined ? {} : { detail }),
        badges: badge(node.outputs[0] as Value),
      })
      ids.set(node, id)
      for (const from of inputIds) edges.push({ from, to: id, directed: true, kind: 'data' })
      return id
    }

    for (const o of outputs) visitValue(o)
    return structure('computation-graph', nodes, edges, { direction: 'right' })
  })
}
