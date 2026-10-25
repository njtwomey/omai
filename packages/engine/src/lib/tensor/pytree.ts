/**
 * Pytrees, values nested in arrays and plain records, as in JAX: flattening a tree into leaves and a stored structure
 * and rebuilding it, mapping functions over corresponding leaves, leaf paths such as `layers[0].w`, zero trees,
 * parameter counts, and `ravel` to pack a tree of tensors into one vector and back.
 */

import { WhiteboxError } from '../base/index.ts'
import { cpu } from './cpu.ts'
import { zerosLike } from './create.ts'
import { sizeOf } from './shape.ts'
import { isValue, type Tensor } from './tensor.ts'
import { isTraced, type Value } from './traced.ts'

/**
 * A pytree: values nested in arrays and plain records. Parameters, gradients and optimiser states are pytrees, so one
 * set of functions maps over all of them (as in JAX).
 *
 * @concept pytree
 */
export type Tree<L = Value> = L | readonly Tree<L>[] | { readonly [key: string]: Tree<L> }

/** Replace every leaf type `From` in a tree type with `To`, keeping the structure. */
export type MapLeaves<T, From, To> = T extends From
  ? To
  : T extends readonly unknown[]
    ? { readonly [K in keyof T]: MapLeaves<T[K], From, To> }
    : T extends object
      ? { readonly [K in keyof T]: MapLeaves<T[K], From, To> }
      : T

/** The structure of a pytree without its leaves: data, so it can be stored and compared. */
export type TreeDef =
  | { readonly kind: 'leaf' }
  | { readonly kind: 'array'; readonly children: readonly TreeDef[] }
  | {
      readonly kind: 'record'
      readonly keys: readonly string[]
      readonly children: readonly TreeDef[]
    }

function isBranchArray(node: unknown): node is readonly unknown[] {
  return Array.isArray(node)
}

function isRecord(node: unknown): node is Readonly<Record<string, unknown>> {
  return typeof node === 'object' && node !== null && !isBranchArray(node) && !isValue(node)
}

/**
 * Split a tree into its leaves (records by sorted key, arrays by index) and its structure.
 *
 * @param tree - The tree.
 * @returns The leaves in order and the structure.
 * @example
 * ```ts
 * import { treeFlatten, zeros } from 'whitebox-engine/tensor'
 * treeFlatten({ b: zeros([1]), a: [zeros([2])] }).leaves.map((l) => l.shape) // => [[2], [1]]
 * ```
 */
export function treeFlatten(tree: unknown): { readonly leaves: Value[]; readonly def: TreeDef } {
  const leaves: Value[] = []
  const visit = (node: unknown): TreeDef => {
    if (isValue(node)) {
      leaves.push(node)
      return { kind: 'leaf' }
    }
    if (isBranchArray(node)) return { kind: 'array', children: node.map(visit) }
    if (isRecord(node)) {
      const keys = Object.keys(node).sort()
      return { kind: 'record', keys, children: keys.map((k) => visit(node[k])) }
    }
    throw new WhiteboxError('value', `pytree leaves must be tensors or traced values, got ${typeof node}`)
  }
  const def = visit(tree)
  return { leaves, def }
}

/**
 * Rebuild a tree from its structure and leaves (the inverse of {@link treeFlatten}).
 *
 * @param def - The structure.
 * @param leaves - The leaves, in flattening order.
 * @returns The tree.
 * @throws WhiteboxError with code `value` when the number of leaves does not match.
 * @example
 * ```ts
 * import { treeFlatten, treeUnflatten, zeros } from 'whitebox-engine/tensor'
 * const { leaves, def } = treeFlatten({ w: zeros([2]) })
 * Object.keys(treeUnflatten(def, leaves) as object) // => ['w']
 * ```
 */
export function treeUnflatten(def: TreeDef, leaves: readonly unknown[]): unknown {
  let i = 0
  const build = (d: TreeDef): unknown => {
    if (d.kind === 'leaf') {
      if (i >= leaves.length) throw new WhiteboxError('value', 'treeUnflatten: too few leaves')
      return leaves[i++]
    }
    if (d.kind === 'array') return d.children.map(build)
    const out: Record<string, unknown> = {}
    d.keys.forEach((k, j) => {
      out[k] = build(d.children[j] as TreeDef)
    })
    return out
  }
  const tree = build(def)
  if (i !== leaves.length) throw new WhiteboxError('value', `treeUnflatten: ${leaves.length - i} leaves left over`)
  return tree
}

/**
 * The leaves of a tree in flattening order.
 *
 * @param tree - The tree.
 * @returns The leaves.
 * @example
 * ```ts
 * import { treeLeaves, zeros } from 'whitebox-engine/tensor'
 * treeLeaves({ b: zeros([1]), a: [zeros([2]), zeros([3])] }).map((t) => t.shape) // => [[2], [3], [1]]
 * ```
 */
export function treeLeaves(tree: unknown): Value[] {
  return treeFlatten(tree).leaves
}

/**
 * Apply a function to corresponding leaves of trees with the same structure.
 *
 * @param fn - Called with the leaves at each position.
 * @param trees - The trees; the first defines the structure.
 * @returns A tree of the results with the same structure.
 * @throws WhiteboxError with code `value` when structures differ.
 * @example
 * ```ts
 * import { treeMap, add, tensor, toArray } from 'whitebox-engine/tensor'
 * const t = treeMap((a, b) => add(a, b), { w: tensor([1]) }, { w: tensor([2]) })
 * toArray(t.w) // => [3]
 * ```
 */
export function treeMap<T>(fn: (...leaves: Value[]) => Value, ...trees: readonly [T, ...T[]]): T {
  const flats = trees.map((t) => treeFlatten(t))
  const first = flats[0] as { leaves: Value[]; def: TreeDef }
  const sig = JSON.stringify(first.def)
  for (const f of flats.slice(1)) {
    if (JSON.stringify(f.def) !== sig) throw new WhiteboxError('value', 'treeMap: trees have different structures')
  }
  const out = first.leaves.map((_, i) => fn(...flats.map((f) => f.leaves[i] as Value)))
  return treeUnflatten(first.def, out) as T
}

/**
 * The leaves of a tree keyed by their path (`layers[0].w`), in flattening order.
 *
 * @param tree - The tree.
 * @returns A record from path to leaf.
 * @example
 * ```ts
 * import { treePaths, zeros } from 'whitebox-engine/tensor'
 * Object.keys(treePaths({ layers: [{ w: zeros([1]) }], b: zeros([1]) })) // => ['b', 'layers[0].w']
 * ```
 */
export function treePaths(tree: unknown): Record<string, Value> {
  const out: Record<string, Value> = {}
  const visit = (node: unknown, path: string): void => {
    if (isValue(node)) out[path === '' ? 'x' : path] = node
    else if (isBranchArray(node)) {
      node.forEach((child, i) => {
        visit(child, `${path}[${i}]`)
      })
    } else if (isRecord(node)) {
      for (const key of Object.keys(node).sort()) visit(node[key], path === '' ? key : `${path}.${key}`)
    }
  }
  visit(tree, '')
  return out
}

/**
 * A tree of zeros with the structure, shapes and dtypes of another tree.
 *
 * @param tree - The template.
 * @returns The zero tree.
 * @example
 * ```ts
 * import { treeZeros, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(treeZeros({ w: tensor([1, 2]) }).w) // => [0, 0]
 * ```
 */
export function treeZeros<T>(tree: T): T {
  return treeMap((leaf) => zerosLike(leaf), tree)
}

/**
 * The total number of elements in a tree's leaves (a model's parameter count).
 *
 * @param tree - The tree.
 * @returns The count.
 * @example
 * ```ts
 * import { treeCount, zeros } from 'whitebox-engine/tensor'
 * treeCount({ w: zeros([3, 2]), b: zeros([2]) }) // => 8
 * ```
 */
export function treeCount(tree: unknown): number {
  return treeLeaves(tree).reduce((n, leaf) => n + sizeOf(leaf.shape), 0)
}

/**
 * Concatenate a tree of concrete float tensors into one float64 vector, with the function that undoes it.
 *
 * @param tree - A tree of concrete tensors.
 * @returns The flat vector and `unravel`, which rebuilds a tree of the same structure from such a vector.
 * @throws WhiteboxError with code `not-differentiable` when a leaf is traced.
 * @example
 * ```ts
 * import { ravel, tensor } from 'whitebox-engine/tensor'
 * const { flat, unravel } = ravel({ a: tensor([1, 2]), b: tensor(3) })
 * Array.from(flat) // => [1, 2, 3]
 * ```
 */
export function ravel<T>(tree: T): { readonly flat: Float64Array; readonly unravel: (flat: ArrayLike<number>) => T } {
  const { leaves, def } = treeFlatten(tree)
  const concrete = leaves.map((l) => {
    if (isTraced(l)) throw new WhiteboxError('not-differentiable', 'ravel needs concrete leaves')
    return l as Tensor
  })
  const sizes = concrete.map((l) => l.data.length)
  const flat = new Float64Array(sizes.reduce((a, b) => a + b, 0))
  let offset = 0
  for (const l of concrete) {
    flat.set(l.data, offset)
    offset += l.data.length
  }
  const unravel = (v: ArrayLike<number>): T => {
    let o = 0
    const rebuilt = concrete.map((l) => {
      const data = new Float64Array(l.data.length)
      for (let i = 0; i < data.length; i++) data[i] = v[o + i] ?? 0
      o += data.length
      return cpu.wrap('float64', l.shape, data)
    })
    return treeUnflatten(def, rebuilt) as T
  }
  return { flat, unravel }
}
