/**
 * Describing a network's architecture: a per-layer table of shapes, parameter counts and estimated FLOPs, memory
 * estimates as torchinfo makes them, a torchinfo-style text `summary`, and `describe`, which turns a module (or, with
 * `describeFunction`, any functional model through its computation graph) into a `Structure` for drawing as a diagram.
 */

import { computationGraph } from '../autodiff/index.ts'
import {
  type Structure,
  type StructureEdge,
  type StructureGroup,
  type StructureNode,
  structure,
} from '../scene/index.ts'
import { item, sizeOf, treeCount, treePaths, type Value, zeros } from '../tensor/index.ts'
import { Parallel, Residual, Sequential } from './module/containers.ts'
import type { Module as BaseModule, Params } from './module/module.ts'

type Module = BaseModule<never, unknown>

const shapeText = (shape: readonly number[]): string => `[${shape.join(', ')}]`

/** Bytes per element of the float64 tensors parameters and activations are stored in. */
const BYTES = 8

// Primitives that only move or reinterpret data.
const MOVEMENT = new Set([
  'broadcastTo',
  'cast',
  'concat',
  'pad',
  'reshape',
  'slice',
  'sliceScatter',
  'stopGradient',
  'take',
  'takeAlongAxis',
  'transpose',
])

const parseShape = (s: unknown): number[] =>
  typeof s === 'string' && s.startsWith('[')
    ? s
        .slice(1, -1)
        .split(',')
        .map((v) => v.trim())
        .filter((v) => v !== '')
        .map(Number)
    : []

/**
 * Floating-point operations of a function at an input, counted from its computation graph: 2 m n k for each matrix
 * product, one per input element for reductions and scatters, one per output element for elementwise operations, none
 * for reshapes, gathers and other data movement (the convention of fvcore and torchinfo, counting multiply-adds as
 * two).
 *
 * @param f - The function.
 * @param input - Its input (a value or a pytree).
 * @returns The estimated count.
 * @concept flops
 * @example
 * ```ts
 * import { functionFlops } from 'whitebox-engine/nn'
 * import { matmul, zeros, type Value } from 'whitebox-engine/tensor'
 * functionFlops((x: Value) => matmul(x, zeros([3, 4])), zeros([2, 3])) // => 48
 * ```
 */
export function functionFlops<P>(f: (input: P) => unknown, input: P): number {
  const g = computationGraph(f, input, { values: false })
  const shapes = new Map(g.nodes.map((n) => [n.id, parseShape(n.badges?.['shape'])]))
  const inputsOf = new Map<string, string[]>()
  for (const e of g.edges) inputsOf.set(e.to, [...(inputsOf.get(e.to) ?? []), e.from])
  let total = 0
  for (const node of g.nodes) {
    if (node.role !== 'operation' || MOVEMENT.has(node.label)) continue
    const out = sizeOf(shapes.get(node.id) ?? [])
    const ins = (inputsOf.get(node.id) ?? []).map((id) => shapes.get(id) ?? [])
    if (node.label === 'matmul') {
      const a = ins[0] ?? []
      total += 2 * out * (a[a.length - 1] ?? 1)
    } else if (node.label.startsWith('reduce_') || node.label.startsWith('arg') || node.label.includes('catter')) {
      total += sizeOf(ins[0] ?? [])
    } else total += out
  }
  return total
}

/** What {@link layerTable} reports for one layer. */
export interface LayerInfo {
  /** The module path (`0`, `encoder.layers.1`). */
  readonly path: string
  readonly kind: string
  readonly detail: string
  readonly inputShape: readonly number[]
  readonly outputShape: readonly number[]
  readonly params: number
  readonly trainableParams: number
  readonly buffers: number
  /** Estimated floating-point operations of one forward pass (undefined when the input cannot be built). */
  readonly flops: number | undefined
  /** The layer's parts (composite modules), with their parameter counts. */
  readonly parts: readonly { readonly name: string; readonly kind: string; readonly params: number }[]
  /** The size of a homogeneous list of parts (a stack of blocks), when there is one. */
  readonly repeat?: number
  readonly depth: number
}

/** Containers whose data flow is known: they are expanded into their children. */
const _isContainer = (m: Module): boolean => m instanceof Sequential || m instanceof Residual || m instanceof Parallel

/** FLOPs of a leaf module at an input shape, by tracing its forward pass in evaluation mode. */
function leafFlops(m: Module, shape: readonly number[]): number | undefined {
  try {
    // Trace the input with the parameters, so operations on the input alone are counted too.
    const f = (v: { readonly p: Params; readonly x: Value }): unknown =>
      (m as unknown as BaseModule<Value, unknown>).apply(v.p, v.x, { train: false })
    return functionFlops(f, { p: m.params, x: zeros(shape) })
  } catch {
    // Modules with structured inputs (pairs, records) cannot be run from a shape alone.
    return undefined
  }
}

function partsOf(m: Module): { parts: LayerInfo['parts']; repeat?: number } {
  const parts = m.children().map(([name, c]) => ({ name, kind: c.kind, params: c.parameterCount() }))
  let repeat: number | undefined
  for (const [, c] of m.children()) {
    const inner = c.children()
    if (c.kind === 'ModuleList' && inner.length > 1 && inner.every(([, x]) => x.kind === inner[0]?.[1].kind))
      repeat = Math.max(repeat ?? 0, inner.length)
  }
  return repeat === undefined ? { parts } : { parts, repeat }
}

/** Parameters whose trainable flag is set. */
function trainableCount(m: Module): number {
  const mask = treePaths(m.trainableMask())
  let n = 0
  for (const [path, p] of Object.entries(treePaths(m.params))) {
    const flag = mask[path]
    if (flag === undefined || item(flag) !== 0) n += sizeOf(p.shape)
  }
  return n
}

/**
 * One row per layer (leaf or composite block), in forward order: shapes propagated through the containers, parameter
 * and buffer counts, estimated FLOPs, the parts of composite blocks and their repeats.
 *
 * @param module - The module.
 * @param inputShape - The input batch shape.
 * @returns The rows.
 * @example
 * ```ts
 * import { layerTable, mlp } from 'whitebox-engine/nn'
 * layerTable(mlp([2, 4, 1]), [8, 2]).map((r) => r.flops) // => [160, 32, 72]
 * ```
 */
export function layerTable(module: Module, inputShape: readonly number[]): LayerInfo[] {
  const rows: LayerInfo[] = []
  const visit = (m: Module, path: string, shape: readonly number[], depth: number): number[] => {
    if (m instanceof Sequential)
      return m
        .children()
        .reduce<number[]>((s, [name, c]) => visit(c, path === '' ? name : `${path}.${name}`, s, depth + 1), [...shape])
    if (m instanceof Residual || m instanceof Parallel) {
      for (const [name, c] of m.children()) visit(c, path === '' ? name : `${path}.${name}`, shape, depth + 1)
      return m.outputShape(shape)
    }
    const out = m.outputShape(shape)
    rows.push({
      path,
      kind: m.kind,
      detail: m.extraRepr(),
      inputShape: [...shape],
      outputShape: out,
      params: m.parameterCount(),
      trainableParams: trainableCount(m),
      buffers: treeCount(m.buffers),
      flops: leafFlops(m, shape),
      ...partsOf(m),
      depth,
    })
    return out
  }
  visit(module, '', inputShape, 0)
  return rows
}

/** Estimated memory of a module, in bytes (float64 storage). */
export interface MemoryEstimate {
  readonly params: number
  readonly buffers: number
  /** The outputs of every layer of one forward pass (what backpropagation keeps). */
  readonly activations: number
  readonly input: number
  readonly total: number
}

/**
 * The memory a forward and backward pass needs, estimated as torchinfo does: parameters, buffers, the input and every
 * layer's output (kept for backpropagation), at 8 bytes per float64 element.
 *
 * @param module - The module.
 * @param inputShape - The input batch shape.
 * @returns Bytes by category and the total.
 * @example
 * ```ts
 * import { memoryEstimate, Linear } from 'whitebox-engine/nn'
 * memoryEstimate(new Linear(4, 2), [8, 4]).params // => 80
 * ```
 */
export function memoryEstimate(module: Module, inputShape: readonly number[]): MemoryEstimate {
  const rows = layerTable(module, inputShape)
  const params = module.parameterCount() * BYTES
  const buffers = treeCount(module.buffers) * BYTES
  const activations = rows.reduce((s, r) => s + sizeOf(r.outputShape), 0) * BYTES
  const input = sizeOf(inputShape) * BYTES
  return { params, buffers, activations, input, total: params + buffers + activations + input }
}

/**
 * The estimated floating-point operations of one forward pass of a module (the sum over its layers).
 *
 * @param module - The module.
 * @param inputShape - The input batch shape.
 * @returns The count (layers whose input cannot be built from a shape count zero).
 * @example
 * ```ts
 * import { flops, Conv2d } from 'whitebox-engine/nn'
 * flops(new Conv2d(1, 1, 3, { bias: false }), [1, 1, 5, 5]) // => 162
 * ```
 */
export function flops(module: Module, inputShape: readonly number[]): number {
  return layerTable(module, inputShape).reduce((s, r) => s + (r.flops ?? 0), 0)
}

/** Options of {@link describe}. */
export interface DescribeOptions {
  /** Show each composite block's parts as nodes in its group (default false: one node per block). */
  readonly expandParts?: boolean
}

/**
 * A module as a {@link Structure}: an input node, one block per layer with its configuration, output shape, parameter
 * count and estimated FLOPs, and the output. Sequential containers become chains (nested ones are groups), residual
 * connections become skip edges into an `add` node, parallel branches fan out and join at their combination, and
 * composite blocks (transformer layers, ResNet blocks) are blocks that list their parts and repeats. whitebox-render
 * draws it as an architecture diagram.
 *
 * @param module - The module.
 * @param inputShape - The shape of one input batch (for example `[batch, features]`).
 * @param options - Whether to show composite blocks' parts.
 * @returns The structure.
 * @concept architectures
 * @example
 * ```ts
 * import { describe, mlp } from 'whitebox-engine/nn'
 * describe(mlp([2, 4, 1]), [8, 2]).nodes.map((n) => n.label) // => ['input', 'Linear', 'ReLU', 'Linear', 'output']
 * ```
 */
export function describe(module: Module, inputShape: readonly number[], options: DescribeOptions = {}): Structure {
  const nodes: StructureNode[] = [
    { id: 'input', label: 'input', role: 'input', badges: { shape: shapeText(inputShape) } },
  ]
  const edges: StructureEdge[] = []
  const groups: StructureGroup[] = []
  const join = (path: string, name: string): string => (path === '' ? name : `${path}.${name}`)
  const inGroup = (group: string | undefined): { group?: string } => (group === undefined ? {} : { group })
  /** Visit a module fed by node `from`; returns the id of its output node and its output shape. */
  const visit = (
    m: Module,
    path: string,
    from: string,
    shape: readonly number[],
    group?: string,
  ): [string, number[]] => {
    if (m instanceof Sequential) {
      const id = `group:${path || m.kind}`
      const g = path === '' ? group : id
      if (path !== '')
        groups.push({
          id,
          label: `${m.kind} ${path}`,
          kind: 'block',
          ...(group === undefined ? {} : { parent: group }),
        })
      let at = from
      let s = [...shape]
      for (const [name, c] of m.children()) [at, s] = visit(c, join(path, name), at, s, g)
      return [at, s]
    }
    if (m instanceof Residual || m instanceof Parallel) {
      const gid = `group:${path || m.kind}`
      groups.push({
        id: gid,
        label: `${m.kind}${path === '' ? '' : ` ${path}`}`,
        kind: 'block',
        ...(group === undefined ? {} : { parent: group }),
      })
      const outs: [string, number[]][] = []
      for (const [name, c] of m.children()) outs.push(visit(c, join(path, name), from, shape, gid))
      const out = m.outputShape(shape)
      const op = m instanceof Residual ? 'add' : typeof m.combine === 'string' ? m.combine : 'combine'
      const id = `op:${path || m.kind}`
      nodes.push({ id, label: op, role: 'operation', group: gid, badges: { shape: shapeText(out) } })
      for (const [o] of outs) edges.push({ from: o, to: id, label: shapeText(out), kind: 'data' })
      // The identity shortcut of a residual block.
      if (m instanceof Residual && m.shortcut === undefined)
        edges.push({ from, to: id, label: shapeText(shape), kind: 'skip' })
      return [id, out]
    }
    const out = m.outputShape(shape)
    const id = `layer:${path || m.kind}`
    const { parts, repeat } = partsOf(m)
    const f = leafFlops(m, shape)
    nodes.push({
      id,
      label: m.kind,
      role: m.parameterCount() > 0 ? 'block' : 'operation',
      ...(m.extraRepr() === '' ? {} : { detail: m.extraRepr() }),
      badges: {
        shape: shapeText(out),
        params: m.parameterCount(),
        ...(f === undefined ? {} : { flops: f }),
        ...(repeat === undefined ? {} : { repeat }),
      },
      ...inGroup(group),
      data: { path, ...(parts.length === 0 ? {} : { parts }) },
    })
    edges.push({ from, to: id, label: shapeText(shape), kind: 'data' })
    if (options.expandParts === true && parts.length > 0) {
      const gid = `parts:${path || m.kind}`
      groups.push({
        id: gid,
        label: `${m.kind} parts`,
        kind: repeat === undefined ? 'cluster' : 'repeat',
        ...(repeat === undefined ? {} : { count: repeat }),
        ...(group === undefined ? {} : { parent: group }),
      })
      for (const p of parts) {
        nodes.push({
          id: `part:${join(path, p.name)}`,
          label: p.kind,
          role: 'internal',
          group: gid,
          badges: { params: p.params },
        })
        edges.push({ from: id, to: `part:${join(path, p.name)}`, kind: 'dependency' })
      }
    }
    return [id, out]
  }
  const [last, out] = visit(module, '', 'input', inputShape)
  nodes.push({ id: 'output', label: 'output', role: 'output', badges: { shape: shapeText(out) } })
  edges.push({ from: last, to: 'output', label: shapeText(out), kind: 'data' })
  return structure('architecture', nodes, edges, { groups, title: module.kind, direction: 'down' })
}

const human = (n: number): string =>
  n >= 1e9
    ? `${(n / 1e9).toFixed(2)}G`
    : n >= 1e6
      ? `${(n / 1e6).toFixed(2)}M`
      : n >= 1e3
        ? `${(n / 1e3).toFixed(2)}K`
        : String(n)

/**
 * A torchinfo-style text summary: one row per layer with its output shape, parameter count and estimated FLOPs, then
 * totals (parameters, trainable and frozen, buffers, FLOPs) and memory estimates in MB.
 *
 * @param module - The module.
 * @param inputShape - The input batch shape.
 * @returns The summary text.
 * @example
 * ```ts
 * import { summary, mlp } from 'whitebox-engine/nn'
 * summary(mlp([2, 4, 1]), [8, 2]).split('\n').find((l) => l.startsWith('Total params')) // => 'Total params: 17'
 * ```
 */
export function summary(module: Module, inputShape: readonly number[]): string {
  const rows = layerTable(module, inputShape)
  const cells = rows.map((r) => [
    `${'  '.repeat(Math.max(0, r.depth - 1))}${r.kind}${r.path === '' ? '' : ` (${r.path})`}${r.repeat === undefined ? '' : ` x${r.repeat}`}`,
    shapeText(r.outputShape),
    String(r.params),
    r.flops === undefined ? '-' : human(r.flops),
  ])
  const header = ['Layer', 'Output shape', 'Params', 'FLOPs']
  const widths = header.map((h, j) => cells.reduce((w, r) => Math.max(w, (r[j] ?? '').length), h.length))
  const line = (row: readonly string[]): string =>
    row
      .map((c, j) => c.padEnd(widths[j] ?? 0))
      .join('  ')
      .trimEnd()
  const total = module.parameterCount()
  const trainable = rows.reduce((s, r) => s + r.trainableParams, 0)
  const mem = memoryEstimate(module, inputShape)
  const mb = (b: number): string => (b / 2 ** 20).toFixed(4)
  return [
    line(header),
    ...cells.map(line),
    `Total params: ${total}`,
    `Trainable params: ${trainable}`,
    `Non-trainable params: ${total - trainable}`,
    `Buffers: ${treeCount(module.buffers)}`,
    `FLOPs (forward): ${human(rows.reduce((s, r) => s + (r.flops ?? 0), 0))}`,
    `Params size (MB): ${mb(mem.params + mem.buffers)}`,
    `Forward/backward pass size (MB): ${mb(mem.activations)}`,
    `Estimated total size (MB): ${mb(mem.total)}`,
  ].join('\n')
}

/**
 * Describe a functional model (any function of parameters and an input) by its computation graph: one node per
 * primitive with shapes, the parameters named by their paths and the input `x`.
 *
 * @param apply - The model `(params, x) => y`.
 * @param params - The parameters.
 * @param input - An input batch.
 * @returns The structure (type `computation-graph`).
 * @example
 * ```ts
 * import { describeFunction, linear } from 'whitebox-engine/nn'
 * import { zeros, type Value } from 'whitebox-engine/tensor'
 * const g = describeFunction((p: { w: Value }, x: Value) => linear(x, p.w), { w: zeros([2, 3]) }, zeros([1, 3]))
 * g.nodes.filter((n) => n.role === 'input').map((n) => n.label).sort() // => ['w', 'x']
 * ```
 */
export function describeFunction<P, X>(apply: (params: P, x: X) => unknown, params: P, input: X): Structure {
  const both = { params, x: input }
  // Leaves flatten in path order: the parameters (under `params.`) and then the input (under `x`).
  const names = Object.keys(treePaths(both)).map((p) => (p.startsWith('params.') ? p.slice('params.'.length) : p))
  return computationGraph((v: { readonly params: P; readonly x: X }) => apply(v.params, v.x), both, {
    inputNames: names,
  })
}
