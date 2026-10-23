import { type Structure, type StructureEdge, type StructureNode, structure } from '../scene/index.ts'
import type { Module as BaseModule } from './module/module.ts'

type Module = BaseModule<never, unknown>

const shapeText = (shape: readonly number[]): string => `[${shape.join(', ')}]`

/**
 * A module as a {@link Structure}: an input node, one block per layer with its configuration, output shape and parameter
 * count, and the output; nested containers become groups. whitebox-render draws it as an architecture diagram.
 *
 * @param module - The module.
 * @param inputShape - The shape of one input batch (for example `[batch, features]`).
 * @returns The structure.
 * @concept architectures
 * @example
 * ```ts
 * import { describe, mlp } from 'whitebox-engine/nn'
 * describe(mlp([2, 4, 1]), [8, 2]).nodes.map((n) => n.label) // => ['input', 'Linear', 'ReLU', 'Linear', 'output']
 * ```
 */
export function describe(module: Module, inputShape: readonly number[]): Structure {
  const nodes: StructureNode[] = [
    { id: 'input', label: 'input', role: 'input', badges: { shape: shapeText(inputShape) } },
  ]
  const edges: StructureEdge[] = []
  const groups: { id: string; label: string; kind: 'block'; parent?: string }[] = []
  let previous = 'input'
  const visit = (m: Module, path: string, shape: readonly number[], group?: string): number[] => {
    const children = m.children()
    if (children.length > 0) {
      const id = `group:${path || m.kind}`
      if (path !== '')
        groups.push({
          id,
          label: `${m.kind} ${path}`,
          kind: 'block',
          ...(group === undefined ? {} : { parent: group }),
        })
      return children.reduce<number[]>(
        (s, [name, child]) => visit(child, path === '' ? name : `${path}.${name}`, s, path === '' ? group : id),
        [...shape],
      )
    }
    const out = m.outputShape(shape)
    const id = `layer:${path || m.kind}`
    nodes.push({
      id,
      label: m.kind,
      role: m.parameterCount() > 0 ? 'block' : 'operation',
      ...(m.extraRepr() === '' ? {} : { detail: m.extraRepr() }),
      badges: { shape: shapeText(out), params: m.parameterCount() },
      ...(group === undefined ? {} : { group }),
      data: { path },
    })
    edges.push({ from: previous, to: id, label: shapeText(shape), kind: 'data' })
    previous = id
    return out
  }
  const out = visit(module, '', inputShape)
  nodes.push({ id: 'output', label: 'output', role: 'output', badges: { shape: shapeText(out) } })
  edges.push({ from: previous, to: 'output', label: shapeText(out), kind: 'data' })
  return structure('architecture', nodes, edges, { groups, title: module.kind, direction: 'down' })
}

/**
 * A torchinfo-style text summary: one row per layer with its output shape and parameter count, and the total.
 *
 * @param module - The module.
 * @param inputShape - The input batch shape.
 * @returns The summary text.
 * @example
 * ```ts
 * import { summary, mlp } from 'whitebox-engine/nn'
 * summary(mlp([2, 4, 1]), [8, 2]).split('\n').at(-1) // => 'Total params: 17'
 * ```
 */
export function summary(module: Module, inputShape: readonly number[]): string {
  const s = describe(module, inputShape)
  const rows = s.nodes
    .filter((n) => n.id.startsWith('layer:'))
    .map((n) => [
      `${n.label} (${String(n.data?.['path'] ?? '')})`,
      String(n.badges?.['shape'] ?? ''),
      String(n.badges?.['params'] ?? 0),
    ])
  const header = ['Layer', 'Output shape', 'Params']
  const widths = header.map((h, j) => Math.max(h.length, ...rows.map((r) => (r[j] ?? '').length)))
  const line = (cells: readonly string[]): string =>
    cells
      .map((c, j) => c.padEnd(widths[j] ?? 0))
      .join('  ')
      .trimEnd()
  return [line(header), ...rows.map(line), `Total params: ${module.parameterCount()}`].join('\n')
}
