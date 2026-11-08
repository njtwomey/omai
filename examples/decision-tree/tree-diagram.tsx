/**
 * The tree of a `TreeRecord` drawn as boxes and arrows in plain SVG, for the decision-tree example: every box shows the
 * question it asks (internal nodes) or the leaf number and class it predicts (leaves), the number of training rows of
 * each class in the class colours, and is filled with the colour of its majority class, paler where the box is less
 * pure. The geometry (positions, numbering of leaves and splits) is computed from the record by {@link layoutTree}, so
 * the plane beside it can use the same numbers.
 */

import type { TreeNode, TreeRecord } from 'mdml-engine/exposition'
import { categoricalColour } from 'mdml-render/theme'
import type { ReactElement } from 'react'

const FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"

/** A node placed on the page, with its numbers. */
export interface PlacedNode {
  readonly node: TreeNode
  /** Centre of the box. */
  readonly x: number
  readonly y: number
  /** `L1`, `L2`, ... for leaves in left-to-right order; absent for internal nodes. */
  readonly leaf?: number
  /** `S1`, `S2`, ... for internal nodes in the order the tree decided them; absent for leaves. */
  readonly split?: number
}

/** The tree's layout. */
export interface TreeLayout {
  readonly nodes: readonly PlacedNode[]
  readonly width: number
  readonly height: number
}

export const BOX = { width: 196, height: 86, gapX: 12, gapY: 56 } as const

/** The share of a node's rows in its majority class, rescaled so that an even split is 0 and a pure node is 1. */
export function purity(value: readonly number[]): number {
  const total = value.reduce((a, b) => a + b, 0)
  const k = value.length
  return total === 0 || k < 2 ? 1 : (Math.max(...value) / total - 1 / k) / (1 - 1 / k)
}

/** The opacity of a class colour for a node of this purity (the tree boxes and the regions of the plane share it). */
export function tint(value: readonly number[]): number {
  return 0.12 + 0.38 * purity(value)
}

/** The index of the majority class. */
export function majority(value: readonly number[]): number {
  return value.indexOf(Math.max(...value))
}

/**
 * Place the nodes: leaves side by side in the order of the splits (`left` before `right`), parents centred above their
 * children, one row per depth.
 */
export function layoutTree(tree: TreeRecord): TreeLayout {
  const byId = new Map(tree.nodes.map((n) => [n.id, n]))
  const root = tree.nodes.find((n) => n.parent === null)
  if (root === undefined) throw new Error('layoutTree: no root')
  let nextLeaf = 0
  const placed = new Map<number, { x: number; leaf?: number }>()
  const walk = (n: TreeNode): number => {
    if (n.split === undefined) {
      const x = BOX.width / 2 + nextLeaf * (BOX.width + BOX.gapX)
      placed.set(n.id, { x, leaf: nextLeaf + 1 })
      nextLeaf += 1
      return x
    }
    const l = walk(byId.get(n.split.left) as TreeNode)
    const r = walk(byId.get(n.split.right) as TreeNode)
    placed.set(n.id, { x: (l + r) / 2 })
    return (l + r) / 2
  }
  walk(root)
  const internal = tree.nodes.filter((n) => n.split !== undefined).sort((a, b) => a.order - b.order)
  const depth = Math.max(...tree.nodes.map((n) => n.depth))
  const nodes = tree.nodes.map((node): PlacedNode => {
    const p = placed.get(node.id) as { x: number; leaf?: number }
    const split = internal.indexOf(node)
    return {
      node,
      x: p.x,
      y: BOX.height / 2 + node.depth * (BOX.height + BOX.gapY),
      ...(p.leaf === undefined ? {} : { leaf: p.leaf }),
      ...(split < 0 ? {} : { split: split + 1 }),
    }
  })
  return {
    nodes,
    width: nextLeaf * BOX.width + (nextLeaf - 1) * BOX.gapX,
    height: (depth + 1) * BOX.height + depth * BOX.gapY,
  }
}

/** A feature name without its unit, for the question in a box (`petal width (cm)` becomes `petal width`). */
const short = (name: string): string => name.replace(/\s*\([^)]*\)\s*$/, '')

/** The tree as SVG elements, positioned at `(x, y)` on the page. */
export function TreeDiagram(props: { tree: TreeRecord; layout: TreeLayout; x: number; y: number }): ReactElement {
  const { tree, layout } = props
  const classes = tree.classes ?? []
  const at = new Map(layout.nodes.map((p) => [p.node.id, p]))
  const colour = (k: number): string => categoricalColour(k)
  return (
    <g transform={`translate(${props.x} ${props.y})`} fontFamily={FONT}>
      {layout.nodes.map((p) => {
        const s = p.node.split
        if (s === undefined) return null
        return [
          [s.left, 'yes'],
          [s.right, 'no'],
        ].map(([child, word]) => {
          const c = at.get(child as number) as PlacedNode
          const [x1, y1, x2, y2] = [p.x, p.y + BOX.height / 2, c.x, c.y - BOX.height / 2]
          const [mx, my] = [(x1 + x2) / 2, (y1 + y2) / 2]
          return (
            <g key={`${p.node.id}-${child}`}>
              <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="#555555" strokeWidth={1.8} />
              <rect x={mx - 17} y={my - 11} width={34} height={22} rx={5} fill="#ffffff" />
              <text x={mx} y={my + 5} textAnchor="middle" fontSize={14} fontWeight={600} fill="#444444">
                {word}
              </text>
            </g>
          )
        })
      })}
      {layout.nodes.map((p) => {
        const { node } = p
        const m = majority(node.value)
        const total = node.value.reduce((a, b) => a + b, 0)
        const left = p.x - BOX.width / 2
        const top = p.y - BOX.height / 2
        const leaf = node.split === undefined
        const head = leaf
          ? `L${p.leaf}: ${classes[m] ?? `class ${m}`}`
          : `S${p.split}: ${short(tree.features[node.split?.feature ?? 0] ?? 'x')} ≤ ${node.split?.threshold}`
        return (
          <g key={node.id}>
            <rect x={left} y={top} width={BOX.width} height={BOX.height} rx={9} fill="#ffffff" />
            <rect
              x={left}
              y={top}
              width={BOX.width}
              height={BOX.height}
              rx={9}
              fill={colour(m)}
              fillOpacity={tint(node.value)}
              stroke={leaf ? colour(m) : '#777777'}
              strokeWidth={leaf ? 3.5 : 1.5}
            />
            <text x={p.x} y={top + 26} textAnchor="middle" fontSize={leaf ? 18 : 16} fontWeight={700} fill="#1a1a1a">
              {head}
            </text>
            <text x={p.x} y={top + 51} textAnchor="middle" fontSize={17} fontWeight={700}>
              {node.value.map((v, k) => [
                k === 0 ? null : (
                  <tspan key={`s${k}`} dx={5} fill="#666666" fontWeight={400}>
                    /
                  </tspan>
                ),
                <tspan key={k} dx={k === 0 ? 0 : 5} fill={colour(k)}>
                  {v}
                </tspan>,
              ])}
            </text>
            <text x={p.x} y={top + 73} textAnchor="middle" fontSize={13} fill="#444444">
              {leaf ? `${node.value[m]} of ${total} are ${classes[m] ?? m}` : `${total} rows reach this question`}
            </text>
          </g>
        )
      })}
    </g>
  )
}

/** A one-row key: the class colours with their names. */
export function ClassKey(props: { classes: readonly string[]; x: number; y: number }): ReactElement {
  return (
    <g transform={`translate(${props.x} ${props.y})`} fontFamily={FONT} fontSize={14} fill="#333333">
      {props.classes.map((c, k) => (
        <g key={c} transform={`translate(${k * 130} 0)`}>
          <rect width={14} height={14} y={-12} rx={3} fill={categoricalColour(k)} />
          <text x={20}>{c}</text>
        </g>
      ))}
    </g>
  )
}
