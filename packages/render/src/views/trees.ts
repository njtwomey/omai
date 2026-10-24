import type { EChartsOption } from 'echarts'
import type { TreeNode, TreeRecord } from 'whitebox-engine/exposition'
import { group, line, type Paint, rect, type Scene, type SceneNode, scene, text } from 'whitebox-engine/scene'
import { referenceLines } from '../charts/functions.ts'
import { lines } from '../charts/lines.ts'
import { cartesian, type Layer } from '../charts/plot.ts'
import { points } from '../charts/points.ts'
import { tidyTree } from '../layout/trees.ts'
import { formatNumber } from '../theme/format.ts'
import { outlined, short, type ViewOptions, viewFigure, viewMode } from './_view.ts'

/** Options of {@link treeScene}. */
export interface TreeSceneOptions {
  /** The node drawn selected (an accent outline), e.g. the one whose candidate splits are shown. */
  readonly selected?: number
  /** Show only the nodes grown up to this step of the growth order (default all): a frame of the growth. */
  readonly grownTo?: number
  /** The deepest level drawn (default all); deeper subtrees are summarised under their node. */
  readonly maxDepth?: number
}

const CARD_W = 136
// Leaves say less (a prediction, not a split), so their cards are narrower and wide trees stay compact.
const LEAF_W = 104
const CARD_H = 58
const FONT = 11

/** The text of a node's split, e.g. `petal width ≤ 0.8`. */
export function splitText(tree: TreeRecord, node: TreeNode): string | undefined {
  if (node.split === undefined) return undefined
  const name = tree.features[node.split.feature] ?? `x${node.split.feature}`
  // Units in brackets go first, then the name is cut to fit a card.
  const bare = name.replace(/\s*\([^)]*\)\s*$/, '')
  const shortName = bare.length > 14 ? `${bare.slice(0, 13)}…` : bare
  return `${shortName} ≤ ${short(node.split.threshold)}`
}

/** What a node predicts: its majority class (classification) or its mean (regression). */
export function nodePrediction(tree: TreeRecord, node: TreeNode): string {
  if (tree.classes === undefined || node.value.length < 2) return `ŷ = ${short(node.value[0] ?? 0)}`
  const k = node.value.indexOf(Math.max(...node.value))
  return tree.classes[k] ?? `class ${k}`
}

/**
 * A grown tree as a scene: a tidy tree of cards, each with its split (or, at a leaf, its prediction), a bar of its
 * class proportions (classification), and its row count and impurity; left edges are the rows that satisfy the split.
 * Card groups have ids `node:<id>` for hit-testing.
 *
 * @param tree - The record.
 * @param options - The selected node, the growth step and the depth limit.
 * @returns The scene.
 * @example
 * ```ts
 * import { treeRecord } from 'whitebox-engine/exposition'
 * import { treeScene } from 'whitebox-render/views'
 * const t = treeRecord({ criterion: 'gini', features: ['x'], classes: ['a', 'b'], nodes: [
 *   { id: 0, parent: null, depth: 0, rows: [0, 1], value: [1, 1], impurity: 0.5, candidates: [], split: { feature: 0, threshold: 0.5, left: 1, right: 2 }, order: 0 },
 *   { id: 1, parent: 0, depth: 1, rows: [0], value: [1, 0], impurity: 0, candidates: [], order: 1 },
 *   { id: 2, parent: 0, depth: 1, rows: [1], value: [0, 1], impurity: 0, candidates: [], order: 2 },
 * ] })
 * treeScene(t).kind // => 'scene'
 * ```
 */
export function treeScene(tree: TreeRecord, options: TreeSceneOptions = {}): Scene {
  const grownTo = options.grownTo ?? Number.POSITIVE_INFINITY
  const maxDepth = options.maxDepth ?? Number.POSITIVE_INFINITY
  const shown = tree.nodes.filter((n) => n.order <= grownTo && n.depth <= maxDepth)
  const visible = new Set(shown.map((n) => n.id))
  // A node is drawn as split only when both children are drawn.
  const isOpen = (n: TreeNode): boolean =>
    n.split !== undefined && visible.has(n.split.left) && visible.has(n.split.right)
  const edges = shown.flatMap((n) =>
    isOpen(n) && n.split !== undefined
      ? [
          { from: `${n.id}`, to: `${n.split.left}` },
          { from: `${n.id}`, to: `${n.split.right}` },
        ]
      : [],
  )
  const layout = tidyTree(
    { nodes: shown.map((n) => ({ id: `${n.id}`, width: isOpen(n) ? CARD_W : LEAF_W, height: CARD_H })), edges },
    { levelSep: 34, siblingSep: 10, margin: 8 },
  )
  const total = Math.max(1, tree.nodes.find((n) => n.parent === null)?.rows.length ?? 1)
  const children: SceneNode[] = []
  // Edges first, under the cards: from the bottom of the parent to the top of the child, labelled yes / no.
  for (const n of shown) {
    if (!isOpen(n) || n.split === undefined) continue
    const a = layout.nodes[`${n.id}`]
    if (a === undefined) continue
    for (const [side, child] of [
      ['yes', n.split.left],
      ['no', n.split.right],
    ] as const) {
      const b = layout.nodes[`${child}`]
      if (b === undefined) continue
      const x1 = a.x
      const y1 = a.y + CARD_H / 2
      const x2 = b.x
      const y2 = b.y - CARD_H / 2
      children.push(line(x1, y1, x2, y2, { stroke: 'grid', strokeWidth: 1.4 }))
      children.push(
        text((x1 + x2) / 2 + (side === 'yes' ? -6 : 6), (y1 + y2) / 2, side, {
          size: FONT - 2,
          fill: 'muted',
          anchor: side === 'yes' ? 'end' : 'start',
          baseline: 'middle',
        }),
      )
    }
  }
  for (const n of shown) {
    const box = layout.nodes[`${n.id}`]
    if (box === undefined) continue
    const cardW = isOpen(n) ? CARD_W : LEAF_W
    const x0 = box.x - cardW / 2
    const y0 = box.y - CARD_H / 2
    const selected = options.selected === n.id
    const parts: SceneNode[] = [
      rect(x0, y0, cardW, CARD_H, {
        fill: 'surface',
        stroke: selected ? 'accent' : 'grid',
        strokeWidth: selected ? 2.2 : 1,
        radius: 6,
      }),
    ]
    const heading = isOpen(n) ? (splitText(tree, n) ?? '') : `→ ${nodePrediction(tree, n)}`
    parts.push(text(box.x, y0 + 15, heading, { size: FONT, fill: 'fg', anchor: 'middle', weight: 'bold' }))
    // The class proportions as a stacked bar; a regression node shows its share of the rows instead.
    const barX = x0 + 10
    const barW = cardW - 20
    const barY = y0 + 24
    if (tree.classes !== undefined && n.value.length > 1) {
      const sum = n.value.reduce((s, v) => s + v, 0) || 1
      let at = barX
      n.value.forEach((v, k) => {
        const w = (barW * v) / sum
        if (w > 0) parts.push(rect(at, barY, w, 8, { fill: `tab${k % 10}` as Paint }))
        at += w
      })
    } else {
      parts.push(rect(barX, barY, barW, 8, { fill: 'grid' }))
      parts.push(rect(barX, barY, (barW * n.rows.length) / total, 8, { fill: 'tab0' }))
    }
    const hidden = n.split !== undefined && !isOpen(n) && n.depth >= maxDepth ? ' · …' : ''
    parts.push(
      text(
        box.x,
        y0 + 47,
        `n ${n.rows.length} · ${tree.criterion} ${formatNumber(n.impurity, { digits: 2 })}${hidden}`,
        {
          size: FONT - 1,
          fill: 'muted',
          anchor: 'middle',
        },
      ),
    )
    children.push(group(parts, { id: `node:${n.id}`, title: describeNode(tree, n) }))
  }
  return scene(Math.max(layout.width, CARD_W + 16), Math.max(layout.height, CARD_H + 16), children, {
    title: `${tree.criterion} tree`,
  })
}

function describeNode(tree: TreeRecord, n: TreeNode): string {
  const split = splitText(tree, n)
  const value =
    tree.classes === undefined
      ? `mean ${short(n.value[0] ?? 0)}`
      : n.value.map((v, k) => `${tree.classes?.[k] ?? k}: ${v}`).join(', ')
  return `node ${n.id} (depth ${n.depth}): ${n.rows.length} rows, ${value}${split === undefined ? '' : `; split ${split}`}`
}

/** The node id of a scene id from {@link treeScene} (`node:3` gives 3), or `undefined`. */
export function treeNodeOf(sceneId: string): number | undefined {
  const m = /^node:(\d+)$/.exec(sceneId)
  return m === null ? undefined : Number(m[1])
}

/** The number of growth steps of a tree (one per node, in growth order). */
export function growthSteps(tree: TreeRecord): number {
  return Math.max(0, ...tree.nodes.map((n) => n.order)) + 1
}

/**
 * The candidate splits a node evaluated: the impurity decrease against the threshold, one curve per feature, with the
 * chosen split marked (a leaf shows its candidates too, none of which was taken).
 *
 * @param tree - The record.
 * @param nodeId - The node.
 * @param options - Mode and figure.
 * @returns The ECharts option.
 */
export function splitCandidatesView(tree: TreeRecord, nodeId: number, options: ViewOptions = {}): EChartsOption {
  const fig = viewFigure(options, { aspect: 2.4 })
  const mode = viewMode({ ...options, mode: fig.mode })
  const node = tree.nodes.find((n) => n.id === nodeId)
  const layers: Layer[] = []
  // Features take the tab10 slots after the classes', so a feature's curve is never mistaken for a class.
  const offset = tree.classes?.length ?? 0
  const byFeature = new Map<number, { t: number; d: number }[]>()
  for (const c of node?.candidates ?? [])
    byFeature.set(c.feature, [...(byFeature.get(c.feature) ?? []), { t: c.threshold, d: c.decrease }])
  layers.push(
    lines(
      [...byFeature.entries()]
        .sort((a, b) => a[0] - b[0])
        .map(([f, cs]) => {
          const sorted = [...cs].sort((a, b) => a.t - b.t)
          return {
            x: sorted.map((c) => c.t),
            y: sorted.map((c) => c.d),
            name: tree.features[f] ?? `x${f}`,
            colour: `tab${(offset + f) % 10}`,
            markers: sorted.length < 40,
            width: 1.5,
          }
        }),
      { mode },
    ),
  )
  if (node?.split !== undefined) {
    const chosen = node.candidates.find(
      (c) => c.feature === node.split?.feature && c.threshold === node.split?.threshold,
    )
    layers.push(referenceLines({ x: [node.split.threshold] }, { mode, colour: 'accent', dash: 'dashed' }))
    if (chosen !== undefined)
      layers.push(
        outlined(
          points(
            [
              {
                x: [chosen.threshold],
                y: [chosen.decrease],
                name: 'chosen',
                colour: `tab${(offset + chosen.feature) % 10}`,
                size: 12,
              },
            ],
            {
              mode,
            },
          ),
          mode,
          1.5,
        ),
      )
  }
  // What the chart is about goes under its threshold axis, so it never meets the legend.
  const about =
    node === undefined
      ? 'no such node'
      : node.split === undefined
        ? `node ${node.id}, a leaf: ${node.candidates.length} candidates, none taken`
        : `node ${node.id}: chose ${splitText(tree, node)}`
  return cartesian(layers, {
    figure: fig,
    mode,
    x: { label: `threshold (${about})` },
    y: { label: 'impurity decrease' },
    tooltip: 'item',
    legend: true,
  })
}
