/**
 * Example "A decision tree and the regions it cuts" (in the index, not in the README): a CART classifier on Fisher's
 * iris data with two features (petal length and petal width), grown to depth 2, one node per step. The left panel is
 * the tree, drawn from the session's `model/structure` record: each box is a question (a split, S1 and S2) or a leaf
 * (L1 to L3), with the count of rows of each class and a fill of the majority class's colour. The right panel is the
 * plane of the same two features: each leaf is one rectangle, shaded as its box is, each split a single line, with the
 * rows over it in the class colours. The rectangles are read from the splits of the record, and checked against what
 * the fitted tree predicts.
 */
import { iris } from 'mdml-engine/datasets'
import type { TreeNode, TreeRecord } from 'mdml-engine/exposition'
import { key } from 'mdml-engine/random'
import { tensor, toArray, toFlat } from 'mdml-engine/tensor'
import { decisionTreeClassifier } from 'mdml-engine/trees'
import { cartesian, type Layer, layer, lines, points, polygons } from 'mdml-render/charts'
import { figure } from 'mdml-render/figure'
import { categoricalColour } from 'mdml-render/theme'
import type { ExampleModule, Outcome } from '../_shared/contract.ts'
import { Figure, Panel } from '../_shared/figure.tsx'
import { type Window, windowOf } from '../_shared/maps.ts'
import { ClassKey, layoutTree, majority, TreeDiagram, tint } from './tree-diagram.tsx'

export const meta: ExampleModule['meta'] = {
  title: 'A decision tree and the regions it cuts',
  summary: 'A depth-2 classification tree on two iris measurements: its questions, and the three rectangles they cut.',
  readme: false,
  alt: 'A decision tree on iris petal length and width drawn as boxes of questions and leaves, beside the plane of the two features cut into the three rectangles of its leaves',
}

interface Result {
  readonly tree: TreeRecord
  readonly rows: number[][]
  readonly labels: number[]
}

const DEPTH = 2

export function run(): Outcome<Result> {
  const all = iris()
  const wanted = [2, 3] // petal length and petal width
  const rows = (toArray(all.x) as number[][]).map((r) => wanted.map((j) => r[j] ?? 0))
  const data = {
    x: tensor(rows),
    y: all.y,
    featureNames: wanted.map((j) => all.featureNames?.[j] ?? `x${j}`),
    classes: all.classes,
  }
  const fit = decisionTreeClassifier({ maxDepth: DEPTH, record: 5 }).session(data as never, { key: key(1) })
  fit.advance() // one node decided per step
  const tree = fit.probe(fit.t, 'model/structure') as TreeRecord
  const model = fit.fittedAt(fit.t)
  const labels = Array.from(toFlat(all.y))

  // Each rectangle must give the class that the tree predicts at its centre and hold the rows its box counts.
  const { rects } = walk(tree, windowOf(rows))
  const centres = rects.map((r) => [(r.x[0] + r.x[1]) / 2, (r.y[0] + r.y[1]) / 2])
  const guess = Array.from(toFlat(model.predict(tensor(centres))))
  const agree = rects.every((r, i) => guess[i] === majority(r.node.value))
  const inside = rects.map(
    (r) =>
      rows.filter((p) => (p[0] ?? 0) > r.x[0] && (p[0] ?? 0) <= r.x[1] && (p[1] ?? 0) > r.y[0] && (p[1] ?? 0) <= r.y[1])
        .length,
  )
  const notes = [
    `${fit.t} steps, ${fit.status}; ${rects.length} leaves; accuracy ${model.score(data as never).toFixed(3)} on the 150 training rows`,
    `rectangles agree with the tree's predictions at their centres: ${agree}; rows per rectangle ${inside.join(', ')} (the boxes say ${rects.map((r) => r.node.rows.length).join(', ')})`,
    ...tree.nodes
      .filter((n) => n.split !== undefined)
      .map((n) => `node ${n.id}: ${tree.features[n.split?.feature ?? 0]} <= ${n.split?.threshold}, counts ${n.value.join('/')}`),
  ]
  return { result: { tree, rows, labels }, tapes: { session: fit }, notes }
}

/** A leaf's rectangle in the plane, read off the splits on the way from the root. */
interface Rect {
  readonly node: TreeNode
  readonly x: readonly [number, number]
  readonly y: readonly [number, number]
}

/** A split's line segment: the feature and value it cuts at, and the extent of the other feature it spans. */
interface Cut {
  readonly node: TreeNode
  readonly feature: number
  readonly at: number
  readonly span: readonly [number, number]
}

/** The leaf rectangles and split lines of a tree, inside a window. */
function walk(tree: TreeRecord, window: Window): { rects: Rect[]; cuts: Cut[] } {
  const byId = new Map(tree.nodes.map((n) => [n.id, n]))
  const rects: Rect[] = []
  const cuts: Cut[] = []
  const go = (n: TreeNode, x: [number, number], y: [number, number]): void => {
    const s = n.split
    if (s === undefined) {
      rects.push({ node: n, x, y })
      return
    }
    const at = s.threshold
    cuts.push({ node: n, feature: s.feature, at, span: s.feature === 0 ? y : x })
    const left = byId.get(s.left) as TreeNode
    const right = byId.get(s.right) as TreeNode
    if (s.feature === 0) {
      go(left, [x[0], at], y)
      go(right, [at, x[1]], y)
    } else {
      go(left, x, [y[0], at])
      go(right, x, [at, y[1]])
    }
  }
  const root = tree.nodes.find((n) => n.parent === null) as TreeNode
  go(root, [...window.x], [...window.y])
  return { rects, cuts }
}

/** Text badges at points of the plane (white boxes with a coloured edge), as one chart layer. */
function badges(
  items: readonly { at: readonly [number, number]; text: string; colour: string; size: number; where: string }[],
): Layer {
  return layer(
    items.map((b) => ({
      type: 'scatter',
      name: `__badge ${b.text}`,
      data: [{ value: [...b.at] }],
      symbolSize: 0,
      silent: true,
      z: 12,
      label: {
        show: true,
        formatter: b.text,
        position: b.where,
        fontSize: b.size,
        fontWeight: 'bold',
        color: '#1a1a1a',
        backgroundColor: '#ffffff',
        borderColor: b.colour,
        borderWidth: 2,
        borderRadius: 5,
        padding: [3, 6, 3, 6],
      },
    })) as never,
  )
}

const PLANE = 520
const FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"

export function View({ result }: { result: Result }) {
  const { tree, rows, labels } = result
  const classes = tree.classes ?? []
  const window = windowOf(rows)
  const { rects, cuts } = walk(tree, window)
  const layout = layoutTree(tree)
  const numberOf = new Map(layout.nodes.map((p) => [p.node.id, p]))
  const [wx, wy] = [window.x[1] - window.x[0], window.y[1] - window.y[0]]
  const feature = (j: number): string => tree.features[j] ?? `x${j}`
  const short = (j: number): string => feature(j).replace(/\s*\([^)]*\)\s*$/, '')

  // Where the labels go: a split's label at the start of its line, a leaf's at the free spot farthest from the rows.
  const taken: [number, number][] = rows.map((r) => [r[0] ?? 0, r[1] ?? 0])
  const splitLabels = cuts.map((c) => {
    const text = `S${numberOf.get(c.node.id)?.split}: ${short(c.feature)} ≤ ${c.at}`
    const at: [number, number] =
      c.feature === 0 ? [c.at + 0.012 * wx, window.y[0] + 0.07 * wy] : [c.span[0] + 0.012 * wx, c.at + 0.08 * wy]
    for (let t = 0; t <= 5; t++) taken.push([at[0] + t * 0.05 * wx, at[1]])
    return { at, text, colour: '#333333', size: 13, where: 'right' }
  })
  const leafLabels = rects.map((r) => {
    const m = majority(r.node.value)
    const text = `L${numberOf.get(r.node.id)?.leaf}: ${classes[m] ?? m}`
    // The label is a box about 0.2 of the plot wide and 0.06 high: it must fit in the rectangle and stay clear of rows.
    const [hw, hh] = [(text.length * 0.0125 + 0.02) * wx, 0.045 * wy]
    let best: [number, number] = [(r.x[0] + r.x[1]) / 2, (r.y[0] + r.y[1]) / 2]
    let bestD = -1
    for (let i = 0; i <= 12; i++)
      for (let j = 0; j <= 12; j++) {
        const c: [number, number] = [
          r.x[0] + hw + ((r.x[1] - r.x[0] - 2 * hw) * i) / 12,
          r.y[0] + hh + ((r.y[1] - r.y[0] - 2 * hh) * j) / 12,
        ]
        const d = Math.min(
          ...taken.map((q) =>
            Math.hypot(Math.max(0, Math.abs(q[0] - c[0]) - hw) / wx, Math.max(0, Math.abs(q[1] - c[1]) - hh) / wy),
          ),
        )
        if (d > bestD) {
          bestD = d
          best = c
        }
      }
    taken.push(best)
    return { at: best, text, colour: categoricalColour(m), size: 15, where: 'inside' }
  })

  const plane = cartesian(
    [
      polygons(
        rects.map((r) => ({
          points: [
            [r.x[0], r.y[0]],
            [r.x[1], r.y[0]],
            [r.x[1], r.y[1]],
            [r.x[0], r.y[1]],
          ] as [number, number][],
          colour: `tab${majority(r.node.value)}`,
          opacity: tint(r.node.value),
          edges: false,
        })),
      ),
      lines(
        cuts.map((c) => ({
          x: c.feature === 0 ? [c.at, c.at] : [...c.span],
          y: c.feature === 0 ? [...c.span] : [c.at, c.at],
          colour: '#222222',
          width: 2.5,
        })),
      ),
      ...[0, 1, 2].map((c) => {
        const idx = labels.flatMap((l, i) => (l === c ? [i] : []))
        return points(
          [
            {
              x: idx.map((i) => rows[i]?.[0] ?? 0),
              y: idx.map((i) => rows[i]?.[1] ?? 0),
              name: classes[c] ?? `class ${c}`,
              colour: `tab${c}`,
              size: 8,
              opacity: 1,
            },
          ],
          { colourOffset: 0 },
        )
      }),
      badges([...splitLabels, ...leafLabels]),
    ],
    {
      figure: figure({
        width: PLANE,
        height: layout.height + 10,
        padding: { top: 36, right: 14, bottom: 46, left: 54 },
      }),
      x: { label: feature(0), limits: [...window.x] },
      y: { label: feature(1), limits: [...window.y] },
      legend: true,
    },
  )
  const top = 30
  return (
    <Figure width={layout.width + 24 + PLANE} height={top + layout.height + 52} title={meta.alt}>
      <text x={layout.width / 2} y={19} textAnchor="middle" fontFamily={FONT} fontSize={15} fontWeight={600} fill="#333333">
        the fitted tree (yes goes left)
      </text>
      <TreeDiagram tree={tree} layout={layout} x={0} y={top} />
      <ClassKey classes={classes} x={(layout.width - 130 * classes.length) / 2 + 20} y={top + layout.height + 36} />
      <Panel x={layout.width + 24} width={PLANE} height={layout.height + 10} title="the same tree on the plane" still={plane} />
    </Figure>
  )
}
