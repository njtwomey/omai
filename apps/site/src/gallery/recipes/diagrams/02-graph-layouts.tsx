import { type ReactNode, useMemo, useState } from 'react'
import { path, rect, type SceneNode, scene, text } from 'whitebox-engine/scene'
import {
  circularLayout,
  forceLayout,
  type GraphLayout,
  gridLayout,
  type LayoutGraph,
  layered,
  radialTree,
  spectralLayout,
  tidyTree,
} from 'whitebox-render/layout'
import { ControlPanel, SceneView, SeedInput, Select } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Graph layouts',
  question: 'How do I place the nodes of a graph: in layers, as a tree, by forces, on a circle?',
  explain:
    'Layouts in `whitebox-render/layout` take nodes with sizes and edges and return node boxes, edge routes and group boxes: `layered` (Sugiyama-style, as dagre), `tidyTree` and `radialTree`, a seeded `forceLayout` that stops when its energy settles, `circularLayout`, `spectralLayout` (Laplacian eigenvectors) and `gridLayout`. Here the layout is drawn by hand as a scene; diagrams do this for any structure.',
}

const names = ['root', 'parse', 'check', 'lower', 'optimise', 'emit', 'cache', 'log', 'report']
const graph: LayoutGraph = {
  nodes: names.map((id) => ({ id, width: 12 + 7 * id.length, height: 26 })),
  edges: [
    ['root', 'parse'],
    ['root', 'cache'],
    ['parse', 'check'],
    ['parse', 'log'],
    ['check', 'lower'],
    ['lower', 'optimise'],
    ['lower', 'emit'],
    ['cache', 'report'],
  ].map(([from, to]) => ({ from: from as string, to: to as string })),
}

const LAYOUTS = ['layered', 'tidy tree', 'radial', 'force', 'circle', 'spectral', 'grid'] as const
type LayoutName = (typeof LAYOUTS)[number]

function place(name: LayoutName, seed: number): GraphLayout {
  switch (name) {
    case 'layered':
      return layered(graph)
    case 'tidy tree':
      return tidyTree(graph)
    case 'radial':
      return radialTree(graph, { levelSep: 90 })
    case 'force':
      return forceLayout(graph, { seed, length: 90 })
    case 'circle':
      return circularLayout(graph)
    case 'spectral':
      return spectralLayout(graph, { size: 360 })
    case 'grid':
      return gridLayout(graph, { columns: 3, cell: 110 })
  }
}

export default function GraphLayouts(): ReactNode {
  const [name, setName] = useState<LayoutName>('layered')
  const [seed, setSeed] = useState(0)
  const drawing = useMemo(() => {
    const l = place(name, seed)
    const nodes: SceneNode[] = Object.entries(l.nodes).flatMap(([id, b]) => [
      rect(b.x - b.width / 2, b.y - b.height / 2, b.width, b.height, {
        id,
        fill: 'surface',
        stroke: 'tab0',
        radius: 6,
      }),
      text(b.x, b.y, id, { anchor: 'middle', baseline: 'middle', size: 12 }),
    ])
    const edges = l.edges.map((e) => path(e.points, { stroke: 'muted', arrow: true, strokeWidth: 1.2 }))
    return scene(l.width, l.height, [...edges, ...nodes])
  }, [name, seed])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <Select label="layout" value={name} onChange={setName} options={[...LAYOUTS]} />
        <SeedInput value={seed} onChange={setSeed} label="seed (force layout)" />
      </ControlPanel>
      <SceneView scene={drawing} width={Math.min(drawing.width, 720)} />
    </div>
  )
}
