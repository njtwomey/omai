import { type ReactNode, useMemo, useState } from 'react'
import { structure } from 'whitebox-engine/scene'
import { Choice, ControlPanel, Player, StructureView, usePlayback } from 'whitebox-render/react'
import { searchOrder, traversalFrames } from 'whitebox-render/structures'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Search trees',
  question: 'How do I draw a tree and animate a search through it?',
  explain:
    'A structure of type `tree` is laid out as a tidy tree, leaves green and inner nodes blue. `searchOrder` gives the breadth-first or depth-first expansion order, and `traversalFrames` turns an order into overlays (the node expanded now active, every expanded node numbered), one frame per expansion for the player.',
}

// A small game tree: each move branches.
const ids = ['root', 'a', 'b', 'c', 'a1', 'a2', 'b1', 'c1', 'c2', 'c3', 'a1x', 'a2x', 'c2x']
const parent: Record<string, string> = {
  a: 'root',
  b: 'root',
  c: 'root',
  a1: 'a',
  a2: 'a',
  b1: 'b',
  c1: 'c',
  c2: 'c',
  c3: 'c',
  a1x: 'a1',
  a2x: 'a2',
  c2x: 'c2',
}
const tree = structure(
  'tree',
  ids.map((id) => ({
    id,
    label: id,
    role: ids.some((c) => parent[c] === id) ? ('internal' as const) : ('leaf' as const),
  })),
  Object.entries(parent).map(([to, from]) => ({ from, to })),
)

export default function SearchTrees(): ReactNode {
  const [kind, setKind] = useState<'bfs' | 'dfs'>('bfs')
  const frames = useMemo(() => traversalFrames(tree, searchOrder(tree, 'root', kind)), [kind])
  const playback = usePlayback(frames.length, { fps: 2 })
  return (
    <div className="flex w-fit max-w-full flex-col gap-3">
      <ControlPanel>
        <Choice
          label="search"
          value={kind}
          onChange={setKind}
          options={[
            { value: 'bfs', label: 'breadth first' },
            { value: 'dfs', label: 'depth first' },
          ]}
        />
      </ControlPanel>
      <StructureView structure={tree} overlay={frames[playback.frame]} frames={frames} />
      <Player controller={playback} stamp={(f) => `expansion ${f + 1} of ${frames.length}`} />
    </div>
  )
}
