import { type MazeAlgorithmName, type MazeState, mazeAlgorithm, mazeOf } from 'omai-engine/worlds'
import { worldView } from 'omai-render/gym'
import { ControlPanel, Player, SceneFrame, SeedInput, Select, useLiveRun } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Maze generation',
  question: 'How do I watch a world being built, step by step?',
  explain:
    'Maze generators are engine `Algorithm`s; `useLiveRun` steps one as the player plays and `worldView` draws each state: the cells already carved tinted, the cell being worked on outlined. Change the algorithm to see the backtracker’s long corridors, Prim’s many short dead ends, or the binary tree’s diagonal bias.',
}

const ALGORITHMS: MazeAlgorithmName[] = [
  'backtracker',
  'prim',
  'kruskal',
  'wilson',
  'aldousBroder',
  'eller',
  'binaryTree',
  'sidewinder',
  'division',
]

export default function MazeGeneration(): ReactNode {
  const [name, setName] = useState<MazeAlgorithmName>('backtracker')
  const [seed, setSeed] = useState(0)
  const algorithm = useMemo(() => mazeAlgorithm({ rows: 12, cols: 18, algorithm: name }), [name])
  const run = useLiveRun<MazeState>(algorithm, { maxSteps: 2000, seed, fps: 30, stepsPerFrame: 1 })
  const view = useMemo(() => {
    const s = run.state
    const carved = Array.from(s.inMaze).flatMap((v, i) => (v === 1 ? [i] : []))
    return worldView(mazeOf(s), { cell: 28, explored: carved, frontier: s.current >= 0 ? [s.current] : [] })
  }, [run.state])
  return (
    <div className="flex w-fit max-w-full flex-col gap-3">
      <ControlPanel columns={2}>
        <Select label="algorithm" value={name} onChange={setName} options={ALGORITHMS} />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <SceneFrame scene={view} filename="maze" playback={run.controller} />
      <Player
        controller={run.controller}
        stamp={() => `step ${run.t}${run.status === undefined ? '' : ` (${run.status})`}`}
      />
    </div>
  )
}
