import { episodeRecord } from 'mdml-engine/exposition'
import { ControlPanel, ExpositionView, NumberField, SeedInput } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { uniforms } from '../../toys/synthetic.ts'

export const recipe: Recipe = {
  title: 'Episodes',
  question: 'How do I show how an agent’s returns improved over episodes, and what it learned about the world?',
  explain:
    'An `episodes` record has the return and the length of every episode and, optionally, a few recorded episodes step by step and snapshots of the value function or policy at chosen episodes. `EpisodeView` draws the returns with a moving average, the lengths and, when the world is a grid, the values and greedy actions as they were at a snapshot you pick. The agent is tabular Q-learning on a small corridor-and-pit grid: with a high exploration rate it learns slowly and keeps falling in the pit; with too little it settles on the first path it finds.',
  shows: ['episodes', 'EpisodeView'],
}

const ROWS = 4
const COLS = 6
const GOAL = COLS - 1
const PIT = 2 * COLS + COLS - 1
const MOVES = [
  [-1, 0],
  [1, 0],
  [0, -1],
  [0, 1],
]

export default function EpisodesRecipe(): ReactNode {
  const [epsilon, setEpsilon] = useState(0.15)
  const [episodes, setEpisodes] = useState(150)
  const [seed, setSeed] = useState(1)
  const record = useMemo(() => {
    const u = uniforms(seed, episodes * 80 * 2)
    const q = Array.from({ length: ROWS * COLS }, () => [0, 0, 0, 0])
    const returns: number[] = []
    const lengths: number[] = []
    const snapshots: { episode: number; values: number[]; policy: number[] }[] = []
    let r = 0
    for (let e = 0; e < episodes; e++) {
      let s = (ROWS - 1) * COLS
      let total = 0
      let len = 0
      while (len < 80) {
        const qs = q[s] ?? [0, 0, 0, 0]
        const a = (u[r++] ?? 0) < epsilon ? Math.floor((u[r++] ?? 0) * 4) : qs.indexOf(Math.max(...qs))
        const [dr = 0, dc = 0] = MOVES[a] ?? [0, 0]
        const row = Math.min(ROWS - 1, Math.max(0, Math.floor(s / COLS) + dr))
        const col = Math.min(COLS - 1, Math.max(0, (s % COLS) + dc))
        const n = row * COLS + col
        const reward = n === GOAL ? 1 : n === PIT ? -1 : -0.02
        const done = n === GOAL || n === PIT
        const best = done ? 0 : Math.max(...(q[n] ?? [0]))
        const qa = q[s] as number[]
        qa[a] = (qa[a] ?? 0) + 0.3 * (reward + 0.95 * best - (qa[a] ?? 0))
        total += reward
        len++
        s = n
        if (done) break
      }
      returns.push(total)
      lengths.push(len)
      if (e % Math.max(1, Math.floor(episodes / 6)) === 0 || e === episodes - 1)
        snapshots.push({
          episode: e,
          values: q.map((v) => Math.max(...v)),
          policy: q.map((v) => v.indexOf(Math.max(...v))),
        })
    }
    return episodeRecord({
      environment: 'cliff grid',
      agent: 'Q-learning',
      returns,
      lengths,
      snapshotShape: [ROWS, COLS],
      actionNames: ['up', 'down', 'left', 'right'],
      snapshots,
    })
  }, [epsilon, episodes, seed])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField label="exploration rate" value={epsilon} onChange={setEpsilon} min={0} max={1} step={0.05} />
        <NumberField label="episodes" value={episodes} onChange={setEpisodes} min={20} max={600} step={10} integer />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <ExpositionView record={record} title="Q-learning on a small grid" />
    </div>
  )
}
