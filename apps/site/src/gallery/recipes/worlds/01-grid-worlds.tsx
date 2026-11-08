import {
  aimaGridWorld,
  cliffWalkingWorld,
  distanceField,
  exits,
  fourRoomsWorld,
  frozenLakeWorld,
  greedyPath,
  keyMazeWorld,
  taxiWorld,
  type World,
  windyGridWorld,
} from 'omai-engine/worlds'
import { worldView } from 'omai-render/gym'
import { Choice, ControlPanel, SceneFrame, Select, Toggle } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Grid worlds',
  question: 'How do I draw a grid world, with values and a policy on its cells?',
  explain:
    '`worldView(world, options)` draws any `worlds` world from above (terrain, walls, items, starts and goals) and lays on what an agent or a search knows: numbers per cell (`labels`), values shaded (`values`), a policy as arrows (`policy`, one action or a distribution per cell). `valueStyle` chooses how the values are drawn: one paint (more opaque the larger), or a function that returns a paint per value, here a signed ramp on a fixed `range`; one NaN no longer switches the shading off. `greedyPath` climbs the values from a cell, and its cells go to `path`. Here the overlays come from the distance to the goal, which the engine computes for any world.',
}

const WORLDS: Readonly<Record<string, () => World>> = {
  'Frozen lake (8 by 8)': () => frozenLakeWorld({ mapName: '8x8' }).world,
  'Cliff walking': () => cliffWalkingWorld().world,
  'AIMA 4 by 3': () => aimaGridWorld().world,
  'Four rooms': () => fourRoomsWorld().world,
  'Windy grid': () => windyGridWorld().world,
  'Key maze': () => keyMazeWorld().world,
  Taxi: () => taxiWorld().world,
}

export default function GridWorlds(): ReactNode {
  const [name, setName] = useState('Four rooms')
  const [overlay, setOverlay] = useState<'none' | 'distances' | 'policy'>('policy')
  const [shade, setShade] = useState(true)
  const [signed, setSigned] = useState(false)
  const [greedy, setGreedy] = useState(true)
  const view = useMemo(() => {
    const w = (WORLDS[name] ?? (() => fourRoomsWorld().world))()
    const goals = w.goals.length > 0 ? w.goals : [w.n - 1]
    const dist = distanceField(w, goals)
    // Head for the neighbour nearest the goal: a policy read straight off the distances.
    const policy = Array.from({ length: w.n }, (_, cell) => {
      let best = -1
      let bestD = dist[cell] ?? Number.POSITIVE_INFINITY
      for (const [direction, to] of exits(w, cell)) {
        const d = dist[to] ?? Number.POSITIVE_INFINITY
        if (d < bestD) {
          bestD = d
          best = direction
        }
      }
      return best
    })
    const finite = Array.from(dist, (d) => (Number.isFinite(d) ? d : Number.NaN))
    // Unreachable cells (walls) are NaN, which the shading skips.
    const closeness = finite.map((d) => (Number.isFinite(d) ? 1 / (1 + d) : Number.NaN))
    const reachable = closeness.filter(Number.isFinite)
    const mean = reachable.reduce((a, b) => a + b, 0) / Math.max(1, reachable.length)
    const start = w.starts[0] ?? 0
    return worldView(w, {
      cell: 34,
      ...(greedy ? { path: greedyPath(w, closeness, start) } : {}),
      ...(overlay === 'distances' ? { labels: finite } : {}),
      ...(overlay === 'policy' ? { policy } : {}),
      ...(shade
        ? signed
          ? {
              values: closeness.map((v) => v - mean),
              valueStyle: {
                fill: (v: number) => (v < 0 ? ('tab3' as const) : ('tab0' as const)),
                range: [-1, 1] as const,
                maxOpacity: 0.5,
              },
            }
          : { values: closeness }
        : {}),
    })
  }, [name, overlay, shade, signed, greedy])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <Select label="world" value={name} onChange={setName} options={Object.keys(WORLDS)} />
        <Choice label="overlay" value={overlay} onChange={setOverlay} options={['none', 'distances', 'policy']} />
        <Toggle label="shade by closeness to the goal" value={shade} onChange={setShade} />
        <Toggle label="signed colours (above and below the mean)" value={signed} onChange={setSigned} />
        <Toggle label="greedy path from the start" value={greedy} onChange={setGreedy} />
      </ControlPanel>
      <SceneFrame scene={view} filename="grid-world" />
    </div>
  )
}
