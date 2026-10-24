import { type ReactNode, useMemo } from 'react'
import { Player, type Point2, usePlayback } from 'whitebox-render/react'
import { Dots, Line, Plot, plotFrame, sampleCurve } from '../../demo/Plot.tsx'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Player',
  question: 'How do I play precomputed frames: play, pause, step and scrub?',
  explain:
    '`usePlayback(count)` is the playhead of a precomputed sequence; `Player` drives it: step back, play or pause with a caret menu for the target frames per second, step forward, a scrubber to drag, and an optional time stamp. Space and the arrow keys work while the player has focus. Here each frame is one position of a point travelling a Lissajous curve.',
}

const frame = plotFrame({ x: [-1.2, 1.2], y: [-1.2, 1.2], width: 420, aspect: 1.2 })
const N = 240
const at = (i: number): Point2 => {
  const t = (2 * Math.PI * i) / N
  return [Math.sin(3 * t), Math.sin(2 * t)]
}

export default function PlayerRecipe(): ReactNode {
  const playback = usePlayback(N, { fps: 30, loop: true })
  const path = useMemo(() => sampleCurve((i) => i, 0, N, N + 1).map(([i]) => at(i)), [])
  const trail = Array.from({ length: 30 }, (_, k) => at(playback.frame - k))
  return (
    <div className="flex w-fit max-w-full flex-col gap-3">
      <Plot frame={frame}>
        <Line frame={frame} points={path} colour="var(--wb-grid)" width={1} />
        <Line frame={frame} points={trail} colour="var(--wb-tab1)" width={2} />
        <Dots frame={frame} points={[at(playback.frame)]} radius={6} colour="var(--wb-tab1)" opacity={1} />
      </Plot>
      <Player controller={playback} stamp stampFps={30} />
    </div>
  )
}
