import { cartesian, lines, points } from 'mdml-render/charts'
import { figure } from 'mdml-render/figure'
import { Chart, FigureFrame, Player, useColourMode, usePlayback } from 'mdml-render/react'
import { type ReactNode, useMemo } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Player',
  question: 'How do I play precomputed frames: play, pause, step and scrub?',
  explain:
    '`usePlayback(count)` is the playhead of a precomputed sequence; `Player` drives it: step back, play or pause with a caret menu for the target frames per second, step forward, a scrubber to drag, and an optional time stamp. Space and the arrow keys work while the player has focus. Given the `playback`, the figure frame’s export menu records the animation as a GIF (or a WebM where the browser encodes video). Here each frame is one position of a point travelling a Lissajous curve.',
}

const FIGURE = figure({ width: 480, aspect: 1.2 })
const N = 120
const at = (i: number): [number, number] => {
  const t = (2 * Math.PI * i) / N
  return [Math.sin(3 * t), Math.sin(2 * t)]
}
const path = Array.from({ length: N + 1 }, (_, i) => at(i))

export default function PlayerRecipe(): ReactNode {
  const mode = useColourMode()
  const playback = usePlayback(N, { fps: 30, loop: true })
  const option = useMemo(() => {
    const trail = Array.from({ length: 30 }, (_, k) => at(playback.frame - k))
    return cartesian(
      [
        lines(
          [
            { x: path.map((p) => p[0]), y: path.map((p) => p[1]), name: 'path', colour: 'grid', width: 1 },
            { x: trail.map((p) => p[0]), y: trail.map((p) => p[1]), name: 'trail', colour: 'tab1' },
          ],
          { mode },
        ),
        points([{ x: [at(playback.frame)], name: 'now', colour: 'tab1', size: 12 }], { mode }),
      ],
      { mode, figure: FIGURE, legend: false, tooltip: false, x: { limits: [-1.2, 1.2] }, y: { limits: [-1.2, 1.2] } },
    )
  }, [playback.frame, mode])
  return (
    <div className="flex w-[480px] max-w-full flex-col gap-3">
      <FigureFrame figure={FIGURE} playback={playback} filename="lissajous">
        <Chart option={option} />
      </FigureFrame>
      <Player controller={playback} stamp stampFps={30} />
    </div>
  )
}
