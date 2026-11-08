import type { Point2 } from 'omai-engine/numerics/geometry'
import type { Algorithm } from 'omai-engine/trace'
import { cartesian, lines, points } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import { Chart, ControlPanel, FigureFrame, NumberField, Player, useColourMode, useLiveRun } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Live algorithm',
  question: 'How do I animate an algorithm as it runs, and still step back through what it did?',
  explain:
    '`useLiveRun(algorithm, { maxSteps })` steps an engine `Algorithm` as the player plays (one step per frame, or `stepsPerFrame`), keeping the states so the reader can step back, scrub and play on into new steps. Here: gradient descent with momentum on an elongated bowl; change the momentum and the run starts afresh.',
}

const FIGURE = figure()
// f(x, y) = (x² + 10 y²) / 2, whose gradient is (x, 10 y).
const levels = [0.5, 2, 5, 10, 18].map((c) =>
  Array.from({ length: 121 }, (_, i): Point2 => {
    const t = (2 * Math.PI * i) / 120
    return [Math.sqrt(2 * c) * Math.cos(t), Math.sqrt((2 * c) / 10) * Math.sin(t)]
  }),
)

interface State {
  readonly at: Point2
  readonly velocity: Point2
  readonly path: readonly Point2[]
}

function descent(rate: number, momentum: number): Algorithm<State> {
  return {
    name: 'momentum descent',
    config: { rate, momentum },
    init: () => ({ at: [-3.5, 2], velocity: [0, 0], path: [[-3.5, 2]] }),
    step: ({ at, velocity, path }) => {
      const v: Point2 = [momentum * velocity[0] - rate * at[0], momentum * velocity[1] - rate * 10 * at[1]]
      const next: Point2 = [at[0] + v[0], at[1] + v[1]]
      return { at: next, velocity: v, path: [...path, next] }
    },
    done: ({ at }) => (Math.hypot(at[0], at[1]) < 1e-3 ? 'converged' : undefined),
  }
}

export default function LiveAlgorithm(): ReactNode {
  const [momentum, setMomentum] = useState(0.8)
  const algorithm = useMemo(() => descent(0.05, momentum), [momentum])
  const run = useLiveRun(algorithm, { maxSteps: 400, fps: 15 })
  const mode = useColourMode()
  const { path, at } = run.state
  const option = useMemo(
    () =>
      cartesian(
        [
          lines(
            [
              ...levels.map((l) => ({ x: l.map((p) => p[0]), y: l.map((p) => p[1]), colour: 'grid', width: 1 })),
              { x: path.map((p) => p[0]), y: path.map((p) => p[1]), name: 'path', colour: 'tab1' },
            ],
            { mode },
          ),
          points([{ x: [[...at]], name: 'now', colour: 'tab1', size: 10 }], { mode }),
        ],
        {
          mode,
          figure: FIGURE,
          legend: false,
          tooltip: false,
          x: { label: 'x', limits: [-4, 4] },
          y: { label: 'y', limits: [-2.5, 2.5] },
        },
      ),
    [path, at, mode],
  )
  return (
    <div className="flex w-[640px] max-w-full flex-col gap-3">
      <ControlPanel>
        <NumberField label="momentum" value={momentum} onChange={setMomentum} min={0} max={0.95} step={0.05} />
      </ControlPanel>
      <FigureFrame figure={FIGURE} playback={run.controller} filename="momentum-descent">
        <Chart option={option} />
      </FigureFrame>
      <Player
        controller={run.controller}
        stamp={(f) =>
          `step ${run.t}${run.status === undefined || f < run.controller.count - 1 ? '' : ` (${run.status})`}`
        }
      />
    </div>
  )
}
