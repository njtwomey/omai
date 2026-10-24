import { type ReactNode, useMemo, useState } from 'react'
import type { Algorithm } from 'whitebox-engine/trace'
import { ControlPanel, NumberField, Player, type Point2, useLiveRun } from 'whitebox-render/react'
import { Dots, Line, Plot, plotFrame } from '../../demo/Plot.tsx'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Live algorithm',
  question: 'How do I animate an algorithm as it runs, and still step back through what it did?',
  explain:
    '`useLiveRun(algorithm, { maxSteps })` steps an engine `Algorithm` as the player plays (one step per frame, or `stepsPerFrame`), keeping the states so the reader can step back, scrub and play on into new steps. Here: gradient descent with momentum on an elongated bowl; change the momentum and the run starts afresh.',
}

const frame = plotFrame({ x: [-4, 4], y: [-2.5, 2.5] })
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
  return (
    <div className="flex w-fit max-w-full flex-col gap-3">
      <ControlPanel>
        <NumberField label="momentum" value={momentum} onChange={setMomentum} min={0} max={0.95} step={0.05} />
      </ControlPanel>
      <Plot frame={frame} xLabel="x" yLabel="y">
        {levels.map((l, i) => (
          // biome-ignore lint/suspicious/noArrayIndexKey: fixed contour levels
          <Line key={i} frame={frame} points={l} colour="var(--wb-grid)" width={1} />
        ))}
        <Line frame={frame} points={run.state.path} colour="var(--wb-tab1)" />
        <Dots frame={frame} points={[run.state.at]} radius={5} colour="var(--wb-tab1)" opacity={1} />
      </Plot>
      <Player
        controller={run.controller}
        stamp={(f) =>
          `step ${run.t}${run.status === undefined || f < run.controller.count - 1 ? '' : ` (${run.status})`}`
        }
      />
    </div>
  )
}
