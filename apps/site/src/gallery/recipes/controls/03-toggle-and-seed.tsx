import { type ReactNode, useMemo, useState } from 'react'
import { key, normal } from 'whitebox-engine/random'
import { toArray } from 'whitebox-engine/tensor'
import { ControlPanel, SeedInput, Toggle } from 'whitebox-render/react'
import { Dots, Line, Plot, plotFrame } from '../../demo/Plot.tsx'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Toggle and seed',
  question: 'How do I switch something on and off, and let the reader redraw random data reproducibly?',
  explain:
    '`Toggle` is a switch with its label. `SeedInput` is a whole-number seed with a die that rerolls it; the new seed is derived from the old one by a key, so a sequence of rerolls is reproducible too. Random draws come from `key(seed)` in whitebox-engine.',
}

const frame = plotFrame({ x: [0, 100], y: [-25, 25] })

export default function ToggleAndSeed(): ReactNode {
  const [seed, setSeed] = useState(1)
  const [path, setPath] = useState(true)
  const [dots, setDots] = useState(false)
  const walk = useMemo(() => {
    const steps = toArray(normal(key(seed), [100])) as number[]
    let y = 0
    return steps.map((s, t): [number, number] => {
      y += s
      return [t, y]
    })
  }, [seed])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <SeedInput value={seed} onChange={setSeed} />
        <Toggle label="draw the path" value={path} onChange={setPath} />
        <Toggle label="draw the steps" value={dots} onChange={setDots} />
      </ControlPanel>
      <Plot frame={frame} xLabel="step" yLabel="position">
        {path ? <Line frame={frame} points={walk} /> : null}
        {dots ? <Dots frame={frame} points={walk} colour="var(--wb-tab1)" /> : null}
      </Plot>
    </div>
  )
}
