import { type ReactNode, useMemo, useState } from 'react'
import { key, normal } from 'whitebox-engine/random'
import { toArray } from 'whitebox-engine/tensor'
import { lineChart } from 'whitebox-render/charts'
import { figure } from 'whitebox-render/figure'
import { Chart, ControlPanel, SeedInput, Toggle, useColourMode } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Toggle and seed',
  question: 'How do I switch something on and off, and let the reader redraw random data reproducibly?',
  explain:
    '`Toggle` is a switch with its label. `SeedInput` is a whole-number seed with a die that rerolls it; the new seed is derived from the old one by a key, so a sequence of rerolls is reproducible too. Random draws come from `key(seed)` in whitebox-engine.',
}

const FIGURE = figure()

export default function ToggleAndSeed(): ReactNode {
  const mode = useColourMode()
  const [seed, setSeed] = useState(1)
  const [markers, setMarkers] = useState(false)
  const [steps, setSteps] = useState(false)
  const option = useMemo(() => {
    const increments = toArray(normal(key(seed), [100])) as number[]
    let y = 0
    const walk = increments.map((s) => {
      y += s
      return y
    })
    return lineChart([{ y: walk, name: 'random walk', markers, ...(steps ? { step: 'end' as const } : {}) }], {
      mode,
      figure: FIGURE,
      x: { label: 'step' },
      y: { label: 'position', limits: [-25, 25] },
    })
  }, [seed, markers, steps, mode])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <SeedInput value={seed} onChange={setSeed} />
        <Toggle label="markers" value={markers} onChange={setMarkers} />
        <Toggle label="draw as steps" value={steps} onChange={setSteps} />
      </ControlPanel>
      <Chart option={option} figure={FIGURE} />
    </div>
  )
}
