import { key, normal } from 'mdml-engine/random'
import { toArray } from 'mdml-engine/tensor'
import { lines, type Panel, points, subplots } from 'mdml-render/charts'
import { figure, gridSpec } from 'mdml-render/figure'
import { Chart, ControlPanel, FigureFrame, NumberField, SeedInput, Toggle, useColourMode } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Small multiples',
  question: 'How do I draw one small chart per group, all on the same axes?',
  explain:
    'A grid of panels from `gridSpec(rows, cols)` filled by `subplots(grid, panels)`, every panel given the same limits, so the eye compares shapes rather than reading scales. Untick shared limits to see how much harder that gets when each panel fits its own data.',
}

const FIGURE = figure({ width: 860, height: 560 })
const REGIONS = ['north', 'south', 'east', 'west', 'centre', 'coast', 'hills', 'islands', 'city']

export default function SmallMultiples(): ReactNode {
  const mode = useColourMode()
  const [shared, setShared] = useState(true)
  const [noise, setNoise] = useState(0.4)
  const [seed, setSeed] = useState(0)
  const option = useMemo(() => {
    const x = Array.from({ length: 40 }, (_, i) => i)
    const grid = gridSpec(3, 3, { wspace: 36, hspace: 48 })
    const panels: Record<string, Panel> = {}
    REGIONS.forEach((name, k) => {
      const z = toArray(normal(key(seed * 10 + k), [40])) as number[]
      const trend = (k - 4) * 0.08
      const y = x.map((t, i) => 5 + trend * t + Math.sin(t / (3 + k)) * (1 + k / 4) + noise * (z[i] ?? 0))
      panels[`r${Math.floor(k / 3)}c${k % 3}`] = {
        layers: [
          lines([{ x, y, name, width: 1.5 }], { mode, colourOffset: k }),
          points([{ x: [39], y: [y[39] ?? 0], name: '__end', size: 5 }], { mode, colourOffset: k }),
        ],
        options: {
          title: name,
          x: { label: '', limits: [0, 39] },
          y: { label: '', ...(shared ? { limits: [0, 14] as [number, number] } : {}) },
        },
      }
    })
    return subplots(grid, panels, { figure: { ...FIGURE, mode }, legend: false })
  }, [shared, noise, seed, mode])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <Toggle label="shared limits" value={shared} onChange={setShared} />
        <NumberField label="noise" value={noise} onChange={setNoise} min={0} max={2} step={0.1} />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <FigureFrame figure={FIGURE} filename="small-multiples">
        <Chart option={option} figure={FIGURE} />
      </FigureFrame>
    </div>
  )
}
