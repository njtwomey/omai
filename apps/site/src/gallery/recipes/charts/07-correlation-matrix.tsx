import { normal } from 'mdml-engine/dists'
import { key } from 'mdml-engine/random'
import { corrcoef } from 'mdml-engine/stats'
import { toArray } from 'mdml-engine/tensor'
import { correlationChart } from 'mdml-render/charts'
import { figure } from 'mdml-render/figure'
import { Chart, ControlPanel, FigureFrame, NumberField, SeedInput, useColourMode } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Correlation matrix',
  question: 'How do I draw the correlations between many variables?',
  explain:
    '`correlationChart(corr, labels)` draws a correlation matrix on a diverging ramp centred on zero, each cell annotated. The toy variables share a hidden factor with loadings that fall away along the list, so correlations fade from the top-left corner; with few rows, sampling noise shows up as spurious small correlations.',
}

const FIGURE = figure({ width: 560, height: 520 })
const NAMES = ['height', 'weight', 'reach', 'shoe', 'grip', 'pace', 'sleep', 'tea']

export default function CorrelationMatrix(): ReactNode {
  const mode = useColourMode()
  const [rows, setRows] = useState(200)
  const [strength, setStrength] = useState(0.9)
  const [seed, setSeed] = useState(0)
  const option = useMemo(() => {
    const factor = toArray(normal(0, 1).sample(key(seed), [rows])) as number[]
    const vars = NAMES.map((_, j) => {
      const load = strength * (1 - j / NAMES.length)
      const e = toArray(normal(0, 1).sample(key(seed * 31 + j + 1), [rows])) as number[]
      return factor.map((f, i) => load * f + Math.sqrt(1 - load * load) * (e[i] ?? 0))
    })
    return correlationChart(corrcoef(vars), NAMES, { mode })
  }, [rows, strength, seed, mode])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField label="rows" value={rows} onChange={setRows} min={10} max={5000} integer log />
        <NumberField label="shared factor" value={strength} onChange={setStrength} min={0} max={0.99} step={0.01} />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <FigureFrame figure={FIGURE} filename="correlation-matrix">
        <Chart option={option} figure={FIGURE} />
      </FigureFrame>
    </div>
  )
}
