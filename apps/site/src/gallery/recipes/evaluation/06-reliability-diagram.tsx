import { calibrationCurve } from 'mdml-engine/metrics'
import { key, uniform } from 'mdml-engine/random'
import { toArray } from 'mdml-engine/tensor'
import { calibrationChart } from 'mdml-render/charts'
import { figure } from 'mdml-render/figure'
import { Chart, ControlPanel, FigureFrame, NumberField, SeedInput, useColourMode } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Reliability diagram',
  question: 'How do I show whether predicted probabilities can be taken at face value?',
  explain:
    "`calibrationCurve(y, p, { nBins })` bins the forecasts and compares each bin's mean forecast with how often the event happened; `calibrationChart` draws it against the diagonal, points sized by count. The toy forecaster here knows the true probability but reports it through a temperature: below 1 it is overconfident (an S below and above the diagonal), above 1 underconfident.",
}

const FIGURE = figure({ width: 520, height: 520 })
const logit = (p: number): number => Math.log(p / (1 - p))
const sigmoid = (z: number): number => 1 / (1 + Math.exp(-z))

export default function ReliabilityDiagram(): ReactNode {
  const mode = useColourMode()
  const [temperature, setTemperature] = useState(0.6)
  const [bins, setBins] = useState(10)
  const [seed, setSeed] = useState(0)
  const option = useMemo(() => {
    const n = 4000
    const truth = toArray(uniform(key(seed), [n])) as number[]
    const u = toArray(uniform(key(seed + 1), [n])) as number[]
    const p = truth.map((q) => Math.min(0.97, Math.max(0.03, q)))
    const y = p.map((q, i) => ((u[i] ?? 1) < q ? 1 : 0))
    const reported = p.map((q) => sigmoid(logit(q) / temperature))
    return calibrationChart(
      [
        { name: 'calibrated', curve: calibrationCurve(y, p, { nBins: bins }) },
        { name: `temperature ${temperature}`, curve: calibrationCurve(y, reported, { nBins: bins }) },
      ],
      { mode },
    )
  }, [temperature, bins, seed, mode])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField
          label="temperature"
          value={temperature}
          onChange={setTemperature}
          min={0.2}
          max={3}
          step={0.05}
          log
        />
        <NumberField label="bins" value={bins} onChange={setBins} min={3} max={30} integer />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <FigureFrame figure={FIGURE} filename="reliability-diagram">
        <Chart option={option} figure={FIGURE} />
      </FigureFrame>
    </div>
  )
}
