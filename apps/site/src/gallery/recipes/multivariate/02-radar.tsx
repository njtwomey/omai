import { type ReactNode, useMemo, useState } from 'react'
import { radarChart } from 'whitebox-render/charts'
import { figure } from 'whitebox-render/figure'
import { Chart, Choice, ControlPanel, FigureFrame, Select, Toggle, useColourMode } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Radar chart',
  question: 'How do I compare a few items on several scores at once?',
  explain:
    '`radarChart` puts one spoke per indicator (each with its own range) and one polygon per series, in tab10 order. `highlight` brings one series forward and fades the rest; `fill` and `shape` set the look.',
}

const FIG = figure({ width: 560, aspect: 1.25 })
const INDICATORS = [
  { name: 'accuracy', max: 1 },
  { name: 'speed', max: 10 },
  { name: 'memory', max: 10 },
  { name: 'interpretability', max: 10 },
  { name: 'robustness', max: 10 },
]
// Made-up profiles of three kinds of model.
const SERIES = [
  { name: 'linear model', values: [0.78, 9, 9, 9, 6] },
  { name: 'random forest', values: [0.9, 6, 4, 5, 8] },
  { name: 'neural network', values: [0.94, 4, 3, 2, 5] },
]

export default function RadarRecipe(): ReactNode {
  const mode = useColourMode()
  const [highlight, setHighlight] = useState('none')
  const [fill, setFill] = useState(true)
  const [shape, setShape] = useState<'polygon' | 'circle'>('polygon')
  const option = useMemo(
    () =>
      radarChart(SERIES, {
        indicators: INDICATORS,
        mode,
        figure: FIG,
        fill,
        shape,
        ...(highlight === 'none' ? {} : { highlight }),
      }),
    [mode, fill, shape, highlight],
  )
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <Select
          label="highlight"
          value={highlight}
          options={['none', ...SERIES.map((s) => s.name)]}
          onChange={setHighlight}
        />
        <Choice label="web" value={shape} options={['polygon', 'circle'] as const} onChange={setShape} />
        <Toggle label="fill" value={fill} onChange={setFill} />
      </ControlPanel>
      <FigureFrame figure={FIG} filename="radar">
        <Chart option={option} />
      </FigureFrame>
    </div>
  )
}
