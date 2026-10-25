import { cartesian, type StackOffset, stackedAreas } from 'mdml-render/charts'
import { figure } from 'mdml-render/figure'
import { Chart, Choice, ControlPanel, FigureFrame, NumberField, useColourMode } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Stacked areas and streamgraphs',
  question: 'How do I stack series to show both the parts and the total over time?',
  explain:
    '`stackedAreas(x, series, { offset })` stacks series as bands: from zero (the total is the top edge), expanded to shares of the total, or about a centred, wiggle-minimising baseline (a streamgraph, which flatters the shapes and hides the total). The toy series are listening hours by genre over a year.',
}

const FIGURE = figure({ width: 820, height: 420 })
const GENRES = ['folk', 'jazz', 'rock', 'electronic', 'classical', 'hip hop']

export default function StackedAreas(): ReactNode {
  const mode = useColourMode()
  const [offset, setOffset] = useState<StackOffset>('wiggle')
  const [seasonality, setSeasonality] = useState(1)
  const option = useMemo(() => {
    const x = Array.from({ length: 53 }, (_, w) => w)
    const series = GENRES.map((name, g) => ({
      name,
      y: x.map((w) => {
        const peak = (g * 9 + 4) % 52
        return Math.max(0, 2 + 6 * seasonality * Math.exp(-(((w - peak) / 7) ** 2)) + 1.5 * Math.sin((w + g * 5) / 4))
      }),
    }))
    return cartesian([stackedAreas(x, series, { offset, mode })], {
      mode,
      x: { label: 'week', limits: [0, 52] },
      y: {
        label: offset === 'expand' ? 'share of listening' : 'hours',
        ...(offset === 'expand' ? { limits: [0, 1] } : {}),
      },
    })
  }, [offset, seasonality, mode])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <Choice
          label="offset"
          value={offset}
          onChange={setOffset}
          options={['zero', 'expand', 'wiggle'] as StackOffset[]}
        />
        <NumberField label="seasonality" value={seasonality} onChange={setSeasonality} min={0} max={3} step={0.1} />
      </ControlPanel>
      <FigureFrame figure={FIGURE} filename="stacked-areas">
        <Chart option={option} figure={FIGURE} />
      </FigureFrame>
    </div>
  )
}
