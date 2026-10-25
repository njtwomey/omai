import { cartesian, lines } from 'mdml-render/charts'
import { figure } from 'mdml-render/figure'
import { Chart, Choice, ControlPanel, FigureFrame, useColourMode } from 'mdml-render/react'
import { categoricalColour, colourBlindCheck, simulateColourVision } from 'mdml-render/theme'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Colour vision',
  question: "How do I check that a chart's colours stay apart for readers with colour-vision deficiencies?",
  explain:
    '`simulateColourVision(colour, kind)` shows a colour as a reader with protanopia, deuteranopia or tritanopia sees it (Machado, Oliveira and Fernandes, 2009), and `colourBlindCheck(colours)` reports the smallest difference between any two colours under each deficiency and the pair that comes closest. The chart redraws in the simulated colours; the title names the closest pair. Red and green are the classic clash.',
}

const FIGURE = figure({ width: 820, height: 380 })
type Vision = 'normal' | 'protanopia' | 'deuteranopia' | 'tritanopia'

export default function ColourVision(): ReactNode {
  const mode = useColourMode()
  const [vision, setVision] = useState<Vision>('deuteranopia')
  const option = useMemo(() => {
    const colours = Array.from({ length: 6 }, (_, k) => categoricalColour(k))
    const shown = vision === 'normal' ? colours : colours.map((c) => simulateColourVision(c, vision))
    const report = colourBlindCheck(colours)
    const x = Array.from({ length: 60 }, (_, i) => i / 6)
    return cartesian(
      [
        lines(
          shown.map((colour, k) => ({
            x,
            y: x.map((t) => Math.sin(t + k * 0.6) + k * 0.5),
            name: `tab${k}`,
            colour,
            width: 3,
          })),
          { mode },
        ),
      ],
      {
        mode,
        title: report.safe
          ? `all six colours stay apart (smallest difference ${report.deuteranopia.toFixed(0)} under deuteranopia)`
          : `tab${report.worstPair[0]} and tab${report.worstPair[1]} come closest (difference ${Math.min(report.protanopia, report.deuteranopia, report.tritanopia).toFixed(0)})`,
        x: { label: 'x' },
        y: { label: 'y' },
      },
    )
  }, [vision, mode])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <Choice
          label="seen with"
          value={vision}
          onChange={setVision}
          options={['normal', 'protanopia', 'deuteranopia', 'tritanopia'] as Vision[]}
        />
      </ControlPanel>
      <FigureFrame figure={FIGURE} filename="colour-vision">
        <Chart option={option} figure={FIGURE} />
      </FigureFrame>
    </div>
  )
}
