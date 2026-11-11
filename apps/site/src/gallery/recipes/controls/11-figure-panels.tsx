import { lineChart } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import { Chart, FigurePanel, PanelsStrip, useColourMode, usePanels } from 'omai-render/react'
import type { ReactNode } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Figure panels',
  question: 'How do I let a viewer fold or hide the figures of a page, and bring them back?',
  explain:
    '`usePanels` keeps which panels are open, folded or hidden (remembered per page in `localStorage`); `FigurePanel` is one panel with a chevron that folds it and a button that hides it; `PanelsStrip` lists them all with a switch each. `panels.active(name)` is true only while a panel is open, so a page asks for a heavy figure’s data only then.',
}

const FIGURE = figure({ width: 420, height: 220 })
const xs = Array.from({ length: 101 }, (_, i) => (2 * Math.PI * i) / 100)

export default function FigurePanelsRecipe(): ReactNode {
  const panels = usePanels('gallery-figure-panels', { sine: 'Sine', cosine: 'Cosine', note: 'Note' })
  const mode = useColourMode()
  const draw = (f: (x: number) => number, name: string): ReactNode => (
    <Chart
      option={lineChart([{ x: xs, y: xs.map(f), name }], { mode, figure: FIGURE, x: { label: 'x' } })}
      figure={FIGURE}
    />
  )
  return (
    <div className="flex flex-col gap-3">
      <PanelsStrip panels={panels} />
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        <FigurePanel panels={panels} name="sine">
          {panels.active('sine') ? draw(Math.sin, 'sin') : null}
        </FigurePanel>
        <FigurePanel panels={panels} name="cosine">
          {draw(Math.cos, 'cos')}
        </FigurePanel>
      </div>
      <FigurePanel panels={panels} name="note">
        <p className="text-sm text-(--wb-muted)">A folded or hidden panel mounts none of its body.</p>
      </FigurePanel>
    </div>
  )
}
