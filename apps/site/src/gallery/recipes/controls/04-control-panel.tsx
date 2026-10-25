import { lineChart } from 'mdml-render/charts'
import { figure } from 'mdml-render/figure'
import {
  Chart,
  Choice,
  ControlLayout,
  ControlPanel,
  type ControlPlacement,
  FigureFrame,
  NumberField,
  PanelArea,
  Toggle,
  useColourMode,
} from 'mdml-render/react'
import { type ReactNode, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Control panel',
  question: 'How do I lay out a figure’s controls: in a sidebar, a bar above, or a bento grid?',
  explain:
    '`ControlPanel` lays controls in `columns`, in `rows`, or in named `areas` (one string per row, as CSS grid areas, with a `PanelArea` per name); a `title` makes it collapsible. `ControlLayout` puts the panel `left`, `right`, `top` or `bottom` of the figure, which keeps its own size.',
}

const FIGURE = figure({ width: 520 })
const ts = Array.from({ length: 401 }, (_, i) => (4 * Math.PI * i) / 400)

export default function ControlPanelRecipe(): ReactNode {
  const [placement, setPlacement] = useState<ControlPlacement>('left')
  const [amplitude, setAmplitude] = useState(1)
  const [frequency, setFrequency] = useState(1)
  const [phase, setPhase] = useState(0)
  const [damped, setDamped] = useState(false)
  const f = (t: number): number => amplitude * Math.sin(frequency * t + phase) * (damped ? Math.exp(-t / 6) : 1)
  const mode = useColourMode()
  const option = lineChart([{ x: ts, y: ts.map(f), name: 'wave' }], {
    mode,
    figure: FIGURE,
    x: { label: 't' },
    y: { label: 'y', limits: [-2.5, 2.5] },
  })
  const side = placement === 'left' || placement === 'right'
  return (
    <ControlLayout
      placement={placement}
      sidebarWidth={280}
      controls={
        <ControlPanel
          title="wave"
          {...(side ? { columns: 1 } : { areas: ['place place amp freq', 'place place phase damp'] })}
        >
          <PanelArea name="place">
            <Choice
              label="panel placement"
              value={placement}
              onChange={setPlacement}
              options={['left', 'top', 'right', 'bottom']}
            />
          </PanelArea>
          <PanelArea name="amp">
            <NumberField label="amplitude" value={amplitude} onChange={setAmplitude} min={0} max={2} />
          </PanelArea>
          <PanelArea name="freq">
            <NumberField label="frequency" value={frequency} onChange={setFrequency} min={0.1} max={5} log />
          </PanelArea>
          <PanelArea name="phase">
            <NumberField label="phase" value={phase} onChange={setPhase} min={-Math.PI} max={Math.PI} />
          </PanelArea>
          <PanelArea name="damp">
            <Toggle label="damped" value={damped} onChange={setDamped} />
          </PanelArea>
        </ControlPanel>
      }
    >
      <FigureFrame figure={FIGURE} filename="wave">
        <Chart option={option} />
      </FigureFrame>
    </ControlLayout>
  )
}
