import { type ReactNode, useState } from 'react'
import {
  Choice,
  ControlLayout,
  ControlPanel,
  type ControlPlacement,
  NumberField,
  PanelArea,
  Toggle,
} from 'whitebox-render/react'
import { Line, Plot, plotFrame, sampleCurve } from '../../demo/Plot.tsx'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Control panel',
  question: 'How do I lay out a figure’s controls: in a sidebar, a bar above, or a bento grid?',
  explain:
    '`ControlPanel` lays controls in `columns`, in `rows`, or in named `areas` (one string per row, as CSS grid areas, with a `PanelArea` per name); a `title` makes it collapsible. `ControlLayout` puts the panel `left`, `right`, `top` or `bottom` of the figure, which keeps its own size.',
}

const frame = plotFrame({ x: [0, 4 * Math.PI], y: [-2.5, 2.5], width: 520 })

export default function ControlPanelRecipe(): ReactNode {
  const [placement, setPlacement] = useState<ControlPlacement>('left')
  const [amplitude, setAmplitude] = useState(1)
  const [frequency, setFrequency] = useState(1)
  const [phase, setPhase] = useState(0)
  const [damped, setDamped] = useState(false)
  const f = (t: number): number => amplitude * Math.sin(frequency * t + phase) * (damped ? Math.exp(-t / 6) : 1)
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
      <Plot frame={frame} xLabel="t" yLabel="y">
        <Line frame={frame} points={sampleCurve(f, 0, 4 * Math.PI, 400)} />
      </Plot>
    </ControlLayout>
  )
}
