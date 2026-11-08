import { type FlowLink, sankeyChart } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import { Chart, ControlPanel, FigureFrame, NumberField, useColourMode } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Sankey diagram',
  question: 'How do I show quantities flowing from sources through stages to uses?',
  explain:
    '`sankeyChart(links)` places the nodes in columns from the links and draws every flow as a band as wide as its value, coloured by its source; hover a node to highlight its flows. Toy energy flows: change the share of electricity from wind and follow it through to the uses.',
}

const FIGURE = figure({ width: 820, height: 480 })

export default function Sankey(): ReactNode {
  const mode = useColourMode()
  const [wind, setWind] = useState(30)
  const option = useMemo(() => {
    const electricity = 100
    const gas = 100 - wind
    const links: FlowLink[] = [
      { source: 'wind', target: 'electricity', value: wind },
      { source: 'gas', target: 'electricity', value: gas * 0.6 },
      { source: 'gas', target: 'heat', value: gas * 0.4 },
      { source: 'solar', target: 'electricity', value: 15 },
      { source: 'oil', target: 'transport', value: 60 },
      { source: 'electricity', target: 'homes', value: electricity * 0.4 },
      { source: 'electricity', target: 'industry', value: electricity * 0.35 },
      { source: 'electricity', target: 'losses', value: electricity * 0.25 + 15 - (100 - wind - gas * 0.6) },
      { source: 'heat', target: 'homes', value: gas * 0.4 },
      { source: 'transport', target: 'losses', value: 40 },
      { source: 'transport', target: 'travel', value: 20 },
    ].filter((l) => l.value > 0)
    return sankeyChart(links, { mode })
  }, [wind, mode])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <NumberField label="wind (of 100)" value={wind} onChange={setWind} min={0} max={100} integer />
      </ControlPanel>
      <FigureFrame figure={FIGURE} filename="sankey">
        <Chart option={option} figure={FIGURE} />
      </FigureFrame>
    </div>
  )
}
