import { type ReactNode, useMemo, useState } from 'react'
import { iris, wine } from 'whitebox-engine/datasets'
import { toArray } from 'whitebox-engine/tensor'
import { type AxisBrush, brushFromAxisEvent, brushRows, parallelCoordinates, type Table } from 'whitebox-render/charts'
import { figure } from 'whitebox-render/figure'
import { Chart, Choice, ControlPanel, FigureFrame, useColourMode } from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Parallel coordinates',
  question: 'How do I compare many columns at once, and brush a range on any of them?',
  explain:
    '`parallelCoordinates` draws one axis per column and one line per row, coloured by class or along a column. Drag along an axis to brush it: `brushFromAxisEvent` turns the chart’s `axisareaselected` event into a brush, the lines outside fade, and `brushRows` gives the rows kept. Click an axis to clear its interval.',
}

const FIG = figure({ width: 720, aspect: 2 })

function tableOf(name: 'iris' | 'wine'): Table {
  const data = name === 'iris' ? iris() : wine()
  const keep = name === 'iris' ? [0, 1, 2, 3] : [0, 1, 4, 5, 6, 9, 11, 12]
  const rows = (toArray(data.x) as number[][]).map((r) => keep.map((j) => r[j] ?? 0))
  return {
    columns: keep.map((j) => (data.featureNames[j] ?? '').replace(/ \(cm\)$/, '').replace(/_/g, ' ')),
    rows,
    groups: Array.from(data.y.data),
    groupNames: Object.fromEntries(data.classes.map((c, i) => [i, c.replace(/_/g, ' ')])),
  }
}

export default function ParallelRecipe(): ReactNode {
  const mode = useColourMode()
  const [name, setName] = useState<'iris' | 'wine'>('iris')
  const [colour, setColour] = useState<'class' | 'last column'>('class')
  const [brush, setBrush] = useState<AxisBrush>({ 'petal length': [3, 5.2] })
  const data = useMemo(() => tableOf(name), [name])
  const kept = brushRows(data, brush).length
  const events = useMemo(() => ({ axisareaselected: (e: unknown) => setBrush((b) => brushFromAxisEvent(b, e)) }), [])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <Choice
          label="data"
          value={name}
          options={['iris', 'wine'] as const}
          onChange={(v) => {
            setName(v)
            setBrush({})
          }}
        />
        <Choice label="colour by" value={colour} options={['class', 'last column'] as const} onChange={setColour} />
      </ControlPanel>
      <FigureFrame
        figure={FIG}
        filename="parallel-coordinates"
        caption={`${kept} of ${data.rows.length} rows inside the brush. Drag along an axis to brush it; click an axis to clear it.`}
      >
        {(size) => (
          <Chart
            option={parallelCoordinates(data, {
              mode,
              figure: figure({ width: size.width, height: size.height }),
              brush,
              ...(colour === 'class' ? {} : { colourBy: data.columns[data.columns.length - 1] ?? '' }),
            })}
            onEvents={events}
          />
        )}
      </FigureFrame>
    </div>
  )
}
