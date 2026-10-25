import { iris, wine } from 'mdml-engine/datasets'
import { toArray } from 'mdml-engine/tensor'
import { type AxisBrush, brushFromAxisEvent, brushRows, parallelCoordinates, type Table } from 'mdml-render/charts'
import { figure } from 'mdml-render/figure'
import { Chart, Choice, ColumnOrder, ControlPanel, FigureFrame, useColourMode } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Parallel coordinates',
  question: 'How do I compare many columns at once, and brush a range on any of them?',
  explain:
    '`parallelCoordinates` draws one axis per column and one line per row, coloured by class or along a column. Drag along an axis to brush it: `brushFromAxisEvent` turns the chart’s `axisareaselected` event into a brush, the lines outside fade, and `brushRows` gives the rows kept. Click an axis to clear its interval, or clear them all. Reorder the axes by dragging their names (`order`), and flip any axis with its ↕ (`inverted`): neighbouring axes show their relationships most clearly, and a flip turns a crossing into parallels.',
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
  const [brush, setBrush] = useState<AxisBrush>({})
  const data = useMemo(() => tableOf(name), [name])
  const [order, setOrder] = useState<readonly string[]>(data.columns)
  const [inverted, setInverted] = useState<readonly string[]>([])
  const brushed = Object.keys(brush).length > 0
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
            setOrder(tableOf(v).columns)
            setInverted([])
          }}
        />
        <Choice label="colour by" value={colour} options={['class', 'last column'] as const} onChange={setColour} />
      </ControlPanel>
      <div className="flex flex-wrap items-end justify-between gap-3">
        <ColumnOrder
          label="axes (drag to reorder, ↕ to flip)"
          order={order}
          onOrder={setOrder}
          inverted={inverted}
          onInvert={setInverted}
        />
        <button
          type="button"
          disabled={!brushed}
          onClick={() => setBrush({})}
          className="rounded-md border border-(--wb-grid) px-2.5 py-1 text-xs text-(--wb-fg) hover:border-(--wb-accent) disabled:opacity-40"
        >
          clear selection
        </button>
      </div>
      <FigureFrame
        figure={FIG}
        filename="parallel-coordinates"
        caption={
          brushed
            ? `${kept} of ${data.rows.length} rows inside the selection. Click an axis to clear its interval.`
            : `${data.rows.length} rows. Drag along an axis to select a range on it.`
        }
      >
        {(size) => (
          <Chart
            option={parallelCoordinates(data, {
              mode,
              figure: figure({ width: size.width, height: size.height }),
              brush,
              order,
              inverted,
              ...(colour === 'class' ? {} : { colourBy: data.columns[data.columns.length - 1] ?? '' }),
            })}
            onEvents={events}
          />
        )}
      </FigureFrame>
    </div>
  )
}
