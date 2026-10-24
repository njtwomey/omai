import { type ReactNode, useMemo, useState } from 'react'
import { scatterChart } from 'whitebox-render/charts'
import { figure } from 'whitebox-render/figure'
import {
  Chart,
  ControlPanel,
  DatasetPicker,
  selectEntry,
  Toggle,
  useColourMode,
  useDataset,
} from 'whitebox-render/react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Scatter plot',
  question: 'How do I draw points coloured by group, with an equal data aspect?',
  explain:
    '`scatterChart(sets, options)` takes point sets `{ x, y, groups }` (or `[n, 2]` points as `x`); each group becomes a series with its own colour, and `shapes` varies the marker too. `aspect: "equal"` makes a unit the same length on both axes, widening a range to fit the figure.',
}

const COLLECTION = 'classification2d'

// The fixed figure the chart fills (the builders' default size).
const FIGURE = figure()

export default function Scatter(): ReactNode {
  const mode = useColourMode()
  const [selection, setSelection] = useState(() => selectEntry(COLLECTION, 'blobs'))
  const [shapes, setShapes] = useState(false)
  const { data } = useDataset(COLLECTION, selection)
  const option = useMemo(() => {
    if (!('x' in data && 'y' in data)) return {}
    return scatterChart([{ x: data.x, groups: Array.from(data.y.data as ArrayLike<number>), shapes }], {
      mode,
      aspect: 'equal',
      x: { label: 'x₁' },
      y: { label: 'x₂' },
    })
  }, [data, shapes, mode])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <DatasetPicker collection={COLLECTION} value={selection} onChange={setSelection} knobs={false} />
        <Toggle label="marker shapes by group" value={shapes} onChange={setShapes} />
      </ControlPanel>
      <Chart option={option} figure={FIGURE} />
    </div>
  )
}
