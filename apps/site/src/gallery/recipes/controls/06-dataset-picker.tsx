import { scatterChart } from 'mdml-render/charts'
import { figure } from 'mdml-render/figure'
import {
  Chart,
  ControlPanel,
  DatasetPicker,
  FigureFrame,
  selectEntry,
  useColourMode,
  useDataset,
} from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Dataset picker',
  question: 'How do I let the reader choose a toy dataset and turn its knobs?',
  explain:
    '`DatasetPicker` offers the entries of a dataset collection from mdml-engine (`classification2d` here), with arrows to step through them, a seed, and the chosen generator’s knobs as controls. `useDataset` generates the selection once per change; the same seed always gives the same data.',
}

const FIGURE = figure({ aspect: 1.25 })
const COLLECTION = 'classification2d'

export default function DatasetPickerRecipe(): ReactNode {
  const mode = useColourMode()
  const [selection, setSelection] = useState(() => selectEntry(COLLECTION, 'moons'))
  const { data } = useDataset(COLLECTION, selection)
  const option = useMemo(
    () =>
      'x' in data && 'y' in data
        ? scatterChart([{ x: data.x, groups: Array.from(data.y.data as ArrayLike<number>) }], {
            mode,
            figure: FIGURE,
            aspect: 'equal',
            x: { label: 'x₁' },
            y: { label: 'x₂' },
          })
        : {},
    [data, mode],
  )
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <DatasetPicker collection={COLLECTION} value={selection} onChange={setSelection} />
      </ControlPanel>
      <FigureFrame figure={FIGURE} filename="dataset">
        <Chart option={option} />
      </FigureFrame>
    </div>
  )
}
