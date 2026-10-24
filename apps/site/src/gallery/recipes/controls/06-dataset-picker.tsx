import { type ReactNode, useState } from 'react'
import { toArray } from 'whitebox-engine/tensor'
import { ControlPanel, DatasetPicker, selectEntry, useDataset } from 'whitebox-render/react'
import { Dots, Plot, plotFrame } from '../../demo/Plot.tsx'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Dataset picker',
  question: 'How do I let the reader choose a toy dataset and turn its knobs?',
  explain:
    '`DatasetPicker` offers the entries of a dataset collection from whitebox-engine (`classification2d` here), with arrows to step through them, a seed, and the chosen generator’s knobs as controls. `useDataset` generates the selection once per change; the same seed always gives the same data.',
}

const frame = plotFrame({ x: [-3, 3], y: [-3, 3], aspect: 1.25 })
const COLLECTION = 'classification2d'

export default function DatasetPickerRecipe(): ReactNode {
  const [selection, setSelection] = useState(() => selectEntry(COLLECTION, 'moons'))
  const { data } = useDataset(COLLECTION, selection)
  const x = 'x' in data ? (toArray(data.x) as number[][]) : []
  const y = 'y' in data ? (toArray(data.y) as number[]) : []
  const points = rescale(x)
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <DatasetPicker collection={COLLECTION} value={selection} onChange={setSelection} />
      </ControlPanel>
      <Plot frame={frame} xLabel="x₁ (standardised)" yLabel="x₂ (standardised)">
        <Dots frame={frame} points={points} colour={(i) => `var(--wb-tab${(y[i] ?? 0) % 10})`} />
      </Plot>
    </div>
  )
}

// Standardise each column so every dataset fits the same axes.
function rescale(rows: readonly number[][]): [number, number][] {
  const n = Math.max(rows.length, 1)
  const [mx, my] = [0, 1].map((j) => rows.reduce((a, r) => a + (r[j] ?? 0), 0) / n) as [number, number]
  const [sx, sy] = [0, 1].map(
    (j, k) => Math.sqrt(rows.reduce((a, r) => a + ((r[j] ?? 0) - (k === 0 ? mx : my)) ** 2, 0) / n) || 1,
  ) as [number, number]
  return rows.map((r) => [((r[0] ?? 0) - mx) / sx, ((r[1] ?? 0) - my) / sy])
}
