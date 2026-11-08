import { type DatasetKind, rowCount } from 'omai-engine/data'
import { DATASET_MODALITIES, type DatasetModality, datasetRegistry } from 'omai-engine/datasets'
import { scatterChart } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import {
  Chart,
  Choice,
  ControlPanel,
  DatasetSelector,
  FigureFrame,
  Select,
  selectEntry,
  tagLabel,
  useColourMode,
  useDataset,
} from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Dataset selector',
  question: 'How do I let the reader choose a dataset by what it is, and turn its knobs?',
  explain:
    '`DatasetSelector` lists the tagged dataset registry of omai-engine (`datasetRegistry`: every generator, vendored real dataset and remote asset with its kind, modality, real or synthetic flag, size and provenance). Its props are the filters (`kind`, `modality`, `real`, `form`), a search box narrows further, and the chosen entry shows its description, its provenance and, when generated, a seed and its knobs. `useDataset` generates or loads the selection once per change; the same seed always gives the same data. `DatasetPicker` is the compact select over one hand-picked collection.',
}

const FIGURE = figure({ aspect: 1.25 })
const KINDS: readonly DatasetKind[] = [...new Set(datasetRegistry.entries.map((e) => e.kind))]

type Any<T extends string> = 'any' | T

export default function DatasetSelectorRecipe(): ReactNode {
  const mode = useColourMode()
  const [kind, setKind] = useState<Any<DatasetKind>>('any')
  const [modality, setModality] = useState<Any<DatasetModality>>('any')
  const [real, setReal] = useState<'any' | 'real' | 'synthetic'>('any')
  const [selection, setSelection] = useState(() => selectEntry(datasetRegistry, 'moons'))
  const { data } = useDataset(datasetRegistry, selection)
  const chart = useMemo(() => {
    if (!('x' in data)) return undefined
    const [n, d] = data.x.shape
    if (n === undefined || d === undefined || n === 0) return undefined
    const xs = data.x.data as ArrayLike<number>
    const ys = 'y' in data ? (data.y.data as ArrayLike<number>) : undefined
    // The first two features, or the one feature against the target.
    const points: number[][] = []
    for (let i = 0; i < n; i++) {
      const x0 = xs[i * d] ?? 0
      const x1 = d >= 2 ? (xs[i * d + 1] ?? 0) : (ys?.[i] ?? 0)
      points.push([x0, x1])
    }
    if (d < 2 && ys === undefined) return undefined
    const groups = data.kind === 'classification' && ys !== undefined ? Array.from(ys) : undefined
    return scatterChart([groups === undefined ? { x: points } : { x: points, groups }], {
      mode,
      figure: FIGURE,
      x: { label: 'x₁' },
      y: { label: d >= 2 ? 'x₂' : 'y' },
    })
  }, [data, mode])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3} title="filters">
        <Select label="kind" value={kind} options={['any', ...KINDS]} onChange={setKind} cycle={false} />
        <Select
          label="modality"
          value={modality}
          options={['any', ...DATASET_MODALITIES.map((m) => ({ value: m, label: tagLabel(m) }))]}
          onChange={setModality}
          cycle={false}
        />
        <Choice label="data" value={real} options={['any', 'real', 'synthetic']} onChange={setReal} />
      </ControlPanel>
      <div className="flex flex-wrap items-start gap-3">
        <ControlPanel columns={1} className="w-72">
          <DatasetSelector
            {...(kind === 'any' ? {} : { kind })}
            {...(modality === 'any' ? {} : { modality })}
            {...(real === 'any' ? {} : { real: real === 'real' })}
            form={['generator', 'vendored']}
            value={selection}
            onChange={setSelection}
          />
        </ControlPanel>
        <FigureFrame figure={FIGURE} filename="dataset">
          {chart === undefined ? (
            <p className="p-4 text-sm text-(--wb-muted)">
              {selection.name}: {data.kind}, {rowCount(data)} rows (no scatter for this kind of data)
            </p>
          ) : (
            <Chart option={chart} />
          )}
        </FigureFrame>
      </div>
    </div>
  )
}
