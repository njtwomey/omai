/**
 * A dataset selector over a tagged registry: the filters are its props (kinds, modalities, real or synthetic, form),
 * a search box narrows further, each entry shows its title and tags, and the chosen entry shows its description,
 * provenance and, when generated, its seed and knobs as controls. Draw the data with `useDataset`.
 */

import type { DatasetKind } from 'mdml-engine/data'
import {
  type DatasetEntry,
  type DatasetForm,
  type DatasetModality,
  type DatasetRegistry,
  datasetRegistry,
  findDataset,
  whereDatasets,
} from 'mdml-engine/datasets'
import type { ReactNode } from 'react'
import { DatasetKnobs } from './DatasetPicker.tsx'
import type { DatasetSelection } from './datasets.ts'
import { selectEntry } from './datasets.ts'
import { EntryList, tagLabel } from './EntryList.tsx'

/** Props of {@link DatasetSelector}: the filters of a registry query, the selection and the controls to draw. */
export interface DatasetSelectorProps {
  /** The registry to choose from (default the library's). */
  readonly registry?: DatasetRegistry
  readonly kind?: DatasetKind | readonly DatasetKind[]
  readonly modality?: DatasetModality | readonly DatasetModality[]
  /** Real data only (`true`), synthetic only (`false`), or both. */
  readonly real?: boolean
  /** Generators, vendored or remote entries only (remote ones load asynchronously, outside `useDataset`). */
  readonly form?: DatasetForm | readonly DatasetForm[]
  readonly value: DatasetSelection
  readonly onChange: (selection: DatasetSelection) => void
  /** Show the search box (default true). */
  readonly search?: boolean
  /** Show the generator's knobs as controls (default true). */
  readonly knobs?: boolean
  /** Knobs to leave out (say `n`, when the page fixes the sample size). */
  readonly exclude?: readonly string[]
  readonly label?: ReactNode
  readonly disabled?: boolean
}

const UNITS: Readonly<Record<DatasetModality, string>> = {
  tabular: 'rows',
  image: 'images',
  audio: 'clips',
  timeSeries: 'samples',
  text: 'documents',
  graph: 'nodes',
  sequence: 'sequences',
  events: 'events',
  interactions: 'interactions',
  trajectories: 'episodes',
}

/**
 * The tags shown for a dataset entry: its kind, modality, real or synthetic, remote when fetched, and its size.
 *
 * @param entry - The entry.
 * @returns The tags, as chips read them.
 */
export function datasetTags(entry: DatasetEntry): string[] {
  const tags = [entry.kind, tagLabel(entry.modality), entry.real ? 'real' : 'synthetic']
  if (entry.form === 'remote') tags.push('remote')
  if (entry.size !== undefined) tags.push(`${entry.size.toLocaleString('en-GB')} ${UNITS[entry.modality]}`)
  return tags
}

/**
 * Choose a dataset from a registry filtered by the props, each entry with its title and tags; the chosen entry's
 * description and provenance are shown under the list, with its seed and knobs when it is generated. Choosing
 * another entry starts it from its starting knobs and keeps the seed.
 *
 * @param props - The filters, the selection and what to call with a new one.
 * @returns The list and the controls, for a `ControlPanel` to lay out.
 */
export function DatasetSelector(props: DatasetSelectorProps): ReactNode {
  const {
    registry = datasetRegistry,
    kind,
    modality,
    real,
    form,
    value,
    onChange,
    search = true,
    knobs = true,
    exclude = [],
    label = 'dataset',
    disabled = false,
  } = props
  const entries = whereDatasets(registry, {
    ...(kind === undefined ? {} : { kind }),
    ...(modality === undefined ? {} : { modality }),
    ...(real === undefined ? {} : { real }),
    ...(form === undefined ? {} : { form }),
  })
  const chosen = findDataset(registry, value.name)
  const provenance = chosen === undefined ? [] : [chosen.source, chosen.licence].filter((s) => s !== undefined)
  return (
    <>
      <EntryList
        label={label}
        rows={entries.map((e) => ({ name: e.name, title: e.title, tags: datasetTags(e), description: e.description }))}
        value={value.name}
        onChange={(name) => onChange(selectEntry(registry, name, value.seed))}
        search={search}
        placeholder="Search datasets"
        disabled={disabled}
      />
      {chosen === undefined ? null : (
        <div className="flex min-w-36 flex-col gap-1 text-xs">
          <p className="text-(--wb-fg)">{chosen.description}</p>
          {provenance.length === 0 ? null : <p className="text-(--wb-muted)">{provenance.join(' · ')}</p>}
        </div>
      )}
      <DatasetKnobs
        entry={chosen}
        value={value}
        onChange={onChange}
        knobs={knobs}
        exclude={exclude}
        disabled={disabled}
      />
    </>
  )
}
