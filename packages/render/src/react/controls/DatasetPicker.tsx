/**
 * A dataset picker: a select over a dataset collection's entries with arrows to step through them, and for generated
 * entries a seed and the generator's knobs as controls (`DatasetKnobs`, which the dataset selector shares);
 * `useDataset` generates or loads the chosen dataset once per change, from a collection or a registry.
 */

import type { Dataset, Generated } from 'mdml-engine/data'
import type { DatasetCollection } from 'mdml-engine/datasets'
import { type ReactNode, useMemo } from 'react'
import { useByContent } from '../_content.ts'
import {
  collectionOf,
  type DatasetSelection,
  type DatasetSource,
  entryKnobSpace,
  loadSelection,
  type PickableEntry,
  selectEntry,
} from './datasets.ts'
import { SeedInput } from './SeedInput.tsx'
import { Select } from './Select.tsx'
import { SpaceControls } from './SpaceControls.tsx'

/** Props of {@link DatasetKnobs}. */
export interface DatasetKnobsProps {
  /** The chosen entry (nothing is drawn for a vendored or remote one, or none). */
  readonly entry: PickableEntry | undefined
  readonly value: DatasetSelection
  readonly onChange: (selection: DatasetSelection) => void
  /** Show the generator's knobs as controls (default true). */
  readonly knobs?: boolean
  /** Knobs to leave out (say `n`, when the page fixes the sample size). */
  readonly exclude?: readonly string[]
  readonly disabled?: boolean
}

/**
 * The seed and the generator's knobs of a selection, as controls: what the picker and the selector draw under the
 * chosen entry when it is generated.
 *
 * @param props - The entry, the selection and what to call with a new one.
 * @returns The controls, or nothing for an entry without a generator.
 */
export function DatasetKnobs(props: DatasetKnobsProps): ReactNode {
  const { entry, value, onChange, knobs = true, exclude = [], disabled = false } = props
  const space = entry === undefined ? undefined : entryKnobSpace(entry)
  if (space === undefined) return null
  return (
    <>
      <SeedInput value={value.seed} onChange={(seed) => onChange({ ...value, seed })} disabled={disabled} />
      {knobs ? (
        <SpaceControls
          space={space}
          value={value.knobs}
          onChange={(k) => onChange({ ...value, knobs: k })}
          exclude={exclude}
          disabled={disabled}
        />
      ) : null}
    </>
  )
}

/** Props of {@link DatasetPicker}. */
export interface DatasetPickerProps {
  /** A collection id in `datasetCollections` (`'classification2d'`, `'regression1d'`, ...), or a collection. */
  readonly collection: string | DatasetCollection
  readonly value: DatasetSelection
  readonly onChange: (selection: DatasetSelection) => void
  /** Show the generator's knobs as controls (default true). */
  readonly knobs?: boolean
  /** Knobs to leave out (say `n`, when the page fixes the sample size). */
  readonly exclude?: readonly string[]
  readonly label?: ReactNode
  readonly disabled?: boolean
}

/**
 * Pick a dataset from a collection: a select of its entries (with arrows to step through them), and for generated
 * entries a seed and the generator's knobs as controls. Choosing another entry starts it from its demo knobs and keeps
 * the seed. Draw the data with {@link useDataset}.
 *
 * @param props - The collection, the selection and what to call with a new one.
 * @returns The controls, side by side for a `ControlPanel` to lay out.
 * @index function React component: pick a dataset and its knobs: `<DatasetPicker collection="classification2d" value={sel} onChange={setSel} />`
 */
export function DatasetPicker(props: DatasetPickerProps): ReactNode {
  const { value, onChange, knobs = true, exclude = [], label = 'dataset', disabled = false } = props
  const collection = collectionOf(props.collection)
  const entry = collection.entries.find((e) => e.name === value.name)
  return (
    <>
      <Select
        label={label}
        value={value.name}
        options={collection.entries.map((e) => ({ value: e.name, label: e.title }))}
        onChange={(name) => onChange(selectEntry(collection, name, value.seed))}
        disabled={disabled}
      />
      <DatasetKnobs
        entry={entry}
        value={value}
        onChange={onChange}
        knobs={knobs}
        exclude={exclude}
        disabled={disabled}
      />
    </>
  )
}

/**
 * The dataset a selection names, generated (or loaded) once per change of selection.
 *
 * @param source - A collection id, a collection or a registry.
 * @param selection - The selection.
 * @returns The data, with its truth for generated entries.
 * @index function React hook: generate the dataset a picker selected: `useDataset('classification2d', selection)`
 */
export function useDataset(
  source: DatasetSource,
  selection: DatasetSelection,
): { readonly data: Dataset } & Partial<Generated> {
  // The selection is compared by content, not identity: a page may rebuild it on every render.
  const knobs = useByContent(selection.knobs)
  const { name, seed } = selection
  return useMemo(() => loadSelection(source, { name, seed, knobs }), [source, name, seed, knobs])
}
