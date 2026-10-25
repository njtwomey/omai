/**
 * A dataset picker: a select over a dataset collection's entries with arrows to step through them, and for generated
 * entries a seed and the generator's knobs as controls; `useDataset` generates or loads the chosen dataset once per
 * change.
 */

import { type ReactNode, useMemo } from 'react'
import type { Dataset, Generated } from 'whitebox-engine/data'
import type { DatasetCollection } from 'whitebox-engine/datasets'
import { collectionOf, type DatasetSelection, entryKnobSpace, loadSelection, selectEntry } from './datasets.ts'
import { SeedInput } from './SeedInput.tsx'
import { Select } from './Select.tsx'
import { SpaceControls } from './SpaceControls.tsx'

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
 */
export function DatasetPicker(props: DatasetPickerProps): ReactNode {
  const { value, onChange, knobs = true, exclude = [], label = 'dataset', disabled = false } = props
  const collection = collectionOf(props.collection)
  const entry = collection.entries.find((e) => e.name === value.name)
  const space = entry === undefined ? undefined : entryKnobSpace(entry)
  return (
    <>
      <Select
        label={label}
        value={value.name}
        options={collection.entries.map((e) => ({ value: e.name, label: e.title }))}
        onChange={(name) => onChange(selectEntry(collection, name, value.seed))}
        disabled={disabled}
      />
      {space === undefined ? null : (
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
      )}
    </>
  )
}

/**
 * The dataset a selection names, generated (or loaded) once per change of selection.
 *
 * @param collection - The collection.
 * @param selection - The selection.
 * @returns The data, with its truth for generated entries.
 */
export function useDataset(
  collection: string | DatasetCollection,
  selection: DatasetSelection,
): { readonly data: Dataset } & Partial<Generated> {
  const id = typeof collection === 'string' ? collection : collection.id
  const knobs = JSON.stringify(selection.knobs)
  // biome-ignore lint/correctness/useExhaustiveDependencies: the selection is compared by content, not identity
  return useMemo(() => loadSelection(collection, selection), [id, selection.name, selection.seed, knobs])
}
