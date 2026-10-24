import { type ReactNode, useEffect, useState } from 'react'
import { datasetCollections } from 'whitebox-engine/datasets'
import type { Exposition } from 'whitebox-engine/exposition'
import type { SpaceValue } from 'whitebox-engine/space'
import { ControlPanel, DatasetPicker, PanelArea, Select, SpaceControls } from 'whitebox-render/react'
import type { HubEntry, ViewSlot } from '../entry/entry.ts'
import { datasetState, demoState, type HubState } from '../entry/run.ts'
import { RecordSummary } from './RecordSummary.tsx'
import { type HubRun, useHubRun } from './useHubRun.ts'

/** Props of {@link HubEntryView}. */
export interface HubEntryViewProps {
  readonly entry: HubEntry
  /** The state to start from (default the entry's demo state). */
  readonly initialState?: HubState
  /** Called with every new state (to keep it in the URL, say). */
  readonly onStateChange?: (state: HubState) => void
}

/** The record a slot shows, if the run returned it. */
function recordFor(records: readonly Exposition[], slot: ViewSlot): Exposition | undefined {
  return records.filter((r) => r.kind === slot.record)[slot.index ?? 0]
}

/** One slot: the record's view, or a summary card while render has no view for its kind. */
function Slot(props: {
  readonly slot: ViewSlot
  readonly record: Exposition | undefined
  readonly pending: boolean
}): ReactNode {
  const { slot, record, pending } = props
  if (record === undefined && pending)
    return <div className="h-24 animate-pulse rounded-lg border border-(--wb-grid) bg-(--wb-surface)" />
  if (record === undefined)
    return (
      <p className="m-0 rounded-lg border border-dashed border-(--wb-grid) px-3 py-2 text-xs text-(--wb-muted)">
        No {slot.record} record in this run.
      </p>
    )
  return <RecordSummary record={record} {...(slot.title === undefined ? {} : { title: slot.title })} />
}

function Status(props: { readonly run: HubRun }): ReactNode {
  const { run } = props
  const last = run.status === 'done' ? run : run.previous
  return (
    <div className="flex min-h-5 flex-wrap items-center gap-x-3 gap-y-1 text-xs text-(--wb-muted)" aria-live="polite">
      {run.status === 'running' ? <span>Running…</span> : null}
      {run.status === 'error' ? <span className="text-[#d62728]">{run.error}</span> : null}
      {last === undefined || run.status !== 'done' ? null : (
        <span className="tabular-nums">
          {last.ms < 10 ? last.ms.toFixed(1) : Math.round(last.ms)} ms on {last.rows} rows ({last.place})
        </span>
      )}
    </div>
  )
}

/**
 * A hub entry, live: a dataset picker over the entry's dataset collections, controls from its parameter space, and
 * its view slots, re-run (debounced, off the main thread where it can) whenever the data or a parameter changes.
 * Usable on any page: `<HubEntryView entry={lasso} />`.
 *
 * @param props - The entry, the state to start from and a listener for state changes.
 * @returns The widget.
 */
export function HubEntryView(props: HubEntryViewProps): ReactNode {
  const { entry, onStateChange } = props
  const [state, setState] = useState<HubState>(() => props.initialState ?? demoState(entry))
  // A new entry starts from its own demo state.
  const [shown, setShown] = useState(entry)
  if (shown !== entry) {
    setShown(entry)
    setState(props.initialState ?? demoState(entry))
  }
  // biome-ignore lint/correctness/useExhaustiveDependencies: report changes of state only
  useEffect(() => onStateChange?.(state), [state])
  const run = useHubRun(entry, state)
  const update = (s: HubState): void => setState(s)
  const records = run.status === 'done' ? run.exposition : (run.previous?.exposition ?? [])
  const slots: readonly ViewSlot[] =
    entry.views.length > 0 ? entry.views : records.map((r) => ({ record: r.kind }) as ViewSlot)
  const main = slots.filter((s) => (s.area ?? 'main') === 'main')
  const side = slots.filter((s) => s.area === 'side')
  const below = slots.filter((s) => s.area === 'below')
  const hasParams = Object.keys(entry.params.spaces).length > 0
  const stale = run.status !== 'done'
  const pending = records.length === 0 && run.status === 'running'
  return (
    <div className="flex flex-col gap-4">
      <div className="grid gap-3 md:grid-cols-2">
        <ControlPanel title="Data" columns={2}>
          {entry.datasets.length > 1 ? (
            <PanelArea columnSpan={2}>
              <Select
                label="collection"
                value={state.collection}
                options={entry.datasets.map((id) => ({ value: id, label: datasetCollections[id]?.title ?? id }))}
                onChange={(id) => update({ ...state, ...datasetState(id, undefined, state.seed) })}
              />
            </PanelArea>
          ) : null}
          <DatasetPicker
            collection={state.collection}
            value={{ name: state.dataset, seed: state.seed, knobs: state.knobs }}
            onChange={(sel) => update({ ...state, dataset: sel.name, seed: sel.seed, knobs: sel.knobs })}
          />
        </ControlPanel>
        {hasParams ? (
          <ControlPanel title="Method" columns={2}>
            <SpaceControls
              space={entry.params}
              value={state.params}
              onChange={(p) => update({ ...state, params: p as Readonly<Record<string, SpaceValue>> })}
            />
          </ControlPanel>
        ) : null}
      </div>
      <Status run={run} />
      <div className={`flex flex-col gap-4 transition-opacity ${stale ? 'opacity-60' : ''}`}>
        {main.map((s) => (
          <Slot key={`${s.record}-${s.index ?? 0}`} slot={s} record={recordFor(records, s)} pending={pending} />
        ))}
        {side.length > 0 ? (
          <div className="grid gap-4 md:grid-cols-2">
            {side.map((s) => (
              <Slot key={`${s.record}-${s.index ?? 0}`} slot={s} record={recordFor(records, s)} pending={pending} />
            ))}
          </div>
        ) : null}
        {below.map((s) => (
          <Slot key={`${s.record}-${s.index ?? 0}`} slot={s} record={recordFor(records, s)} pending={pending} />
        ))}
      </div>
    </div>
  )
}
