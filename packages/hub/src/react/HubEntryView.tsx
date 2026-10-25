/**
 * The live page of a hub entry as a React component: a dataset picker over the entry's dataset collections, controls
 * generated from its parameter space, and its view slots drawn with whitebox-render, re-run (debounced, off the main
 * thread where it can) whenever the data or a control changes.
 */

import { type ComponentType, type ReactNode, type RefObject, useEffect, useMemo, useRef, useState } from 'react'
import { datasetCollections } from 'whitebox-engine/datasets'
import type { Exposition } from 'whitebox-engine/exposition'
import type { SpaceValue } from 'whitebox-engine/space'
import { asTensor, type TensorLike, toArray, toFlat } from 'whitebox-engine/tensor'
import { viewFor } from 'whitebox-render'
import { type Figure, figure } from 'whitebox-render/figure'
import * as renderReact from 'whitebox-render/react'
import {
  ControlPanel,
  DatasetPicker,
  ExpositionView,
  PanelArea,
  SeedInput,
  Select,
  SpaceControls,
} from 'whitebox-render/react'
import type { HubEntry, SlotContext, ViewSlot } from '../entry/entry.ts'
import { datasetState, demoState, entryData, type HubState } from '../entry/run.ts'
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

/**
 * The input rows of a run's dataset with, for classification, their labels and, for regression, their targets: what
 * views draw beside a record.
 */
interface Rows {
  readonly points?: readonly (readonly number[])[]
  readonly labels?: readonly number[]
  readonly targets?: readonly number[]
}

function rowsOf(entry: HubEntry, state: HubState): Rows {
  try {
    const data = entryData(entry, state) as
      | { readonly kind: string; readonly x?: TensorLike; readonly y?: TensorLike }
      | undefined
    if (data?.x === undefined) return {}
    const points = toArray(asTensor(data.x)) as number[][]
    if (data.y === undefined) return { points }
    if (data.kind === 'classification') return { points, labels: toFlat(asTensor(data.y)) }
    const y = asTensor(data.y)
    // Single-output targets only (a multi-output target has no one colour per row).
    return y.shape.length === 1 || y.shape[1] === 1 ? { points, targets: toFlat(y) } : { points }
  } catch {
    return {}
  }
}

/** The width of an element, followed as it resizes. */
function useWidth(ref: RefObject<HTMLElement | null>): number {
  const [width, setWidth] = useState(0)
  useEffect(() => {
    const el = ref.current
    if (el === null || typeof ResizeObserver === 'undefined') return
    const observer = new ResizeObserver(([e]) => setWidth(Math.floor(e?.contentRect.width ?? 0)))
    observer.observe(el)
    return () => observer.disconnect()
  }, [ref])
  return width
}

/** A component of `whitebox-render/react` by name, if there is one. */
function renderComponent(name: string): ComponentType<Record<string, unknown>> | undefined {
  const c = (renderReact as Readonly<Record<string, unknown>>)[name]
  return typeof c === 'function' ? (c as ComponentType<Record<string, unknown>>) : undefined
}

/**
 * One slot: the named render view (with the slot's props), else the record's standard view, else a summary card while
 * render has no view for its kind.
 */
function Slot(props: {
  readonly slot: ViewSlot
  readonly record: Exposition | undefined
  readonly pending: boolean
  readonly rows: Rows
  readonly params: SlotContext['params']
  readonly figure: Figure | undefined
}): ReactNode {
  const { slot, record, pending, rows } = props
  if (record === undefined && pending)
    return <div className="h-24 animate-pulse rounded-lg border border-(--wb-grid) bg-(--wb-surface)" />
  if (record === undefined)
    return (
      <p className="m-0 rounded-lg border border-dashed border-(--wb-grid) px-3 py-2 text-xs text-(--wb-muted)">
        No {slot.record} record in this run.
      </p>
    )
  const Named = slot.view === undefined ? undefined : renderComponent(slot.view)
  if (Named !== undefined) {
    const extra = slot.props?.(record, { ...rows, params: props.params }) ?? {}
    return <Named record={record} {...(props.figure === undefined ? {} : { figure: props.figure })} {...extra} />
  }
  if (viewFor(record.kind) !== undefined)
    return (
      <ExpositionView
        record={record}
        {...(props.figure === undefined ? {} : { figure: props.figure })}
        {...(rows.points === undefined ? {} : { points: rows.points })}
        {...(rows.labels === undefined ? {} : { labels: rows.labels })}
      />
    )
  return <RecordSummary record={record} {...(slot.title === undefined ? {} : { title: slot.title })} />
}

function Status(props: { readonly run: HubRun }): ReactNode {
  const { run } = props
  const last = run.status === 'done' ? run : run.previous
  return (
    <div className="flex min-h-5 flex-wrap items-center gap-x-3 gap-y-1 text-xs text-(--wb-muted)" aria-live="polite">
      {run.status === 'running' && run.progress === undefined ? <span>Running…</span> : null}
      {run.status === 'running' && run.progress !== undefined ? (
        <span className="flex items-center gap-2">
          <span className="h-1.5 w-32 overflow-hidden rounded-full bg-(--wb-grid)">
            <span
              className="block h-full bg-(--wb-accent)"
              style={{ width: `${Math.round(100 * Math.min(1, Math.max(0, run.progress.fraction)))}%` }}
            />
          </span>
          <span className="tabular-nums">{run.progress.label ?? `${Math.round(100 * run.progress.fraction)}%`}</span>
        </span>
      ) : null}
      {run.status === 'error' ? <span className="text-[#d62728]">{run.error}</span> : null}
      {last === undefined || run.status !== 'done' ? null : (
        <span className="tabular-nums">
          {last.ms < 10 ? last.ms.toFixed(1) : Math.round(last.ms)} ms{last.rows > 0 ? ` on ${last.rows} rows` : ''} (
          {last.place})
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
  const last = run.status === 'done' ? run : run.previous
  // A run that learns live shows its records so far, on the state it runs on.
  const live = run.status === 'running' ? run.progress?.exposition : undefined
  const records = live ?? last?.exposition ?? []
  const shownState = live !== undefined ? state : last?.state
  const shownKey = shownState === undefined ? '' : JSON.stringify(shownState)
  // biome-ignore lint/correctness/useExhaustiveDependencies: the shown state is compared by content
  const rows = useMemo(() => (shownState === undefined ? {} : rowsOf(entry, shownState)), [entry, shownKey])
  const params = shownState?.params ?? state.params
  const area = useRef<HTMLDivElement | null>(null)
  const width = useWidth(area)
  const sideWidth = width >= 720 ? Math.floor((width - 16) / 2) : width
  const mainFigure = useMemo(() => (width > 0 ? figure({ width: Math.min(width, 900) }) : undefined), [width])
  const sideFigure = useMemo(() => (sideWidth > 0 ? figure({ width: sideWidth }) : undefined), [sideWidth])
  const slots: readonly ViewSlot[] =
    entry.views.length > 0 ? entry.views : records.map((r) => ({ record: r.kind }) as ViewSlot)
  const main = slots.filter((s) => (s.area ?? 'main') === 'main')
  const side = slots.filter((s) => s.area === 'side')
  const below = slots.filter((s) => s.area === 'below')
  const hasParams = Object.keys(entry.params.spaces).length > 0
  const stale = run.status !== 'done' && live === undefined
  const pending = records.length === 0 && run.status === 'running'
  return (
    <div className="flex flex-col gap-4">
      <div className="grid gap-3 md:grid-cols-2">
        {entry.datasets.length === 0 ? null : (
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
        )}
        {hasParams || entry.datasets.length === 0 ? (
          <ControlPanel title="Method" columns={2}>
            {entry.datasets.length === 0 ? (
              <SeedInput value={state.seed} onChange={(seed) => update({ ...state, seed })} />
            ) : null}
            <SpaceControls
              space={entry.params}
              value={state.params}
              onChange={(p) => update({ ...state, params: p as Readonly<Record<string, SpaceValue>> })}
            />
          </ControlPanel>
        ) : null}
      </div>
      <Status run={run} />
      <div ref={area} className={`flex flex-col gap-4 transition-opacity ${stale ? 'opacity-60' : ''}`}>
        {main.map((s) => (
          <Slot
            key={`${s.record}-${s.index ?? 0}-${s.view ?? ''}`}
            slot={s}
            record={recordFor(records, s)}
            pending={pending}
            rows={rows}
            params={params}
            figure={mainFigure}
          />
        ))}
        {side.length > 0 ? (
          <div className={`grid items-start gap-4 ${width >= 720 ? 'grid-cols-2' : ''}`}>
            {side.map((s) => (
              <Slot
                key={`${s.record}-${s.index ?? 0}-${s.view ?? ''}`}
                slot={s}
                record={recordFor(records, s)}
                pending={pending}
                rows={rows}
                params={params}
                figure={sideFigure}
              />
            ))}
          </div>
        ) : null}
        {below.map((s) => (
          <Slot
            key={`${s.record}-${s.index ?? 0}-${s.view ?? ''}`}
            slot={s}
            record={recordFor(records, s)}
            pending={pending}
            rows={rows}
            params={params}
            figure={mainFigure}
          />
        ))}
      </div>
    </div>
  )
}
