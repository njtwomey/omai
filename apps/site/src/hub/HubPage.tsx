import { datasetCollections } from 'mdml-engine/datasets'
import { key } from 'mdml-engine/random'
import { type DictSpace, defaults } from 'mdml-engine/space'
import type { Tensor } from 'mdml-engine/tensor'
import { type Config, contractOf, type Method } from 'mdml-engine/trace'
import {
  ControlPanel,
  DatasetPicker,
  initialSettings,
  loadSelection,
  logsOf,
  type Panel,
  SessionControls,
  SessionPanels,
  SessionPlayer,
  type SessionSettings,
  SpaceControls,
  selectEntry,
  useSession,
} from 'mdml-render/react'
import { type ReactNode, useEffect, useMemo, useState } from 'react'
import { PageHeader } from '../shell/layout.tsx'
import type { HubEntry } from './entries.ts'

type Params = Record<string, never>

const TAPE: readonly Panel[] = [{ kind: 'tape' }]

/**
 * The hub's host: an entry run as a session. An estimator runs on a chosen dataset with its hyperparameters as
 * controls; a method of nothing but its configuration runs on that. Session settings are controls, the player scrubs
 * the tape, and the entry's layout alone decides which panels are drawn.
 */
export function HubPage(props: { readonly entry: HubEntry; readonly nav?: ReactNode }): ReactNode {
  const { entry, nav } = props
  return 'estimator' in entry ? <EstimatorPage entry={entry} nav={nav} /> : <MethodPage entry={entry} nav={nav} />
}

type EstimatorEntry = Extract<HubEntry, { readonly estimator: unknown }>
type MethodEntry = Extract<HubEntry, { readonly method: unknown }>

function MethodPage(props: { readonly entry: MethodEntry; readonly nav?: ReactNode }): ReactNode {
  const { entry } = props
  const space = Object.keys(entry.method.config.spaces).length === 0 ? undefined : entry.method.config
  const [params, setParams] = useState<Params>(() => defaults(entry.method.config) as Params)
  return (
    <SessionPage
      entry={entry}
      method={entry.method}
      space={space}
      params={params}
      onParams={setParams}
      nav={props.nav}
    />
  )
}

function EstimatorPage(props: { readonly entry: EstimatorEntry; readonly nav?: ReactNode }): ReactNode {
  const { entry } = props
  const estimator = entry.estimator
  const space = estimator.hyperSpace as DictSpace | undefined
  const collectionId = entry.datasets[0] ?? ''
  const [selection, setSelection] = useState(() => selectEntry(collectionId))
  const [params, setParams] = useState<Params>(() => (space === undefined ? {} : (defaults(space) as Params)))
  const loaded = useMemo(() => loadSelection(collectionId, selection), [collectionId, selection])
  const data = useMemo(() => {
    const d = loaded.data as unknown as { readonly x: Tensor; readonly y?: Tensor }
    return d.y === undefined ? { x: d.x } : { x: d.x, y: d.y }
  }, [loaded])
  const method = useMemo(() => estimator.with(params).method(data), [estimator, params, data])
  return (
    <SessionPage
      entry={entry}
      method={method}
      space={space}
      params={params}
      onParams={setParams}
      data={data}
      nav={props.nav}
      controls={
        <DatasetPicker
          collection={datasetCollections[collectionId] ?? collectionId}
          value={selection}
          onChange={setSelection}
        />
      }
    />
  )
}

// The page itself, given what runs: header, the entry's own controls, its knobs, the session settings, the player and
// the layout's panels.
function SessionPage(props: {
  readonly entry: HubEntry
  // biome-ignore lint/suspicious/noExplicitAny: the host is generic over methods
  readonly method: Method<any, Config>
  readonly space: DictSpace | undefined
  readonly params: Params
  readonly onParams: (p: Params) => void
  readonly data?: { readonly x: Tensor; readonly y?: Tensor }
  readonly controls?: ReactNode
  /** The previous and next links, in the header. */
  readonly nav?: ReactNode
}): ReactNode {
  const { entry, method, space, params, data } = props
  const [settings, setSettings] = useState<SessionSettings>(() =>
    initialSettings(entry.mode ?? 'live', entry.logs ?? 'defaults', entry.steps),
  )
  const [at, setAt] = useState<number | null>(null)
  // A remade method (another dataset, other knobs) is a new run: follow its head.
  // biome-ignore lint/correctness/useExhaustiveDependencies: reset exactly when the method is remade
  useEffect(() => setAt(null), [method])
  const paths = useMemo(() => Object.keys(contractOf(method, params, key(0))), [method, params])
  const logs = logsOf(settings)
  const live = useSession(
    method,
    { config: params, seed: settings.seed, mode: settings.mode, steps: settings.steps, logs },
    [method, JSON.stringify(params), settings.seed, settings.mode, settings.steps, JSON.stringify(logs)],
  )
  const layout = useMemo(() => (settings.tape ? [...entry.layout, ...TAPE] : entry.layout), [entry, settings.tape])
  return (
    <div className="flex flex-col gap-4">
      <PageHeader
        title={entry.title}
        lede={entry.summary}
        eyebrow={{ label: 'Hub', path: '/hub' }}
        actions={props.nav}
      />
      {props.controls === undefined ? null : <ControlPanel columns={4}>{props.controls}</ControlPanel>}
      {space === undefined ? null : (
        <ControlPanel columns={4}>
          <SpaceControls space={space} value={params} onChange={(p) => props.onParams(p as Params)} />
        </ControlPanel>
      )}
      <SessionControls paths={paths} settings={settings} onChange={setSettings} />
      <SessionPlayer live={live} at={at} onScrub={setAt} />
      <SessionPanels
        id={entry.id}
        live={live}
        at={at}
        onScrub={setAt}
        {...(data === undefined ? {} : { x: data.x })}
        {...(data?.y === undefined ? {} : { y: data.y })}
        layout={layout}
      />
    </div>
  )
}
