import { datasetCollections } from 'mdml-engine/datasets'
import { key } from 'mdml-engine/random'
import { type DictSpace, defaults } from 'mdml-engine/space'
import type { Tensor } from 'mdml-engine/tensor'
import { contractOf } from 'mdml-engine/trace'
import { ControlPanel, DatasetPicker, loadSelection, SpaceControls, selectEntry } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import { initialSettings, logsOf, SessionControls, type SessionSettings } from '../session/SessionControls.tsx'
import { SessionPlayer } from '../session/SessionPlayer.tsx'
import { useSession } from '../session/useSession.ts'
import { PageHeader } from '../shell/layout.tsx'
import type { HubEntry, Panel } from './entries.ts'
import { Panels } from './Panels.tsx'

type Params = Record<string, never>

const TAPE: readonly Panel[] = [{ kind: 'tape' }]

/**
 * The hub's host: any entry's estimator on a chosen dataset, run as a session. Dataset, hyperparameters and session
 * settings are controls, the player scrubs the tape, and the entry's layout alone decides which panels are drawn.
 */
export function HubPage(props: { readonly entry: HubEntry }): ReactNode {
  const { entry } = props
  const estimator = entry.estimator
  const space = estimator.hyperSpace as DictSpace | undefined
  const collectionId = entry.datasets[0] ?? ''
  const [selection, setSelection] = useState(() => selectEntry(collectionId))
  const [params, setParams] = useState<Params>(() => (space === undefined ? {} : (defaults(space) as Params)))
  const [settings, setSettings] = useState<SessionSettings>(() =>
    initialSettings(entry.mode ?? 'live', entry.logs ?? 'defaults'),
  )
  const [at, setAt] = useState<number | null>(null)
  const loaded = useMemo(() => loadSelection(collectionId, selection), [collectionId, selection])
  const data = useMemo(() => {
    const d = loaded.data as unknown as { readonly x: Tensor; readonly y?: Tensor }
    return d.y === undefined ? { x: d.x } : { x: d.x, y: d.y }
  }, [loaded])
  const method = useMemo(() => estimator.with(params).method(data), [estimator, params, data])
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
      <PageHeader title={entry.title} lede={entry.summary} eyebrow={{ label: 'Hub', path: '/hub' }} />
      <ControlPanel columns={4}>
        <DatasetPicker
          collection={datasetCollections[collectionId] ?? collectionId}
          value={selection}
          onChange={(s) => {
            setSelection(s)
            setAt(null)
          }}
        />
      </ControlPanel>
      {space === undefined ? null : (
        <ControlPanel columns={4}>
          <SpaceControls space={space} value={params} onChange={(p) => setParams(p as Params)} />
        </ControlPanel>
      )}
      <SessionControls paths={paths} settings={settings} onChange={setSettings} />
      <SessionPlayer live={live} at={at} onScrub={setAt} />
      <Panels
        entry={entry}
        live={live}
        at={at}
        x={data.x}
        {...(data.y === undefined ? {} : { y: data.y })}
        layout={layout}
      />
    </div>
  )
}
