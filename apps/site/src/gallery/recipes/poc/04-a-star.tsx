import { key } from 'mdml-engine/random'
import { contractOf } from 'mdml-engine/trace'
import { ControlPanel, Select } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { type AStarConfig, aStarMethod } from './_lib/a-star.ts'
import { logsOf, SessionControls, type SessionSettings } from './_lib/controls.tsx'
import { useSession } from './_lib/session.ts'
import { TapeView } from './_lib/tape.tsx'

export const recipe: Recipe = {
  title: 'A* on a contract',
  question: 'What does the trivial end look like: a search that needs almost nothing from the contract?',
  explain:
    'The core search is untouched; its expansion messages and counts are renamed to `search/...` paths, and the map with the node being expanded is a derived frame. `logs: off` is the raw search; `all` shows every expansion. No `serialise`, so nothing to seek: a search is rerun in a blink.',
}

export default function AStarPoc(): ReactNode {
  const [config, setConfig] = useState<AStarConfig>({ map: 'detour' })
  const [settings, setSettings] = useState<SessionSettings>({
    mode: 'live',
    seed: 0,
    steps: 2000,
    preset: 'all',
    chosen: ['search/expand', 'world/frame'],
    every: 1,
  })
  const [at, setAt] = useState<number | null>(null)
  const paths = useMemo(() => Object.keys(contractOf(aStarMethod, config, key(0))), [config])
  const logs = logsOf(settings)
  const live = useSession(
    aStarMethod,
    { config, seed: settings.seed, mode: settings.mode, steps: settings.steps, logs },
    [JSON.stringify(config), settings.seed, settings.mode, settings.steps, JSON.stringify(logs)],
  )
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <Select
          label="map"
          value={config.map}
          options={['open', 'detour', 'river', 'rooms']}
          onChange={(map) => setConfig({ map })}
        />
      </ControlPanel>
      <SessionControls paths={paths} settings={settings} onChange={setSettings} />
      <TapeView live={live} at={at} onScrub={setAt} />
    </div>
  )
}
