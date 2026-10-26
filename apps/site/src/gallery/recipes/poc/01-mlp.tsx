import { isExposition } from 'mdml-engine/exposition'
import { key } from 'mdml-engine/random'
import { contractOf } from 'mdml-engine/trace'
import { figure } from 'mdml-render/figure'
import { ControlPanel, ExpositionView, NumberField, Select } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { logsOf, SessionControls, type SessionSettings } from './_lib/controls.tsx'
import { DATASETS } from './_lib/data.ts'
import { type MlpConfig, mlpMethod } from './_lib/mlp.ts'
import { useSession } from './_lib/session.ts'
import { TapeView } from './_lib/tape.tsx'

export const recipe: Recipe = {
  title: 'MLP on a contract',
  question: 'What does a torch-like method look like when it only declares what can be known about it?',
  explain:
    'The method logs `training/loss`, its parameters and activations by path, fires `training/epoch/end`, and derives its decision surface from any state. Choose `logs` to see what the tape keeps, scrub it, and probe the surface at any step: nothing was decided in advance.',
}

const FIG = figure({ width: 420, height: 320 })

export default function MlpPoc(): ReactNode {
  const [config, setConfig] = useState<MlpConfig>({ dataset: 'moons', n: 200, hidden: 16, lr: 0.01, batch: 32 })
  const [settings, setSettings] = useState<SessionSettings>({
    mode: 'live',
    seed: 0,
    steps: 1500,
    preset: 'defaults',
    chosen: ['training/loss', 'model/decision-surface'],
    every: 10,
  })
  const [at, setAt] = useState<number | null>(null)
  const paths = useMemo(() => Object.keys(contractOf(mlpMethod, config, key(0))), [config])
  const logs = logsOf(settings)
  const live = useSession(
    mlpMethod,
    { config, seed: settings.seed, mode: settings.mode, steps: settings.steps, logs },
    [JSON.stringify(config), settings.seed, settings.mode, settings.steps, JSON.stringify(logs)],
  )
  const viewed = at === null ? live.session.t : Math.min(at, live.session.t)
  // biome-ignore lint/correctness/useExhaustiveDependencies: probes re-run as the tape grows
  const probe = useMemo(() => {
    if (settings.preset === 'off' && viewed !== live.session.t) return undefined
    try {
      const surface = live.session.probe(viewed, 'model/decision-surface', { resolution: 40 })
      const accuracy = live.session.probe(viewed, 'model/accuracy') as number
      return { surface, accuracy }
    } catch {
      return undefined
    }
  }, [live.session, live.version, viewed, settings.preset])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={5}>
        <Select
          label="dataset"
          value={config.dataset}
          options={[...DATASETS]}
          onChange={(dataset) => setConfig({ ...config, dataset })}
        />
        <NumberField
          label="rows"
          value={config.n}
          onChange={(n) => setConfig({ ...config, n })}
          min={50}
          max={1000}
          integer
        />
        <NumberField
          label="hidden"
          value={config.hidden}
          onChange={(hidden) => setConfig({ ...config, hidden })}
          min={2}
          max={64}
          integer
        />
        <NumberField
          label="learning rate"
          value={config.lr}
          onChange={(lr) => setConfig({ ...config, lr })}
          min={1e-4}
          max={0.1}
          log
        />
        <NumberField
          label="batch"
          value={config.batch}
          onChange={(batch) => setConfig({ ...config, batch })}
          min={4}
          max={200}
          integer
        />
      </ControlPanel>
      <SessionControls paths={paths} settings={settings} onChange={setSettings} />
      <TapeView live={live} at={at} onScrub={setAt}>
        {probe !== undefined && isExposition(probe.surface) ? (
          <div className="flex flex-col gap-1">
            <div className="text-sm opacity-70">
              probe: decision surface at step {viewed}, accuracy {(probe.accuracy * 100).toFixed(1)}%
            </div>
            <ExpositionView record={probe.surface} figure={FIG} />
          </div>
        ) : null}
      </TapeView>
    </div>
  )
}
