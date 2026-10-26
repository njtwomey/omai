import { isExposition } from 'mdml-engine/exposition'
import { key } from 'mdml-engine/random'
import { contractOf } from 'mdml-engine/trace'
import { figure } from 'mdml-render/figure'
import { ControlPanel, ExpositionView, NumberField, Select } from 'mdml-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { logsOf, SessionControls, type SessionSettings } from './_lib/controls.tsx'
import { DATASETS } from './_lib/data.ts'
import { useSession } from './_lib/session.ts'
import { type SvmConfig, svmMethod } from './_lib/svm.ts'
import { TapeView } from './_lib/tape.tsx'

export const recipe: Recipe = {
  title: 'SMO on a contract',
  question: 'Does the contract work for a solver whose innards are the pairs it picks, without touching the solver?',
  explain:
    'The core SMO is untouched: its native channels (`objective`, `violation`, `alpha`, `workingSet`) are renamed to `smo/...` paths with `from`. The margin at any step is a probe: the standard `margin` record with the support vectors, the pair being optimised and the decision function, drawn by the margin view.',
}

const FIG = figure({ width: 640, height: 360 })

export default function SvmPoc(): ReactNode {
  const [config, setConfig] = useState<SvmConfig>({ dataset: 'moons', n: 120, C: 1, gamma: 1 })
  const [settings, setSettings] = useState<SessionSettings>({
    mode: 'live',
    seed: 0,
    steps: 400,
    preset: 'defaults',
    chosen: ['smo/dual-objective', 'smo/working-set', 'smo/margin'],
    every: 5,
  })
  const [at, setAt] = useState<number | null>(null)
  const paths = useMemo(() => Object.keys(contractOf(svmMethod, config, key(0))), [config])
  const logs = logsOf(settings)
  const live = useSession(
    svmMethod,
    { config, seed: settings.seed, mode: settings.mode, steps: settings.steps, logs },
    [JSON.stringify(config), settings.seed, settings.mode, settings.steps, JSON.stringify(logs)],
  )
  const viewed = at === null ? live.session.t : Math.min(at, live.session.t)
  // biome-ignore lint/correctness/useExhaustiveDependencies: probes re-run as the tape grows
  const margin = useMemo(() => {
    if (settings.preset === 'off' && viewed !== live.session.t) return undefined
    try {
      return live.session.probe(viewed, 'smo/margin', { resolution: 40 })
    } catch {
      return undefined
    }
  }, [live.session, live.version, viewed, settings.preset])
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={4}>
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
          min={20}
          max={400}
          integer
        />
        <NumberField
          label="C"
          value={config.C}
          onChange={(C) => setConfig({ ...config, C })}
          min={0.01}
          max={100}
          log
        />
        <NumberField
          label="gamma"
          value={config.gamma}
          onChange={(gamma) => setConfig({ ...config, gamma })}
          min={0.05}
          max={20}
          log
        />
      </ControlPanel>
      <SessionControls paths={paths} settings={settings} onChange={setSettings} />
      <TapeView live={live} at={at} onScrub={setAt}>
        {margin !== undefined && isExposition(margin) && margin.kind === 'margin' ? (
          <div className="flex flex-col gap-1">
            <div className="text-sm opacity-70">
              probe: the margin at step {viewed}: {margin.support.length} support vectors
              {margin.workingSets?.[0] === undefined ? '' : `, optimising the pair ${margin.workingSets[0].join(', ')}`}
            </div>
            <ExpositionView record={margin} figure={FIG} />
          </div>
        ) : null}
      </TapeView>
    </div>
  )
}
