import { key } from 'mdml-engine/random'
import type { Scene } from 'mdml-engine/scene'
import { isTensor, toFlat } from 'mdml-engine/tensor'
import { contractOf } from 'mdml-engine/trace'
import { heatmapChart } from 'mdml-render/charts'
import { figure } from 'mdml-render/figure'
import { Chart, ControlPanel, NumberField, Player, SceneView, useColourMode, usePlayback } from 'mdml-render/react'
import { type ReactElement, type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'
import { logsOf, SessionControls, type SessionSettings } from './_lib/controls.tsx'
import { rowsOf } from './_lib/data.ts'
import { type QConfig, qLearningMethod } from './_lib/q-learning.ts'
import { useSession } from './_lib/session.ts'
import { TapeView } from './_lib/tape.tsx'

export const recipe: Recipe = {
  title: 'Q-learning on a contract',
  question: 'Can I click an episode on the learning curve and watch the policy it had then?',
  explain:
    'The core training loop fires `rl/episode/end`; schedules count it (`every: [rl/episode/end, 10]`). Clicking an episode seeks the state at its end and runs `agent/rollout` there: greedy episodes with that policy, played back with frames. The Q-table at the scrubbed step is another probe.',
}

const SMALL = figure({ width: 360, height: 200 })

interface Rollout {
  readonly episode: number
  readonly step: number
  readonly plays: readonly { readonly return: number; readonly length: number; readonly frames: readonly Scene[] }[]
}

export default function QLearningPoc(): ReactNode {
  const mode = useColourMode()
  const [config, setConfig] = useState<QConfig>({ epsilon: 0.1, alpha: 0.5, gamma: 0.99 })
  const [settings, setSettings] = useState<SessionSettings>({
    mode: 'eager',
    seed: 0,
    steps: 6000,
    preset: 'defaults',
    chosen: ['rl/episode/end', 'agent/q', 'env/frame'],
    every: 25,
  })
  const [at, setAt] = useState<number | null>(null)
  const [rollout, setRollout] = useState<Rollout | undefined>(undefined)
  const paths = useMemo(() => Object.keys(contractOf(qLearningMethod, config, key(0))), [config])
  const logs = logsOf(settings)
  const live = useSession(
    qLearningMethod,
    { config, seed: settings.seed, mode: settings.mode, steps: settings.steps, logs },
    [JSON.stringify(config), settings.seed, settings.mode, settings.steps, JSON.stringify(logs)],
  )
  const viewed = at === null ? live.session.t : Math.min(at, live.session.t)
  // biome-ignore lint/correctness/useExhaustiveDependencies: probes re-run as the tape grows
  const q = useMemo(() => {
    try {
      const t = live.session.probe(viewed, 'agent/q')
      if (!isTensor(t)) return undefined
      const cols = t.shape[1] ?? 1
      const rows = rowsOf(toFlat(t), cols)
      // Cliff walking is 4 by 12: the best value per state, as the grid.
      const best = rows.map((r) => Math.max(...r))
      return rowsOf(best, 12)
    } catch {
      return undefined
    }
  }, [live.session, live.version, viewed])
  const onEpisodeClick = (episode: number): void => {
    const step = live.session.stepsOf('rl/episode/end')[episode]
    if (step === undefined) return
    const plays = live.session.probe(step, 'agent/rollout', { n: 3, seed: 1 }) as unknown as Rollout['plays']
    setRollout({ episode, step, plays })
  }
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <NumberField
          label="epsilon"
          value={config.epsilon}
          onChange={(epsilon) => setConfig({ ...config, epsilon })}
          min={0}
          max={1}
          step={0.05}
        />
        <NumberField
          label="alpha"
          value={config.alpha}
          onChange={(alpha) => setConfig({ ...config, alpha })}
          min={0.05}
          max={1}
          step={0.05}
        />
        <NumberField
          label="gamma"
          value={config.gamma}
          onChange={(gamma) => setConfig({ ...config, gamma })}
          min={0.8}
          max={1}
          step={0.01}
        />
      </ControlPanel>
      <SessionControls paths={paths} settings={settings} onChange={setSettings} />
      <TapeView live={live} at={at} onScrub={setAt} onEpisodeClick={onEpisodeClick}>
        {q === undefined ? null : (
          <Chart
            figure={SMALL}
            option={heatmapChart(q, {}, { mode, figure: SMALL, title: `probe: max Q per cell at step ${viewed}` })}
          />
        )}
        {rollout === undefined ? null : <RolloutPlayer rollout={rollout} />}
      </TapeView>
    </div>
  )
}

function RolloutPlayer(props: { readonly rollout: Rollout }): ReactElement {
  const { rollout } = props
  const mode = useColourMode()
  const frames = rollout.plays.flatMap((p) => p.frames)
  const c = usePlayback(frames.length, { fps: 8, loop: true, autoplay: true })
  const scene = frames[Math.min(c.frame, frames.length - 1)]
  return (
    <div className="flex flex-col gap-1">
      <div className="text-sm opacity-70">
        probe: the policy after episode {rollout.episode} (step {rollout.step}) playing {rollout.plays.length} greedy
        episodes: returns {rollout.plays.map((p) => p.return).join(', ')}
      </div>
      {scene === undefined ? null : <SceneView scene={scene} width={360} mode={mode} />}
      <Player controller={c} />
    </div>
  )
}
