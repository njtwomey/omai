import { foldIn, normal } from 'omai-engine/random'
import { type DictSpace, defaults, dict, int } from 'omai-engine/space'
import { toFlat } from 'omai-engine/tensor'
import { type Config, defineMethod, Log, type Method, scalar, stateFromJson, stateToJson } from 'omai-engine/trace'
import {
  ControlPanel,
  type Panel,
  SessionPanels,
  SessionPlayer,
  SpaceControls,
  useSessionPreferences,
  useSessions,
  type Variant,
} from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Curve families',
  question: 'How do I draw many runs with their mean, bands, a truth line, rates on log-log axes and alarms?',
  explain:
    'The `curves` and `sweep` panels take options for the families a study of many runs needs: `runs` draws every variant faint with the mean across them in bold (the variants are repeats of one thing, not rivals), `bands` shades the region between a lower and an upper path, `truth` draws a dashed line (a constant or a path), `log` and `slopes` put the curve on log-log axes beside a reference slope, `events` marks the steps where a path fired, and a `sweep` with `intervals` draws every trial as a bar against the truth, the misses in red. Here a running mean of noisy draws, repeated under different keys; the same layout serves any method that logs these paths.',
}

interface State {
  readonly n: number
  readonly sum: number
}

const TRUTH = 2
const SPREAD = 3
const STEPS = 160
const SPACE: DictSpace = dict({ runs: int(2, 12, { default: 6, label: 'repeats' }) })

// An estimator of a mean from noisy draws: the running mean, its 95% interval, the error against the truth (which
// falls as one over the root of n), and an alarm whenever the estimate is further from the truth than its interval.
const estimator: Method<State, Config> = defineMethod<State, Config>({
  name: 'running-mean',
  config: dict({ run: int(0, 99, { default: 0 }) }),
  make: (c) => ({
    name: 'running-mean',
    config: { run: c['run'] as number },
    init: () => ({ n: 0, sum: 0 }),
    step: (s, ctx) => {
      const draw = toFlat(normal(foldIn(ctx.key, c['run'] as number), [1], { loc: TRUTH, scale: SPREAD }))[0] ?? 0
      const n = s.n + 1
      const sum = s.sum + draw
      const mean = sum / n
      const half = (1.96 * SPREAD) / Math.sqrt(n)
      ctx.log('estimate', mean)
      ctx.log('error', Math.abs(mean - TRUTH))
      ctx.log('lo', mean - half)
      ctx.log('hi', mean + half)
      if (n % 4 === 0) {
        ctx.log('trial/estimate', mean)
        ctx.log('trial/lo', mean - half)
        ctx.log('trial/hi', mean + half)
      }
      ctx.log('alarm', Math.abs(mean - TRUTH) > (2.6 * SPREAD) / Math.sqrt(n) ? 1 : 0)
      return { n, sum }
    },
    done: (s) => (s.n >= STEPS ? 'completed' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as State,
  }),
  contract: {
    estimate: scalar({ description: 'the running mean of the draws' }),
    error: scalar({ description: 'its distance from the truth' }),
    lo: scalar({ description: 'the lower end of its 95% interval' }),
    hi: scalar({ description: 'the upper end of its 95% interval' }),
    'trial/estimate': scalar({ description: 'the estimate, every fourth step' }),
    'trial/lo': scalar({ description: 'its interval, every fourth step' }),
    'trial/hi': scalar({ description: 'its interval, every fourth step' }),
    alarm: scalar({ description: '1 when the estimate leaves the truth by more than 2.6 standard errors' }),
  },
  defaults: {
    estimate: Log(),
    error: Log(),
    lo: Log(),
    hi: Log(),
    'trial/estimate': Log(),
    'trial/lo': Log(),
    'trial/hi': Log(),
    alarm: Log(),
  },
})

const LAYOUT: readonly Panel[] = [
  {
    kind: 'curves',
    paths: ['estimate'],
    runs: true,
    truth: { value: TRUTH, label: 'truth' },
    title: 'many runs, faint, with their mean in bold',
  },
  {
    kind: 'curves',
    paths: ['error'],
    runs: true,
    log: 'xy',
    slopes: [{ slope: -0.5, label: 'slope -1/2' }],
    title: 'the error on log-log axes, against the rate 1/√n',
  },
  {
    kind: 'curves',
    paths: ['estimate'],
    runs: true,
    bands: [{ path: 'estimate', lower: 'lo', upper: 'hi' }],
    truth: { value: TRUTH },
    events: [{ path: 'alarm', label: 'alarm' }],
    title: 'every run with its 95% band; alarms where the first run left the truth',
  },
  {
    kind: 'sweep',
    path: 'trial/estimate',
    intervals: { lower: 'trial/lo', upper: 'trial/hi' },
    truth: { value: TRUTH },
    title: 'the first run as intervals, one every fourth step, the misses in red',
  },
]

export default function CurveFamiliesRecipe(): ReactNode {
  const [params, setParams] = useState<Config>(() => defaults(SPACE) as Config)
  const [at, setAt] = useState<number | null>(null)
  const prefs = useSessionPreferences()
  const runs = params['runs'] as number
  const variants = useMemo<readonly Variant<Config>[]>(
    () => Array.from({ length: runs }, (_, i) => ({ label: `run ${i + 1}`, config: { run: i } })),
    [runs],
  )
  const live = useSessions(estimator, { config: {}, mode: prefs.mode, steps: STEPS }, variants, [], {
    hold: prefs.hold,
  })
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <SpaceControls space={SPACE} value={params} onChange={(p) => setParams(p as Config)} />
      </ControlPanel>
      <SessionPlayer live={live} at={at} onScrub={setAt} onReset={() => setParams(defaults(SPACE) as Config)} />
      <SessionPanels id="gallery/curve-families" live={live} at={at} onScrub={setAt} layout={LAYOUT} />
    </div>
  )
}
