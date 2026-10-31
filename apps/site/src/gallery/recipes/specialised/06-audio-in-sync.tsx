import { audioClip } from 'mdml-engine/exposition'
import { rfft } from 'mdml-engine/numerics/fourier'
import { choice, type DictSpace, defaults, dict } from 'mdml-engine/space'
import { toArray } from 'mdml-engine/tensor'
import {
  type Config,
  defineMethod,
  derived,
  Log,
  type Method,
  scalar,
  stateFromJson,
  stateToJson,
} from 'mdml-engine/trace'
import {
  ControlPanel,
  type Panel,
  SessionPanels,
  SessionPlayer,
  SpaceControls,
  useSession,
  useSessionPreferences,
} from 'mdml-render/react'
import { type ReactNode, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Audio in sync',
  question: 'How do I show a clip with its spectrogram, with a playhead locked to the step cursor, and play it?',
  explain:
    'A probe that returns an `audio` record (the samples and their rate, the `frames` that say what time of the clip each tape step stands at) and the panel `{ kind: "audio", path }`: the waveform above the spectrogram, the playhead at the time of the viewed step, and a button that plays the clip through WebAudio from there while the step cursor follows the sound. Clicking the picture moves the cursor, and so every other panel. Here a method analyses the clip one window per step, logging the dominant frequency of the window under the playhead, and the curve beside it marks the same step.',
  shows: ['audio', 'AudioView'],
}

const RATE = 8000
const SECONDS = 2
const STEPS = 100
const WINDOW = 256

const SIGNALS = ['chirp', 'two tones', 'clicks'] as const
const signal = (kind: string): number[] =>
  Array.from({ length: RATE * SECONDS }, (_, i) => {
    const t = i / RATE
    if (kind === 'chirp') return 0.8 * Math.sin(2 * Math.PI * (300 * t + (2200 / (2 * SECONDS)) * t * t))
    if (kind === 'two tones')
      return (
        0.45 *
        (Math.sin(2 * Math.PI * 440 * t) * (t < 1 ? 1 : 0) + Math.sin(2 * Math.PI * 1320 * t) * (t >= 0.6 ? 1 : 0))
      )
    return (i % 2000 < 40 ? 0.9 : 0) * Math.sin(2 * Math.PI * 1800 * t) + 0.02 * Math.sin(2 * Math.PI * 97 * t)
  })

interface State {
  readonly k: number
  readonly kind: string
}

// The frequency with the most energy in the window centred on the time of a step.
function dominant(samples: readonly number[], step: number): number {
  const centre = Math.round((step / STEPS) * (samples.length - 1))
  const from = Math.min(Math.max(0, centre - WINDOW / 2), samples.length - WINDOW)
  const spec = rfft(samples.slice(from, from + WINDOW))
  const re = toArray(spec.re) as number[]
  const im = toArray(spec.im) as number[]
  let best = 1
  for (let f = 1; f < re.length; f++)
    if (Math.hypot(re[f] ?? 0, im[f] ?? 0) > Math.hypot(re[best] ?? 0, im[best] ?? 0)) best = f
  return (best * RATE) / WINDOW
}

const analyser: Method<State, Config> = defineMethod<State, Config>({
  name: 'analyser',
  config: dict({ kind: choice([...SIGNALS], { default: 'chirp', label: 'signal' }) }),
  make: (c) => {
    const kind = c['kind'] as string
    const samples = signal(kind)
    return {
      name: 'analyser',
      config: { kind },
      init: () => ({ k: 0, kind }),
      step: (s, ctx) => {
        ctx.log('dominant frequency (Hz)', dominant(samples, s.k + 1))
        return { k: s.k + 1, kind }
      },
      done: (s) => (s.k >= STEPS ? 'completed' : undefined),
      serialise: (s) => stateToJson(s),
      deserialise: (j) => stateFromJson(j) as State,
    }
  },
  contract: {
    'dominant frequency (Hz)': scalar({
      description: 'the frequency with the most energy in the window under the playhead',
    }),
    clip: derived<State>({
      description: 'the clip, with the time each step stands at',
      compute: (s) =>
        audioClip({
          sampleRate: RATE,
          samples: signal(s.kind),
          frames: [
            { step: 0, time: 0 },
            { step: STEPS, time: SECONDS },
          ],
          caption: `window ${WINDOW} samples, one per step`,
        }),
    }),
  },
  defaults: { 'dominant frequency (Hz)': Log() },
})

const LAYOUT: readonly Panel[] = [
  { kind: 'audio', path: 'clip', title: 'the clip' },
  { kind: 'curves', paths: ['dominant frequency (Hz)'], title: 'dominant frequency (click to scrub)' },
]

export default function AudioInSyncRecipe(): ReactNode {
  const [params, setParams] = useState<Config>(() => defaults(analyser.config) as Config)
  const [at, setAt] = useState<number | null>(null)
  const prefs = useSessionPreferences()
  const live = useSession(analyser, { config: params, mode: prefs.mode, steps: STEPS }, [JSON.stringify(params)], {
    hold: prefs.hold,
  })
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <SpaceControls space={analyser.config as DictSpace} value={params} onChange={(p) => setParams(p as Config)} />
      </ControlPanel>
      <SessionPlayer
        live={live}
        at={at}
        onScrub={setAt}
        onReset={() => setParams(defaults(analyser.config) as Config)}
      />
      <SessionPanels id="gallery/audio-in-sync" live={live} at={at} onScrub={setAt} layout={LAYOUT} />
    </div>
  )
}
