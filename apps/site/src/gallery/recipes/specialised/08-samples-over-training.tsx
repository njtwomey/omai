import { sampleGrid } from 'mdml-engine/exposition'
import { choice, type DictSpace, defaults, dict } from 'mdml-engine/space'
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
  title: 'Samples over training',
  question: 'How do I show what a generative model produces from fixed noise as training goes on?',
  explain:
    'A probe that returns a `sampleGrid` record (images, or text, drawn from the same noise at recorded steps) and the panel `{ kind: "samples", path, filmstrip }`: the grid of the latest recorded step at or before the viewed one, with the filmstrip beneath, a row per earlier step spread over the run on a log scale (the early steps, where the samples change most, are close together), so the reader sees the samples sharpen. Here a toy generator turns noise into a shape (or garbled text into a sentence) as its quality rises over the epochs.',
  shows: ['sampleGrid', 'SampleGridView'],
}

const STEPS = 40
const SIDE = 10
const COUNT = 8
const SENTENCE = 'the quick brown fox jumps over the lazy dog'
const ALPHABET = 'abcdefghijklmnopqrstuvwxyz '

// A stream of numbers in [0, 1) from a linear congruential generator.
function stream(seed: number): () => number {
  let s = seed >>> 0
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0
    return s / 2 ** 32
  }
}

const SHAPES: Record<string, (x: number, y: number) => number> = {
  disc: (x, y) => (Math.hypot(x, y) < 0.6 ? 1 : 0),
  ring: (x, y) => (Math.abs(Math.hypot(x, y) - 0.6) < 0.2 ? 1 : 0),
  cross: (x, y) => (Math.abs(x) < 0.18 || Math.abs(y) < 0.18 ? 1 : 0),
  stripes: (x) => (Math.sin(x * 9) > 0 ? 1 : 0),
}

// The same noise at every step, so that only the model's quality changes what is drawn.
const NOISE = Array.from({ length: COUNT }, (_, i) => {
  const next = stream(100 + i)
  return {
    pixels: Array.from({ length: SIDE * SIDE }, () => next()),
    shift: [Math.round((next() - 0.5) * 3), Math.round((next() - 0.5) * 3)] as const,
    chars: Array.from({ length: SENTENCE.length }, () => [next(), Math.floor(next() * ALPHABET.length)] as const),
  }
})

const quality = (k: number): number => 1 - (1 - k / STEPS) ** 2

interface State {
  readonly k: number
  readonly modality: string
  readonly shape: string
}

function imageAt(i: number, q: number, shape: string): number[] {
  const f = SHAPES[shape] ?? SHAPES['disc']
  const n = NOISE[i]
  return Array.from({ length: SIDE * SIDE }, (_, p) => {
    const r = Math.floor(p / SIDE)
    const c = p % SIDE
    const x = ((c - (n?.shift[0] ?? 0)) / (SIDE - 1)) * 2 - 1
    const y = ((r - (n?.shift[1] ?? 0)) / (SIDE - 1)) * 2 - 1
    return (1 - q) * (n?.pixels[p] ?? 0) + q * (f?.(x, y) ?? 0)
  })
}

const textAt = (i: number, q: number): string =>
  [...SENTENCE]
    .map((ch, p) => ((NOISE[i]?.chars[p]?.[0] ?? 0) < q ? ch : (ALPHABET[NOISE[i]?.chars[p]?.[1] ?? 0] ?? '?')))
    .join('')

const generator: Method<State, Config> = defineMethod<State, Config>({
  name: 'generator',
  config: dict({
    modality: choice(['images', 'text'], { default: 'images' }),
    shape: choice(Object.keys(SHAPES), { default: 'ring', label: 'shape to learn' }),
  }),
  make: (c) => {
    const modality = c['modality'] as string
    const shape = c['shape'] as string
    return {
      name: 'generator',
      config: { modality, shape },
      init: () => ({ k: 0, modality, shape }),
      step: (s, ctx) => {
        ctx.log('quality', quality(s.k + 1))
        return { ...s, k: s.k + 1 }
      },
      done: (s) => (s.k >= STEPS ? 'completed' : undefined),
      serialise: (s) => stateToJson(s),
      deserialise: (j) => stateFromJson(j) as State,
    }
  },
  contract: {
    quality: scalar({ description: 'how close the samples are to the target, from 0 to 1' }),
    samples: derived<State>({
      description: 'the samples from the same noise at every step so far',
      compute: (s) => {
        const steps = Array.from({ length: s.k + 1 }, (_, k) => ({
          step: k,
          samples: Array.from({ length: COUNT }, (_, i) =>
            s.modality === 'text' ? textAt(i, quality(k)) : imageAt(i, quality(k), s.shape),
          ),
        }))
        return s.modality === 'text'
          ? sampleGrid({ modality: 'text', steps, columns: 1, stepName: 'epoch' })
          : sampleGrid({ modality: 'image', height: SIDE, width: SIDE, steps, columns: 4, stepName: 'epoch' })
      },
    }),
  },
  defaults: { quality: Log() },
})

const LAYOUT: readonly Panel[] = [
  { kind: 'samples', path: 'samples', filmstrip: 4, title: 'samples from fixed noise' },
  { kind: 'curves', paths: ['quality'], title: 'quality (click to scrub)' },
]

export default function SamplesOverTrainingRecipe(): ReactNode {
  const [params, setParams] = useState<Config>(() => defaults(generator.config) as Config)
  const [at, setAt] = useState<number | null>(null)
  const prefs = useSessionPreferences()
  const live = useSession(generator, { config: params, mode: prefs.mode, steps: STEPS }, [JSON.stringify(params)], {
    hold: prefs.hold,
  })
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <SpaceControls space={generator.config as DictSpace} value={params} onChange={(p) => setParams(p as Config)} />
      </ControlPanel>
      <SessionPlayer
        live={live}
        at={at}
        onScrub={setAt}
        onReset={() => setParams(defaults(generator.config) as Config)}
      />
      <SessionPanels id="gallery/samples-over-training" live={live} at={at} onScrub={setAt} layout={LAYOUT} />
    </div>
  )
}
