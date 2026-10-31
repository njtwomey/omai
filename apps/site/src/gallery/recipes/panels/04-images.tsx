import { imageSteps } from 'mdml-engine/exposition'
import { type DictSpace, defaults, dict, real } from 'mdml-engine/space'
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
  title: 'Grid of images',
  question: 'How do I show many small images at the viewed step, with captions, as a run goes on?',
  explain:
    'A probe that returns an `imageSteps` record (a row of images per group, optionally a caption under each image) and the panel `{ kind: "images", path }`, which draws the latest step at or before the viewed one as a grid with the group names down the side. Here a toy denoiser learns to clean six noisy glyphs: the first row is the input, the second the estimate, the third what is left of the error; the caption under each image is its error. Raise the noise to make the task harder.',
  shows: ['imageSteps', 'ImagePlayer'],
}

const W = 7
const GLYPHS: readonly (readonly string[])[] = [
  ['...#...', '...#...', '...#...', '...#...', '...#...', '...#...', '...#...'],
  ['#.....#', '.#...#.', '..#.#..', '...#...', '..#.#..', '.#...#.', '#.....#'],
  ['#######', '#.....#', '#.....#', '#.....#', '#.....#', '#.....#', '#######'],
  ['#......', '.#.....', '..#....', '...#...', '....#..', '.....#.', '......#'],
  ['.#####.', '#.....#', '......#', '..###..', '......#', '#.....#', '.#####.'],
  ['...#...', '...#...', '#######', '...#...', '...#...', '...#...', '...#...'],
]
const CLEAN = GLYPHS.map((g) =>
  g
    .join('')
    .split('')
    .map((c) => (c === '#' ? 1 : 0)),
)
// A fixed pseudo-random noise value per glyph and pixel, in [-1, 1].
const NOISE = CLEAN.map((g, i) =>
  g.map((_, j) => Math.sin(12.9898 * (i * W * W + j) + 4.1) * 2 - Math.sin(78.233 * (i + 1) * (j + 1))),
)
const STEPS = 80

interface State {
  readonly k: number
  readonly noise: number
}

const clamp = (v: number): number => Math.min(1, Math.max(0, v))
// The input: the glyph plus noise, and the estimate: the input pulled towards the glyph as training goes on.
const inputOf = (s: State, g: number): number[] =>
  (CLEAN[g] ?? []).map((v, j) => clamp(v + s.noise * 0.5 * (NOISE[g]?.[j] ?? 0)))
const estimateOf = (s: State, g: number): number[] => {
  const w = 1 - Math.exp(-s.k / 20)
  return inputOf(s, g).map((v, j) => (1 - w) * v + w * (CLEAN[g]?.[j] ?? 0))
}
const mse = (a: readonly number[], b: readonly number[]): number =>
  a.reduce((t, v, j) => t + (v - (b[j] ?? 0)) ** 2, 0) / a.length

const denoiser: Method<State, Config> = defineMethod<State, Config>({
  name: 'denoiser',
  config: dict({ noise: real(0, 1.6, { default: 0.5, label: 'noise level' }) }),
  make: (c) => {
    const noise = c['noise'] as number
    return {
      name: 'denoiser',
      config: { noise },
      init: () => ({ k: 0, noise }),
      step: (s, ctx) => {
        const next = { k: s.k + 1, noise }
        const err = CLEAN.reduce((t, g, i) => t + mse(estimateOf(next, i), g), 0) / CLEAN.length
        ctx.log('error', err)
        return next
      },
      done: (s) => (s.k >= STEPS ? 'completed' : undefined),
      serialise: (s) => stateToJson(s),
      deserialise: (j) => stateFromJson(j) as State,
    }
  },
  contract: {
    error: scalar({ description: 'mean squared error of the estimates' }),
    glyphs: derived<State>({
      description: 'the noisy inputs, the estimates and the residuals as rows of small images, with their errors',
      compute: (s) => {
        const inputs = CLEAN.map((_, i) => inputOf(s, i))
        const estimates = CLEAN.map((_, i) => estimateOf(s, i))
        return imageSteps({
          method: 'denoiser',
          height: W,
          width: W,
          groups: ['input', 'estimate', 'residual'],
          stepName: 'step',
          steps: [
            {
              iteration: s.k,
              images: [inputs, estimates, estimates.map((e, i) => e.map((v, j) => Math.abs(v - (CLEAN[i]?.[j] ?? 0))))],
              labels: [
                inputs.map((v, i) => mse(v, CLEAN[i] ?? []).toFixed(2)),
                estimates.map((v, i) => mse(v, CLEAN[i] ?? []).toFixed(2)),
                estimates.map(() => ''),
              ],
            },
          ],
        })
      },
    }),
  },
  defaults: { error: Log() },
})

const LAYOUT: readonly Panel[] = [
  { kind: 'images', path: 'glyphs', title: 'inputs, estimates and what is left' },
  { kind: 'curves', paths: ['error'], title: 'error (click to scrub)' },
]

export default function ImagesRecipe(): ReactNode {
  const [params, setParams] = useState<Config>(() => defaults(denoiser.config) as Config)
  const [at, setAt] = useState<number | null>(null)
  const prefs = useSessionPreferences()
  const live = useSession(denoiser, { config: params, mode: prefs.mode, steps: STEPS }, [JSON.stringify(params)], {
    hold: prefs.hold,
  })
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <SpaceControls space={denoiser.config as DictSpace} value={params} onChange={(p) => setParams(p as Config)} />
      </ControlPanel>
      <SessionPlayer
        live={live}
        at={at}
        onScrub={setAt}
        onReset={() => setParams(defaults(denoiser.config) as Config)}
      />
      <SessionPanels id="gallery/images" live={live} at={at} onScrub={setAt} layout={LAYOUT} />
    </div>
  )
}
