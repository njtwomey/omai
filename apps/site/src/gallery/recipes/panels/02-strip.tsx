import { stripRecord } from 'omai-engine/exposition'
import { type DictSpace, defaults, dict, real } from 'omai-engine/space'
import {
  type Config,
  defineMethod,
  derived,
  Log,
  type Method,
  scalar,
  stateFromJson,
  stateToJson,
} from 'omai-engine/trace'
import {
  ControlPanel,
  type Panel,
  SessionPanels,
  SessionPlayer,
  SpaceControls,
  useSession,
  useSessionPreferences,
} from 'omai-render/react'
import { type ReactNode, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Strip of tokens',
  question: 'How do I show per-position values under a sequence: tag probabilities, a chosen path, the errors?',
  explain:
    'A probe that returns a `strip` record (the tokens, a track of values per class, rows of tags, the chosen path through the tracks, marked positions) and the panel `{ kind: "strip", path }`, which draws the record at the viewed step: one row of cells per track under the tokens, the argmax path as a line through its cells and the positions it gets wrong outlined. Here a toy tagger learns the parts of speech of one sentence by gradient steps; scrub back to see it start out unsure.',
  shows: ['strip', 'StripView'],
}

const TOKENS = ['the', 'quick', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
const CLASSES = ['det', 'adj', 'noun', 'verb', 'adp']
const TRUTH = [0, 1, 2, 3, 4, 0, 1, 2]
const STEPS = 60

interface State {
  readonly k: number
  /** logits[i * 5 + c]: the tagger's score of class c at position i. */
  readonly logits: readonly number[]
  readonly loss: number
}

const softmax = (row: readonly number[]): number[] => {
  const top = Math.max(...row)
  const e = row.map((v) => Math.exp(v - top))
  const z = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / z)
}
const probabilities = (s: State): number[][] =>
  TOKENS.map((_, i) => softmax(s.logits.slice(i * CLASSES.length, (i + 1) * CLASSES.length)))

const tagger: Method<State, Config> = defineMethod<State, Config>({
  name: 'tagger',
  config: dict({ rate: real(0.05, 1, { default: 0.25, label: 'learning rate' }) }),
  make: (c) => {
    const rate = c['rate'] as number
    return {
      name: 'tagger',
      config: { rate },
      // A start that is wrong in places: a fixed pseudo-random lean towards some class at every position.
      init: () => ({
        k: 0,
        loss: 0,
        logits: TOKENS.flatMap((_, i) => CLASSES.map((_, c) => 1.6 * Math.sin(12.9898 * i + 78.233 * c + 3))),
      }),
      step: (s, ctx) => {
        const p = probabilities(s)
        const logits = s.logits.map((v, j) => {
          const i = Math.floor(j / CLASSES.length)
          const c = j % CLASSES.length
          return v + rate * ((TRUTH[i] === c ? 1 : 0) - (p[i]?.[c] ?? 0))
        })
        const loss = -p.reduce((a, row, i) => a + Math.log(row[TRUTH[i] ?? 0] ?? 1e-12), 0) / TOKENS.length
        ctx.log('loss', loss)
        return { k: s.k + 1, logits, loss }
      },
      done: (s) => (s.k >= STEPS ? 'completed' : undefined),
      serialise: (s) => stateToJson(s),
      deserialise: (j) => stateFromJson(j) as State,
    }
  },
  contract: {
    loss: scalar({ description: 'cross-entropy of the tags' }),
    tagging: derived<State>({
      description: 'the tag probabilities per position, the chosen tags and the positions it gets wrong',
      compute: (s) => {
        const p = probabilities(s)
        const chosen = p.map((row) => row.indexOf(Math.max(...row)))
        return stripRecord({
          tokens: TOKENS,
          tracks: CLASSES.map((name, c) => ({ name, values: p.map((row) => row[c] ?? 0) })),
          tags: [
            { name: 'truth', labels: TRUTH.map((c) => CLASSES[c] ?? '') },
            { name: 'chosen', labels: chosen.map((c) => CLASSES[c] ?? '') },
          ],
          path: chosen,
          marks: chosen.flatMap((c, i) => (c === TRUTH[i] ? [] : [i])),
          valueName: 'probability',
        })
      },
    }),
  },
  defaults: { loss: Log() },
})

const LAYOUT: readonly Panel[] = [
  { kind: 'strip', path: 'tagging', title: 'tag probabilities, the chosen tags, the mistakes outlined' },
  { kind: 'curves', paths: ['loss'], title: 'loss (click to scrub)' },
]

export default function StripRecipe(): ReactNode {
  const [params, setParams] = useState<Config>(() => defaults(tagger.config) as Config)
  const [at, setAt] = useState<number | null>(null)
  const prefs = useSessionPreferences()
  const live = useSession(tagger, { config: params, mode: prefs.mode, steps: STEPS }, [JSON.stringify(params)], {
    hold: prefs.hold,
  })
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <SpaceControls space={tagger.config as DictSpace} value={params} onChange={(p) => setParams(p as Config)} />
      </ControlPanel>
      <SessionPlayer live={live} at={at} onScrub={setAt} onReset={() => setParams(defaults(tagger.config) as Config)} />
      <SessionPanels id="gallery/strip" live={live} at={at} onScrub={setAt} layout={LAYOUT} />
    </div>
  )
}
