import { transcript } from 'omai-engine/exposition'
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
  title: 'Transcript',
  question: 'How do I show the tokens of a conversation or a generation as it grows, with attention on them?',
  explain:
    'A probe that returns a `transcript` record (the tokens so far with their role, a weight per token for each named track, the index of the token emitted at this step) and the panel `{ kind: "transcript", path }`: the tokens in turns by role, each tinted by its weight on the chosen track (buttons switch heads), the token emitted at the viewed step outlined and marked with an arrow, every weight in the token\'s tooltip and the heaviest tokens named under the figure. Here a toy agent answers a question, calls a tool and answers again, one token per step; two "heads" attend to the recent tokens and to earlier occurrences of the same word.',
  shows: ['transcript', 'TranscriptView'],
}

const SCRIPT: readonly (readonly [string, string])[] = [
  ['user', 'What is the capital of France ?'],
  ['assistant', 'I will look it up .'],
  ['tool', 'search ( capital of France ) -> Paris'],
  ['assistant', 'The capital of France is Paris .'],
]
// Every token with its role, the first turn being the prompt.
const TOKENS = SCRIPT.flatMap(([role, text]) =>
  text.split(' ').map((word, i) => ({ text: i === 0 ? word : ` ${word}`, role })),
)
const PROMPT = (SCRIPT[0]?.[1] ?? '').split(' ').length
const STEPS = TOKENS.length - PROMPT

interface State {
  readonly k: number
  readonly span: number
}

// Two toy heads over the tokens so far: attention falls off with distance, or sits on earlier copies of the same word.
function heads(n: number, span: number): { name: string; weights: number[] }[] {
  const last = n - 1
  const word = (i: number): string => (TOKENS[i]?.text ?? '').trim().toLowerCase()
  const recent = Array.from({ length: n }, (_, i) => Math.exp(-(last - i) / span))
  const repeat = Array.from({ length: n }, (_, i) => (i < last && word(i) === word(last) ? 1 : 0.02))
  const norm = (w: number[]): number[] => {
    const s = w.reduce((a, b) => a + b, 0) || 1
    return w.map((v) => v / s)
  }
  return [
    { name: 'head 1: recent tokens', weights: norm(recent) },
    { name: 'head 2: same word', weights: norm(repeat) },
  ]
}

const agent: Method<State, Config> = defineMethod<State, Config>({
  name: 'agent',
  config: dict({ span: real(0.5, 12, { default: 3, label: 'reach of head 1 (tokens)' }) }),
  make: (c) => {
    const span = c['span'] as number
    return {
      name: 'agent',
      config: { span },
      init: () => ({ k: 0, span }),
      step: (s, ctx) => {
        ctx.log('tokens', PROMPT + s.k + 1)
        return { k: s.k + 1, span }
      },
      done: (s) => (s.k >= STEPS ? 'completed' : undefined),
      serialise: (s) => stateToJson(s),
      deserialise: (j) => stateFromJson(j) as State,
    }
  },
  contract: {
    tokens: scalar({ description: 'tokens in the transcript' }),
    text: derived<State>({
      description: 'the transcript so far, with the weights of two heads on the token just emitted',
      compute: (s) => {
        const n = PROMPT + s.k
        return transcript({
          tokens: TOKENS.slice(0, n),
          tracks: heads(n, s.span),
          emitted: n - 1,
          weightName: 'attention of the emitted token',
          caption: `step ${s.k}.`,
        })
      },
    }),
  },
  defaults: { tokens: Log() },
})

const LAYOUT: readonly Panel[] = [
  { kind: 'transcript', path: 'text', title: 'the transcript' },
  { kind: 'curves', paths: ['tokens'], title: 'tokens so far (click to scrub)' },
]

export default function TranscriptRecipe(): ReactNode {
  const [params, setParams] = useState<Config>(() => defaults(agent.config) as Config)
  const [at, setAt] = useState<number | null>(null)
  const prefs = useSessionPreferences()
  const live = useSession(agent, { config: params, mode: prefs.mode, steps: STEPS }, [JSON.stringify(params)], {
    hold: prefs.hold,
  })
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <SpaceControls space={agent.config as DictSpace} value={params} onChange={(p) => setParams(p as Config)} />
      </ControlPanel>
      <SessionPlayer live={live} at={at} onScrub={setAt} onReset={() => setParams(defaults(agent.config) as Config)} />
      <SessionPanels id="gallery/transcript" live={live} at={at} onScrub={setAt} layout={LAYOUT} />
    </div>
  )
}
