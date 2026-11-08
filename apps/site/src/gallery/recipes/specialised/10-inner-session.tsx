import { dict, real } from 'omai-engine/space'
import {
  type Config,
  defineMethod,
  derived,
  Log,
  type Method,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from 'omai-engine/trace'
import {
  type InnerSessionSpec,
  type Panel,
  SessionPanels,
  SessionPlayer,
  useSession,
  useSessionPreferences,
} from 'omai-render/react'
import { type ReactNode, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Inner session',
  question: 'How do I show the inner loop of a method, an optimiser inside a training step, with its own cursor?',
  explain:
    'The panel `{ kind: "inner", name, layout }` embeds a session inside a step of the outer one. The page registers the inner session by name on `SessionPanels` (`inner={{ name: { method, options } }}`): its options come from what the outer run holds at the viewed step, here a learning rate the outer loop adapts. The panel runs the inner session for the viewed outer step and draws its own layout over it (curves, a trajectory), with a slider for the inner cursor; with `follow: "outer"` the inner cursor is the outer one instead. Here a meta-learner raises the inner learning rate while the inner descent keeps improving and cuts it when it does not; scrub the outer cursor and the inner run is remade for that step.',
}

const OUTER_STEPS = 20
const INNER_STEPS = 25
const grad = (x: number, y: number): [number, number] => [4 * x, y]
const loss = (x: number, y: number): number => (4 * x * x + y * y) / 2

interface Inner {
  readonly k: number
  readonly x: number
  readonly y: number
  readonly rate: number
}

// The inner loop: gradient descent on a valley from (1.5, 1.5), at a learning rate its outer step chose.
const inner: Method<Inner, Config> = defineMethod<Inner, Config>({
  name: 'inner descent',
  config: dict({ rate: real(0.001, 1, { default: 0.05, label: 'learning rate' }) }),
  make: (c) => {
    const rate = c['rate'] as number
    return {
      name: 'inner descent',
      config: { rate },
      init: () => ({ k: 0, x: 1.5, y: 1.5, rate }),
      step: (s, ctx) => {
        const g = grad(s.x, s.y)
        const next = { k: s.k + 1, x: s.x - rate * g[0], y: s.y - rate * g[1], rate }
        ctx.log('loss', loss(next.x, next.y))
        ctx.log('position', [next.x, next.y])
        return next
      },
      done: (s) => (s.k >= INNER_STEPS ? 'completed' : undefined),
      serialise: (s) => stateToJson(s),
      deserialise: (j) => stateFromJson(j) as Inner,
    }
  },
  contract: {
    loss: scalar({ description: 'the loss' }),
    position: tensor({ role: 'state', shape: [2], description: 'where the iterate is' }),
  },
  defaults: { loss: Log(), position: Log({ detail: 'full' }) },
})

// The loss the inner loop ends at, for a learning rate (the loop is cheap, so the outer step just runs it).
function finalLoss(rate: number): number {
  let x = 1.5
  let y = 1.5
  for (let i = 0; i < INNER_STEPS; i++) {
    const g = grad(x, y)
    x -= rate * g[0]
    y -= rate * g[1]
  }
  return loss(x, y)
}

// The learning rate the meta-learner has at the start of outer step k: it grows while the inner loop improves and
// is cut when it does not.
function rateAt(k: number): number {
  let rate = 0.02
  let last = finalLoss(rate)
  for (let i = 0; i < k; i++) {
    const next = rate * (last > 1e-3 ? 1.35 : 0.8)
    const l = finalLoss(next)
    rate = l < last ? next : rate * 0.6
    last = Math.min(last, l)
  }
  return Math.min(rate, 0.6)
}

interface Outer {
  readonly k: number
}

const meta: Method<Outer, Config> = defineMethod<Outer, Config>({
  name: 'meta-learner',
  config: dict({}),
  make: () => ({
    name: 'meta-learner',
    config: {},
    init: () => ({ k: 0 }),
    step: (s, ctx) => {
      ctx.log('inner learning rate', rateAt(s.k + 1))
      ctx.log('inner final loss', finalLoss(rateAt(s.k + 1)))
      return { k: s.k + 1 }
    },
    done: (s) => (s.k >= OUTER_STEPS ? 'completed' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as Outer,
  }),
  contract: {
    'inner learning rate': scalar({ description: 'the learning rate the inner loop uses at this outer step' }),
    'inner final loss': scalar({ description: 'the loss the inner loop ends at' }),
    rate: derived<Outer>({
      description: 'the inner learning rate at the viewed outer step',
      compute: (s) => rateAt(s.k),
    }),
  },
  defaults: { 'inner learning rate': Log(), 'inner final loss': Log() },
})

// The inner session at an outer step: its learning rate is read from the outer run by a probe.
const INNER: Record<string, InnerSessionSpec> = {
  descent: {
    method: inner,
    options: ({ session, t }) => ({ config: { rate: session.probe(t, 'rate') as number }, steps: INNER_STEPS }),
  },
}

const INNER_LAYOUT: readonly Panel[] = [
  { kind: 'curves', paths: ['loss'], log: 'y', title: 'inner loss (click to move the inner cursor)' },
  { kind: 'trajectory', paths: ['position'], title: 'the iterate in the valley' },
]

const LAYOUT: readonly Panel[] = [
  { kind: 'curves', paths: ['inner learning rate'], title: 'the learning rate the outer loop chose (click to scrub)' },
  { kind: 'curves', paths: ['inner final loss'], log: 'y', title: 'where the inner loop ended' },
  { kind: 'inner', name: 'descent', layout: INNER_LAYOUT, title: 'the inner loop' },
]

export default function InnerSessionRecipe(): ReactNode {
  const [at, setAt] = useState<number | null>(null)
  const prefs = useSessionPreferences()
  const live = useSession(meta, { config: {}, mode: prefs.mode, steps: OUTER_STEPS }, [], { hold: prefs.hold })
  return (
    <div className="flex flex-col gap-3">
      <SessionPlayer live={live} at={at} onScrub={setAt} />
      <SessionPanels id="gallery/inner-session" live={live} at={at} onScrub={setAt} layout={LAYOUT} inner={INNER} />
    </div>
  )
}
