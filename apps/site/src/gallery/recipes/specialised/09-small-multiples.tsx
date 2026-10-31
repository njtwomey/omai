import { sampleVectorField } from 'mdml-engine/exposition'
import { type DictSpace, defaults, dict, int } from 'mdml-engine/space'
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
  useSessionPreferences,
  useSessions,
  type Variant,
} from 'mdml-render/react'
import { type ReactNode, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Small multiples',
  question: 'How do I draw one panel many times, once per variant, per quantity or per value, on shared axes?',
  explain:
    'The layout entry `{ kind: "multiples", panel, over }` repeats one panel spec over a facet in a grid of titled cells under the one step cursor. `over: { by: "variants" }` makes a cell of each variant of a comparison (here four learning rates), with the vertical axis of the curves shared so that the cells compare by eye (`shared: false` lets each find its own scale); `{ by: "paths" }` makes a chart of each path of a curves panel; `{ by: "args", key, values }` asks the panel\'s probe once per value (a flow per curvature). Every cell follows the player.',
  shows: ['vectorField'],
}

const STEPS = 50
const GRID = Array.from({ length: 13 }, (_, i) => -2 + (4 * i) / 12)
// A quadratic bowl with a steep and a shallow direction: f = (4 x² + y²) / 2.
const grad = (x: number, y: number): [number, number] => [4 * x, y]

interface State {
  readonly k: number
  readonly rate: number
  readonly x: number
  readonly y: number
}

const SHARED = dict({ start: int(1, 4, { default: 2, label: 'start distance' }) }) as DictSpace

const descent: Method<State, Config> = defineMethod<State, Config>({
  name: 'descent',
  config: dict({ rate: int(1, 30, { default: 8, label: 'rate (hundredths)' }), start: int(1, 4, { default: 2 }) }),
  make: (c) => {
    const rate = (c['rate'] as number) / 100
    const start = c['start'] as number
    return {
      name: 'descent',
      config: { rate, start },
      init: () => ({ k: 0, rate, x: start, y: start }),
      step: (s, ctx) => {
        const g = grad(s.x, s.y)
        const next = { k: s.k + 1, rate, x: s.x - rate * g[0], y: s.y - rate * g[1] }
        ctx.log('loss', (4 * next.x ** 2 + next.y ** 2) / 2)
        ctx.log('gradient norm', Math.hypot(...grad(next.x, next.y)))
        ctx.log('distance', Math.hypot(next.x, next.y))
        return next
      },
      done: (s) => (s.k >= STEPS ? 'completed' : undefined),
      serialise: (s) => stateToJson(s),
      deserialise: (j) => stateFromJson(j) as State,
    }
  },
  contract: {
    loss: scalar({ description: 'the loss' }),
    'gradient norm': scalar({ description: 'the norm of the gradient' }),
    distance: scalar({ description: 'the distance from the minimum' }),
    flow: derived<State>({
      description: 'the direction of descent from each point, for a curvature k along x asked for as an argument',
      compute: (_, args) => {
        const k = (args as { curvature?: number } | undefined)?.curvature ?? 4
        return sampleVectorField((x, y) => [-k * x, -y], GRID, GRID, { features: ['x', 'y'] })
      },
    }),
  },
  defaults: { loss: Log(), 'gradient norm': Log(), distance: Log() },
})

const VARIANTS: readonly Variant<Config>[] = [4, 10, 20, 26].map((rate) => ({
  label: `rate ${rate / 100}`,
  config: { rate },
}))

const LAYOUT: readonly Panel[] = [
  {
    kind: 'multiples',
    title: 'the loss for each learning rate, on one scale',
    panel: { kind: 'curves', paths: ['loss'] },
    over: { by: 'variants' },
    columns: 2,
  },
  {
    kind: 'multiples',
    title: 'three quantities, every variant overlaid, each chart on its own scale',
    panel: { kind: 'curves', paths: ['loss', 'gradient norm', 'distance'] },
    over: { by: 'paths' },
    shared: false,
    columns: 3,
  },
  {
    kind: 'multiples',
    title: 'the flow of descent for three curvatures (a probe argument)',
    panel: { kind: 'field', path: 'flow', style: 'streamlines' },
    over: { by: 'args', key: 'curvature', values: [1, 4, 16], labels: ['curvature 1', 'curvature 4', 'curvature 16'] },
    columns: 3,
  },
]

export default function SmallMultiplesRecipe(): ReactNode {
  const [params, setParams] = useState<Config>(() => defaults(SHARED) as Config)
  const [at, setAt] = useState<number | null>(null)
  const prefs = useSessionPreferences()
  const live = useSessions(
    descent,
    { config: params, mode: prefs.mode, steps: STEPS },
    VARIANTS,
    [JSON.stringify(params)],
    {
      hold: prefs.hold,
    },
  )
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <SpaceControls space={SHARED} value={params} onChange={(p) => setParams(p as Config)} />
      </ControlPanel>
      <SessionPlayer live={live} at={at} onScrub={setAt} onReset={() => setParams(defaults(SHARED) as Config)} />
      <SessionPanels id="gallery/small-multiples" live={live} at={at} onScrub={setAt} layout={LAYOUT} />
    </div>
  )
}
