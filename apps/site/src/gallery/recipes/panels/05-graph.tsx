import { structureSteps } from 'mdml-engine/exposition'
import { structure } from 'mdml-engine/scene'
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
  title: 'Graph that follows the run',
  question: 'How do I show one graph whose nodes and edges change as the run goes on?',
  explain:
    'A probe that returns a `structureSteps` record (the structure once, and at a step the looks of its nodes and edges: a value as colour, a size, a label, the width of an edge) and the panel `{ kind: "graph", path }`, which draws the same nodes in the same places with the looks of the viewed step. Here heat spreads over a ring with chords from the node at the top: a node is brighter and larger the hotter it is, an edge thicker the more heat crosses it. Scrub to watch it arrive; the spread rate is a control.',
  shows: ['structureSteps', 'StructureStepsView'],
}

const N = 10
const NAMES = Array.from({ length: N }, (_, i) => `n${i}`)
// A ring with two chords.
const EDGES: readonly (readonly [number, number])[] = [
  ...Array.from({ length: N }, (_, i) => [i, (i + 1) % N] as const),
  [0, 5],
  [2, 7],
]
const GRAPH = structure(
  'graph',
  NAMES.map((id) => ({ id, label: id, role: 'latent' as const })),
  EDGES.map(([a, b]) => ({ from: NAMES[a] ?? '', to: NAMES[b] ?? '', directed: false })),
)
const STEPS = 60

interface State {
  readonly k: number
  readonly heat: readonly number[]
  readonly rate: number
}

const flows = (s: State): number[] => EDGES.map(([a, b]) => s.rate * ((s.heat[a] ?? 0) - (s.heat[b] ?? 0)))

const diffusion: Method<State, Config> = defineMethod<State, Config>({
  name: 'diffusion',
  config: dict({ rate: real(0.02, 0.3, { default: 0.15, label: 'spread rate' }) }),
  make: (c) => {
    const rate = c['rate'] as number
    return {
      name: 'diffusion',
      config: { rate },
      init: () => ({ k: 0, rate, heat: NAMES.map((_, i) => (i === 0 ? 1 : 0)) }),
      step: (s, ctx) => {
        const heat = [...s.heat]
        for (const [e, [a, b]] of EDGES.entries()) {
          const f = flows(s)[e] ?? 0
          heat[a] = (heat[a] ?? 0) - f
          heat[b] = (heat[b] ?? 0) + f
        }
        ctx.log('far node', heat[3] ?? 0)
        return { k: s.k + 1, rate, heat }
      },
      done: (s) => (s.k >= STEPS ? 'completed' : undefined),
      serialise: (s) => stateToJson(s),
      deserialise: (j) => stateFromJson(j) as State,
    }
  },
  contract: {
    'far node': scalar({ description: 'heat at n3, three hops from the source' }),
    graph: derived<State>({
      description: 'the graph with each node coloured and sized by its heat and each edge by the heat crossing it',
      compute: (s) =>
        structureSteps({
          structure: GRAPH,
          range: [0, 0.5],
          valueName: 'heat',
          steps: [
            {
              step: s.k,
              caption: `step ${s.k}`,
              nodes: Object.fromEntries(
                NAMES.map((id, i) => [id, { value: s.heat[i] ?? 0, size: 0.7 + 1.1 * (s.heat[i] ?? 0), label: id }]),
              ),
              edges: EDGES.map(([a, b], e) => ({
                from: NAMES[a] ?? '',
                to: NAMES[b] ?? '',
                width: 0.4 + 30 * Math.abs(flows(s)[e] ?? 0),
              })),
            },
          ],
        }),
    }),
  },
  defaults: { 'far node': Log() },
})

const LAYOUT: readonly Panel[] = [
  { kind: 'graph', path: 'graph', layout: 'circular', title: 'heat on the graph' },
  { kind: 'curves', paths: ['far node'], title: 'heat at n3, three hops away (click to scrub)' },
]

export default function GraphRecipe(): ReactNode {
  const [params, setParams] = useState<Config>(() => defaults(diffusion.config) as Config)
  const [at, setAt] = useState<number | null>(null)
  const prefs = useSessionPreferences()
  const live = useSession(diffusion, { config: params, mode: prefs.mode, steps: STEPS }, [JSON.stringify(params)], {
    hold: prefs.hold,
  })
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <SpaceControls space={diffusion.config as DictSpace} value={params} onChange={(p) => setParams(p as Config)} />
      </ControlPanel>
      <SessionPlayer
        live={live}
        at={at}
        onScrub={setAt}
        onReset={() => setParams(defaults(diffusion.config) as Config)}
      />
      <SessionPanels id="gallery/graph" live={live} at={at} onScrub={setAt} layout={LAYOUT} />
    </div>
  )
}
