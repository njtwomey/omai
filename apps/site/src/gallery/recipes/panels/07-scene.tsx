import { sceneSteps } from 'omai-engine/exposition'
import { arrow, circle, type SceneNode, scene, text } from 'omai-engine/scene'
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
  title: 'Scene that follows the run',
  question: 'How do I show agents on a plane, or any drawing, as it was at the viewed step?',
  explain:
    'A probe that returns a `sceneSteps` record (a `Scene` at a step, with a caption) and the panel `{ kind: "scene", path }`, which draws the latest scene recorded at or before the viewed step. Any drawing the engine can build as a scene goes this way: agents, a plant and its horizon, a game. Here twenty-four oscillators with their own natural frequencies sit on a circle, each pulled towards the mean phase with the coupling you choose; the arrow from the centre is the mean field, longer the more they agree. Below the critical coupling they drift apart; above it they lock.',
  shows: ['sceneSteps', 'SceneStepsView'],
}

const N = 24
const STEPS = 200
const DT = 0.05
const TABS = ['tab0', 'tab1', 'tab2', 'tab3', 'tab4', 'tab5', 'tab6', 'tab7', 'tab8', 'tab9'] as const
const OMEGA = Array.from({ length: N }, (_, i) => 1 + 0.8 * Math.sin(2.399 * i + 1))

interface State {
  readonly k: number
  readonly phase: readonly number[]
  readonly coupling: number
}

// The order parameter: the length and direction of the mean of the unit vectors at the phases.
const meanField = (phase: readonly number[]): { r: number; psi: number } => {
  const x = phase.reduce((t, a) => t + Math.cos(a), 0) / phase.length
  const y = phase.reduce((t, a) => t + Math.sin(a), 0) / phase.length
  return { r: Math.hypot(x, y), psi: Math.atan2(y, x) }
}

const oscillators: Method<State, Config> = defineMethod<State, Config>({
  name: 'oscillators',
  config: dict({ coupling: real(0, 3, { default: 1.6, label: 'coupling' }) }),
  make: (c) => {
    const coupling = c['coupling'] as number
    return {
      name: 'oscillators',
      config: { coupling },
      init: () => ({ k: 0, coupling, phase: OMEGA.map((_, i) => 6.283 * ((i * 0.618) % 1)) }),
      step: (s, ctx) => {
        const { r, psi } = meanField(s.phase)
        const phase = s.phase.map((a, i) => a + DT * ((OMEGA[i] ?? 1) + s.coupling * r * Math.sin(psi - a)))
        ctx.log('order', meanField(phase).r)
        return { k: s.k + 1, coupling, phase }
      },
      done: (s) => (s.k >= STEPS ? 'completed' : undefined),
      serialise: (s) => stateToJson(s),
      deserialise: (j) => stateFromJson(j) as State,
    }
  },
  contract: {
    order: scalar({ description: 'how much the phases agree, from 0 to 1' }),
    ring: derived<State>({
      description: 'the oscillators on a circle with the mean field, as a scene',
      compute: (s) => {
        const size = 320
        const c = size / 2
        const R = 125
        const { r, psi } = meanField(s.phase)
        const dots: SceneNode[] = s.phase.map((a, i) =>
          circle(c + R * Math.cos(a), c - R * Math.sin(a), 7, {
            fill: TABS[Math.min(9, Math.floor((((OMEGA[i] ?? 1) - 0.2) / 1.6) * 10))] ?? 'tab0',
            stroke: 'fg',
            strokeWidth: 1,
            title: `natural frequency ${(OMEGA[i] ?? 1).toFixed(2)}`,
          }),
        )
        return sceneSteps({
          stepName: 'step',
          steps: [
            {
              step: s.k,
              caption: `step ${s.k}, agreement r = ${r.toFixed(2)}`,
              scene: scene(size, size, [
                circle(c, c, R, { fill: 'none', stroke: 'grid', strokeWidth: 1 }),
                ...dots,
                arrow(c, c, c + R * r * Math.cos(psi), c - R * r * Math.sin(psi), { stroke: 'accent', strokeWidth: 3 }),
                text(c, 16, 'phases on the circle; colour is the natural frequency', {
                  size: 10,
                  anchor: 'middle',
                  fill: 'muted',
                }),
              ]),
            },
          ],
        })
      },
    }),
  },
  defaults: { order: Log() },
})

const LAYOUT: readonly Panel[] = [
  { kind: 'scene', path: 'ring', title: 'the oscillators' },
  { kind: 'curves', paths: ['order'], title: 'agreement (click to scrub)' },
]

export default function SceneRecipe(): ReactNode {
  const [params, setParams] = useState<Config>(() => defaults(oscillators.config) as Config)
  const [at, setAt] = useState<number | null>(null)
  const prefs = useSessionPreferences()
  const live = useSession(oscillators, { config: params, mode: prefs.mode, steps: STEPS }, [JSON.stringify(params)], {
    hold: prefs.hold,
  })
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <SpaceControls
          space={oscillators.config as DictSpace}
          value={params}
          onChange={(p) => setParams(p as Config)}
        />
      </ControlPanel>
      <SessionPlayer
        live={live}
        at={at}
        onScrub={setAt}
        onReset={() => setParams(defaults(oscillators.config) as Config)}
      />
      <SessionPanels id="gallery/scene" live={live} at={at} onScrub={setAt} layout={LAYOUT} />
    </div>
  )
}
