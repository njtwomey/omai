import { opticsClustersAt, opticsOrdering } from 'mdml-engine/cluster'
import { reachabilityRecord } from 'mdml-engine/exposition'
import { key, normal } from 'mdml-engine/random'
import { type DictSpace, defaults, dict, int } from 'mdml-engine/space'
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
  title: 'Reachability plot',
  question:
    'How do I show the order a density-based clustering visits the points in, and where a cut would split them?',
  explain:
    "A probe that returns a `reachability` record (each position's reachability distance in the order OPTICS visited the points, the cut as a threshold, and optionally the clusters) drawn as bars, valleys being clusters and the dashed line the cut; shown with the probe panel. Here three blobs of different density; the cut slides down from the top over the run, and the curve beneath counts the clusters it gives, so you can see the three valleys appear and then break up. Move the core size to change how smooth the plot is.",
  shows: ['reachability', 'ReachabilityPlotView'],
}

const STEPS = 60

// Three blobs of different spread, from a fixed key.
const X: number[][] = (() => {
  const z = toArray(normal(key(7), [3 * 40 * 2])) as number[]
  const centres = [
    [0, 0, 0.4],
    [4, 0.5, 0.7],
    [1.5, 4, 0.3],
  ] as const
  return centres.flatMap((c, k) =>
    Array.from({ length: 40 }, (_, i) => [
      c[0] + c[2] * (z[2 * (k * 40 + i)] ?? 0),
      c[1] + c[2] * (z[2 * (k * 40 + i) + 1] ?? 0),
    ]),
  )
})()

interface State {
  readonly k: number
  readonly minSamples: number
}

const orderingOf = (minSamples: number): ReturnType<typeof opticsOrdering> => opticsOrdering(X, { minSamples })
// The cut slides from the top of the plot towards its floor over the run.
const cutAt = (reach: readonly number[], k: number): number => {
  const finite = reach.filter(Number.isFinite)
  const top = Math.max(...finite)
  return top * (1 - k / STEPS) ** 2 + 0.02
}

const optics: Method<State, Config> = defineMethod<State, Config>({
  name: 'optics',
  config: dict({ minSamples: int(3, 15, { default: 6, label: 'core size' }) }),
  make: (c) => {
    const minSamples = c['minSamples'] as number
    const o = orderingOf(minSamples)
    return {
      name: 'optics',
      config: { minSamples },
      init: () => ({ k: 0, minSamples }),
      step: (s, ctx) => {
        const labels = opticsClustersAt(o, cutAt(o.reachability, s.k + 1))
        ctx.log('clusters', new Set(labels.filter((l) => l >= 0)).size)
        return { k: s.k + 1, minSamples }
      },
      done: (s) => (s.k >= STEPS ? 'completed' : undefined),
      serialise: (s) => stateToJson(s),
      deserialise: (j) => stateFromJson(j) as State,
    }
  },
  contract: {
    clusters: scalar({ description: 'clusters found at the cut' }),
    plot: derived<State>({
      description: 'the reachability plot with the cut of the step',
      compute: (s) => {
        const o = orderingOf(s.minSamples)
        return reachabilityRecord({
          order: o.ordering,
          reach: o.ordering.map((i) => {
            const r = o.reachability[i] ?? Number.POSITIVE_INFINITY
            return Number.isFinite(r) ? r : null
          }),
          threshold: cutAt(o.reachability, s.k),
        })
      },
    }),
  },
  defaults: { clusters: Log() },
})

const LAYOUT: readonly Panel[] = [
  { kind: 'probe', path: 'plot', title: 'reachability distance in visiting order' },
  { kind: 'curves', paths: ['clusters'], title: 'clusters at the cut (click to scrub)' },
]

export default function ReachabilityRecipe(): ReactNode {
  const [params, setParams] = useState<Config>(() => defaults(optics.config) as Config)
  const [at, setAt] = useState<number | null>(null)
  const prefs = useSessionPreferences()
  const live = useSession(optics, { config: params, mode: prefs.mode, steps: STEPS }, [JSON.stringify(params)], {
    hold: prefs.hold,
  })
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <SpaceControls space={optics.config as DictSpace} value={params} onChange={(p) => setParams(p as Config)} />
      </ControlPanel>
      <SessionPlayer live={live} at={at} onScrub={setAt} onReset={() => setParams(defaults(optics.config) as Config)} />
      <SessionPanels id="gallery/reachability" live={live} at={at} onScrub={setAt} layout={LAYOUT} />
    </div>
  )
}
