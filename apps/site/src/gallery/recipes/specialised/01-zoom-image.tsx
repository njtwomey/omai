import { decisionSurface } from 'omai-engine/exposition'
import { choice, type DictSpace, defaults, dict } from 'omai-engine/space'
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
  title: 'Zoomable picture',
  question: 'How do I let the reader pan and zoom a picture of a grid, with the method re-rendering it deeper?',
  explain:
    'The panel `{ kind: "zoom", path, view }` asks a derived path for the grid it needs by passing the viewport and the raster size as probe arguments, and draws the `decisionSurface` with `values` that comes back. Drag to pan, scroll or pinch to zoom, double-click to reset; while the gesture runs the last picture is moved and scaled, and once it stands still the probe is asked again at the new viewport. The minimap shows where the viewport sits in the whole domain, and the line under the figure reads the value under the cursor. Here the escape-time picture of the Mandelbrot set (or the burning ship) is refined by the run: each step raises the iteration budget, so playing sharpens the boundary, and zooming in at any step re-renders it.',
  shows: ['decisionSurface'],
}

const STEPS = 60
const MAX = (k: number): number => 24 + 4 * k

interface State {
  readonly k: number
  readonly set: string
}

// The smooth escape time of c = x + iy under z -> z² + c (or its burning-ship variant), `max` at points that stay.
function escapeTime(x: number, y: number, max: number, ship: boolean): number {
  let zr = 0
  let zi = 0
  for (let n = 0; n < max; n++) {
    const ar = ship ? Math.abs(zr) : zr
    const ai = ship ? Math.abs(zi) : zi
    const nr = ar * ar - ai * ai + x
    zi = 2 * ar * ai + y
    zr = nr
    const m = zr * zr + zi * zi
    if (m > 256) return n + 1 - Math.log2(Math.log(m) / 2)
  }
  return max
}

const fractal: Method<State, Config> = defineMethod<State, Config>({
  name: 'fractal',
  config: dict({ set: choice(['mandelbrot', 'burning ship'], { default: 'mandelbrot', label: 'set' }) }),
  make: (c) => {
    const set = c['set'] as string
    return {
      name: 'fractal',
      config: { set },
      init: () => ({ k: 0, set }),
      step: (s, ctx) => {
        ctx.log('iterations', MAX(s.k + 1))
        return { k: s.k + 1, set }
      },
      done: (s) => (s.k >= STEPS ? 'completed' : undefined),
      serialise: (s) => stateToJson(s),
      deserialise: (j) => stateFromJson(j) as State,
    }
  },
  contract: {
    iterations: scalar({ description: 'the iteration budget of the picture' }),
    picture: derived<State>({
      description: 'the escape time over the viewport asked for: args { view: { x, y }, size: [columns, rows] }',
      compute: (s, args) => {
        const a = (args ?? {}) as { view?: { x: [number, number]; y: [number, number] }; size?: [number, number] }
        const view = a.view ?? { x: [-2.2, 0.9], y: [-1.1, 1.1] }
        const [cols, rows] = a.size ?? [120, 90]
        const xs = Array.from({ length: cols }, (_, j) => view.x[0] + ((j + 0.5) / cols) * (view.x[1] - view.x[0]))
        const ys = Array.from({ length: rows }, (_, i) => view.y[0] + ((i + 0.5) / rows) * (view.y[1] - view.y[0]))
        const max = MAX(s.k)
        const ship = s.set === 'burning ship'
        return decisionSurface({
          features: ['re c', 'im c'],
          x: xs,
          y: ys,
          values: ys.map((y) => xs.map((x) => escapeTime(x, ship ? -y : y, max, ship))),
          field: { name: 'escape time', smooth: true },
        })
      },
    }),
  },
  defaults: { iterations: Log() },
})

const WHOLE = { x: [-2.2, 0.9], y: [-1.1, 1.1] } as const

const LAYOUT: readonly Panel[] = [
  { kind: 'zoom', path: 'picture', view: WHOLE, title: 'escape time (drag, scroll, pinch)' },
  { kind: 'curves', paths: ['iterations'], title: 'iteration budget (click to scrub)' },
]

export default function ZoomImageRecipe(): ReactNode {
  const [params, setParams] = useState<Config>(() => defaults(fractal.config) as Config)
  const [at, setAt] = useState<number | null>(null)
  const prefs = useSessionPreferences()
  const live = useSession(fractal, { config: params, mode: prefs.mode, steps: STEPS }, [JSON.stringify(params)], {
    hold: prefs.hold,
  })
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <SpaceControls space={fractal.config as DictSpace} value={params} onChange={(p) => setParams(p as Config)} />
      </ControlPanel>
      <SessionPlayer
        live={live}
        at={at}
        onScrub={setAt}
        onReset={() => setParams(defaults(fractal.config) as Config)}
      />
      <SessionPanels id="gallery/zoom-image" live={live} at={at} onScrub={setAt} layout={LAYOUT} />
    </div>
  )
}
