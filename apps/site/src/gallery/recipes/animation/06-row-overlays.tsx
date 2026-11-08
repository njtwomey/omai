import { type DictSpace, defaults, dict, int } from 'omai-engine/space'
import { type Tensor, toFlat } from 'omai-engine/tensor'
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
  DatasetPicker,
  type Panel,
  SessionPanels,
  SessionPlayer,
  SpaceControls,
  selectEntry,
  useDataset,
  useSession,
  useSessionPreferences,
} from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Recipe } from '../../recipe.ts'

export const recipe: Recipe = {
  title: 'Row overlays',
  question: 'How do I colour, mark, size and explain the training rows at the viewed step?',
  explain:
    'The rows panel takes optional channels, each read at the viewed step from a path of the session (a probe, or a tensor of the tape): `values` a number per row as colour with a colour bar (`diverging` for weights and residuals), `size` a number per row as marker size, `marks` a category per row (or named sets of rows) as a hollow marker shape in the legend, `regions` boxes and ellipses over the plane, and `focus` one chosen row ringed with its neighbours, a path through rows, a radius and a related point. Here a toy active learner on a two-class data set: it labels two rows, then each step queries the row a one-nearest-neighbour rule is least sure of. Drag the player to watch the uncertainty, the queried rows, the boxes around what is labelled and the next query with the labelled rows nearest to it.',
}

interface State {
  readonly order: readonly number[]
}

const STEPS = 40
const SPACE: DictSpace = dict({ seeds: int(1, 4, { default: 1, label: 'labelled per class at the start' }) })

// What the learner knows at a step: the labelled rows (in query order) and, for every row, the distance to the nearest
// labelled row of each class.
function knowledge(points: readonly number[][], classes: readonly number[], order: readonly number[]) {
  const dist = (i: number, j: number): number =>
    Math.hypot((points[i]?.[0] ?? 0) - (points[j]?.[0] ?? 0), (points[i]?.[1] ?? 0) - (points[j]?.[1] ?? 0))
  const nearest = [0, 1].map((c) =>
    points.map((_, i) => {
      let best = -1
      for (const j of order) if (classes[j] === c && (best < 0 || dist(i, j) < dist(i, best))) best = j
      return best
    }),
  )
  const gap = nearest.map((rows) => rows.map((j, i) => (j < 0 ? Number.POSITIVE_INFINITY : dist(i, j))))
  const uncertainty = points.map((_, i) => {
    const a = gap[0]?.[i] ?? 0
    const b = gap[1]?.[i] ?? 0
    return order.includes(i) || !Number.isFinite(a + b) ? 0 : 1 - Math.abs(a - b) / (a + b)
  })
  const predicted = points.map((_, i) => ((gap[1]?.[i] ?? 0) < (gap[0]?.[i] ?? 0) ? 1 : 0))
  let next = -1
  uncertainty.forEach((u, i) => {
    if (next < 0 || u > (uncertainty[next] ?? 0)) next = i
  })
  return { nearest, gap, uncertainty, predicted, next }
}

// The learner, over the rows of a data set.
function learner(points: number[][], classes: number[]): Method<State, Config> {
  const first = (c: number, k: number): number[] => classes.flatMap((v, i) => (v === c ? [i] : [])).slice(0, k)
  return defineMethod<State, Config>({
    name: 'active-learner',
    config: SPACE,
    make: (c) => ({
      name: 'active-learner',
      config: { seeds: c['seeds'] as number },
      init: () => ({ order: [...first(0, c['seeds'] as number), ...first(1, c['seeds'] as number)] }),
      step: (s, ctx) => {
        const k = knowledge(points, classes, s.order)
        const order = [...s.order, k.next]
        const after = knowledge(points, classes, order)
        ctx.log('labelled', order.length)
        ctx.log('accuracy', after.predicted.filter((p, i) => p === classes[i]).length / points.length)
        return { order }
      },
      done: (s) => (s.order.length >= points.length ? 'completed' : undefined),
      serialise: (s) => stateToJson(s),
      deserialise: (j) => stateFromJson(j) as State,
    }),
    contract: {
      labelled: scalar({ description: 'how many rows are labelled' }),
      accuracy: scalar({ description: 'the share of all rows the one-nearest-neighbour rule gets right' }),
      'rows/uncertainty': derived<State>({
        description: 'how unsure the rule is of each row (1: equidistant from the two classes)',
        compute: (s) => knowledge(points, classes, s.order).uncertainty,
      }),
      'rows/queried': derived<State>({
        description: 'the labelled rows and the next query',
        compute: (s) => ({
          sets: { labelled: [...s.order], 'next query': [knowledge(points, classes, s.order).next] },
        }),
      }),
      'rows/errors': derived<State>({
        description: 'the rows the rule gets wrong',
        compute: (s) => ({
          sets: {
            wrong: knowledge(points, classes, s.order).predicted.flatMap((p, i) => (p === classes[i] ? [] : [i])),
          },
        }),
      }),
      'rows/boxes': derived<State>({
        description: 'a box around the labelled rows of each class',
        compute: (s) =>
          [0, 1].flatMap((c) => {
            const own = s.order.filter((i) => classes[i] === c).map((i) => points[i] as number[])
            return own.length === 0
              ? []
              : [
                  {
                    lower: [Math.min(...own.map((p) => p[0] ?? 0)), Math.min(...own.map((p) => p[1] ?? 0))],
                    upper: [Math.max(...own.map((p) => p[0] ?? 0)), Math.max(...own.map((p) => p[1] ?? 0))],
                    label: `labelled, class ${c}`,
                  },
                ]
          }),
      }),
      'rows/next': derived<State>({
        description: 'the next query with the nearest labelled row of each class, the last queries as a path',
        compute: (s) => {
          const k = knowledge(points, classes, s.order)
          return {
            row: k.next,
            neighbours: k.nearest.map((rows) => rows[k.next] ?? -1).filter((j) => j >= 0),
            path: [...s.order.slice(-5), k.next],
            radius: Math.min(k.gap[0]?.[k.next] ?? 0, k.gap[1]?.[k.next] ?? 0),
          }
        },
      }),
    },
    defaults: { labelled: Log(), accuracy: Log() },
  })
}

const LAYOUT: readonly Panel[] = [
  {
    kind: 'rows',
    title: 'uncertainty (colour), labelled rows and the next query (hollow marks)',
    values: { path: 'rows/uncertainty', label: 'uncertainty', range: [0, 1] },
    marks: { path: 'rows/queried' },
    focus: { path: 'rows/next' },
  },
  {
    kind: 'rows',
    title: 'what is labelled (boxes), what the rule gets wrong, uncertain rows large',
    size: { path: 'rows/uncertainty', range: [3, 14] },
    regions: { path: 'rows/boxes' },
    marks: { path: 'rows/errors' },
  },
  { kind: 'curves', paths: ['accuracy'], title: 'accuracy of the rule on all rows (click to scrub)' },
]

export default function RowOverlaysRecipe(): ReactNode {
  const [selection, setSelection] = useState(() => selectEntry('classification2d', 'moons'))
  const [params, setParams] = useState<Config>(() => defaults(SPACE) as Config)
  const [at, setAt] = useState<number | null>(null)
  const prefs = useSessionPreferences()
  const { data } = useDataset('classification2d', selection)
  const rows = useMemo(() => {
    if (!('x' in data) || !('y' in data)) return undefined
    const flat = toFlat(data.x)
    const points = Array.from({ length: flat.length / 2 }, (_, i) => [flat[2 * i] ?? 0, flat[2 * i + 1] ?? 0])
    return { points, classes: Array.from(toFlat(data.y), (v) => (v > 0 ? 1 : 0)) }
  }, [data])
  const method = useMemo(() => (rows === undefined ? undefined : learner(rows.points, rows.classes)), [rows])
  if (method === undefined || !('x' in data) || !('y' in data)) return <p>This recipe needs a two-class data set.</p>
  return (
    <Run
      method={method}
      selection={selection}
      setSelection={setSelection}
      params={params}
      setParams={setParams}
      at={at}
      setAt={setAt}
      prefs={prefs}
      x={data.x}
      y={data.y}
    />
  )
}

// The session needs the method, so it lives below the guard that makes it.
function Run(props: {
  readonly method: Method<State, Config>
  readonly selection: ReturnType<typeof selectEntry>
  readonly setSelection: (s: ReturnType<typeof selectEntry>) => void
  readonly params: Config
  readonly setParams: (c: Config) => void
  readonly at: number | null
  readonly setAt: (t: number | null) => void
  readonly prefs: ReturnType<typeof useSessionPreferences>
  readonly x: Tensor
  readonly y: Tensor
}): ReactNode {
  const { method, params, at, setAt, prefs } = props
  const live = useSession(
    method,
    { config: params, mode: prefs.mode, steps: STEPS },
    [method, JSON.stringify(params)],
    {
      hold: prefs.hold,
    },
  )
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={3}>
        <DatasetPicker
          collection="classification2d"
          value={props.selection}
          onChange={props.setSelection}
          exclude={['n']}
        />
        <SpaceControls space={SPACE} value={params} onChange={(p) => props.setParams(p as Config)} />
      </ControlPanel>
      <SessionPlayer live={live} at={at} onScrub={setAt} onReset={() => props.setParams(defaults(SPACE) as Config)} />
      <SessionPanels
        id="gallery/row-overlays"
        live={live}
        at={at}
        onScrub={setAt}
        layout={LAYOUT}
        x={props.x}
        y={props.y}
      />
    </div>
  )
}
