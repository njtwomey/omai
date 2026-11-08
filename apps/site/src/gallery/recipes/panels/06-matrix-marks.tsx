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
  tensor,
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
  title: 'Matrix with labels and marks',
  question: 'How do I show a matrix that changes with the run, with named rows and columns and marks on it?',
  explain:
    'The `tensor` panel draws a tensor path of the tape at the viewed step as a heatmap; given `rowLabels` and `colLabels` it writes the values in the cells, `marks` outline a cell, a row or a column or draw a dashed boundary (`{ cell }`, `{ row }`, `{ col }`, `{ rowEdge }`, `{ colEdge }`), and `marksPath` is a probe at the viewed step that returns more marks. Here a ratings matrix is completed by gradient steps on two factors, one user per step: the user just updated is outlined and the dashed line fences the last two films, which no one rated yet.',
}

const USERS = ['ann', 'ben', 'cat', 'dev', 'eli', 'fay']
const FILMS = ['alien', 'brazil', 'casino', 'dune', 'emma', 'fargo', 'gandhi', 'heat']
const U = USERS.length
const V = FILMS.length
const RANK = 2
// The truth: two tastes, every user a mix of them and every film too.
const taste = (i: number, k: number): number => Math.sin(1.7 * i + 2.1 * k + 0.4)
const RATING = (u: number, f: number): number =>
  3 + 1.6 * (taste(u, 0) * taste(f + 11, 0) + taste(u, 1) * taste(f + 11, 1))
// The last two films are not rated by anyone.
const SEEN = V - 2
const STEPS = 120

interface State {
  readonly k: number
  readonly users: readonly number[]
  readonly films: readonly number[]
  /** The user updated by this step, or -1 before the first. */
  readonly row: number
  readonly rate: number
}

const predict = (s: State): number[][] =>
  USERS.map((_, u) =>
    FILMS.map((__, f) =>
      (s.users.slice(u * RANK, (u + 1) * RANK) as number[]).reduce(
        (t, a, k) => t + a * (s.films[f * RANK + k] ?? 0),
        0,
      ),
    ),
  )

const completion: Method<State, Config> = defineMethod<State, Config>({
  name: 'completion',
  config: dict({ rate: real(0.05, 1.5, { default: 0.8, label: 'step size' }) }),
  make: (c) => {
    const rate = c['rate'] as number
    return {
      name: 'completion',
      config: { rate },
      init: () => ({
        k: 0,
        row: -1,
        rate,
        users: Array.from({ length: U * RANK }, (_, j) => 0.4 * Math.sin(5.3 * j + 1)),
        films: Array.from({ length: V * RANK }, (_, j) => 0.4 * Math.sin(3.7 * j + 2)),
      }),
      step: (s, ctx) => {
        // One user's factors, and the rated films', move down the squared error of the observed ratings, each by a
        // step normalised by the size of the other side's factors (the old values drive both).
        const u = s.k % U
        const users = [...s.users]
        const films = [...s.films]
        const mine = s.users.slice(u * RANK, (u + 1) * RANK)
        const mineSize = mine.reduce((t, a) => t + a * a, 0) + 0.5
        let theirSize = 0.5
        for (let f = 0; f < SEEN; f++) for (let k = 0; k < RANK; k++) theirSize += (s.films[f * RANK + k] ?? 0) ** 2
        const p0 = predict(s)
        for (let f = 0; f < SEEN; f++) {
          const e = RATING(u, f) - (p0[u]?.[f] ?? 0)
          for (let k = 0; k < RANK; k++) {
            users[u * RANK + k] = (users[u * RANK + k] ?? 0) + (s.rate * e * (s.films[f * RANK + k] ?? 0)) / theirSize
            films[f * RANK + k] = (films[f * RANK + k] ?? 0) + (s.rate * e * (mine[k] ?? 0)) / mineSize
          }
        }
        const next = { k: s.k + 1, row: u, rate: s.rate, users, films }
        const p = predict(next)
        let err = 0
        for (let a = 0; a < U; a++) for (let f = 0; f < SEEN; f++) err += (RATING(a, f) - (p[a]?.[f] ?? 0)) ** 2
        ctx.log('error', err / (U * SEEN))
        ctx.log('predicted', p)
        return next
      },
      done: (s) => (s.k >= STEPS ? 'completed' : undefined),
      serialise: (s) => stateToJson(s),
      deserialise: (j) => stateFromJson(j) as State,
    }
  },
  contract: {
    error: scalar({ description: 'mean squared error on the rated films' }),
    predicted: tensor({ role: 'state', shape: [U, V], description: 'the predicted ratings' }),
    updated: derived<State>({
      description: 'the marks of the step: the user whose factors were just updated',
      compute: (s) => (s.row < 0 ? [] : [{ row: s.row }]),
    }),
  },
  defaults: { error: Log(), predicted: Log({ detail: 'full' }) },
})

const LAYOUT: readonly Panel[] = [
  {
    kind: 'tensor',
    path: 'predicted',
    rowLabels: USERS,
    colLabels: FILMS,
    marks: [{ colEdge: SEEN }],
    marksPath: 'updated',
    title: 'predicted ratings',
  },
  { kind: 'curves', paths: ['error'], title: 'error on the rated films (click to scrub)' },
]

export default function MatrixMarksRecipe(): ReactNode {
  const [params, setParams] = useState<Config>(() => defaults(completion.config) as Config)
  const [at, setAt] = useState<number | null>(null)
  const prefs = useSessionPreferences()
  const live = useSession(completion, { config: params, mode: prefs.mode, steps: STEPS }, [JSON.stringify(params)], {
    hold: prefs.hold,
  })
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <SpaceControls space={completion.config as DictSpace} value={params} onChange={(p) => setParams(p as Config)} />
      </ControlPanel>
      <SessionPlayer
        live={live}
        at={at}
        onScrub={setAt}
        onReset={() => setParams(defaults(completion.config) as Config)}
      />
      <SessionPanels id="gallery/matrix-marks" live={live} at={at} onScrub={setAt} layout={LAYOUT} />
    </div>
  )
}
