import { type SimplexFixedPoint, simplexRecord } from 'mdml-engine/exposition'
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
  title: 'Simplex',
  question: 'How do I show the state of a game as a point on the simplex, with the flow over it and the path it takes?',
  explain:
    'A probe that returns a `simplex` record: a triangle for three strategies (barycentric states) or a square for two populations of two (a pair of probabilities), with the paths of the state, a field of directions, and the fixed points classified by stability. Shown with the probe panel. Here replicator dynamics of rock, paper and scissors in a triangle, and of matching pennies in a square. A loss costs more than a win earns when the penalty is above zero, and the state spirals in to the mixed point; below zero it spirals out to the edge; at zero it cycles. Matching pennies always circles its centre.',
  shows: ['simplex', 'SimplexView'],
}

const STEPS = 400
const DT = 0.05

interface State {
  readonly k: number
  readonly penalty: number
  readonly rps: readonly number[]
  readonly pennies: readonly number[]
  readonly rpsPath: readonly (readonly number[])[]
  readonly penniesPath: readonly (readonly number[])[]
}

// The payoffs of rock, paper, scissors: a win is worth 1, a loss costs 1 + penalty.
const payoff = (penalty: number): number[][] => [
  [0, -1 - penalty, 1],
  [1, 0, -1 - penalty],
  [-1 - penalty, 1, 0],
]

// The replicator field at a state: each share grows by how much its payoff beats the average.
const replicator = (x: readonly number[], penalty: number): number[] => {
  const a = payoff(penalty)
  const f = a.map((row) => row.reduce((t, v, j) => t + v * (x[j] ?? 0), 0))
  const mean = f.reduce((t, v, i) => t + v * (x[i] ?? 0), 0)
  return x.map((xi, i) => xi * ((f[i] ?? 0) - mean))
}

// Matching pennies: one population wants to match, the other to mismatch, so the shares chase each other round.
const pennies = (p: readonly number[]): number[] => {
  const [x = 0.5, y = 0.5] = p
  return [x * (1 - x) * (2 * y - 1) * 2, -y * (1 - y) * (2 * x - 1) * 2]
}

const stepBy = (x: readonly number[], f: (x: readonly number[]) => number[]): number[] => {
  // Midpoint rule, then back onto the simplex against rounding.
  const k1 = f(x)
  const mid = x.map((v, i) => v + (DT / 2) * (k1[i] ?? 0))
  const k2 = f(mid)
  return x.map((v, i) => Math.max(0, v + DT * (k2[i] ?? 0)))
}
const normalised = (x: number[]): number[] => {
  const total = x.reduce((a, b) => a + b, 0)
  return x.map((v) => v / total)
}

const GRID: number[][] = []
for (let i = 1; i < 10; i++) for (let j = 1; j < 10 - i; j++) GRID.push([i / 10, j / 10, (10 - i - j) / 10])

const games: Method<State, Config> = defineMethod<State, Config>({
  name: 'games',
  config: dict({ penalty: real(-0.4, 0.4, { default: 0.15, label: 'cost of a loss beyond 1' }) }),
  make: (c) => {
    const penalty = c['penalty'] as number
    return {
      name: 'games',
      config: { penalty },
      init: () => ({
        k: 0,
        penalty,
        rps: [0.6, 0.25, 0.15],
        pennies: [0.8, 0.3],
        rpsPath: [[0.6, 0.25, 0.15]],
        penniesPath: [[0.8, 0.3]],
      }),
      step: (s, ctx) => {
        const rps = normalised(stepBy(s.rps, (x) => replicator(x, penalty)))
        const pen = stepBy(s.pennies, pennies).map((v) => Math.min(1, v))
        ctx.log(
          'distance from the mixed point',
          Math.hypot((rps[0] ?? 0) - 1 / 3, (rps[1] ?? 0) - 1 / 3, (rps[2] ?? 0) - 1 / 3),
        )
        return {
          k: s.k + 1,
          penalty,
          rps,
          pennies: pen,
          rpsPath: [...s.rpsPath, rps],
          penniesPath: [...s.penniesPath, pen],
        }
      },
      done: (s) => (s.k >= STEPS ? 'completed' : undefined),
      serialise: (s) => stateToJson(s),
      deserialise: (j) => stateFromJson(j) as State,
    }
  },
  contract: {
    'distance from the mixed point': scalar({ description: 'how far rock-paper-scissors is from one third each' }),
    triangle: derived<State>({
      description: 'the three-strategy game on the simplex, with its field, path and mixed point',
      compute: (s) => {
        const mixed: SimplexFixedPoint = {
          at: [1 / 3, 1 / 3, 1 / 3],
          stability: Math.abs(s.penalty) < 0.02 ? 'centre' : s.penalty > 0 ? 'stable' : 'unstable',
          label: 'mixed',
        }
        return simplexRecord({
          shape: 'triangle',
          labels: ['rock', 'paper', 'scissors'],
          caption: 'replicator dynamics of rock, paper, scissors',
          paths: [{ name: 'state', points: s.rpsPath as number[][] }],
          field: { at: GRID, by: GRID.map((x) => replicator(x, s.penalty)) },
          fixedPoints: [mixed],
        })
      },
    }),
    square: derived<State>({
      description: 'matching pennies: two populations, the share of heads in each',
      compute: (s) =>
        simplexRecord({
          shape: 'square',
          labels: ['row plays heads', 'column plays heads'],
          caption: 'matching pennies',
          paths: [{ name: 'state', points: s.penniesPath as number[][] }],
          field: {
            at: Array.from({ length: 81 }, (_, i) => [0.05 + 0.1 * (i % 9), 0.05 + 0.1 * Math.floor(i / 9)]),
            by: Array.from({ length: 81 }, (_, i) => pennies([0.05 + 0.1 * (i % 9), 0.05 + 0.1 * Math.floor(i / 9)])),
          },
          fixedPoints: [{ at: [0.5, 0.5], stability: 'centre', label: 'mixed' }],
        }),
    }),
  },
  defaults: { 'distance from the mixed point': Log() },
})

const LAYOUT: readonly Panel[] = [
  { kind: 'probe', path: 'triangle', title: 'rock, paper, scissors' },
  { kind: 'probe', path: 'square', title: 'matching pennies' },
  { kind: 'curves', paths: ['distance from the mixed point'], title: 'distance from the mixed point (click to scrub)' },
]

export default function SimplexRecipe(): ReactNode {
  const [params, setParams] = useState<Config>(() => defaults(games.config) as Config)
  const [at, setAt] = useState<number | null>(null)
  const prefs = useSessionPreferences()
  const live = useSession(games, { config: params, mode: prefs.mode, steps: STEPS }, [JSON.stringify(params)], {
    hold: prefs.hold,
  })
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <SpaceControls space={games.config as DictSpace} value={params} onChange={(p) => setParams(p as Config)} />
      </ControlPanel>
      <SessionPlayer live={live} at={at} onScrub={setAt} onReset={() => setParams(defaults(games.config) as Config)} />
      <SessionPanels id="gallery/simplex" live={live} at={at} onScrub={setAt} layout={LAYOUT} />
    </div>
  )
}
