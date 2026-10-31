import { type CellMark, markedTable } from 'mdml-engine/exposition'
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
  title: 'Table with marks',
  question:
    'How do I show a small table at the viewed step, with cells coloured by status and the rule that decided outlined?',
  explain:
    'A probe that returns a `markedTable` record (columns, cells, and marks on cells, rows or columns with a status and a note) and the panel `{ kind: "table", path }`. Here the Benjamini-Hochberg step-up procedure looks at ten p-values one rank at a time: each examined row is tinted by whether its p-value is under its own threshold, the rows the procedure rejects so far are tinted, and the largest rank under its threshold is outlined. Hover a tinted cell for its note; move the false discovery rate to see the line move.',
  shows: ['markedTable', 'MarkedTableView'],
}

const P = [0.001, 0.008, 0.012, 0.021, 0.034, 0.041, 0.18, 0.35, 0.62, 0.9]
const M = P.length

interface State {
  readonly k: number
  /** The false discovery rate being controlled. */
  readonly q: number
}

// The largest rank (0-based) among the first k that is under its threshold, or -1.
const largest = (k: number, q: number): number => {
  let at = -1
  for (let i = 0; i < k; i++) if ((P[i] ?? 1) <= ((i + 1) / M) * q) at = i
  return at
}

const procedure: Method<State, Config> = defineMethod<State, Config>({
  name: 'step-up',
  config: dict({ q: real(0.01, 0.4, { default: 0.1, label: 'false discovery rate q' }) }),
  make: (c) => {
    const q = c['q'] as number
    return {
      name: 'step-up',
      config: { q },
      init: () => ({ k: 0, q }),
      step: (s, ctx) => {
        const k = s.k + 1
        ctx.log('rejected', largest(k, q) + 1)
        return { k, q }
      },
      done: (s) => (s.k >= M ? 'completed' : undefined),
      serialise: (s) => stateToJson(s),
      deserialise: (j) => stateFromJson(j) as State,
    }
  },
  contract: {
    rejected: scalar({ description: 'hypotheses rejected so far' }),
    ranks: derived<State>({
      description: 'the ranked p-values with the threshold of each rank, as examined so far',
      compute: (s) => {
        const at = largest(s.k, s.q)
        const marks: CellMark[] = []
        for (const [i, p] of P.entries()) {
          const threshold = ((i + 1) / M) * s.q
          if (i >= s.k) marks.push({ row: i, status: 'muted' })
          else
            marks.push({
              row: i,
              col: 3,
              status: p <= threshold ? 'ok' : 'warn',
              note: `p = ${p}, threshold ${threshold.toFixed(3)}`,
            })
          // Everything up to the largest rank under its threshold is rejected.
          if (i <= at) marks.push({ row: i, col: 1, status: 'ok', note: 'rejected' })
        }
        if (at >= 0) marks.push({ row: at, status: 'decision', note: 'the largest rank under its threshold' })
        return markedTable({
          caption: `rank ${s.k} of ${M} examined, q = ${s.q.toFixed(2)}`,
          columns: ['rank', 'p-value', 'threshold (rank / m) q', 'p under threshold'],
          cells: P.map((p, i) =>
            i < s.k ? [i + 1, p, ((i + 1) / M) * s.q, p <= ((i + 1) / M) * s.q ? 'yes' : 'no'] : [i + 1, p, null, null],
          ),
          digits: 3,
          marks,
        })
      },
    }),
  },
  defaults: { rejected: Log() },
})

const LAYOUT: readonly Panel[] = [
  { kind: 'table', path: 'ranks', title: 'the step-up procedure' },
  { kind: 'curves', paths: ['rejected'], title: 'rejections so far (click to scrub)' },
]

export default function TableRecipe(): ReactNode {
  const [params, setParams] = useState<Config>(() => defaults(procedure.config) as Config)
  const [at, setAt] = useState<number | null>(null)
  const prefs = useSessionPreferences()
  const live = useSession(procedure, { config: params, mode: prefs.mode, steps: M }, [JSON.stringify(params)], {
    hold: prefs.hold,
  })
  return (
    <div className="flex flex-col gap-3">
      <ControlPanel columns={2}>
        <SpaceControls space={procedure.config as DictSpace} value={params} onChange={(p) => setParams(p as Config)} />
      </ControlPanel>
      <SessionPlayer
        live={live}
        at={at}
        onScrub={setAt}
        onReset={() => setParams(defaults(procedure.config) as Config)}
      />
      <SessionPanels id="gallery/table" live={live} at={at} onScrub={setAt} layout={LAYOUT} />
    </div>
  )
}
