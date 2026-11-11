import { modelRunFromTape } from 'omai-engine/exposition'
import { type BoundModel, toStructure } from 'omai-engine/probprog/model'
import type { Session } from 'omai-engine/trace'
import { band, type Curve, cartesian, lines, type PointSet, points } from 'omai-render/charts'
import type { Figure } from 'omai-render/figure'
import {
  FigurePanel,
  ModelRunPlayer,
  ResponsiveChart,
  StructureView,
  useColourMode,
  usePanels,
} from 'omai-render/react'
import { modelGraph } from 'omai-render/structures'
import { type ReactNode, useCallback, useMemo } from 'react'
import { CodeBlock } from '../../../../site/src/shell/CodeBlock.tsx'
import { snippetOf } from '../../../../site/src/shell/snippet.ts'
import { type Agreement, passes, type Recovery, TOLERANCE, type Tolerance } from './recovery.ts'
import type { ReferenceState } from './useReference.ts'

/** A number to three significant digits, a dash when it is not one. */
export const sig = (v: number, digits = 3): string => (Number.isFinite(v) ? String(Number(v.toPrecision(digits))) : '-')

/** One band with its centre line: what EP believes about a curve, drawn as the posterior mean inside its interval. */
export interface BandSeries {
  readonly name: string
  readonly x: readonly number[]
  readonly mean: readonly number[]
  readonly lower: readonly number[]
  readonly upper: readonly number[]
  /** A tab10 name (`tab0`...) or CSS colour. */
  readonly colour?: string
  readonly dash?: 'solid' | 'dashed' | 'dotted'
}

/**
 * The picture of a recovery: EP's posterior bands with their centres, the true curve and the data points. `bands`, `truth`
 * and `dots` must be stable (memoised).
 *
 * @param props - The bands, the truth (a curve), data points, steps (draw the truth as steps) and the axis labels.
 * @returns The chart.
 */
export function RecoveryChart(props: {
  readonly bands: readonly BandSeries[]
  readonly truth?: { readonly x: readonly number[]; readonly y: readonly number[]; readonly name?: string }
  readonly dots?: readonly PointSet[]
  readonly steps?: boolean
  readonly xLabel?: string
  readonly yLabel?: string
  readonly aspect?: number
}): ReactNode {
  const { bands, truth, dots, steps = false, xLabel, yLabel, aspect = 2.2 } = props
  const mode = useColourMode()
  const build = useCallback(
    (figure: Figure) => {
      const edge = steps ? ({ step: 'middle' } as const) : {}
      const curves: Curve[] = [
        ...bands.map((b, i) => ({
          name: b.name,
          x: b.x,
          y: b.mean,
          colour: b.colour ?? `tab${i}`,
          width: 2.2,
          dash: b.dash ?? 'solid',
          ...edge,
        })),
        ...(truth === undefined
          ? []
          : [
              {
                name: truth.name ?? 'truth',
                x: truth.x,
                y: truth.y,
                colour: 'fg',
                width: 2.4,
                dash: 'dashed' as const,
                ...edge,
              },
            ]),
      ]
      return cartesian(
        [
          band(
            bands.map((b, i) => ({
              x: b.x,
              lower: b.lower,
              upper: b.upper,
              colour: b.colour ?? `tab${i}`,
              opacity: 0.22,
              name: `${b.name} 90%`,
            })),
            { mode },
          ),
          lines(curves, { mode }),
          ...(dots === undefined ? [] : [points(dots, { mode, colourOffset: 3 })]),
        ],
        {
          mode,
          figure,
          tooltip: 'axis',
          ...(xLabel === undefined ? {} : { x: { label: xLabel } }),
          ...(yLabel === undefined ? {} : { y: { label: yLabel } }),
        },
      )
    },
    [bands, truth, dots, steps, xLabel, yLabel, mode],
  )
  return <ResponsiveChart aspect={aspect} narrowAspect={1.5} build={build} />
}

/**
 * The verdict of a recovery check: a PASS or FAIL badge, the numbers, the tolerance and a note (why EP is poor, when it is).
 *
 * @param props - The recoveries (a labelled row each), the tolerance and the note.
 * @returns The check.
 */
export function RecoveryCheck(props: {
  readonly rows: readonly { readonly label: string; readonly recovery: Recovery; readonly unit?: string }[]
  readonly tolerance?: Tolerance
  readonly note?: ReactNode
}): ReactNode {
  const { rows, tolerance = TOLERANCE, note } = props
  const ok = rows.every((r) => passes(r.recovery, tolerance))
  const colour = ok ? 'var(--wb-success)' : 'var(--wb-danger)'
  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2">
        <span
          className="inline-block rounded border px-2 py-0.5 text-xs font-semibold uppercase tracking-wide"
          style={{ color: colour, borderColor: colour }}
        >
          {ok ? 'pass' : 'fail'}
        </span>
        <span className="text-xs text-muted">
          the truth inside the 90% intervals at least {Math.round(100 * tolerance.coverage)}% of the time, and the error
          at most {tolerance.zrms} posterior sds (root mean square)
        </span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm tabular-nums">
          <thead className="text-xs text-muted">
            <tr>
              <th className="py-1 font-normal">quantity</th>
              <th className="font-normal">inside the 90% interval</th>
              <th className="font-normal">rmse</th>
              <th className="font-normal">mean posterior sd</th>
              <th className="font-normal">error / sd</th>
              <th className="font-normal" />
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const good = passes(r.recovery, tolerance)
              return (
                <tr key={r.label} className="border-t border-line">
                  <td className="py-1">{r.label}</td>
                  <td>{sig(100 * r.recovery.coverage, 3)}%</td>
                  <td>{sig(r.recovery.rmse, 3)}</td>
                  <td>{sig(r.recovery.meanSd, 3)}</td>
                  <td>{sig(r.recovery.zrms, 3)}</td>
                  <td style={{ color: good ? 'var(--wb-success)' : 'var(--wb-danger)' }}>{good ? 'pass' : 'FAIL'}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      {note === undefined ? null : <div className="max-w-3xl text-xs text-muted">{note}</div>}
    </div>
  )
}

/**
 * The program's own file at the bottom of the page, titled with its name: the source is imported with `?raw`, so the code
 * on screen is the code that ran. `// region` markers pick what is shown, `// hide` drops a line.
 *
 * @param props - The file's source and its name.
 * @returns The code block.
 */
export function ProgramSource(props: { readonly source: string; readonly file: string }): ReactNode {
  return (
    <section className="flex flex-col gap-2">
      <h2 className="text-sm font-semibold">The program</h2>
      <CodeBlock code={snippetOf(props.source)} title={props.file} />
    </section>
  )
}

/**
 * The frame of a program page: what the model is, the controls that change the data, the recovery figure and its check, the
 * model as a graph and the messages EP sent (folded panels), and the program's source at the bottom.
 *
 * @param props - The sections and the run: the bound model and the session whose tape is shown as messages.
 * @returns The page.
 */
export function ProgramFrame(props: {
  readonly id: string
  readonly intro: ReactNode
  readonly controls: ReactNode
  readonly status: ReactNode
  readonly figures: ReactNode
  readonly check: ReactNode
  readonly bound: BoundModel
  // biome-ignore lint/suspicious/noExplicitAny: the session's state and config types are the method's
  readonly run: Session<any, any, any>
  readonly extra?: ReactNode
  readonly source: string
  readonly file: string
}): ReactNode {
  const { id, bound, run } = props
  const panels = usePanels(`ep-program-${id}`, {
    graph: 'The model as a diagram',
    messages: 'The messages, step by step',
  })
  const graph = useMemo(() => modelGraph(toStructure(bound)), [bound])
  const record = useMemo(
    () =>
      panels.active('messages')
        ? modelRunFromTape(
            toStructure(bound, { view: 'factor' }),
            { run: run.run() },
            { maxSteps: 120, maxMessages: 6 },
          )
        : undefined,
    [bound, run, panels],
  )
  return (
    <div className="flex flex-col gap-5">
      <section className="flex max-w-3xl flex-col gap-2 text-sm">{props.intro}</section>
      {props.controls}
      <section className="flex flex-col gap-2">
        <h2 className="text-sm font-semibold">What EP recovered</h2>
        <p className="text-xs text-muted" role="status">
          {props.status}
        </p>
        {props.figures}
      </section>
      <section className="flex flex-col gap-2">
        <h2 className="text-sm font-semibold">Did it recover the truth?</h2>
        {props.check}
      </section>
      {props.extra}
      <FigurePanel panels={panels} name="graph">
        <StructureView structure={graph} frame={{ filename: `ep-program-${id}-model` }} />
      </FigurePanel>
      <FigurePanel panels={panels} name="messages">
        {record === undefined ? null : <ModelRunPlayer record={record} filename={`ep-program-${id}-messages`} />}
      </FigurePanel>
      <ProgramSource source={props.source} file={props.file} />
    </div>
  )
}

/** How a run ended, in a sentence. */
export const endedText = (run: { readonly status?: string | undefined; readonly t: number }, ms: number): string =>
  `${run.status === 'converged' ? 'Converged' : 'Stopped without converging'} after ${run.t} sweeps (${sig(ms, 2)} ms).`

/**
 * The reference run (a long NUTS in a worker): a button to start it, its state, and when it is in, how closely EP's means
 * and spreads agree with it.
 *
 * @param props - The reference's state, how to start and stop it, a name and the agreement (when the result is in).
 * @returns The bar.
 */
export function ReferenceBar(props: {
  readonly name: string
  readonly state: ReferenceState<unknown>
  readonly start: () => void
  readonly stop: () => void
  readonly agreement: Agreement | undefined
  readonly note?: ReactNode
}): ReactNode {
  const { state, agreement: a } = props
  return (
    <div className="flex flex-col gap-1">
      <div className="flex flex-wrap items-center gap-3 text-sm">
        {state.status === 'running' ? (
          <button type="button" className="rounded border border-line px-2 py-1 text-xs" onClick={props.stop}>
            stop
          </button>
        ) : (
          <button type="button" className="rounded border border-line px-2 py-1 text-xs" onClick={props.start}>
            run {props.name}
          </button>
        )}
        <span className="text-xs text-muted" role="status">
          {state.status === 'idle'
            ? `${props.name} runs in a worker, on request.`
            : state.status === 'running'
              ? `${props.name} is running in a worker...`
              : state.status === 'failed'
                ? `${props.name} failed: ${state.error}`
                : `${props.name} done in ${sig(state.ms / 1000, 2)} s.`}
        </span>
      </div>
      {a === undefined ? null : (
        <p className="text-sm tabular-nums">
          EP against {props.name}: the means differ by at most {sig(a.meanZ, 2)} reference sds; EP&apos;s sd is{' '}
          {sig(a.sdRatio, 3)} times the reference&apos;s on average (from {sig(a.sdRatioRange[0], 2)} to{' '}
          {sig(a.sdRatioRange[1], 2)}).
        </p>
      )}
      {props.note === undefined ? null : <div className="max-w-3xl text-xs text-muted">{props.note}</div>}
    </div>
  )
}
