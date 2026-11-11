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
import { RunProgress } from '../RunProgress.tsx'
import type { UseRun } from '../useRun.ts'
import { type Agreement, passes, type Recovery, TOLERANCE, type Tolerance } from './recovery.ts'

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
  /** Further curves (a reference, other series), drawn as given. */
  readonly curves?: readonly Curve[]
  readonly yLimits?: readonly [number, number]
  readonly dotOffset?: number
  readonly steps?: boolean
  readonly xLabel?: string
  readonly yLabel?: string
  readonly aspect?: number
}): ReactNode {
  const {
    bands,
    truth,
    dots,
    curves: more,
    yLimits,
    dotOffset = 3,
    steps = false,
    xLabel,
    yLabel,
    aspect = 2.2,
  } = props
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
        ...(more ?? []),
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
          ...(dots === undefined ? [] : [points(dots, { mode, colourOffset: dotOffset })]),
        ],
        {
          mode,
          figure,
          tooltip: 'axis',
          ...(xLabel === undefined ? {} : { x: { label: xLabel } }),
          ...(yLabel === undefined && yLimits === undefined
            ? {}
            : {
                y: {
                  ...(yLabel === undefined ? {} : { label: yLabel }),
                  ...(yLimits === undefined ? {} : { limits: yLimits }),
                },
              }),
        },
      )
    },
    [bands, truth, dots, more, yLimits, dotOffset, steps, xLabel, yLabel, mode],
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
  /** The bar of the run that makes the figures (a `RunProgress`), shown above them. */
  readonly progress?: ReactNode
  readonly figures: ReactNode
  readonly check: ReactNode
  /** The bound model and the session of EP's run, undefined while the run has not produced them. */
  readonly bound: BoundModel | undefined
  // biome-ignore lint/suspicious/noExplicitAny: the session's state and config types are the method's
  readonly run: Session<any, any, any> | undefined
  readonly extra?: ReactNode
  readonly source: string
  readonly file: string
}): ReactNode {
  const { id, bound, run } = props
  const panels = usePanels(`ep-program-${id}`, {
    graph: 'The model as a diagram',
    messages: 'The messages, step by step',
  })
  const graph = useMemo(() => (bound === undefined ? undefined : modelGraph(toStructure(bound))), [bound])
  const record = useMemo(
    () =>
      panels.active('messages') && bound !== undefined && run !== undefined
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
        {props.progress}
        {props.figures}
      </section>
      <section className="flex flex-col gap-2">
        <h2 className="text-sm font-semibold">Did it recover the truth?</h2>
        {props.check}
      </section>
      {props.extra}
      <FigurePanel panels={panels} name="graph">
        {graph === undefined ? null : (
          <StructureView structure={graph} frame={{ filename: `ep-program-${id}-model` }} />
        )}
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
 * The reference run (a long NUTS in a worker): a button to start it, its progress bar with a timer and a Stop button while it
 * goes, and when it is in, how closely EP's means and spreads agree with it.
 *
 * @param props - The reference's run (`useRun` with `auto: false`), a name and the agreement (when the result is in).
 * @returns The bar.
 */
export function ReferenceBar(props: {
  readonly name: string
  readonly run: UseRun<unknown>
  readonly agreement: Agreement | undefined
  readonly note?: ReactNode
}): ReactNode {
  const { run, agreement: a } = props
  return (
    <div className="flex flex-col gap-2">
      <RunProgress
        run={run}
        label={props.name}
        onStart={run.start}
        startText={`Run ${props.name}`}
        hint={`${props.name} runs in a worker, on request.`}
      />
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

/** One line of a check: what was measured, its value, the rule it is held to and whether it passes. */
export interface CheckLine {
  readonly label: string
  readonly value: string
  readonly rule: string
  readonly ok: boolean
}

/**
 * A check as a table of lines with an overall PASS or FAIL badge, for programs whose recovery is not one set of intervals
 * (regime probabilities, agreement with a reference, whether EP converged).
 *
 * @param props - The lines and a note (why EP is poor, when it is).
 * @returns The check.
 */
export function CheckLines(props: { readonly lines: readonly CheckLine[]; readonly note?: ReactNode }): ReactNode {
  const ok = props.lines.every((l) => l.ok)
  const colour = ok ? 'var(--wb-success)' : 'var(--wb-danger)'
  return (
    <div className="flex flex-col gap-2">
      <span
        className="inline-block w-fit rounded border px-2 py-0.5 text-xs font-semibold uppercase tracking-wide"
        style={{ color: colour, borderColor: colour }}
      >
        {ok ? 'pass' : 'fail'}
      </span>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm tabular-nums">
          <thead className="text-xs text-muted">
            <tr>
              <th className="py-1 font-normal">what</th>
              <th className="font-normal">value</th>
              <th className="font-normal">held to</th>
              <th className="font-normal" />
            </tr>
          </thead>
          <tbody>
            {props.lines.map((l) => (
              <tr key={l.label} className="border-t border-line">
                <td className="py-1">{l.label}</td>
                <td>{l.value}</td>
                <td className="text-muted">{l.rule}</td>
                <td style={{ color: l.ok ? 'var(--wb-success)' : 'var(--wb-danger)' }}>{l.ok ? 'pass' : 'FAIL'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {props.note === undefined ? null : <div className="max-w-3xl text-xs text-muted">{props.note}</div>}
    </div>
  )
}
