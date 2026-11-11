import { choice, type DictSpace, defaults, dict } from 'omai-engine/space'
import { type Config, Log, metricSeries } from 'omai-engine/trace'
import { cartesian, lines } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import type { ProducerFactory } from 'omai-render/producer'
import {
  Chart,
  ControlPanel,
  FigurePanel,
  type LiveSession,
  type Panels,
  PanelsStrip,
  SeedInput,
  SessionPlayer,
  SpaceControls,
  useColourMode,
  usePanels,
  useProbe,
  useRun,
  useSession,
} from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import type { Example, ExampleProblem } from '../examples/index.ts'
import { ExampleSource, SOURCE_PANEL } from './ExampleSource.tsx'
import { exampleCode } from './exampleCode.ts'
import { StaleVeil } from './StaleVeil.tsx'

// biome-ignore lint/suspicious/noExplicitAny: a session is generic over its method
export type Live = LiveSession<any, any>

const CURVE = figure({ width: 520, height: 300 })

/** The knobs of a configuration named in `keys`. */
export const pick = (config: Config, keys: readonly string[]): Config =>
  Object.fromEntries(keys.filter((k) => k in config).map((k) => [k, config[k]])) as Config

/** A number for a table: `digits` decimals, or a dash when it is not a finite number. */
export const fmt = (v: unknown, digits = 3): string =>
  typeof v === 'number' && Number.isFinite(v) ? v.toFixed(digits) : '-'

/** A probe's record when it is the kind asked for (`path` undefined asks for nothing: a panel that is not on show). */
export function useRecord<R>(
  live: Live,
  t: number,
  path: string | undefined,
  kind: string,
  args?: never,
): R | undefined {
  const p = useProbe(live, t, path, args)
  const v = p.value as { kind?: string } | null | undefined
  return v?.kind === kind ? (v as R) : undefined
}

/** What an example's output panels get: the chosen run, the reference run, the viewed step and the panels' state. */
export interface ExampleView {
  readonly live: Live
  /** The reference run, when the page has one (a check against an exact answer needs none). */
  readonly reference: Live | undefined
  readonly t: number
  readonly config: Config
  readonly panels: Panels
  /** The chosen method's curve on the tape (variational methods), when it has one. */
  readonly objective: { readonly path: string; readonly label: string } | undefined
  /** Whether the chosen run has ended (finished, converged or stopped by a budget). */
  readonly done: boolean
}

/**
 * The shape of every Bayesian example page: the experiment (data and model panels), the inference (one method, its knobs
 * and the budget) and, under the run, what to observe and how we know it works. The page gives the example's method,
 * the producer that computes it in a worker, a reference run (its inference knobs: a long run of the same problem,
 * started once the chosen run has ended, in a session of its own) and the two output parts; this runs the chosen
 * session and the reference, the player, the status line and the strip of panels that can be folded or hidden.
 *
 * @param props - See the fields.
 * @returns The page.
 */
export function ExamplePage(props: {
  /** The page's slug, the key of what its panels remember. */
  readonly id: string
  readonly example: Example<ExampleProblem>
  /** The method as the page's worker serves it (the page's own `method.ts`), and the producer of that worker. */
  readonly producer: ProducerFactory
  /** The panels of the page, by name with their titles; the panel named `check` is the reference check. */
  readonly panels: Readonly<Record<string, string>>
  /** The inference knobs of the reference run, over the defaults of the chosen method; none when the check is against an exact answer. */
  readonly reference?: Config
  readonly referenceSeed?: number
  /** The time budget of the reference run in milliseconds (default 20 s). */
  readonly referenceBudgetMs?: number
  /** Whether a configuration's check needs the reference run (default always, when there is one): a check against an exact answer needs none. */
  readonly referenceFor?: (config: Config) => boolean
  /** The knob that is the data seed (shown as a seed input, never a button group). Default `dataSeed`. */
  readonly seedKnob?: string
  /** Knobs to leave out of the data and model panels (a list, or a function of the configuration). */
  readonly exclude?: readonly string[] | ((config: Config) => readonly string[])
  /** What to observe: the panels of the chosen method's output. */
  readonly observe: (view: ExampleView) => ReactNode
  /** How we know it works: the reference check (usually {@link ReferenceCheck}). */
  readonly check: (view: ExampleView) => ReactNode
}): ReactNode {
  const { example, producer, seedKnob = 'dataSeed' } = props
  const method = example.method
  const start = useMemo(() => defaults(method.config) as Config, [method])
  const [config, setConfig] = useState<Config>(start)
  const [at, setAt] = useState<number | null>(null)
  const code = exampleCode[props.id]
  const panels = usePanels(
    props.id,
    code === undefined ? props.panels : { ...props.panels, [SOURCE_PANEL]: 'The code that ran' },
  )
  // The tape keeps what the page draws from it: the objective curve of the chosen method (a sampler's draws come from probes).
  const objective = example.objective(config)
  // Every method that runs on this problem (the verdicts are of the model the knobs make), and why the others do not.
  const offer = example.offer(config)
  const inference = (config['inference'] ?? {}) as Record<string, unknown>
  const chosen = String(inference['method'] ?? 'nuts')
  // The method is chosen from one list (the ones that cannot run are drawn disabled, with the library's reason) and its
  // own knobs follow in a panel of their own.
  const chooser = useMemo(
    () =>
      dict({
        method: choice(Object.keys(offer.space.options), {
          default: 'nuts',
          label: 'inference method',
          descriptions: Object.fromEntries(
            Object.entries(offer.space.options).map(([k, o]) => [k, offer.unavailable[k] ?? o.description ?? '']),
          ),
        }),
      }),
    [offer],
  )
  const knobSpace = (offer.space.options[chosen] ?? dict({})) as DictSpace
  const { method: _method, ...ownKnobs } = inference
  const set = (v: Config): void => setConfig((c) => ({ ...c, ...v }))
  const live = useSession(
    method,
    {
      config,
      seed: Number(config[seedKnob]),
      steps: example.steps(config),
      logs: objective === undefined ? {} : { [objective.path]: Log() },
    },
    [JSON.stringify(config)],
    { start: 'end', hold: false, producer },
  )
  const problem = JSON.stringify(pick(config, example.problemKnobs))
  // biome-ignore lint/correctness/useExhaustiveDependencies: the reference depends on the problem alone, not on the method
  const referenceConfig = useMemo(
    () => example.reference(config, props.reference ?? {}, props.referenceBudgetMs),
    [problem],
  )
  // A reference nobody can see is not computed: hiding the check panel spares the run.
  const reference = useSession(
    method,
    { config: referenceConfig, seed: props.referenceSeed ?? 7, steps: example.steps(referenceConfig) },
    [problem],
    {
      start: 'end',
      hold: false,
      producer,
      active: props.reference !== undefined && live.done && panels.active('check'),
    },
  )
  const t = Math.max(0, Math.min(at ?? live.head, live.head))
  const status = live.session.status
  const done = status !== undefined
  const view: ExampleView = {
    live,
    reference: props.reference === undefined ? undefined : reference,
    t,
    config,
    panels,
    objective,
    done,
  }
  const exclude = typeof props.exclude === 'function' ? props.exclude(config) : (props.exclude ?? [])
  return (
    <div className="flex flex-col gap-5">
      <section className="flex flex-col gap-2">
        <h2 className="text-sm font-semibold">The experiment</h2>
        <div className="grid grid-cols-1 items-start gap-3 md:grid-cols-2">
          <ControlPanel title="data" columns={2}>
            <SpaceControls
              space={example.dataKnobs}
              value={pick(config, Object.keys(example.dataKnobs.spaces))}
              onChange={set}
              exclude={[seedKnob, ...exclude]}
              segmentedUpTo={1}
            />
            {seedKnob in example.dataKnobs.spaces ? (
              <SeedInput value={Number(config[seedKnob])} onChange={(v) => set({ [seedKnob]: v })} />
            ) : null}
          </ControlPanel>
          <ControlPanel title="model" columns={2}>
            <SpaceControls
              space={example.modelKnobs}
              value={pick(config, Object.keys(example.modelKnobs.spaces))}
              onChange={set}
              exclude={exclude}
              segmentedUpTo={1}
            />
          </ControlPanel>
        </div>
      </section>
      <section className="flex flex-col gap-2">
        <h2 className="text-sm font-semibold">The inference</h2>
        <ControlPanel title="one method and the budget (0 for none)" columns={4}>
          <SpaceControls
            space={chooser}
            value={{ method: chosen }}
            onChange={(v) =>
              set({
                inference: {
                  ...(defaults(offer.space.options[String(v['method'])] as DictSpace) as Config),
                  method: v['method'],
                },
              })
            }
            segmentedUpTo={0}
            admissible={(_, option) => !(String(option) in offer.unavailable)}
            why={(_, option) => offer.unavailable[String(option)]}
          />
          <SpaceControls
            space={example.budgetKnobs}
            value={pick(config, Object.keys(example.budgetKnobs.spaces))}
            onChange={set}
          />
          <details className="col-span-full text-xs text-muted">
            <summary className="cursor-pointer">Methods that cannot run on this model, and why</summary>
            <ul className="mt-1 flex flex-col gap-1">
              {Object.entries(offer.unavailable).map(([kind, why]) => (
                <li key={kind}>
                  <b>{kind}</b>: {why}
                </li>
              ))}
            </ul>
          </details>
        </ControlPanel>
        {Object.keys(knobSpace.spaces).length === 0 ? null : (
          <ControlPanel title={`${chosen}: its knobs`} columns={4}>
            <SpaceControls
              space={knobSpace}
              value={ownKnobs as Config}
              onChange={(v) => set({ inference: { ...v, method: chosen } })}
              segmentedUpTo={0}
            />
          </ControlPanel>
        )}
        <StaleVeil stale={live.stale}>
          <div className="flex flex-col gap-4">
            <p className="text-xs text-muted" role="status">
              {live.error !== undefined
                ? live.error
                : `${live.session.t} of about ${example.steps(config)} steps ${
                    !done ? 'computing' : status === 'stopped' ? 'stopped by the budget' : 'finished'
                  }${live.producer?.kind === 'worker' ? ' (in a worker)' : ''}`}
            </p>
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h2 className="text-sm font-semibold">What to observe</h2>
              <PanelsStrip panels={panels} />
            </div>
            {props.observe(view)}
            <SessionPlayer live={live} at={at} onScrub={setAt} onReset={() => setConfig(start)} start="end" />
            {props.check(view)}
            {code === undefined ? null : <ExampleSource panels={panels} entry={code} name={props.id} config={config} />}
          </div>
        </StaleVeil>
      </section>
    </div>
  )
}

/** What to say while a panel has nothing to draw: still computing, or the budget ended before the first draws. */
export const Waiting = (props: { readonly done: boolean }): ReactNode => (
  <p className="text-sm text-muted">
    {props.done
      ? 'The budget ended before the first draw: raise the time budget or shorten the warm-up.'
      : 'Waiting for the first draws...'}
  </p>
)

/** A small two-column table of labelled numbers. */
export function Table(props: {
  readonly title?: string
  readonly rows: readonly (readonly [string, string])[]
}): ReactNode {
  return (
    <div>
      {props.title === undefined || props.title === '' ? null : (
        <h3 className="mb-1 text-xs font-semibold text-muted">{props.title}</h3>
      )}
      <table className="text-sm">
        <tbody>
          {props.rows.map(([k, v]) => (
            <tr key={k}>
              <td className="pr-6 text-muted">{k}</td>
              <td className="tabular-nums">{v}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/** The objective curve of the chosen method (its path on the tape) up to the viewed step. */
export function Objective(props: {
  readonly live: Live
  readonly path: string
  readonly label: string
  readonly t: number
}): ReactNode {
  const { live, path, label, t } = props
  const mode = useColourMode()
  const run = useRun(live)
  const series = useMemo(() => {
    try {
      return metricSeries(run, path)
    } catch {
      return undefined
    }
  }, [run, path])
  const option = useMemo(() => {
    if (series === undefined) return undefined
    const keep = series.steps.map((s, i) => [s, series.values[i] ?? Number.NaN] as const).filter(([s]) => s <= t)
    return cartesian(
      [lines([{ x: keep.map((k) => k[0]), y: keep.map((k) => k[1]), name: label, width: 2 }], { mode })],
      { mode, figure: CURVE, x: { label: 'step' }, y: { label } },
    )
  }, [series, t, label, mode])
  return option === undefined ? (
    <p className="text-sm text-muted">No {label} on the tape yet.</p>
  ) : (
    <Chart option={option} figure={CURVE} />
  )
}

/**
 * The fourth part of an example, how we know it works: the panel `check` that holds the explanation of the reference and
 * a table of the chosen method's check against it. The reference's own probes (up to three, named) are asked of the
 * reference run once it has ended and handed to the method's check path as one object; nothing is asked while the panel
 * is folded or hidden.
 *
 * @param props - The view, the reference's probes, the check path and the table's rows.
 * @returns The panel.
 */
export function ReferenceCheck(props: {
  readonly view: ExampleView
  /** The probes of the reference run, by name (the check path gets `{ name: value }` of all of them). */
  readonly probes: Readonly<Record<string, { readonly path: string; readonly args?: unknown }>>
  /** The method's derived path that compares it to the reference. */
  readonly checkPath: string
  /** What the reference is and what the check measures. */
  readonly explain: ReactNode
  /** The table of the check's numbers. */
  readonly rows: (check: Record<string, unknown>) => readonly (readonly [string, string])[]
  /** More under the table (a figure of the check's record). */
  readonly more?: (check: Record<string, unknown>) => ReactNode
  /** What it says when the reference never finished (its own budget). */
  readonly unfinished?: string
}): ReactNode {
  const { view, probes, checkPath, explain, rows } = props
  const { live, t, panels } = view
  // An exact reference is no run: the method's check path reads the truth itself.
  const reference = view.reference ?? live
  const open = panels.active('check')
  const ready = view.reference === undefined || reference.done
  const rt = reference.head
  const names = Object.keys(probes)
  const slot = (i: number): { path: string | undefined; args: never } => {
    const p = names[i] === undefined ? undefined : probes[names[i]]
    return { path: open && ready ? p?.path : undefined, args: p?.args as never }
  }
  const s0 = slot(0)
  const s1 = slot(1)
  const s2 = slot(2)
  const v0 = useProbe(reference, rt, s0.path, s0.args).value
  const v1 = useProbe(reference, rt, s1.path, s1.args).value
  const v2 = useProbe(reference, rt, s2.path, s2.args).value
  const values = [v0, v1, v2]
  const args =
    open && names.every((_, i) => values[i] !== undefined && values[i] !== null)
      ? Object.fromEntries(names.map((n, i) => [n, values[i]]))
      : undefined
  const check = useProbe(live, t, args === undefined ? undefined : checkPath, args as never).value as
    | Record<string, unknown>
    | null
    | undefined
  const table = check === null || check === undefined ? [] : rows(check)
  return (
    <FigurePanel panels={panels} name="check" title="How we know it works">
      <div className="flex flex-col gap-1">
        <p className="text-xs text-muted">
          {explain}{' '}
          {!live.done
            ? 'Waiting for the run to end.'
            : !ready
              ? 'The reference is computing...'
              : args === undefined
                ? (props.unfinished ?? 'The reference run did not reach its draws within its own budget (20 s).')
                : ''}
        </p>
        {table.length > 0 ? <Table rows={table} /> : null}
        {check === null || check === undefined ? null : props.more?.(check)}
      </div>
    </FigurePanel>
  )
}
