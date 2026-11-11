import {
  type CoverageRow,
  type EpDirection,
  type EpGap,
  type EpGapKind,
  epAtom,
  epCoverage,
  epDirections,
  epGaps,
  type VmpCoverage,
  type VmpCoverageRow,
  verifyAtom,
  vmpCoverage,
} from 'omai-engine/probprog/ep'
import { type ReactNode, useMemo, useState } from 'react'
import { Link } from '../../../../site/src/shell/Link.tsx'
import { RunProgress } from '../../shared/RunProgress.tsx'
import type { RunHost } from '../../shared/run.ts'
import { useDeferred } from '../../shared/useDeferred.ts'
import { useRun } from '../../shared/useRun.ts'

/** The variational coverage of every setup, a module-level function so that the deferred computation keeps its identity. */
const variationalCoverage = (): VmpCoverage => vmpCoverage()

const sig = (v: number): string => (Number.isFinite(v) ? String(Number(v.toPrecision(2))) : '-')

/** The outcome of running an atom's check: pass or fail with the worst error in exact sds. */
type Outcome =
  | { readonly pass: boolean; readonly worstZ: number; readonly worstKs: number }
  | { readonly error: string }

function Status(props: { readonly row: CoverageRow }): ReactNode {
  const { row } = props
  const ok = row.status === 'runs'
  const off = row.declared !== undefined && row.declared !== row.status
  return (
    <span style={{ color: off ? 'var(--wb-danger)' : ok ? 'var(--wb-success)' : 'var(--wb-warning)' }}>
      {row.status}
      {off ? ` (declared ${row.declared})` : ''}
    </span>
  )
}

/** Whether variational message passing runs a setup: runs (settled or not), or planned with the operators that lack a mean-field message as the reason. */
function VariationalStatus(props: { readonly row: VmpCoverageRow | undefined }): ReactNode {
  const { row } = props
  if (row === undefined) return <span className="text-muted">-</span>
  if (row.status === 'planned')
    return (
      <span title={row.reason} style={{ color: row.epRuns ? 'var(--wb-warning)' : 'var(--wb-muted)' }}>
        planned
      </span>
    )
  return (
    <span style={{ color: row.converged === true ? 'var(--wb-success)' : 'var(--wb-warning)' }}>
      runs{row.converged === true ? '' : ' (not settled)'}
    </span>
  )
}

const KINDS: readonly { readonly kind: EpGapKind; readonly title: string; readonly note: string }[] = [
  { kind: 'missing-rule', title: 'Missing rules', note: 'A factor EP has no message rule for.' },
  {
    kind: 'missing-mapping',
    title: 'Missing compiler mappings',
    note: 'The rule or operator exists; the model language cannot reach it yet.',
  },
  {
    kind: 'quality',
    title: 'Where EP is less good than it should be',
    note: 'It runs, and the answer is overconfident, one-sided or mislabelled.',
  },
  {
    kind: 'infrastructure',
    title: 'Infrastructure and integration',
    note: 'Evidence, variational rules, where the code lives, keeping the docs honest.',
  },
]

const STATUS_COLOUR: Readonly<Record<EpGap['status'], string>> = {
  queued: 'var(--wb-warning)',
  investigating: 'var(--wb-success)',
  open: 'var(--wb-muted)',
}

const AREAS: readonly EpDirection['area'][] = [
  'time series',
  'counts and survival',
  'ranking and preference',
  'latent structure',
  'regression and classification',
  'spatial and nonparametric',
  'inference quality',
]

const DIRECTION_COLOUR: Readonly<Record<EpDirection['status'], string>> = {
  covered: 'var(--wb-success)',
  partial: 'var(--wb-warning)',
  'in progress': 'var(--wb-accent)',
  'not yet': 'var(--wb-muted)',
}

/** One direction the suite could cover: what it is, how far it is covered and what supporting it needs. */
function Direction(props: { readonly d: EpDirection }): ReactNode {
  const { d } = props
  return (
    <div className="flex flex-col gap-1 rounded-lg border border-line p-3">
      <div className="flex flex-wrap items-baseline gap-2">
        <h4 className="text-sm font-semibold">{d.title}</h4>
        <span className="text-xs" style={{ color: DIRECTION_COLOUR[d.status] }}>
          {d.status}
        </span>
        <span className="text-xs text-muted">{d.size} to add</span>
      </div>
      <p className="m-0 text-sm text-muted">{d.what}</p>
      <p className="m-0 text-sm">
        <span className="text-muted">What it takes: </span>
        {d.needs}
      </p>
      {d.pages.length === 0 ? null : (
        <p className="m-0 text-xs text-muted">
          Shown so far in:{' '}
          {d.pages.map((s, i) => (
            <span key={s}>
              {i === 0 ? '' : ', '}
              <Link to={`/poc/${s}`} className="underline">
                {s}
              </Link>
            </span>
          ))}
        </p>
      )}
    </div>
  )
}

/** The wider suite by area: the directions the setups and programs could grow into, and what each would need. */
function Directions(): ReactNode {
  return (
    <section className="flex flex-col gap-3">
      <h2 className="text-sm font-semibold">The wider suite: what else we could cover</h2>
      <p className="max-w-3xl text-xs text-muted">
        The gaps above are only what is clearly broken today. This is the larger map: the kinds of model and problem the
        suite could grow into, how far each is covered, and what supporting it would take. It is for choosing the next
        programs and rules, not a promise; add a direction when a new idea comes up.
      </p>
      {AREAS.map((a) => {
        const items = epDirections.filter((d) => d.area === a)
        return items.length === 0 ? null : (
          <div key={a} className="flex flex-col gap-2">
            <h3 className="text-sm font-medium capitalize">{a}</h3>
            <div className="grid gap-2 lg:grid-cols-2">
              {items.map((d) => (
                <Direction key={d.id} d={d} />
              ))}
            </div>
          </div>
        )
      })}
    </section>
  )
}

/** One gap: what is missing, why it matters, how it would be closed, and which refused setups it holds up. */
function Gap(props: { readonly gap: EpGap; readonly atoms: ReadonlySet<string> }): ReactNode {
  const { gap } = props
  return (
    <div className="flex flex-col gap-1 rounded-lg border border-line p-3">
      <div className="flex flex-wrap items-baseline gap-2">
        <h3 className="text-sm font-semibold">{gap.title}</h3>
        <span className="text-xs" style={{ color: STATUS_COLOUR[gap.status] }}>
          {gap.status}
        </span>
        <span className="text-xs text-muted">{gap.size} piece of work</span>
      </div>
      <p className="m-0 text-sm text-muted">{gap.summary}</p>
      <p className="m-0 text-sm">
        <span className="text-muted">How: </span>
        {gap.plan}
      </p>
      {gap.refuses.length === 0 ? null : (
        <p className="m-0 text-xs text-muted">
          Holds up:{' '}
          {gap.refuses.map((n, i) => (
            <span key={n}>
              {i === 0 ? '' : ', '}
              {props.atoms.has(n) ? (
                <Link to={`/poc/ep-${n}`} className="underline">
                  {n}
                </Link>
              ) : (
                n
              )}
            </span>
          ))}
        </p>
      )}
    </div>
  )
}

/** The gaps of the suite by kind: nothing EP refuses is without an entry (a test holds the list to the coverage above). */
function Gaps(props: { readonly atoms: ReadonlySet<string> }): ReactNode {
  return (
    <section className="flex flex-col gap-3">
      <h2 className="text-sm font-semibold">Gaps and limitations</h2>
      <p className="max-w-3xl text-xs text-muted">
        Nothing EP refuses is abandoned: each refused setup or model above is named by a gap here, with why, how it
        would be closed and how big that is, and a test fails if one is left out or a closed gap is left in. The order
        of work is: make what runs honest first (the quality items: overconfidence, one-mode answers, false
        convergence), then give every rule its log evidence so models can be compared and hyperparameters learned, then
        close the refused models, then the remaining rules by leverage, and move the suite into the library proper.
      </p>
      {KINDS.map((k) => {
        const gaps = epGaps.filter((g) => g.kind === k.kind)
        return gaps.length === 0 ? null : (
          <div key={k.kind} className="flex flex-col gap-2">
            <h3 className="text-sm font-medium">
              {k.title}{' '}
              <span className="font-normal text-muted">
                ({gaps.length}): {k.note}
              </span>
            </h3>
            <div className="grid gap-2 lg:grid-cols-2">
              {gaps.map((g) => (
                <Gap key={g.id} gap={g} atoms={props.atoms} />
              ))}
            </div>
          </div>
        )
      })}
    </section>
  )
}

export default function EpCoverage(): ReactNode {
  const coverage = useDeferred(epCoverage)
  const variational = useDeferred(variationalCoverage)
  const variationalRows = useMemo(
    () => new Map((variational?.atoms ?? []).map((r) => [r.name, r] as const)),
    [variational],
  )
  // The checks run one setup at a time, each a slice of work, so the page paints, shows how far it is, and Stop keeps the
  // outcomes so far.
  const [queue, setQueue] = useState<readonly string[] | undefined>(undefined)
  const task = useMemo(
    () =>
      queue === undefined
        ? undefined
        : async (host: RunHost<Readonly<Record<string, Outcome>>>): Promise<Readonly<Record<string, Outcome>>> => {
            const done: Record<string, Outcome> = {}
            for (const [i, name] of queue.entries()) {
              host.report({
                stage: 'checks',
                done: i,
                total: queue.length,
                detail: `${name}: check ${i + 1} of ${queue.length}`,
              })
              await host.yieldToPage()
              if (host.stopped()) break
              try {
                const v = verifyAtom(epAtom(name))
                done[name] =
                  v.check === undefined
                    ? { error: 'EP did not run' }
                    : {
                        pass: v.check.pass,
                        worstZ: Math.max(...v.check.rows.map((r) => r.meanZ)),
                        worstKs: Math.max(...v.check.rows.map((r) => r.ks)),
                      }
              } catch (e) {
                done[name] = { error: e instanceof Error ? e.message : String(e) }
              }
              host.partial({ ...done })
            }
            return done
          },
    [queue],
  )
  const checks = useRun<Readonly<Record<string, Outcome>>>(task)
  const outcomes = checks.value ?? {}
  const running = checks.going
  if (coverage === undefined)
    return <p className="text-sm text-muted">Compiling every setup and every benchmark model...</p>
  const runnable = coverage.atoms.filter((r) => r.status === 'runs').map((r) => r.name)
  return (
    <div className="flex flex-col gap-6">
      <section className="flex flex-col gap-2">
        <h2 className="text-sm font-semibold">
          Setups: EP runs on {coverage.atomsRun} of {coverage.atoms.length}
        </h2>
        <div className="flex items-center gap-3">
          <button
            type="button"
            className="rounded border border-line px-3 py-1 text-xs hover:bg-surface disabled:opacity-50"
            disabled={running}
            onClick={() => setQueue([...runnable])}
          >
            {running ? 'Running the checks...' : `Run the ${runnable.length} checks`}
          </button>
          <span className="text-xs text-muted">
            Each check runs EP and compares it with the exact answer; a few take a second or two.
          </span>
        </div>
        <p className="max-w-3xl text-xs text-muted">
          The variational column is the mean-field method on the same setups: every operator of the graph needs a
          variational message, and the ones that have none say why (hover a planned entry).{' '}
          {variational === undefined
            ? 'Running the mean-field method on every setup...'
            : `Variational message passing runs on ${variational.atomsRun} of ${variational.atoms.length}, ${variational.atomsConverged} of them to a fixed point.`}
        </p>
        <RunProgress run={checks} label="Checks" />
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="text-xs text-muted">
              <tr>
                <th className="py-1 font-normal">setup</th>
                <th className="font-normal">family</th>
                <th className="font-normal">status</th>
                <th className="font-normal">variational</th>
                <th className="font-normal">check</th>
                <th className="font-normal">the factors, or the library’s reason</th>
              </tr>
            </thead>
            <tbody>
              {coverage.atoms.map((r) => {
                const o = outcomes[r.name]
                return (
                  <tr key={r.name} className="border-t border-line align-top">
                    <td className="py-1 pr-3">
                      <Link to={`/poc/ep-${r.name}`} className="underline">
                        {r.title}
                      </Link>
                    </td>
                    <td className="pr-3 text-muted">{r.group}</td>
                    <td className="pr-3">
                      <Status row={r} />
                    </td>
                    <td className="pr-3">
                      <VariationalStatus row={variationalRows.get(r.name)} />
                    </td>
                    <td className="pr-3 tabular-nums">
                      {o === undefined ? (
                        <span className="text-muted">-</span>
                      ) : 'error' in o ? (
                        <span style={{ color: 'var(--wb-danger)' }}>{o.error}</span>
                      ) : (
                        <span style={{ color: o.pass ? 'var(--wb-success)' : 'var(--wb-danger)' }}>
                          {o.pass ? 'pass' : 'FAIL'} (mean {sig(o.worstZ)} sd, KS {sig(o.worstKs)})
                        </span>
                      )}
                    </td>
                    <td className="max-w-xl text-xs text-muted">
                      {r.status === 'runs' ? r.factors.join('; ') : r.reason}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </section>
      <section className="flex flex-col gap-2">
        <h2 className="text-sm font-semibold">
          Benchmark models: EP runs on {coverage.benchmarksRun} of {coverage.benchmarks.length}
        </h2>
        <p className="max-w-3xl text-xs text-muted">
          Each benchmark model is compiled to a factor graph and the EP method is asked for its verdict (no run): the
          audit of docs/probprog/ep-plan.md section 1, now a tested figure that only goes up.
        </p>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="text-xs text-muted">
              <tr>
                <th className="py-1 font-normal">model</th>
                <th className="font-normal">status</th>
                <th className="font-normal">the library’s reason</th>
              </tr>
            </thead>
            <tbody>
              {coverage.benchmarks.map((r) => (
                <tr key={r.name} className="border-t border-line align-top">
                  <td className="py-1 pr-3">{r.title}</td>
                  <td className="pr-3">
                    <Status row={r} />
                  </td>
                  <td className="max-w-2xl text-xs text-muted">{r.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
      <Gaps atoms={new Set(coverage.atoms.map((r) => r.name))} />
      <Directions />
    </div>
  )
}
