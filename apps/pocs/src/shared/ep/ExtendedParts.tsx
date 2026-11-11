import {
  type AtomProblem,
  atomRecovery,
  type Belief,
  type CalibrationSample,
  calibrationSample,
  calibrationTable,
  type EpAtom,
  epAtom,
} from 'omai-engine/bayesian-examples'
import type { Config } from 'omai-engine/trace'
import { type ReactNode, useEffect, useMemo, useState } from 'react'
import { Link } from '../../../../site/src/shell/Link.tsx'

/** A number to three significant digits, a dash when it is not one. */
const sig = (v: number, digits = 3): string => (Number.isFinite(v) ? String(Number(v.toPrecision(digits))) : '-')

const GOOD = 'var(--wb-success)'
const BAD = 'var(--wb-danger)'

function Badge(props: { readonly ok: boolean; readonly children: ReactNode }): ReactNode {
  const colour = props.ok ? GOOD : BAD
  return (
    <span
      className="inline-block rounded border px-2 py-0.5 text-xs font-semibold uppercase tracking-wide"
      style={{ color: colour, borderColor: colour }}
    >
      {props.children}
    </span>
  )
}

/** The line under the verdict banner of an extended atom: what it is the twin of, with a link to the basic page. */
export function TwinNote(props: { readonly atom: EpAtom }): ReactNode {
  const basic = props.atom.extends === undefined ? undefined : epAtom(props.atom.extends)
  if (basic === undefined) return null
  return (
    <p className="max-w-3xl text-sm">
      <b>Parameters learned.</b> This is the extended twin of{' '}
      <Link to={`/poc/ep-${basic.name}`} className="underline">
        {basic.title}
      </Link>
      : the same setup, but the numbers the basic page fixes are unknown here, with the priors written below, and the
      data are simulated from a stated truth (the values and the seed are knobs). The page checks that EP recovers the
      truth, here for one seed and, below, over many.
    </p>
  )
}

/** The recovery of one run: each parameter's truth beside EP's mean, sd and 90% interval, with the verdict on coverage. */
export function RecoveryPanel(props: {
  readonly problem: AtomProblem
  readonly beliefs: Readonly<Record<string, Belief>>
}): ReactNode {
  const rows = useMemo(() => atomRecovery(props.problem, props.beliefs), [props.problem, props.beliefs])
  const pools = new Map(props.problem.quantities.map((q) => [q.name, q.pool]))
  const single = rows.filter((r) => pools.get(r.name) === undefined)
  const pooled = [
    ...new Set(rows.flatMap((r) => (pools.get(r.name) === undefined ? [] : [pools.get(r.name) as string]))),
  ]
  const inside = rows.filter((r) => r.covered).length
  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2">
        <Badge ok={inside >= Math.floor(0.75 * rows.length)}>
          {inside} of {rows.length} inside
        </Badge>
        <span className="text-xs text-muted">
          this seed: the truth inside EP&apos;s 90% interval (about nine in ten expected; one seed says little, the
          calibration below says more)
        </span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm tabular-nums">
          <thead className="text-xs text-muted">
            <tr>
              <th className="py-1 pr-4 font-normal">parameter</th>
              <th className="pr-4 font-normal">truth</th>
              <th className="pr-4 font-normal">EP mean</th>
              <th className="pr-4 font-normal">EP sd</th>
              <th className="pr-4 font-normal">90% interval</th>
              <th className="pr-4 font-normal">error (sds)</th>
              <th className="font-normal" />
            </tr>
          </thead>
          <tbody>
            {single.map((r) => (
              <tr key={r.name} className="border-t border-line">
                <td className="py-1 pr-4">{r.label}</td>
                <td className="pr-4">{sig(r.truth)}</td>
                <td className="pr-4">{sig(r.mean)}</td>
                <td className="pr-4">{sig(r.sd)}</td>
                <td className="pr-4">
                  {sig(r.lower)} to {sig(r.upper)}
                </td>
                <td className="pr-4">{sig(r.z, 2)}</td>
                <td style={{ color: r.covered ? GOOD : BAD }}>{r.covered ? 'inside' : 'OUTSIDE'}</td>
              </tr>
            ))}
            {pooled.map((pool) => {
              const members = rows.filter((r) => pools.get(r.name) === pool)
              const hit = members.filter((r) => r.covered).length
              const rms = Math.sqrt(members.reduce((a, r) => a + r.z * r.z, 0) / members.length)
              return (
                <tr key={pool} className="border-t border-line">
                  <td className="py-1 pr-4">{pool}</td>
                  <td colSpan={3} className="text-muted">
                    {members.length} values
                  </td>
                  <td className="pr-4">
                    {hit} of {members.length} inside
                  </td>
                  <td className="pr-4">{sig(rms, 2)} (rms)</td>
                  <td style={{ color: hit >= 0.75 * members.length ? GOOD : BAD }}>
                    {hit >= 0.75 * members.length ? 'inside' : 'MANY OUTSIDE'}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

/** The samples of a calibration, filled seed by seed (a slice of work per tick) so the page stays alive while it runs. */
function useCalibration(
  atom: EpAtom,
  knobs: Config,
  active: boolean,
): { samples: readonly CalibrationSample[]; done: boolean } {
  const [state, setState] = useState<{ samples: readonly CalibrationSample[]; done: boolean }>({
    samples: [],
    done: false,
  })
  const knobKey = JSON.stringify({ ...knobs, seed: 0 })
  useEffect(() => {
    if (!active) return
    const seeds = atom.recovery?.seeds ?? 0
    const base = JSON.parse(knobKey) as Config
    const samples: CalibrationSample[] = []
    let cancelled = false
    let timer = 0
    setState({ samples: [], done: false })
    const tick = (): void => {
      if (cancelled) return
      const t0 = performance.now()
      do {
        samples.push(calibrationSample(atom, samples.length + 1, base))
      } while (samples.length < seeds && performance.now() - t0 < 60)
      setState({ samples: [...samples], done: samples.length >= seeds })
      if (samples.length < seeds) timer = window.setTimeout(tick, 0)
    }
    timer = window.setTimeout(tick, 150)
    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [atom, knobKey, active])
  return state
}

/** A dot per seed: filled green when the truth was inside the interval, red when it was not (amber when some members of a pool were). */
function Dots(props: { readonly flags: readonly number[] }): ReactNode {
  return (
    <span className="inline-flex flex-wrap gap-0.5">
      {props.flags.map((f, i) => (
        <span
          // biome-ignore lint/suspicious/noArrayIndexKey: one dot per seed, in order
          key={i}
          className="inline-block h-2 w-2 rounded-full"
          style={{ background: f >= 1 ? GOOD : f > 0 ? 'var(--wb-warning)' : BAD }}
        />
      ))}
    </span>
  )
}

/**
 * The calibration of an extended atom: EP is run over the atom's seeds (data simulated from the same truth with seed 1, 2, ...),
 * and for each parameter the table shows how often the truth fell inside EP's 90% interval (about 0.9 if the posterior is honest),
 * the bias and the size of the error in posterior sds, and the actual error beside EP's own spread. The badge is the atom's own
 * promise (the floors it declares), never tuned: a parameter that EP is poor at shows as FAIL with the numbers.
 *
 * @param props - The atom, the knobs of the page (the truth and the setup; the seed is replaced) and whether EP runs on it.
 * @returns The panel.
 */
export function CalibrationPanel(props: {
  readonly atom: EpAtom
  readonly knobs: Config
  readonly runs: boolean
}): ReactNode {
  const { atom, knobs, runs } = props
  const spec = atom.recovery
  const { samples, done } = useCalibration(atom, knobs, runs && spec !== undefined)
  const table = useMemo(() => (spec === undefined ? undefined : calibrationTable(spec, samples)), [spec, samples])
  if (spec === undefined) return null
  if (!runs)
    return (
      <p className="text-sm text-muted">
        There is no calibration yet: EP does not run on this model. When the rule lands, the same simulation is repeated
        over {spec.seeds} seeds here.
      </p>
    )
  const dots = (key: string): number[] =>
    samples.map((s) => {
      const rows = s.rows.filter((r) => (r.pool ?? r.name) === key)
      return rows.length === 0 ? 0 : rows.filter((r) => r.covered).length / rows.length
    })
  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2">
        {done && table !== undefined ? (
          <Badge ok={table.pass}>{table.pass ? 'pass' : 'fail'}</Badge>
        ) : (
          <span className="text-xs text-muted" role="status">
            running seed {samples.length} of {spec.seeds}...
          </span>
        )}
        <span className="text-xs text-muted">
          over {spec.seeds} seeds of the same truth: the truth inside the 90% interval at least{' '}
          {Math.round(100 * spec.coverage)}% of the time (nominal 90%), error at most {spec.zrms} posterior sds (root
          mean square; near 1 when EP&apos;s spread is honest)
        </span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm tabular-nums">
          <thead className="text-xs text-muted">
            <tr>
              <th className="py-1 pr-4 font-normal">parameter</th>
              <th className="pr-4 font-normal">truth</th>
              <th className="pr-4 font-normal">inside the 90% interval</th>
              <th className="pr-4 font-normal">bias (sds)</th>
              <th className="pr-4 font-normal">error (sds, rms)</th>
              <th className="pr-4 font-normal">rmse</th>
              <th className="pr-4 font-normal">EP sd (rms)</th>
              <th className="pr-4 font-normal">one dot per seed</th>
              <th className="font-normal" />
            </tr>
          </thead>
          <tbody>
            {(table?.rows ?? []).map((r) => (
              <tr key={r.key} className="border-t border-line">
                <td className="py-1 pr-4">{r.label}</td>
                <td className="pr-4">{sig(r.truth)}</td>
                <td className="pr-4">{sig(100 * r.coverage, 3)}%</td>
                <td className="pr-4">{sig(r.meanZ, 2)}</td>
                <td className="pr-4">{sig(r.zrms, 2)}</td>
                <td className="pr-4">{sig(r.rmse)}</td>
                <td className="pr-4">{sig(r.postSd)}</td>
                <td className="pr-4">
                  <Dots flags={dots(r.key)} />
                </td>
                <td style={{ color: r.pass ? GOOD : BAD }}>{done ? (r.pass ? 'pass' : 'FAIL') : ''}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="max-w-3xl text-xs text-muted">
        {spec.reason}. A parameter pooled over a population (the values of the groups, the players) counts every member
        of every seed. The truth of a parameter is the value stated in the knobs; a value drawn from it (a group&apos;s
        rate) is the draw of that seed. The spread of EP against the real error: when the &quot;EP sd&quot; is near the
        rmse the posterior is honest, when it is smaller EP is too sure.
      </p>
    </div>
  )
}
