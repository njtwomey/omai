import { type AtomCalibration, type CalibrationSample, calibrationTable } from 'omai-engine/bayesian-examples'
import { type ReactNode, useMemo } from 'react'
import { sig } from '../../shared/ep/ProgramParts.tsx'
import { BARELY, type SeedRun } from './calibration.ts'

const GOOD = 'var(--wb-success)'
const BAD = 'var(--wb-danger)'

/** What the calibration holds EP to, on the seeds where every regime is visited. */
export const FLOOR = { coverage: 0.7, zrms: 1.6 }

/** The spread of EP's belief about the rates against the sampler's, narrowest over the three rates, for one seed. */
const narrowest = (r: SeedRun): number => {
  const ratios = r.ep.rows
    .filter((row) => row.name.startsWith('rate'))
    .map((row) => row.sd / (r.gibbs.rows.find((g) => g.name === row.name)?.sd ?? Number.NaN))
  return ratios.length === 0 ? Number.NaN : Math.min(...ratios)
}

const tableOf = (samples: readonly CalibrationSample[]): AtomCalibration =>
  calibrationTable({ seeds: samples.length, ...FLOOR, reason: '' }, samples)

/** One group of seeds as a table: per parameter, how often the truth was inside EP's 90% interval, and the sampler's. */
function GroupTable(props: { readonly runs: readonly SeedRun[]; readonly gating: boolean }): ReactNode {
  const ep = useMemo(() => tableOf(props.runs.map((r) => r.ep)), [props.runs])
  const gibbs = useMemo(() => tableOf(props.runs.map((r) => r.gibbs)), [props.runs])
  if (props.runs.length === 0) return <p className="text-xs text-muted">No seed in this group yet.</p>
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm tabular-nums">
        <thead className="text-xs text-muted">
          <tr>
            <th className="py-1 pr-4 font-normal">parameter</th>
            <th className="pr-4 font-normal">truth</th>
            <th className="pr-4 font-normal">EP: truth inside the 90% interval</th>
            <th className="pr-4 font-normal">EP error (sds, rms)</th>
            <th className="pr-4 font-normal">sampler: inside</th>
            <th className="pr-4 font-normal">sampler error (sds, rms)</th>
            <th className="font-normal" />
          </tr>
        </thead>
        <tbody>
          {ep.rows.map((r) => {
            const g = gibbs.rows.find((x) => x.key === r.key)
            return (
              <tr key={r.key} className="border-t border-line">
                <td className="py-1 pr-4">{r.label}</td>
                <td className="pr-4">{sig(r.truth)}</td>
                <td className="pr-4">{sig(100 * r.coverage, 3)}%</td>
                <td className="pr-4">{sig(r.zrms, 2)}</td>
                <td className="pr-4">{g === undefined ? '-' : `${sig(100 * g.coverage, 3)}%`}</td>
                <td className="pr-4">{g === undefined ? '-' : sig(g.zrms, 2)}</td>
                <td style={{ color: r.pass ? GOOD : BAD }}>{props.gating ? (r.pass ? 'pass' : 'FAIL') : ''}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function Dots(props: { readonly runs: readonly SeedRun[] }): ReactNode {
  return (
    <span className="inline-flex flex-wrap gap-1">
      {props.runs.map((r) => (
        <span
          key={r.seed}
          className="inline-flex h-4 min-w-4 items-center justify-center rounded-full px-1 text-[9px] text-bg"
          style={{ background: r.agrees ? GOOD : BAD }}
          title={`seed ${r.seed}: ${r.visits.join(', ')} steps in the three regimes; EP ${r.agrees ? 'agrees with' : 'differs from'} the sampler (gap ${sig(r.gap, 2)})`}
        >
          {r.seed}
        </span>
      ))}
    </span>
  )
}

/**
 * The calibration of the program over many seeds, in two groups: the seeds on which every regime is visited (held to a floor
 * on coverage and error, with a PASS or FAIL) and the seeds on which a regime is barely visited (the known failure, shown
 * with the same numbers and no verdict).
 *
 * @param props - The seeds done, how many were asked, and whether the worker has finished.
 * @returns The panel.
 */
export function CalibrationPanel(props: {
  readonly runs: readonly SeedRun[]
  readonly asked: number
  readonly done: boolean
  readonly failed: string | undefined
  readonly steps: number
}): ReactNode {
  const { runs, asked, done, failed, steps } = props
  const visited = useMemo(() => runs.filter((r) => !r.barely), [runs])
  const barely = useMemo(() => runs.filter((r) => r.barely), [runs])
  const verdict = useMemo(() => tableOf(visited.map((r) => r.ep)), [visited])
  const agreeing = visited.filter((r) => r.agrees).length
  const ok = done && visited.length > 0 && verdict.pass && agreeing === visited.length
  const gapBarely = barely.length === 0 ? Number.NaN : barely.reduce((s, r) => s + r.gap, 0) / barely.length
  const widths = barely.map(narrowest).filter(Number.isFinite)
  return (
    <section className="flex flex-col gap-3">
      <h2 className="text-sm font-semibold">Calibration over many seeds</h2>
      <p className="max-w-3xl text-sm">
        The same simulation (the truth above, the number of steps and EP&apos;s damping as set) repeated over seeds 1 to{' '}
        {asked}, in a worker: EP and the Gibbs sampler on each. For every parameter, how often the true value falls
        inside the 90% interval (about 9 in 10 for an honest posterior; for the transition probabilities the prior pulls
        the exact posterior towards its own mean, so the sampler itself covers the truth a little less often) and how
        large the error is in posterior sds (1 when the spread is honest).
      </p>
      <div className="flex flex-wrap items-center gap-2">
        {done ? (
          <span
            className="inline-block rounded border px-2 py-0.5 text-xs font-semibold uppercase tracking-wide"
            style={{ color: ok ? GOOD : BAD, borderColor: ok ? GOOD : BAD }}
          >
            {ok ? 'pass' : 'fail'}
          </span>
        ) : (
          <span className="text-xs text-muted" role="status">
            {failed === undefined ? `running seed ${runs.length + 1} of ${asked}...` : `the worker failed: ${failed}`}
          </span>
        )}
        <span className="text-xs text-muted">
          held to, on the seeds where every regime is visited: the truth inside EP&apos;s 90% interval at least{' '}
          {Math.round(100 * FLOOR.coverage)}% of the time, error at most {FLOOR.zrms} sds (rms), and EP&apos;s regime
          probabilities within 0.05 of the sampler&apos;s on every one of those seeds
        </span>
      </div>
      <p className="text-xs text-muted">
        One dot per seed (green: EP&apos;s regime probabilities are within 0.05 of the sampler&apos;s on average; red:
        they are not; hover for the steps in each regime).
      </p>
      <Dots runs={runs} />
      <h3 className="text-xs font-semibold text-muted">
        Every regime visited (at least {Math.round(100 * BARELY)}% of the {steps} steps): {visited.length} seeds, EP
        agrees with the sampler on {agreeing}
      </h3>
      <GroupTable runs={visited} gating />
      <h3 className="text-xs font-semibold text-muted">
        A regime barely visited (fewer than {Math.round(BARELY * steps)} steps): {barely.length} seeds, EP agrees with
        the sampler on {barely.filter((r) => r.agrees).length}; not held to the floor
      </h3>
      <GroupTable runs={barely} gating={false} />
      {barely.length === 0 ? null : (
        <p className="max-w-3xl text-xs text-muted">
          On these seeds the mean gap between EP&apos;s regime probabilities and the sampler&apos;s is{' '}
          {sig(gapBarely, 2)}
          {widths.length === 0
            ? ''
            : `, and the narrowest rate spread of EP is ${sig(Math.min(...widths), 2)} to ${sig(Math.max(...widths), 2)} of the sampler's`}
          . The exact posterior over which steps belong to a regime that occurs in a handful of them is spread over
          several readings (a step or two could be that regime or its neighbour); one message per variable commits to
          one of them, reports the rate of that regime from a few steps as if they were certain, and sometimes does not
          converge. Damping, power EP and alpha-divergences were measured and do not change that
          (docs/probprog/ep-plan.md, section 11).
        </p>
      )}
    </section>
  )
}
