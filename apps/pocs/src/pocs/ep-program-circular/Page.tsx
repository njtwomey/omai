import { toStructure } from 'omai-engine/probprog/model'
import { Choice, NumberField, SeedInput, Select } from 'omai-render/react'
import { type ReactNode, useCallback, useMemo, useState } from 'react'
import { IntervalPlot } from '../../shared/ep/ProgramCharts.tsx'
import { ProgramPage } from '../../shared/ep/ProgramPage.tsx'
import { sig } from '../../shared/ep/program-check.ts'
import { RunProgress } from '../../shared/RunProgress.tsx'
import { workerTask } from '../../shared/runWorker.ts'
import { type UseRun, useRun } from '../../shared/useRun.ts'
import { ClockChart, DayChart, ShiftPicture } from './Charts.tsx'
import type { CircularReply, CircularRequest } from './circular.worker.ts'
import {
  boundOf,
  type CalibrationRow,
  type CircularRun,
  type CircularSetup,
  checkCircular,
  copiesError,
  defaultSetup,
  type Equivalence,
  heldOutHours,
  heldOutLoss,
  lossesAt,
  matchToTruth,
  mixtureOf,
  parameterNames,
  truthScale,
} from './compute.ts'
import { TRUTHS, type TruthName, type Variant } from './program.ts'
import source from './program.ts?raw'

const makeWorker = (): Worker => new Worker(new URL('./circular.worker.ts', import.meta.url), { type: 'module' })

const VARIANTS: readonly { value: Variant; label: string }[] = [
  { value: 'ignore', label: 'ignore the circle (a plain Gaussian mixture)' },
  { value: 'means', label: 'shift the means (the paper)' },
  { value: 'data', label: 'shift the data (the post)' },
  { value: 'rows', label: 'copies as extra rows (wrong)' },
]

const TRUTH_LABELS: Record<TruthName, string> = {
  one: 'one normal at 23:00, sd 3 h',
  two: 'two normals: 22 h (0.6, sd 2.5 h) and 9 h (0.4, sd 2 h)',
}

const EQUATIONS = [
  'x_i in [0, 24) hours (period 24)      w ~ Dirichlet(1, ..., 1)      tau = 1 / s² ~ Gamma(2, 18)',
  'mu_k ~ N(start_k, 8²), held in [0, 24)      start_k: spread evenly round the day',
  'ignore:         x_i ~ sum_k w_k N(mu_k, s²)      (23:50 and 00:10 are 23 h 40 min apart)',
  'shift means:    x_i ~ sum_k w_k (1/3) sum_c N(x_i; mu_k + 24 c, s²),  c in {-1, 0, 1}',
  'shift data:     x_i ~ sum_k w_k (1/3) sum_c N(x_i - 24 c; mu_k, s²)    (the same terms)',
  'extra rows:     x_i - 24, x_i, x_i + 24 are three readings: sum_c log p(.), not log sum_c p(.)',
]

const hoursOf = (v: number): string => `${sig(v, 3)} h`

/** The program as a page: clock times, three ways to treat the circle and a wrong one, run by EP against the truth. */
export default function Page(): ReactNode {
  const [setup, setSetup] = useState<CircularSetup>(defaultSetup)
  const set = (patch: Partial<CircularSetup>): void => setSetup((s) => ({ ...s, ...patch }))
  // Every sweep of EP takes a fifth of a second to a second, so the fit, the comparison and the calibration compute in a worker
  // (circular.worker.ts); the page keeps painting, shows how far each is, and Stop keeps the sweeps done.
  const fitTask = useMemo(() => workerTask<CircularRequest, CircularReply>(makeWorker, { kind: 'fit', setup }), [setup])
  const fit = useRun<CircularReply>(fitTask)
  const run = fit.value as CircularRun | undefined
  const held = useMemo(() => heldOutHours(setup.truth, setup.seed), [setup.truth, setup.seed])
  const [compare, setCompare] = useState(false)
  const eqTask = useMemo(
    () =>
      compare && run !== undefined
        ? workerTask<CircularRequest, CircularReply>(makeWorker, { kind: 'equivalence', setup, current: run })
        : undefined,
    [compare, run, setup],
  )
  const eqRun = useRun<CircularReply>(eqTask)
  const eq = eqRun.value as Equivalence | undefined
  const check = useMemo(() => (run === undefined ? undefined : checkCircular(run, held, eq)), [run, held, eq])
  const structure = useCallback(() => (run === undefined ? undefined : toStructure(boundOf(run))), [run])
  const record = useCallback(() => run?.record, [run])
  const calibTask = useMemo(
    () =>
      workerTask<CircularRequest, CircularReply>(makeWorker, {
        kind: 'calibration',
        setup: {
          ...setup,
          variant: 'means',
          components: TRUTHS[setup.truth].means.length,
          draws: Math.min(setup.draws, 200),
        },
        seeds: Array.from({ length: 8 }, (_, i) => i + 1),
      }),
    [setup],
  )
  const calib = useRun<CircularReply>(calibTask, { auto: false })
  const sd = run?.found.sd.mean ?? truthScale(setup.truth)
  return (
    <ProgramPage
      id="circular"
      equations={EQUATIONS}
      source={source}
      file="program.ts"
      structure={structure}
      record={record}
      check={check}
      status={
        run === undefined
          ? 'Simulating and running EP...'
          : `EP ${run.fit.run.status ?? 'stopped'} after ${run.fit.run.t} sweeps (${sig(run.ms / 1000, 2)} s) on ${run.hours.length * (run.setup.variant === 'rows' ? 3 : 1)} readings.`
      }
      progress={<RunProgress run={fit} label="EP" />}
      controls={
        <>
          <Select
            label="model"
            value={setup.variant}
            options={VARIANTS.map((v) => ({ value: v.value, label: v.label }))}
            onChange={(variant) => set({ variant })}
          />
          <Select
            label="truth"
            value={setup.truth}
            options={(Object.keys(TRUTH_LABELS) as TruthName[]).map((t) => ({ value: t, label: TRUTH_LABELS[t] }))}
            onChange={(t) => set({ truth: t, components: TRUTHS[t].means.length })}
          />
          <Choice
            label="components"
            value={setup.components}
            options={[1, 2, 3]}
            onChange={(components) => set({ components })}
          />
          <NumberField
            label="draws"
            value={setup.draws}
            onChange={(draws) => set({ draws })}
            min={50}
            max={500}
            step={50}
            integer
          />
          <SeedInput value={setup.seed} onChange={(seed) => set({ seed })} />
        </>
      }
    >
      {run === undefined ? (
        <p className="text-sm text-muted">
          Computing in a worker (the discrete latents are a gate per reading, so a sweep takes a fraction of a second to
          a second)...
        </p>
      ) : (
        <Results run={run} held={held} truthName={setup.truth} sd={sd} />
      )}
      <EquivalencePanel
        setup={setup}
        run={run}
        eq={eq}
        eqRun={eqRun}
        comparing={compare}
        onCompare={() => setCompare(true)}
      />
      <Calibration setup={setup} run={calib} />
      <Verdict />
    </ProgramPage>
  )
}

function Results(props: {
  readonly run: CircularRun
  readonly held: readonly number[]
  readonly truthName: TruthName
  readonly sd: number
}): ReactNode {
  const { run, held, truthName, sd } = props
  const mix = useMemo(() => mixtureOf(run.found), [run])
  const truth = TRUTHS[truthName]
  const matched = matchToTruth(truthName, run.found)
  const labels = [...truth.means.map((m) => `mean ${m} h`), 'sd (shared)']
  const pick = (k: number) => run.found.means[matched[k] as number]
  const means = truth.means.map((_, k) => pick(k))
  const rowsOf = (key: 'mean' | 'lower' | 'upper'): number[] => [
    ...means.map((m) => (m === undefined ? Number.NaN : m[key])),
    run.found.sd[key],
  ]
  const loss = heldOutLoss(run.setup.variant, truthName, mix, held)
  const wrapped = run.wrapped.filter((p) => p > 0.5).length
  const edge = run.hours.filter((x) => x < 3 || x > 21).length
  return (
    <>
      <div className="grid grid-cols-1 items-start gap-x-6 gap-y-4 lg:grid-cols-3">
        <div className="flex min-w-0 flex-col gap-1 lg:col-span-2">
          <h3 className="text-xs font-semibold text-muted">The readings over the day, the truth and the fit</h3>
          <DayChart hours={run.hours} truth={truthName} variant={run.setup.variant} mix={mix} />
        </div>
        <div className="flex min-w-0 flex-col items-center gap-1">
          <h3 className="self-start text-xs font-semibold text-muted">The same, as a clock</h3>
          <ClockChart hours={run.hours} truth={truthName} variant={run.setup.variant} mix={mix} />
        </div>
      </div>
      <div className="grid grid-cols-1 items-start gap-x-6 gap-y-4 lg:grid-cols-2">
        <div className="flex min-w-0 flex-col gap-1">
          <h3 className="text-xs font-semibold text-muted">The means and the sd recovered, in hours</h3>
          <IntervalPlot
            labels={labels}
            truth={[...truth.means, truthScale(truthName)]}
            ep={{ name: 'EP posterior mean', mean: rowsOf('mean'), lower: rowsOf('lower'), upper: rowsOf('upper') }}
            yLabel="hours"
          />
          <p className="max-w-xl text-xs text-muted">
            The truth’s sd is one number a shared sd can be held to: the root mean square of the components’ scales (
            {sig(truthScale(truthName), 3)} h).
          </p>
        </div>
        <div className="flex min-w-0 flex-col gap-1">
          <h3 className="text-xs font-semibold text-muted">The weights recovered</h3>
          {run.setup.components > 1 && truth.weights.length > 1 ? (
            <IntervalPlot
              labels={truth.weights.map((_w, k) => `weight of ${truth.means[k]} h`)}
              truth={[...truth.weights]}
              ep={{
                name: 'EP posterior mean',
                mean: truth.weights.map((_, k) => run.found.weights[matched[k] as number]?.mean ?? Number.NaN),
                lower: truth.weights.map((_, k) => run.found.weights[matched[k] as number]?.lower ?? Number.NaN),
                upper: truth.weights.map((_, k) => run.found.weights[matched[k] as number]?.upper ?? Number.NaN),
              }}
            />
          ) : (
            <p className="text-sm text-muted">One component: its weight is 1.</p>
          )}
          <p className="max-w-xl text-sm tabular-nums">
            Held-out log density: {sig(loss, 2)} nats per reading below the truth’s own (3000 fresh readings).
            {run.setup.variant === 'means' || run.setup.variant === 'data'
              ? ` Of the ${edge} readings within three hours of midnight, EP reads ${wrapped} as an early or late copy.`
              : ''}
          </p>
        </div>
      </div>
      <p className="max-w-4xl text-sm tabular-nums">
        How many copies: three are accurate while sd is small against the period. The relative error of the density is
        at most 2 e<sup>-3a/4</sup> / (1 - e<sup>-3a</sup>) with a = 24² / (2 sd²): at the fitted sd of {hoursOf(sd)}{' '}
        that is {sig(copiesError(sd), 2)} (8e-11 at 3 h, under 0.005 at 6 h, about 0.4 by 12 h, where the wrapped
        normal’s Fourier series takes over). On a torus with d periodic coordinates the copies are a grid of (2K + 1)
        <sup>d</sup>: nine in two, twenty-seven in three.
      </p>
    </>
  )
}

function EquivalencePanel(props: {
  readonly setup: CircularSetup
  readonly run: CircularRun | undefined
  readonly eq: Equivalence | undefined
  readonly eqRun: UseRun<CircularReply>
  readonly comparing: boolean
  readonly onCompare: () => void
}): ReactNode {
  const { run, eq, comparing } = props
  const mix = useMemo(() => (run === undefined ? undefined : mixtureOf(run.found)), [run])
  const losses = useMemo(
    () => (run === undefined || mix === undefined ? undefined : lossesAt(run.hours, mix)),
    [run, mix],
  )
  return (
    <section className="flex flex-col gap-3">
      <h2 className="text-sm font-semibold">Shift the means or shift the data: the same arithmetic</h2>
      <p className="max-w-4xl text-xs text-muted">
        N(x; mu + k·24, s²) = N(x − k·24; mu, s²), term by term. So the mean can move and the data stay (the paper), or
        the one normal can stay and each reading be copied a day early and late (the post), the copies scored together
        in one log-sum-exp. Drag the mean: the two sums never differ.
      </p>
      {run === undefined ? null : (
        <ShiftPicture hours={run.hours} sd={run.found.sd.mean} start={run.found.means[0]?.mean ?? 12} />
      )}
      {losses === undefined ? null : (
        <div className="overflow-x-auto">
          <table className="w-full max-w-3xl text-left text-sm tabular-nums">
            <thead className="text-xs text-muted">
              <tr>
                <th className="py-1 font-normal">at the fitted parameters</th>
                <th className="font-normal">total log likelihood</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-t border-line">
                <td className="py-1">shift the means</td>
                <td>{sig(losses.shiftMeans, 14)}</td>
              </tr>
              <tr className="border-t border-line">
                <td className="py-1">shift the data</td>
                <td>{sig(losses.shiftData, 14)}</td>
              </tr>
              <tr className="border-t border-line">
                <td className="py-1">copies as extra rows (the sum of the logs: wrong)</td>
                <td>{sig(losses.extraRows, 8)}</td>
              </tr>
            </tbody>
          </table>
        </div>
      )}
      <p className="max-w-4xl text-xs text-muted">
        Extra rows score each copy separately, maximising Σ<sub>k</sub> log p(x − k·24) rather than log Σ<sub>k</sub>{' '}
        p(x − k·24): every copy has to be likely at once, so a single normal is pushed to a sd of about 21.7 hours
        against a true 3 (the “copies as extra rows” model above shows it under EP).
      </p>
      <div className="flex flex-wrap items-center gap-3">
        <button
          type="button"
          className="rounded border border-line px-2 py-1 text-xs"
          disabled={
            comparing || run === undefined || props.setup.variant === 'ignore' || props.setup.variant === 'rows'
          }
          onClick={props.onCompare}
        >
          fit both forms by EP and compare
        </button>
        <span className="text-xs text-muted" role="status">
          {props.setup.variant === 'ignore' || props.setup.variant === 'rows'
            ? 'Choose “shift the means” or “shift the data” above first.'
            : comparing && eq === undefined
              ? 'Fitting the other form in a worker...'
              : ''}
        </span>
      </div>
      <RunProgress run={props.eqRun} label="Comparison" />
      {eq === undefined ? null : (
        <p className="max-w-4xl text-sm tabular-nums">
          EP on both forms: the log likelihoods differ by {sig(eq.lossGapAtFit, 2)} at the fitted parameters and by at
          most {sig(eq.lossGapRandom, 2)} at six random ones. The fits differ by {sig(eq.meanGap, 2)} h in the means,{' '}
          {sig(eq.sdGap, 2)} h in the sd and {sig(eq.weightGap, 2)} in the weights: EP is not invariant to the way the
          same arithmetic is written (with the shift on the mean the shifted means are variables shared by every
          reading; on the data every reading has its own). Fit times: {sig(eq.means.ms / 1000, 2)} s and{' '}
          {sig(eq.data.ms / 1000, 2)} s.
        </p>
      )}
    </section>
  )
}

function Calibration(props: { readonly setup: CircularSetup; readonly run: UseRun<CircularReply> }): ReactNode {
  const { setup, run } = props
  const names = parameterNames(setup.truth)
  const rows = run.value as CalibrationRow[] | undefined
  return (
    <section className="flex flex-col gap-2">
      <h2 className="text-sm font-semibold">Calibration: does EP’s 90% interval cover the truth?</h2>
      <p className="max-w-4xl text-xs text-muted">
        Eight seeds of the same setup (the true number of components, shift the means, at most 200 readings), run in a
        worker. A calibrated 90% interval covers the truth about 90% of the time. The table fills seed by seed.
      </p>
      <RunProgress
        run={run}
        label="Calibration"
        onStart={run.start}
        startText="Run the calibration"
        hint="Runs on request (about a minute or two)."
      />
      {rows === undefined ? null : (
        <div className="overflow-x-auto">
          <table className="w-full max-w-3xl text-left text-sm tabular-nums">
            <thead className="text-xs text-muted">
              <tr>
                <th className="py-1 font-normal">parameter</th>
                <th className="font-normal">truth inside the 90% interval</th>
              </tr>
            </thead>
            <tbody>
              {names.map((n, i) => {
                const hit = rows.filter((r) => r.covered[i]).length
                return (
                  <tr key={n} className="border-t border-line">
                    <td className="py-1">{n}</td>
                    <td>
                      {hit} of {rows.length} ({sig((100 * hit) / rows.length, 3)}%)
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
          <p className="mt-1 max-w-4xl text-xs text-muted">
            {rows.filter((r) => !r.converged).length} of {rows.length} runs did not converge. Intervals under 90% are EP
            being overconfident; the sd row is also biased when the truth has two scales and the model one.
          </p>
        </div>
      )}
    </section>
  )
}

function Verdict(): ReactNode {
  return (
    <section className="flex flex-col gap-2">
      <h2 className="text-sm font-semibold">The verdict</h2>
      <ul className="max-w-4xl list-disc pl-5 text-sm text-muted">
        <li>
          Wrapping is a property of the loss, not of the distribution: an ordinary normal scored at three copies of each
          reading (or, the same thing, a normal copied three times at the mean) is the wrapped normal to the accuracy of
          the copy count, and EP fits it as it fits any mixture with a discrete latent. The shifted forms recover the
          truth where the plain mixture cannot (it puts a component on each side of midnight, or a huge sd).
        </li>
        <li>
          Extra rows are the wrong way: the three copies of a reading are one observation, not three. Under EP, as under
          Adam, the sd runs to the width of the day.
        </li>
        <li>
          EP is a fixed-point method with one Gaussian per variable, so a mixture with unknown means has local modes and
          EP commits to one. The means are started spread evenly round the day (as in the paper) and, when the number of
          components is more than the truth’s, one component can be left with little weight, a wide posterior and a
          narrow-looking interval: read its weight before its mean. The changepoint program shows the same
          overconfidence on a barely visited component.
        </li>
        <li>
          Two ways of writing the same arithmetic do not give the same EP fit: the likelihoods agree to rounding, the
          posteriors to a few hundredths. And one sd is shared by the components (a sd per component, chosen by the same
          index as the mean, is a gate over likelihoods with a random scale; the compiler maps that with a known scale
          only).
        </li>
      </ul>
    </section>
  )
}
