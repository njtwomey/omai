import { beliefCredibleInterval } from 'omai-engine/probprog/ep'
import type { Curve, PointSet } from 'omai-render/charts'
import { ControlPanel, NumberField, SeedInput } from 'omai-render/react'
import { type ReactNode, useCallback, useMemo, useState } from 'react'
import { Link } from '../../../../../../site/src/shell/Link.tsx'
import { IntervalPlot } from '../../../../shared/ep/ProgramCharts.tsx'
import { CheckLines, endedText, ProgramFrame, RecoveryChart, sig } from '../../../../shared/ep/ProgramParts.tsx'
import { epHooks } from '../../../../shared/ep/sweeps.ts'
import { RunProgress } from '../../../../shared/RunProgress.tsx'
import type { HostBase } from '../../../../shared/run.ts'
import { workerTask } from '../../../../shared/runWorker.ts'
import { useRun } from '../../../../shared/useRun.ts'
import { CalibrationPanel } from './Calibration.tsx'
import { assess, NAMES, type SeedRun } from './calibration.ts'
import { infer, RATES, simulate, transitions } from './program.ts'
import source from './program.ts?raw'
import { gibbsReference } from './reference.ts'
import type { CalibrationRequest } from './reference.worker.ts'

const SEEDS = 24

type Made = Awaited<ReturnType<typeof fit>>

const makeWorker = (): Worker => new Worker(new URL('./reference.worker.ts', import.meta.url), { type: 'module' })

/** Simulates the data, runs EP and then the Gibbs reference, reporting each sweep to the host. */
async function fit(seed: number, stay: number, steps: number, damping: number, host: HostBase) {
  const sim = simulate(seed, stay, steps)
  const t0 = performance.now()
  const run = await infer(sim.y, damping, epHooks(host, 'EP'))
  const ms = performance.now() - t0
  const gibbs = await gibbsReference(sim.y, {}, epHooks(host, 'Gibbs reference'))
  return { sim, ...run, ms, gibbs, result: assess(seed, sim, run, gibbs) }
}
const argmax = (p: readonly number[]): number => p.indexOf(Math.max(...p))
const CELLS = NAMES.flatMap((a, i) => NAMES.map((b, j) => ({ label: `${a} to ${b}`, i, j })))
const STAYS = CELLS.filter((m) => m.i === m.j)
const MOVES = CELLS.filter((m) => m.i !== m.j)

export default function Page(): ReactNode {
  const [seed, setSeed] = useState(1)
  const [stay, setStay] = useState(0.93)
  const [steps, setSteps] = useState(100)
  const [damping, setDamping] = useState(0.5)
  const task = useCallback((host: HostBase) => fit(seed, stay, steps, damping, host), [seed, stay, steps, damping])
  const ep = useRun<Made>(task)
  const made = ep.value
  const calibrationTask = useMemo(
    () => workerTask<CalibrationRequest, SeedRun[]>(makeWorker, { stay, steps, damping, seeds: SEEDS }),
    [stay, steps, damping],
  )
  const calibration = useRun<SeedRun[]>(calibrationTask)
  return (
    <ProgramFrame
      id="changepoint-extended"
      intro={
        <>
          <p>
            The program of{' '}
            <Link to="/poc/ep-program-changepoint" className="underline">
              a Poisson rate that changes
            </Link>
            , with both sets of numbers unknown. A count at each of {steps} steps; a hidden regime (low, middle or high)
            follows a sticky Markov chain and sets the Poisson rate of the count. The three <b>rates</b> are unknown
            (broad gamma priors, means 4, 12 and 30), and so are the <b>transition probabilities</b>: each row of the
            transition matrix has a Dirichlet prior that leans towards staying put (mean stay probability 0.67) and is
            open about where a move goes. EP infers the regime of each step, the rates and the whole matrix, so it
            recovers how sticky the regimes are as well as how fast each one counts.
          </p>
          <p>
            The transition rows are random, which the compiler maps to one <code>gate</code> per step: the previous
            regime selects which row the next regime is drawn from, so each row hears only the transitions out of its
            own regime, weighted by the belief that the chain was there. The rates are tied to the regimes by their
            priors (a low, a middle and a high mean), which names the regimes; there is no hyperprior and no ordering
            constraint, and the section on label switching below says why that matters.
          </p>
          <p className="text-xs text-muted">
            The reference is Gibbs sampling of the same model (forward filtering and backward sampling for the regimes,
            the gamma posterior for the rates, a Dirichlet posterior on the transition counts for each row), 2700
            sweeps. EP is damped by half; set the damping to 0 to see why.
          </p>
        </>
      }
      controls={
        <ControlPanel title="change the data" columns={4}>
          <SeedInput value={seed} onChange={setSeed} />
          <NumberField
            label="stay probability (truth)"
            value={stay}
            onChange={setStay}
            min={0.8}
            max={0.99}
            step={0.01}
          />
          <NumberField
            label="steps"
            value={steps}
            onChange={(v) => setSteps(Math.round(v))}
            min={40}
            max={200}
            step={10}
          />
          <NumberField label="EP damping" value={damping} onChange={setDamping} min={0} max={0.8} step={0.1} />
        </ControlPanel>
      }
      status={
        made === undefined
          ? 'Simulating and running EP...'
          : `${endedText(made.run, made.ms)}${made.run.status === 'converged' ? '' : ' It did not converge: read the check below.'}`
      }
      progress={<RunProgress run={ep} label="EP" />}
      figures={made === undefined ? null : <Figures made={made} stay={stay} />}
      check={made === undefined ? null : <Check made={made} />}
      extra={<CalibrationPanel runs={calibration.value ?? []} asked={SEEDS} run={calibration} steps={steps} />}
      bound={made?.bound}
      run={made?.run}
      source={source}
      file="ep-program-changepoint-extended/program.ts"
    />
  )
}

function Figures(props: { readonly made: Made; readonly stay: number }): ReactNode {
  const { made, stay } = props
  const { sim, regimes, rates, gibbs, summary, result } = made
  const t = useMemo(() => sim.y.map((_, i) => i), [sim])
  const A = useMemo(() => transitions(stay), [stay])

  const rateCurve = useMemo(
    () => regimes.map((p) => p.reduce((s, v, k) => s + v * (rates[k]?.mean ?? 0), 0)),
    [regimes, rates],
  )
  const counts = useMemo(() => [{ x: t, y: sim.y, name: 'counts' } satisfies PointSet], [t, sim])
  const rateCurves = useMemo(
    () =>
      [{ name: 'EP: expected rate', x: t, y: rateCurve, colour: 'tab0', width: 2.2, step: 'middle' }] satisfies Curve[],
    [t, rateCurve],
  )
  const truthRate = useMemo(() => ({ x: t, y: sim.z.map((k) => RATES[k] as number), name: 'true rate' }), [t, sim])
  const probCurves = useMemo(
    (): Curve[] => [
      ...[0, 1, 2].map((k) => ({
        name: `EP: ${NAMES[k]} regime`,
        x: t,
        y: regimes.map((p) => p[k] as number),
        colour: `tab${k}`,
        width: 2,
        step: 'middle' as const,
      })),
      ...[0, 1, 2].map((k) => ({
        name: `Gibbs: ${NAMES[k]}`,
        x: t,
        y: gibbs.regimes.map((p) => p[k] as number),
        colour: `tab${k}`,
        width: 1,
        dash: 'dotted' as const,
        step: 'middle' as const,
      })),
    ],
    [t, regimes, gibbs],
  )
  const strip = useMemo(
    () => [
      {
        x: t,
        y: sim.z.map(() => -0.06),
        groups: sim.z,
        groupNames: { 0: 'true: low', 1: 'true: middle', 2: 'true: high' },
      } satisfies PointSet,
    ],
    [t, sim],
  )

  // The parameters as the interval plots read them: EP's from its summary, the sampler's as a matched interval.
  const epRate = useMemo(
    () => ({
      name: 'EP',
      mean: rates.map((r) => r.mean),
      lower: rates.map((r) => r.lower),
      upper: rates.map((r) => r.upper),
    }),
    [rates],
  )
  const gibbsRate = useMemo(() => {
    const iv = gibbs.rateMean.map((m, k) => beliefCredibleInterval('positive', m, gibbs.rateSd[k] as number))
    return { name: 'Gibbs', mean: gibbs.rateMean, lower: iv.map((v) => v[0]), upper: iv.map((v) => v[1]) }
  }, [gibbs])
  const matrixSeries = useMemo(() => {
    const rows = summary.filter((r) => r.site === 'A')
    const of = (cells: typeof CELLS) => {
      const gMean = cells.map((m) => (gibbs.matrixMean[m.i] as number[])[m.j] as number)
      const iv = cells.map((m, n) =>
        beliefCredibleInterval('unit', gMean[n] as number, (gibbs.matrixSd[m.i] as number[])[m.j] as number),
      )
      return {
        labels: cells.map((m) => (m.i === m.j ? `${NAMES[m.i]} stays` : `${NAMES[m.i]} to ${NAMES[m.j]}`)),
        truth: cells.map((m) => (A[m.i] as number[])[m.j] as number),
        ep: {
          name: 'EP',
          mean: cells.map((m) => rows[3 * m.i + m.j]?.mean ?? Number.NaN),
          lower: cells.map((m) => rows[3 * m.i + m.j]?.lower ?? Number.NaN),
          upper: cells.map((m) => rows[3 * m.i + m.j]?.upper ?? Number.NaN),
        },
        reference: { name: 'Gibbs', mean: gMean, lower: iv.map((v) => v[0]), upper: iv.map((v) => v[1]) },
      }
    }
    return { stays: of(STAYS), moves: of(MOVES) }
  }, [summary, gibbs, A])

  return (
    <div className="flex flex-col gap-4">
      <div>
        <h3 className="text-xs font-semibold text-muted">
          The counts, the true rate and the rate EP expects at each step
        </h3>
        <RecoveryChart
          bands={[]}
          curves={rateCurves}
          truth={truthRate}
          dots={counts}
          steps
          dotOffset={4}
          xLabel="step"
          yLabel="rate"
          aspect={3}
        />
      </div>
      <div>
        <h3 className="text-xs font-semibold text-muted">
          The probability of each regime at each step: EP (solid) and Gibbs (dotted); the dots underneath are the true
          regime
        </h3>
        <RecoveryChart
          bands={[]}
          curves={probCurves}
          dots={strip}
          dotOffset={0}
          yLimits={[-0.12, 1.05]}
          xLabel="step"
          yLabel="probability"
          aspect={3}
        />
      </div>
      <div className="grid gap-4 md:grid-cols-3">
        <div>
          <h3 className="text-xs font-semibold text-muted">The three rates: truth, EP and Gibbs</h3>
          <IntervalPlot
            labels={NAMES}
            truth={RATES}
            ep={epRate}
            reference={gibbsRate}
            yLabel="rate (counts per step)"
          />
        </div>
        <div>
          <h3 className="text-xs font-semibold text-muted">The probability of staying in each regime</h3>
          <IntervalPlot {...matrixSeries.stays} yLabel="probability" />
        </div>
        <div>
          <h3 className="text-xs font-semibold text-muted">The six probabilities of moving to another regime</h3>
          <IntervalPlot {...matrixSeries.moves} yLabel="probability" />
        </div>
      </div>
    </div>
  )
}

function Check(props: { readonly made: Made }): ReactNode {
  const { made } = props
  const { sim, regimes, rates, gibbs, summary, result } = made
  const accuracy = result.accuracy
  const gibbsAccuracy = gibbs.regimes.filter((p, i) => argmax(p) === sim.z[i]).length / sim.z.length
  const inside = result.ep.rows.filter((r) => r.covered).length
  const total = result.ep.rows.length
  const worst = Math.max(0, ...result.ep.rows.map((r) => Math.abs(r.z)))
  const sdRatio = result.ep.rows.map((r) => r.sd / (result.gibbs.rows.find((g) => g.name === r.name)?.sd ?? Number.NaN))
  const converged = made.run.status === 'converged'
  const barelyNames = result.visits.flatMap((v, k) => (v < 6 ? [`${NAMES[k]} (${v} steps)`] : []))
  const swapped = result.gap > 0.5 && accuracy < 0.2

  return (
    <div className="flex flex-col gap-3">
      <CheckLines
        lines={[
          {
            label: 'EP converged',
            value: converged ? `yes, ${made.run.t} sweeps` : `no, stopped at ${made.run.t}`,
            rule: 'converged',
            ok: converged,
          },
          {
            label: 'the most probable regime is the true one',
            value: `${sig(100 * accuracy, 3)}% of steps (Gibbs: ${sig(100 * gibbsAccuracy, 3)}%)`,
            rule: 'at least 90% of steps',
            ok: accuracy >= 0.9,
          },
          {
            label: 'EP against Gibbs: the gap in regime probabilities',
            value: `${sig(result.gap, 2)} on average`,
            rule: 'at most 0.05',
            ok: result.agrees,
          },
          {
            label: 'the true rates and transition probabilities inside EP’s 90% intervals',
            value: `${inside} of ${total} inside; largest miss ${sig(worst, 2)} sds`,
            rule: 'at least 9 of 12 (the sampler itself misses about 1 in 6 here: the prior pulls the transitions)',
            ok: inside >= 9,
          },
          {
            label: 'spread, EP against Gibbs (sd ratio) over the rates and the stay probabilities',
            value: `${sig(Math.min(...sdRatio.slice(0, 6)), 2)} to ${sig(Math.max(...sdRatio.slice(0, 6)), 2)}`,
            rule: 'between 0.7 and 1.4',
            ok: sdRatio.slice(0, 6).every((r) => r >= 0.7 && r <= 1.4),
          },
        ]}
        note={
          <>
            <p>
              Rates: EP{' '}
              {rates.map((r, k) => `${NAMES[k]} ${sig(r.mean, 3)} (sd ${sig(r.sd, 2)}, true ${RATES[k]})`).join('; ')}.
              Gibbs {gibbs.rateMean.map((m, k) => `${sig(m, 3)} (sd ${sig(gibbs.rateSd[k] as number, 2)})`).join('; ')}.
            </p>
            <p className="mt-1">
              Identifiability: ordering the rates (<code>constraint: ordered</code> on log rates, which the compiler
              maps to greater-than factors) would remove the labelling problem, but only for Gaussian latents, so the
              Poisson rate goes through <code>exp</code> and a numerical site; that form took 180 sweeps and about 9 s
              on an easy seed, and returned absurd rates (hundreds) on the two seeds tried where the labelling was at
              stake. A hyperprior tying the three rates would make the regimes exchangeable and so make the labelling
              worse. The program keeps the gamma priors of the basic page.
            </p>
            {barelyNames.length === 0 ? null : (
              <p className="mt-1">
                The {barelyNames.join(' and ')} regime is barely visited in this data. The posterior over which steps
                belong to it is spread over several readings, and EP commits to one, reports a narrow rate for the
                regime from a handful of counts and may not converge: the known limit of EP on a mixture, which the
                calibration below measures.
              </p>
            )}
            {swapped ? (
              <p className="mt-1">
                EP has settled on a relabelling of the regimes: the rates it calls low, middle and high are not in that
                order, so its regime probabilities are the sampler&apos;s with the labels permuted. The likelihood
                cannot tell the labellings apart, and only the broad priors on the rates prefer the right one. Damping
                by half avoids this on every seed tried; undamped EP takes the swapped reading on about one seed in ten.
              </p>
            ) : null}
          </>
        }
      />
    </div>
  )
}
