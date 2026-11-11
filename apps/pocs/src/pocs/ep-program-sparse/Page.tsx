import { toStructure } from 'omai-engine/probprog/model'
import type { SpikeSlabPosterior } from 'omai-engine/probprog/validation'
import { NumberField, SeedInput } from 'omai-render/react'
import { type ReactNode, useCallback, useMemo, useState } from 'react'
import { IntervalPlot, ProbabilityPlot } from '../../shared/ep/ProgramCharts.tsx'
import { ProgramPage } from '../../shared/ep/ProgramPage.tsx'
import { sig } from '../../shared/ep/program-check.ts'
import { epHooks } from '../../shared/ep/sweeps.ts'
import { RunProgress } from '../../shared/RunProgress.tsx'
import type { HostBase } from '../../shared/run.ts'
import { workerTask } from '../../shared/runWorker.ts'
import { useRun } from '../../shared/useRun.ts'
import {
  checkSparse,
  defaultSetup,
  MAX_EXACT,
  recordOf,
  runSparse,
  type SparseRun,
  type SparseSetup,
} from './compute.ts'
import source from './program.ts?raw'

const makeExact = (): Worker => new Worker(new URL('./exact.worker.ts', import.meta.url), { type: 'module' })

const EQUATIONS = [
  'z_j ~ Bernoulli(0.2)   slab_j ~ N(0, 3²)   spike_j ~ N(0, 0.05²)   w_j = z_j ? slab_j : spike_j   (a gate per coefficient)',
  'y_i ~ N(x_i · w, 1/τ),  τ ~ Gamma(2, 0.5)          (Gaussian likelihood, unknown noise precision)',
  'y_i ~ Student-t(3, x_i · w, s)  with s the known noise sd          (heavy-tailed likelihood: the compiler needs s known)',
  'simulated: a few w_j non-zero (the truth), X Gaussian, y = x·w + noise, a share of the rows shifted by 10 noise sds (outliers)',
]

/** The program as a page: the simulated sparse regression, run by EP with a Gaussian and a Student-t likelihood. */
export default function Page(): ReactNode {
  const [setup, setSetup] = useState<SparseSetup>(defaultSetup)
  const [shown, setShown] = useState<'gaussian' | 'student'>('student')
  const set = (patch: Partial<SparseSetup>): void => setSetup((s) => ({ ...s, ...patch }))
  const epRun = useRun<SparseRun>(
    useCallback((host: HostBase) => runSparse(setup, (stage) => epHooks(host, stage)), [setup]),
  )
  const run = epRun.value
  const exactTask = useMemo(
    () =>
      run === undefined || run.setup.features > MAX_EXACT
        ? undefined
        : workerTask<unknown, SpikeSlabPosterior | undefined>(
            makeExact,
            { features: run.setup.features, X: run.data.X, y: run.data.y },
            0,
          ),
    [run],
  )
  const exactRun = useRun<SpikeSlabPosterior | undefined>(exactTask)
  const exact = exactRun.value
  const check = useMemo(
    () =>
      run === undefined || (exact === undefined && run.setup.features <= 14) ? undefined : checkSparse(run, exact),
    [run, exact],
  )
  const labels = useMemo(() => Array.from({ length: setup.features }, (_, j) => String(j + 1)), [setup.features])
  const fit = run === undefined ? undefined : shown === 'gaussian' ? run.gaussian : run.student
  const structure = useCallback(() => (fit === undefined ? undefined : toStructure(fit.bound)), [fit])
  const record = useCallback(() => (fit === undefined ? undefined : recordOf(fit)), [fit])
  return (
    <ProgramPage
      id="sparse"
      equations={EQUATIONS}
      source={source}
      progress={
        <>
          <RunProgress run={epRun} label="EP" />
          <RunProgress run={exactRun} label="Exact posterior (all subsets)" />
        </>
      }
      file="program.ts"
      structure={structure}
      record={record}
      check={check}
      status={
        run === undefined
          ? 'Simulating and running EP...'
          : `Gaussian: ${run.gaussian.run.status} after ${run.gaussian.run.t} sweeps (${sig(run.gaussian.ms, 2)} ms). Student-t: ${run.student.run.status} after ${run.student.run.t} sweeps (${sig(run.student.ms, 2)} ms).${exact === undefined ? '' : ' Exact posterior by enumeration.'}`
      }
      controls={
        <>
          <SeedInput value={setup.seed} onChange={(seed) => set({ seed })} />
          <NumberField label="rows" value={setup.rows} onChange={(rows) => set({ rows })} min={20} max={200} integer />
          <NumberField
            label="coefficients"
            value={setup.features}
            onChange={(features) => set({ features, active: Math.min(setup.active, features) })}
            min={6}
            max={20}
            integer
          />
          <NumberField
            label="truly non-zero"
            value={setup.active}
            onChange={(active) => set({ active })}
            min={1}
            max={Math.min(8, setup.features)}
            integer
          />
          <NumberField
            label="noise sd (inliers)"
            value={setup.noise}
            onChange={(noise) => set({ noise })}
            min={0.1}
            max={2}
            step={0.05}
          />
          <NumberField
            label="outliers (share of rows)"
            value={setup.outliers}
            onChange={(outliers) => set({ outliers })}
            min={0}
            max={0.4}
            step={0.01}
          />
        </>
      }
    >
      {run === undefined ? (
        <p className="text-sm text-muted">Computing...</p>
      ) : (
        <>
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span className="text-muted">The diagram and the messages below are those of the</span>
            {(['student', 'gaussian'] as const).map((k) => (
              <button
                key={k}
                type="button"
                onClick={() => setShown(k)}
                className={`rounded border px-2 py-0.5 ${shown === k ? 'border-fg text-fg' : 'border-line text-muted'}`}
              >
                {k === 'student' ? 'Student-t model' : 'Gaussian model'}
              </button>
            ))}
          </div>
          <div className="grid grid-cols-1 items-start gap-x-6 gap-y-4 lg:grid-cols-2">
            {(
              [
                ['Student-t likelihood, known scale', run.student],
                ['Gaussian likelihood, unknown precision', run.gaussian],
              ] as const
            ).map(([title, f]) => (
              <div key={title} className="flex min-w-0 flex-col gap-3">
                <h3 className="text-xs font-semibold text-muted">{title}</h3>
                <ProbabilityPlot
                  labels={labels}
                  ep={f.coef.pip}
                  truth={run.active}
                  {...(f === run.gaussian && exact !== undefined ? { exact: exact.inclusion } : {})}
                  epName="EP: P(slab)"
                  exactName="exact: all 2^p subsets summed"
                />
                <IntervalPlot
                  labels={labels}
                  truth={run.data.truth}
                  ep={{ name: 'EP posterior mean', mean: f.coef.mean, lower: f.coef.lower, upper: f.coef.upper }}
                  {...(f === run.gaussian && exact !== undefined
                    ? {
                        reference: {
                          name: 'exact posterior mean',
                          mean: exact.mean,
                          lower: exact.mean.map((m, j) => m - 1.645 * (exact.sd[j] as number)),
                          upper: exact.mean.map((m, j) => m + 1.645 * (exact.sd[j] as number)),
                        },
                      }
                    : {})}
                  yLabel="coefficient"
                />
              </div>
            ))}
          </div>
          <p className="max-w-4xl text-xs text-muted">
            {run.data.isOutlier.filter(Boolean).length} of {setup.rows} rows are outliers. The Gaussian model reads them
            as noise: it infers a noise sd of {sig(1 / Math.sqrt(run.gaussian.precision ?? Number.NaN), 3)} where the
            inliers have {setup.noise}, which widens its intervals (EP and the exact posterior agree on that, the exact
            one summed over all subsets); the Student-t model keeps the noise at its known scale and discounts the
            outliers. The exact intervals drawn are the mean plus or minus 1.645 sd (a spike and slab marginal is not
            Gaussian, so they are only a guide); EP's are the exact 90% intervals of the mixture of its two branches.
          </p>
        </>
      )}
    </ProgramPage>
  )
}
