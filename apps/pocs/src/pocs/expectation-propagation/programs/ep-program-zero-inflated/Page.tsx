import { toStructure } from 'omai-engine/probprog/model'
import type { ZeroInflatedPosterior } from 'omai-engine/probprog/validation'
import { NumberField, SeedInput } from 'omai-render/react'
import { type ReactNode, useCallback, useMemo, useState } from 'react'
import { CurveChart } from '../../../../shared/CurveChart.tsx'
import { IntervalPlot, ProbabilityPlot } from '../../../../shared/ep/ProgramCharts.tsx'
import { ProgramPage } from '../../../../shared/ep/ProgramPage.tsx'
import { sig } from '../../../../shared/ep/program-check.ts'
import { epHooks } from '../../../../shared/ep/sweeps.ts'
import { RunProgress } from '../../../../shared/RunProgress.tsx'
import type { HostBase } from '../../../../shared/run.ts'
import { workerTask } from '../../../../shared/runWorker.ts'
import { useRun } from '../../../../shared/useRun.ts'
import { checkZip, defaultSetup, recordOf, runZip, type ZipRun, type ZipSetup, zeroRows } from './compute.ts'
import source from './program.ts?raw'

const makeExact = (): Worker => new Worker(new URL('./exact.worker.ts', import.meta.url), { type: 'module' })

const EQUATIONS = [
  'w ~ Dirichlet(1, 1)   (w[1] = the inflation, with a uniform Beta(1, 1) prior)      b0, b1 ~ N(0, 2²)',
  'z_i ~ Categorical(w)   (0: a count, 1: a structural zero)      rate_i = on[z_i] · exp(b0 + b1 x_i),  on = [1, 0]',
  'y_i ~ Poisson(rate_i)   (a structural zero has rate 0: it is zero; a count is Poisson)',
  'simulated: each y_i is a structural zero with probability π, else Poisson(exp(b0 + b1 x_i)), x_i ~ N(0, 1)',
]

/** The program as a page: the simulated zero-inflated counts, run by EP, against the truth and the exact posterior. */
export default function Page(): ReactNode {
  const [setup, setSetup] = useState<ZipSetup>(defaultSetup)
  const set = (patch: Partial<ZipSetup>): void => setSetup((s) => ({ ...s, ...patch }))
  const fit = useRun<ZipRun>(useCallback((host: HostBase) => runZip(setup, epHooks(host, 'EP')), [setup]))
  const run = fit.value
  const exactTask = useMemo(
    () =>
      run === undefined
        ? undefined
        : workerTask<unknown, ZeroInflatedPosterior>(makeExact, { x: run.data.x, y: run.data.y }, 0),
    [run],
  )
  const exactRun = useRun<ZeroInflatedPosterior>(exactTask)
  const exact = exactRun.value
  const check = useMemo(
    () => (run === undefined || exact === undefined ? undefined : checkZip(run, exact)),
    [run, exact],
  )
  const structure = useCallback(() => (run === undefined ? undefined : toStructure(run.fit.bound)), [run])
  const record = useCallback(() => (run === undefined ? undefined : recordOf(run)), [run])
  const curves = useMemo(() => {
    if (run === undefined) return undefined
    const xs = Array.from({ length: 60 }, (_, i) => -2.6 + (5.2 * i) / 59)
    const { intercept, slope } = run.found
    return {
      curves: [
        {
          name: 'true rate',
          colour: 'tab3',
          width: 2.4,
          x: xs,
          y: xs.map((x) => Math.exp(setup.intercept + setup.slope * x)),
        },
        {
          name: 'EP rate',
          colour: 'tab0',
          width: 2.4,
          dash: 'dashed' as const,
          x: xs,
          y: xs.map((x) => Math.exp(intercept.mean + slope.mean * x)),
        },
      ],
      dots: [
        {
          name: 'structural zeros',
          x: run.data.x.filter((_, i) => run.data.structural[i]),
          y: run.data.y.filter((_, i) => run.data.structural[i]),
          colour: 'tab3',
          size: 5,
          opacity: 0.6,
        },
        {
          name: 'counts',
          x: run.data.x.filter((_, i) => !run.data.structural[i]),
          y: run.data.y.filter((_, i) => !run.data.structural[i]),
          colour: 'tab0',
          size: 5,
          opacity: 0.6,
        },
      ],
    }
  }, [run, setup])
  const zeros = useMemo(() => (run === undefined ? [] : zeroRows(run)), [run])
  const band = (m: { mean: number; sd: number }): { lower: number; upper: number } => ({
    lower: m.mean - 1.645 * m.sd,
    upper: m.mean + 1.645 * m.sd,
  })
  return (
    <ProgramPage
      id="zero-inflated"
      equations={EQUATIONS}
      source={source}
      progress={
        <>
          <RunProgress run={fit} label="EP" />
          <RunProgress run={exactRun} label="Exact posterior" />
        </>
      }
      file="program.ts"
      structure={structure}
      record={record}
      check={check}
      status={
        run === undefined
          ? 'Simulating and running EP...'
          : `EP ${run.fit.run.status} after ${run.fit.run.t} sweeps (${sig(run.ms, 2)} ms). ${exact === undefined ? 'Computing the exact posterior on a grid...' : 'Exact posterior on a refined grid.'}`
      }
      controls={
        <>
          <SeedInput value={setup.seed} onChange={(seed) => set({ seed })} />
          <NumberField
            label="counts"
            value={setup.rows}
            onChange={(rows) => set({ rows })}
            min={30}
            max={400}
            integer
          />
          <NumberField
            label="true inflation"
            value={setup.inflation}
            onChange={(inflation) => set({ inflation })}
            min={0}
            max={0.8}
            step={0.01}
          />
          <NumberField
            label="true intercept"
            value={setup.intercept}
            onChange={(intercept) => set({ intercept })}
            min={-1}
            max={2.5}
            step={0.1}
          />
          <NumberField
            label="true slope"
            value={setup.slope}
            onChange={(slope) => set({ slope })}
            min={-1.5}
            max={1.5}
            step={0.1}
          />
        </>
      }
    >
      {run === undefined || curves === undefined ? (
        <p className="text-sm text-muted">Computing...</p>
      ) : (
        <>
          <div className="grid grid-cols-1 items-start gap-x-6 gap-y-4 lg:grid-cols-2">
            <div className="flex min-w-0 flex-col gap-1">
              <h3 className="text-xs font-semibold text-muted">The counts, the true rate and the rate EP recovered</h3>
              <CurveChart
                curves={curves.curves}
                dots={curves.dots}
                x={{ label: 'covariate x' }}
                y={{ label: 'count' }}
              />
            </div>
            <div className="flex min-w-0 flex-col gap-1">
              <h3 className="text-xs font-semibold text-muted">
                The inflation and the two coefficients: the truth, EP and the exact posterior
              </h3>
              <IntervalPlot
                labels={['inflation', 'intercept', 'slope']}
                truth={[setup.inflation, setup.intercept, setup.slope]}
                ep={{
                  name: 'EP posterior mean',
                  mean: [run.found.inflation.mean, run.found.intercept.mean, run.found.slope.mean],
                  lower: [run.found.inflation.lower, run.found.intercept.lower, run.found.slope.lower],
                  upper: [run.found.inflation.upper, run.found.intercept.upper, run.found.slope.upper],
                }}
                {...(exact === undefined
                  ? {}
                  : {
                      reference: {
                        name: 'exact posterior mean (and 90%)',
                        mean: [exact.pi.mean, exact.b0.mean, exact.b1.mean],
                        lower: [band(exact.pi).lower, band(exact.b0).lower, band(exact.b1).lower],
                        upper: [band(exact.pi).upper, band(exact.b0).upper, band(exact.b1).upper],
                      },
                    })}
              />
            </div>
          </div>
          <div className="flex min-w-0 flex-col gap-1">
            <h3 className="text-xs font-semibold text-muted">
              Each zero count: the probability that it is a structural zero (a count above zero cannot be one), against
              which zeros truly were
            </h3>
            <ProbabilityPlot
              labels={zeros.map((i) => String(i + 1))}
              ep={zeros.map((i) => run.found.structural[i] ?? 0)}
              truth={zeros.map((i) => run.data.structural[i] === true)}
              {...(exact === undefined ? {} : { exact: zeros.map((i) => exact.structural[i] ?? 0) })}
              epName="EP: P(structural zero)"
              exactName="exact: structural zeros summed over the grid posterior"
            />
            <p className="max-w-4xl text-xs text-muted">
              A zero with a large covariate (a large rate) is probably structural; a zero at a small rate may be a
              Poisson zero. The truth (red crosses) is the simulation&apos;s own flag; a posterior probability is not
              expected to match it, only to match the exact posterior (the circles).
            </p>
          </div>
        </>
      )}
    </ProgramPage>
  )
}
