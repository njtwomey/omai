import { ControlPanel, NumberField, SeedInput } from 'omai-render/react'
import { type ReactNode, useCallback, useMemo, useState } from 'react'
import { endedText, ProgramFrame, RecoveryChart, RecoveryCheck, ReferenceBar } from '../../shared/ep/ProgramParts.tsx'
import { agreement, recovery } from '../../shared/ep/recovery.ts'
import { epHooks } from '../../shared/ep/sweeps.ts'
import { RunProgress } from '../../shared/RunProgress.tsx'
import type { HostBase } from '../../shared/run.ts'
import { workerTask } from '../../shared/runWorker.ts'
import { useRun } from '../../shared/useRun.ts'
import { BASE, infer, simulate, WIDTH } from './program.ts'
import source from './program.ts?raw'
import type { Reference } from './reference.ts'

const intensity = (f: number): number => BASE * Math.exp(f)
const makeWorker = (): Worker => new Worker(new URL('./reference.worker.ts', import.meta.url), { type: 'module' })

type Made = Awaited<ReturnType<typeof fit>>

/** Simulates the data and runs EP on them, reporting each sweep to the host. */
async function fit(seed: number, amplitude: number, lengthScale: number, host: HostBase) {
  const sim = simulate(seed, { amplitude, lengthScale })
  const t0 = performance.now()
  const run = await infer(sim.counts, sim.prior, epHooks(host, 'EP'))
  return { sim, ...run, ms: performance.now() - t0 }
}

export default function Page(): ReactNode {
  const [seed, setSeed] = useState(1)
  const [amplitude, setAmplitude] = useState(0.8)
  const [lengthScale, setLengthScale] = useState(1.5)
  const task = useCallback((host: HostBase) => fit(seed, amplitude, lengthScale, host), [seed, amplitude, lengthScale])
  const ep = useRun<Made>(task)
  const made = ep.value
  const nutsTask = useMemo(
    () =>
      made === undefined
        ? undefined
        : workerTask<unknown, Reference>(makeWorker, { counts: made.sim.counts, prior: made.sim.prior }),
    [made],
  )
  const nuts = useRun<Reference>(nutsTask, { auto: false })
  return (
    <ProgramFrame
      id="cox"
      intro={
        <>
          <p>
            Events fall in 40 bins on an interval. The intensity is {BASE} exp(f(x)) with f a draw from a Gaussian
            process; the counts are Poisson. The model puts the Gaussian-process prior on f as one dense multivariate
            normal (the kernel matrix is a constant) and a Poisson likelihood on each bin, and expectation propagation
            infers f.
          </p>
          <p className="text-xs text-muted">
            The kernel&apos;s spread and length scale are taken as known: the program recovers the intensity, not the
            kernel.
          </p>
        </>
      }
      controls={
        <ControlPanel title="change the data" columns={3}>
          <SeedInput value={seed} onChange={setSeed} />
          <NumberField label="spread of f" value={amplitude} onChange={setAmplitude} min={0.2} max={1.5} step={0.1} />
          <NumberField
            label="length scale"
            value={lengthScale}
            onChange={setLengthScale}
            min={0.5}
            max={4}
            step={0.25}
          />
        </ControlPanel>
      }
      status={made === undefined ? 'Simulating and running EP...' : endedText(made.run, made.ms)}
      progress={<RunProgress run={ep} label="EP" />}
      figures={made === undefined ? null : <Figures made={made} nuts={nuts.value} />}
      check={made === undefined ? null : <Check made={made} />}
      extra={
        made === undefined ? null : (
          <section className="flex flex-col gap-2">
            <h2 className="text-sm font-semibold">Against a long NUTS run</h2>
            <ReferenceBar
              name="NUTS (2 chains, 300 warmup, 1000 draws)"
              run={nuts}
              agreement={nuts.value === undefined ? undefined : agreement(made.rows, nuts.value)}
              note="The same model with the GP written in its whitened form (f = L z, z standard normal), which NUTS samples well. Its band appears in the figure above as a second band (dashed centre)."
            />
          </section>
        )
      }
      bound={made?.bound}
      run={made?.run}
      source={source}
      file="ep-program-cox/program.ts"
    />
  )
}

function Figures(props: { readonly made: Made; readonly nuts: Reference | undefined }): ReactNode {
  const { made, nuts } = props
  const { sim, rows } = made
  const x = sim.x
  const bands = useMemo(
    () => [
      {
        name: 'EP posterior',
        x,
        mean: rows.map((r) => intensity(r.mean)),
        lower: rows.map((r) => intensity(r.lower)),
        upper: rows.map((r) => intensity(r.upper)),
        colour: 'tab0',
      },
      ...(nuts === undefined
        ? []
        : [
            {
              name: 'NUTS',
              x,
              mean: nuts.mean.map(intensity),
              lower: nuts.mean.map((m, i) => intensity(m - 1.645 * (nuts.sd[i] ?? 0))),
              upper: nuts.mean.map((m, i) => intensity(m + 1.645 * (nuts.sd[i] ?? 0))),
              colour: 'tab1',
              dash: 'dashed' as const,
            },
          ]),
    ],
    [rows, x, nuts],
  )
  const truth = useMemo(() => ({ x, y: sim.f.map(intensity), name: 'true intensity' }), [x, sim])
  const dots = useMemo(() => [{ x, y: sim.counts.map((c) => c / WIDTH), name: 'counts per unit length' }], [x, sim])
  return (
    <RecoveryChart bands={bands} truth={truth} dots={dots} xLabel="x" yLabel="intensity (events per unit length)" />
  )
}

function Check(props: { readonly made: Made }): ReactNode {
  const { sim, rows } = props.made
  const rec = useMemo(() => recovery(rows, sim.f), [rows, sim])
  return <RecoveryCheck rows={[{ label: 'log-intensity f, 40 bins', recovery: rec }]} />
}
