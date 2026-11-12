import {
  abs,
  argmax,
  cast,
  equal,
  item,
  matmul,
  max,
  reshape,
  sub,
  sum,
  takeAlongAxis,
  tensor,
  toFlat,
} from 'omai-engine/tensor'
import type { Curve, PointSet } from 'omai-render/charts'
import { ControlPanel, NumberField, SeedInput } from 'omai-render/react'
import { type ReactNode, useCallback, useMemo, useState } from 'react'
import { CheckLines, endedText, ProgramFrame, RecoveryChart, sig } from '../../../../shared/ep/ProgramParts.tsx'
import { epHooks } from '../../../../shared/ep/sweeps.ts'
import { RunProgress } from '../../../../shared/RunProgress.tsx'
import type { HostBase } from '../../../../shared/run.ts'
import { useRun } from '../../../../shared/useRun.ts'
import { infer, RATES, simulate } from './program.ts'
import source from './program.ts?raw'
import { gibbsReference } from './reference.ts'

type Made = Awaited<ReturnType<typeof fit>>

/** Simulates the data, runs EP and then the Gibbs reference, reporting each sweep to the host. */
async function fit(seed: number, stay: number, steps: number, host: HostBase) {
  const sim = simulate(seed, stay, steps)
  const t0 = performance.now()
  const run = await infer(sim.y, stay, epHooks(host, 'EP'))
  const ms = performance.now() - t0
  const gibbs = await gibbsReference(sim.y, stay, {}, epHooks(host, 'Gibbs reference'))
  return { sim, ...run, ms, gibbs }
}

const NAMES = ['low', 'middle', 'high']

export default function Page(): ReactNode {
  const [seed, setSeed] = useState(1)
  const [stay, setStay] = useState(0.95)
  const [steps, setSteps] = useState(80)
  const task = useCallback((host: HostBase) => fit(seed, stay, steps, host), [seed, stay, steps])
  const ep = useRun<Made>(task)
  const made = ep.value
  return (
    <ProgramFrame
      id="changepoint"
      intro={
        <>
          <p>
            A count at each of {steps} steps. A hidden regime (low, middle or high) follows a sticky Markov chain and
            sets the Poisson rate of the count. The three rates are unknown, each with a broad gamma prior (means 3, 10,
            33); the chain&apos;s stickiness is known. The model is a scan over a categorical state whose transition row
            is chosen by the previous state and whose rate is read out of a gamma vector by the current one. EP infers
            which regime each step is in and the three rates.
          </p>
          <p className="text-xs text-muted">
            The reference is Gibbs sampling of the same model (forward filtering and backward sampling for the regimes,
            the gamma posterior for the rates), 2700 sweeps.
          </p>
        </>
      }
      controls={
        <ControlPanel title="change the data" columns={3}>
          <SeedInput value={seed} onChange={setSeed} />
          <NumberField
            label="stickiness (stay probability)"
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
            min={30}
            max={200}
            step={10}
          />
        </ControlPanel>
      }
      status={
        made === undefined
          ? 'Simulating and running EP...'
          : `${endedText(made.run, made.ms)}${made.run.status === 'converged' ? '' : ' It did not converge: read the check below.'}`
      }
      progress={<RunProgress run={ep} label="EP" />}
      figures={made === undefined ? null : <Figures made={made} />}
      check={made === undefined ? null : <Check made={made} />}
      bound={made?.bound}
      run={made?.run}
      source={source}
      file="ep-program-changepoint/program.ts"
    />
  )
}

function Figures(props: { readonly made: Made }): ReactNode {
  const { made } = props
  const { sim, regimes, rates, gibbs } = made
  const t = useMemo(() => sim.y.map((_, i) => i), [sim])

  const rateCurve = useMemo(
    () => toFlat(matmul(tensor(regimes), tensor([0, 1, 2].map((k) => rates[k]?.mean ?? 0)))),
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
    </div>
  )
}

function Check(props: { readonly made: Made }): ReactNode {
  const { made } = props
  const { sim, regimes, rates, gibbs } = made
  const visited = useMemo(() => [0, 1, 2].map((k) => sim.z.includes(k)), [sim])
  const probs = tensor(regimes)
  const truthIndex = cast(tensor(sim.z), 'int32')
  const accuracy = item(sum(equal(argmax(probs, 1), truthIndex))) / regimes.length
  const onTruth = item(sum(takeAlongAxis(probs, reshape(truthIndex, [-1, 1]), 1))) / regimes.length
  const gibbsAccuracy = item(sum(equal(argmax(tensor(gibbs.regimes), 1), truthIndex))) / regimes.length
  const gap = item(sum(max(abs(sub(probs, tensor(gibbs.regimes))), 1))) / regimes.length
  const sdRatio = rates.map((r, k) => r.sd / (gibbs.rateSd[k] as number))
  const misses = rates.map((r, k) => Math.abs(r.mean - (RATES[k] as number)) / r.sd)
  const inside = rates.filter((r, k) => (RATES[k] as number) >= r.lower && (RATES[k] as number) <= r.upper).length
  const worst = Math.max(0, ...misses.filter((_, k) => visited[k]))
  const converged = made.run.status === 'converged'
  const unused = visited.flatMap((v, k) => (v ? [] : [NAMES[k]]))

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
            label: 'probability EP gives the true regime, on average',
            value: sig(onTruth, 3),
            rule: 'at least 0.8',
            ok: onTruth >= 0.8,
          },
          {
            label: 'EP against Gibbs: the gap in regime probabilities',
            value: `${sig(gap, 2)} on average`,
            rule: 'at most 0.05',
            ok: gap <= 0.05,
          },
          {
            label: 'the true rates against EP: inside the 90% interval, and the largest miss in posterior sds',
            value: `${inside} of 3 inside; largest miss ${sig(worst, 2)} sds`,
            rule: 'no visited regime more than 3 sds off (the 90% interval misses 1 in 10 by chance)',
            ok: worst <= 3,
          },
          {
            label: 'rate spread, EP against Gibbs (sd ratio)',
            value: sdRatio.map((r) => sig(r, 2)).join(', '),
            rule: 'between 0.7 and 1.4',
            ok: sdRatio.every((r) => r >= 0.7 && r <= 1.4),
          },
        ]}
        note={
          <>
            <p>
              Rates: EP{' '}
              {rates.map((r, k) => `${NAMES[k]} ${sig(r.mean, 3)} (sd ${sig(r.sd, 2)}, true ${RATES[k]})`).join('; ')}.
              Gibbs {gibbs.rateMean.map((m, k) => `${sig(m, 3)} (sd ${sig(gibbs.rateSd[k] as number, 2)})`).join('; ')}.
            </p>
            {unused.length === 0 ? null : (
              <p className="mt-1">
                The {unused.join(' and ')} regime never occurs in this data. The posterior of its rate should then be
                its prior, which is what Gibbs finds; EP instead lets the unused regime take over part of a visited one,
                chooses a side arbitrarily and reports a narrow rate. It may also not converge. This is the known limit
                of EP on a mixture: it commits to one mode where the exact posterior is spread over several.
              </p>
            )}
          </>
        }
      />
    </div>
  )
}
