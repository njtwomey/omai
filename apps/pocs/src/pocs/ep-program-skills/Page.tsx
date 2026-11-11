import { toStructure } from 'omai-engine/probprog/model'
import { NumberField, SeedInput } from 'omai-render/react'
import { type ReactNode, useCallback, useMemo, useState } from 'react'
import { IntervalPlot } from '../../shared/ep/ProgramCharts.tsx'
import { ProgramPage } from '../../shared/ep/ProgramPage.tsx'
import { sig } from '../../shared/ep/program-check.ts'
import { epHooks } from '../../shared/ep/sweeps.ts'
import { RunProgress } from '../../shared/RunProgress.tsx'
import type { HostBase } from '../../shared/run.ts'
import { workerTask } from '../../shared/runWorker.ts'
import { useRun } from '../../shared/useRun.ts'
import {
  checkSkills,
  defaultSetup,
  recordOf,
  runSkills,
  type SkillsReference,
  type SkillsRun,
  type SkillsSetup,
} from './compute.ts'
import source from './program.ts?raw'

const makeReference = (): Worker => new Worker(new URL('./reference.worker.ts', import.meta.url), { type: 'module' })

const EQUATIONS = [
  'τ ~ Gamma(2, 2)   s_i ~ N(0, 1/τ)   (the skill of each player; τ is how spread out the players are)',
  'P(a beats b) = Φ((s_a − s_b) / √2)   (a probit factor on a difference of two skills: performance noise 1 on each side)',
  'simulated: s_i ~ N(0, spread²), a round robin in which every pair meets `rounds` times, each game drawn from the model',
]

/** The program as a page: a simulated tournament, run by EP, against the true skills and a long NUTS run. */
export default function Page(): ReactNode {
  const [setup, setSetup] = useState<SkillsSetup>(defaultSetup)
  const set = (patch: Partial<SkillsSetup>): void => setSetup((s) => ({ ...s, ...patch }))
  const fit = useRun<SkillsRun>(useCallback((host: HostBase) => runSkills(setup, epHooks(host, 'EP')), [setup]))
  const run = fit.value
  const referenceTask = useMemo(
    () => (run === undefined ? undefined : workerTask<unknown, SkillsReference>(makeReference, run.data)),
    [run],
  )
  const referenceRun = useRun<SkillsReference>(referenceTask)
  const reference = referenceRun.value
  const check = useMemo(
    () => (run === undefined || reference === undefined ? undefined : checkSkills(run, reference)),
    [run, reference],
  )
  const structure = useCallback(() => (run === undefined ? undefined : toStructure(run.fit.bound)), [run])
  const record = useCallback(() => (run === undefined ? undefined : recordOf(run)), [run])
  // The players in the order of their true skill, so that a good recovery is a rising line.
  const order = useMemo(
    () =>
      run === undefined
        ? []
        : run.data.skills
            .map((s, i) => [s, i] as const)
            .sort((x, y) => x[0] - y[0])
            .map(([, i]) => i),
    [run],
  )
  const pick = (v: readonly number[]): number[] => order.map((i) => v[i] ?? 0)
  return (
    <ProgramPage
      id="skills"
      equations={EQUATIONS}
      source={source}
      progress={
        <>
          <RunProgress run={fit} label="EP" />
          <RunProgress run={referenceRun} label="NUTS reference" />
        </>
      }
      file="program.ts"
      structure={structure}
      record={record}
      check={check}
      status={
        run === undefined
          ? 'Simulating and running EP...'
          : `EP ${run.fit.run.status} after ${run.fit.run.t} sweeps (${sig(run.ms, 2)} ms) on ${run.data.win.length} games. ${reference === undefined ? 'Running NUTS for the reference in a worker...' : `Reference: NUTS, ${sig(reference.ms / 1000, 2)} s.`}`
      }
      controls={
        <>
          <SeedInput value={setup.seed} onChange={(seed) => set({ seed })} />
          <NumberField
            label="players"
            value={setup.players}
            onChange={(players) => set({ players })}
            min={4}
            max={14}
            integer
          />
          <NumberField
            label="games per pair"
            value={setup.rounds}
            onChange={(rounds) => set({ rounds })}
            min={1}
            max={8}
            integer
          />
          <NumberField
            label="true spread of skills"
            value={setup.spread}
            onChange={(spread) => set({ spread })}
            min={0.3}
            max={2.5}
            step={0.1}
          />
        </>
      }
    >
      {run === undefined ? (
        <p className="text-sm text-muted">Computing...</p>
      ) : (
        <div className="flex min-w-0 flex-col gap-1">
          <h3 className="text-xs font-semibold text-muted">
            The skill of each player, players in the order of their true skill: the truth, EP with its 90% band and the
            long NUTS run
          </h3>
          <IntervalPlot
            labels={order.map((i) => String.fromCharCode(65 + i))}
            truth={pick(run.data.skills)}
            ep={{
              name: 'EP posterior mean',
              mean: pick(run.found.mean),
              lower: pick(run.found.lower),
              upper: pick(run.found.upper),
            }}
            {...(reference === undefined
              ? {}
              : {
                  reference: {
                    name: 'NUTS (long run) mean and 90%',
                    mean: pick(reference.mean),
                    lower: pick(reference.mean).map((m, k) => m - 1.645 * (pick(reference.sd)[k] ?? 0)),
                    upper: pick(reference.mean).map((m, k) => m + 1.645 * (pick(reference.sd)[k] ?? 0)),
                  },
                })}
            yLabel="skill"
          />
          <p className="max-w-4xl text-xs text-muted">
            The bands are wide because a player has only {(run.setup.players - 1) * run.setup.rounds} games. EP also
            infers how spread out the players are (the precision τ: {sig(run.found.tau.mean, 3)} ±{' '}
            {sig(run.found.tau.sd, 2)}; the truth is {sig(1 / run.setup.spread ** 2, 3)}).
          </p>
        </div>
      )}
    </ProgramPage>
  )
}
