import { type IrtSummary, irtHiddenKnobs, type Recovery } from 'omai-engine/bayesian-examples'
import type { ChainDiagnostics, PosteriorPairs } from 'omai-engine/exposition'
import { ChainDiagnosticsView, FigurePanel, NumberField, PosteriorPairsView, useProbe } from 'omai-render/react'
import { type ReactNode, useState } from 'react'
import {
  ExamplePage,
  type ExampleView,
  fmt,
  Objective,
  ReferenceCheck,
  Table,
  useRecord,
  Waiting,
} from '../../shared/ExamplePage.tsx'
import { irtExample, irtProducer } from './method.ts'
import { RecoveryChart } from './RecoveryChart.tsx'

const PANELS = {
  abilities: 'Abilities against the truth',
  difficulties: 'Difficulties against the truth',
  pairs: 'Pair plot of three abilities and the population',
  curve: 'Chains or objective',
  numbers: 'The population and the intervals',
  calibration: 'Calibration against the truth',
  check: 'How we know it works against a long run',
}
// The reference: a long NUTS run of the same answers (population priors non-centred), in a worker of its own.
const REFERENCE = { method: 'nuts', numChains: 2, numWarmup: 200, numDraws: 300, thin: 1 }

interface Recoveries {
  readonly students: number
  readonly questions: number
  readonly totalStudents: number
  readonly totalQuestions: number
  readonly abilities: Recovery
  readonly difficulties: Recovery
  readonly discriminations: Recovery | null
}

interface Calibration {
  readonly abilities: number
  readonly difficulties: number
  readonly discriminations: number | null
  readonly hyperparameters: number | null
  readonly abilityCorrelation: number
  readonly difficultyCorrelation: number
  readonly counts: { readonly students: number; readonly questions: number }
}

const NAMES: Readonly<Record<string, string>> = {
  tau_t: 'ability sd',
  mu_b: 'average difficulty',
  tau_b: 'difficulty sd',
  mu_a: 'centre of the log-discriminations',
  tau_a: 'sd of the log-discriminations',
  prec_t: 'ability precision',
  prec_b: 'difficulty precision',
  prec_a: 'precision of the discriminations',
}

const percent = (v: number | null | undefined): string => (typeof v === 'number' ? `${fmt(100 * v, 0)}%` : '-')

export default function ItemResponsePage(): ReactNode {
  return (
    <ExamplePage
      id="item-response"
      example={irtExample as never}
      producer={irtProducer}
      panels={PANELS}
      exclude={irtHiddenKnobs}
      reference={REFERENCE}
      observe={(view) => <Outputs view={view} />}
      check={(view) => <Checks view={view} />}
    />
  )
}

function Outputs(props: { readonly view: ExampleView }): ReactNode {
  const { live, t, panels, done, config, objective } = props.view
  const totalStudents = Math.max(2, Math.round(Number(config['students'])))
  const totalQuestions = Math.max(2, Math.round(Number(config['questions'])))
  const [askedStudents, setStudents] = useState<number | null>(null)
  const [askedQuestions, setQuestions] = useState<number | null>(null)
  const students = Math.max(2, Math.min(totalStudents, askedStudents ?? totalStudents))
  const questions = Math.max(2, Math.min(totalQuestions, askedQuestions ?? totalQuestions))
  const wantsRecovery = panels.active('abilities') || panels.active('difficulties') || panels.active('numbers')
  const recovery = useProbe(live, t, wantsRecovery ? 'irt/recovery' : undefined, { students, questions } as never)
    .value as Recoveries | null | undefined
  const pairs = useRecord<PosteriorPairs>(live, t, panels.active('pairs') ? 'irt/pairs' : undefined, 'posteriorPairs')
  const diagnostics = useRecord<ChainDiagnostics>(
    live,
    t,
    panels.active('curve') ? 'irt/diagnostics' : undefined,
    'chainDiagnostics',
  )
  const summary = useProbe(live, t, panels.active('numbers') ? 'irt/summary' : undefined).value as
    | IrtSummary
    | null
    | undefined
  const rows: [string, string][] = []
  if (summary !== null && summary !== undefined) {
    rows.push(['draws used', String(summary.draws)])
    if (summary.divergences !== null) rows.push(['divergent transitions', String(summary.divergences)])
    if (summary.maxRhat !== null) rows.push(['largest R-hat', fmt(summary.maxRhat)])
    if (summary.minEss !== null) rows.push(['smallest effective sample size', fmt(summary.minEss, 0)])
    for (const name of Object.keys(NAMES)) {
      const r = summary.rows.find((x) => x.name === name)
      if (r === undefined) continue
      const truth = summary.truth[name]
      rows.push([
        `${NAMES[name]}: true, mean (sd)`,
        `${truth === undefined ? '-' : fmt(truth, 2)}, ${fmt(r.mean, 2)} (${fmt(r.sd, 2)})`,
      ])
    }
  }
  if (recovery !== null && recovery !== undefined) {
    rows.push([
      `abilities (${recovery.students} students): mean 90% interval width, rmse`,
      `${fmt(recovery.abilities.width, 2)}, ${fmt(recovery.abilities.rmse, 2)}`,
    ])
    rows.push([
      `difficulties (${recovery.questions} questions): mean 90% interval width, rmse`,
      `${fmt(recovery.difficulties.width, 2)}, ${fmt(recovery.difficulties.rmse, 2)}`,
    ])
  }
  const sliders = (
    <div className="mb-2 grid max-w-xl grid-cols-1 gap-2 sm:grid-cols-2">
      <NumberField
        label="students seen"
        value={students}
        onChange={(v) => setStudents(Math.max(2, Math.min(totalStudents, Math.round(v))))}
        min={2}
        max={totalStudents}
        step={1}
        slider
      />
      <NumberField
        label="questions seen"
        value={questions}
        onChange={(v) => setQuestions(Math.max(2, Math.min(totalQuestions, Math.round(v))))}
        min={2}
        max={totalQuestions}
        step={1}
        slider
      />
    </div>
  )
  return (
    <div className="flex flex-col gap-4">
      {sliders}
      <div className="grid grid-cols-1 items-start gap-4 lg:grid-cols-2">
        <FigurePanel panels={panels} name="abilities">
          {recovery === undefined || recovery === null ? (
            <Waiting done={done} />
          ) : (
            <RecoveryChart
              record={recovery.abilities}
              title={`${recovery.students} students, ${recovery.questions} questions`}
            />
          )}
          <p className="mt-1 text-xs text-muted">
            Each dot is a student's posterior mean ability against the ability that generated the answers, with its 90%
            interval. The method is run again on the students and questions seen: move the sliders and the intervals
            narrow as questions are added (more answers per student) and the difficulties tighten as students are added.
            Abilities are on the scale the model fixes: mean 0, and for the Rasch model every discrimination 1, for 2PL
            ability sd 1 (the truth is shifted and scaled the same way).
          </p>
        </FigurePanel>
        <FigurePanel panels={panels} name="difficulties">
          {recovery === undefined || recovery === null ? (
            <Waiting done={done} />
          ) : (
            <>
              <RecoveryChart record={recovery.difficulties} title={`${recovery.questions} questions`} />
              {recovery.discriminations === null ? null : (
                <RecoveryChart record={recovery.discriminations} title="discriminations" />
              )}
            </>
          )}
          <p className="mt-1 text-xs text-muted">
            Difficulties are what the average difficulty moves: with the abilities pinned to mean 0, the average
            difficulty is identified. Hard questions are those few students answer; their intervals are wider when few
            students answered them correctly or when the abilities are far from the question.
          </p>
        </FigurePanel>
      </div>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <FigurePanel panels={panels} name="pairs">
          {pairs === undefined ? <Waiting done={done} /> : <PosteriorPairsView record={pairs} controls={false} />}
          <p className="mt-1 text-xs text-muted">
            The first three students' abilities with the average difficulty and the spreads of the population (log
            scale). The spread of the population is only weakly known from a few dozen students: with the centred priors
            the abilities and their spread form a funnel whose neck draws the divergent transitions (red).
          </p>
        </FigurePanel>
        <div className="flex flex-col gap-4">
          <FigurePanel panels={panels} name="numbers">
            {summary === undefined || summary === null ? <Waiting done={done} /> : <Table rows={rows} />}
          </FigurePanel>
          <FigurePanel panels={panels} name="curve">
            {diagnostics !== undefined ? (
              <ChainDiagnosticsView record={diagnostics} />
            ) : objective !== undefined ? (
              <Objective live={live} path={objective.path} label={objective.label} t={t} />
            ) : (
              <p className="text-sm text-muted">
                No chains or objective to show for this method: the samplers that keep chains show them once they have
                draws, the variational methods their objective.
              </p>
            )}
          </FigurePanel>
        </div>
      </div>
    </div>
  )
}

function Checks(props: { readonly view: ExampleView }): ReactNode {
  const { view } = props
  const { live, t, panels, done } = view
  const calibration = useProbe(live, t, panels.active('calibration') ? 'irt/calibration' : undefined).value as
    | Calibration
    | null
    | undefined
  return (
    <div className="flex flex-col gap-4">
      <FigurePanel panels={panels} name="calibration">
        <p className="text-xs text-muted">
          Against the truth that generated the answers: of the abilities, difficulties and hyperparameters the run gives
          a 90% interval, about nine in ten should have the true value inside. The fraction of a few dozen values has a
          sampling error of about 5 points, so this reads best with more students and questions; a method that
          approximates the posterior (a mean-field guide, a Laplace fit) is often too narrow and falls short. The model
          is right here when the Rasch model is chosen on equal discriminations; with varied discriminations the Rasch
          model is wrong, and the intervals say it.
        </p>
        {calibration === undefined || calibration === null ? (
          <Waiting done={done} />
        ) : (
          <Table
            rows={[
              [
                `abilities inside their 90% interval (of ${calibration.counts.students})`,
                percent(calibration.abilities),
              ],
              [
                `difficulties inside their 90% interval (of ${calibration.counts.questions})`,
                percent(calibration.difficulties),
              ],
              ...(calibration.discriminations === null
                ? []
                : [
                    ['discriminations inside their 90% interval', percent(calibration.discriminations)] as [
                      string,
                      string,
                    ],
                  ]),
              ['hyperparameters inside their 90% interval', percent(calibration.hyperparameters)],
              ['correlation of estimated and true abilities', fmt(calibration.abilityCorrelation, 2)],
              ['correlation of estimated and true difficulties', fmt(calibration.difficultyCorrelation, 2)],
            ]}
          />
        )}
      </FigurePanel>
      <ReferenceCheck
        view={view}
        probes={{ summary: { path: 'irt/summary' } }}
        checkPath="irt/check"
        explain={
          <>
            The reference is a long NUTS run (2 chains, 200 warm-up, 300 draws) of the same answers with the population
            priors non-centred, in a worker of its own once this run has ended. The check puts the chosen run against it
            coordinate by coordinate: the difference of the means in the reference's spreads, and in Monte Carlo errors
            for samplers. It is feasible only for small tables (the default is 40 students by 12 questions): the cost of
            NUTS grows with the students and questions, so for a large table the calibration against the truth above is
            the check.
          </>
        }
        unfinished="The reference run did not reach its draws within its own budget (a minute): reduce the students or the questions."
        rows={(c) => {
          const x = c as unknown as Record<string, number | string | null>
          const out: [string, string][] = [
            ['coordinates compared', String(x['compared'])],
            [
              'largest difference of means, in reference sds',
              `${fmt(x['maxMeanError'])} (${typeof x['worst'] === 'string' ? x['worst'] : '-'})`,
            ],
            [
              'largest difference, in Monte Carlo errors',
              typeof x['maxZ'] === 'number' ? fmt(x['maxZ'], 1) : 'not defined (no chains)',
            ],
            ['median ratio of spreads (method / reference)', fmt(x['medianSdRatio'])],
          ]
          for (const name of Object.keys(NAMES))
            if (typeof x[`${name}Error`] === 'number')
              out.push([
                `${NAMES[name]}: mean difference in reference sds, spread ratio`,
                `${fmt(x[`${name}Error`])}, ${fmt(x[`${name}Ratio`])}`,
              ])
          out.push(['draws in the reference', String(x['referenceDraws'])])
          return out
        }}
      />
    </div>
  )
}
