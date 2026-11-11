import { classifierExample } from 'omai-engine/bayesian-examples'
import type { ChainDiagnostics, DecisionSurface, PosteriorPairs } from 'omai-engine/exposition'
import { ChainDiagnosticsView, DecisionSurfaceView, FigurePanel, PosteriorPairsView, useProbe } from 'omai-render/react'
import type { ReactNode } from 'react'
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
import { classifyProducer } from './method.ts'

const SURFACE = 41
// The reference: a long NUTS run of the same problem, in a worker of its own once the chosen run has ended.
const REFERENCE_SIZE = 25
const REFERENCE = { method: 'nuts', numChains: 2, numWarmup: 200, numDraws: 300, thin: 1 }

const PANELS = {
  surface: 'Decision surface',
  pairs: 'Pair plot',
  curve: 'Chains or objective',
  numbers: 'The numbers',
  check: 'How we know it works',
}

export default function ClassifyPage(): ReactNode {
  return (
    <ExamplePage
      id="classify"
      example={classifierExample}
      producer={classifyProducer}
      panels={PANELS}
      reference={REFERENCE}
      observe={(view) => <Outputs view={view} />}
      check={(view) => (
        <ReferenceCheck
          view={view}
          probes={{
            scores: { path: 'classify/scores' },
            surface: { path: 'classify/surface', args: { size: REFERENCE_SIZE } },
          }}
          checkPath="classify/check"
          explain={
            <>
              The reference is a long NUTS run (2 chains, 200 warm-up, 300 draws) of the same problem, computed in a
              worker of its own once this run has ended and kept while only the method changes. The exact posterior has
              no closed form, and a grid of it is out of reach beyond three coefficients, so the check is against the
              reference: the error of each coefficient's mean in units of the reference spread, and how far the
              predictive surface moves.
            </>
          }
          unfinished="The reference run did not reach its draws within its own budget (20 s): lower the degree or the rows."
          rows={(c) => [
            ['largest mean error (in reference sds)', fmt(c['maxMeanError'])],
            ['median spread ratio (method / reference)', fmt(c['medianSdRatio'])],
            ['largest boundary difference (probability)', fmt(c['maxSurfaceDifference'])],
            ['mean boundary difference', fmt(c['meanSurfaceDifference'], 4)],
            ['reference test accuracy', fmt(c['referenceAccuracy'])],
            ['reference held-out log predictive', fmt(c['referenceLogPredictive'])],
          ]}
        />
      )}
    />
  )
}

function Outputs(props: { readonly view: ExampleView }): ReactNode {
  const { live, t, panels, objective, done } = props.view
  // A panel that is folded or hidden asks the session for nothing.
  const surface = useRecord<DecisionSurface>(
    live,
    t,
    panels.active('surface') ? 'classify/surface' : undefined,
    'decisionSurface',
    { size: SURFACE } as never,
  )
  const pairs = useRecord<PosteriorPairs>(
    live,
    t,
    panels.active('pairs') ? 'classify/pairs' : undefined,
    'posteriorPairs',
  )
  const diagnostics = useRecord<ChainDiagnostics>(
    live,
    t,
    panels.active('curve') ? 'classify/diagnostics' : undefined,
    'chainDiagnostics',
  )
  const scores = useProbe(live, t, panels.active('numbers') ? 'classify/scores' : undefined).value as
    | Record<string, number | null>
    | null
    | undefined
  const opt = (k: string, digits: number): [string, string][] =>
    scores?.[k] === null || scores?.[k] === undefined ? [] : [[k, fmt(scores[k], digits)]]
  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <FigurePanel panels={panels} name="surface">
          {surface === undefined ? <Waiting done={done} /> : <DecisionSurfaceView record={surface} />}
          <p className="mt-1 text-xs text-muted">
            The boundary is where the predictive probability crosses one half, and the fading colours mark where the
            posterior is unsure; the pair plot shows how the coefficients trade off against each other, which is where
            that uncertainty comes from.
          </p>
        </FigurePanel>
        <FigurePanel panels={panels} name="pairs">
          {pairs === undefined ? <Waiting done={done} /> : <PosteriorPairsView record={pairs} controls={false} />}
        </FigurePanel>
      </div>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <FigurePanel panels={panels} name="curve">
          {diagnostics !== undefined ? (
            <ChainDiagnosticsView record={diagnostics} />
          ) : objective !== undefined ? (
            <Objective live={live} path={objective.path} label={objective.label} t={t} />
          ) : (
            <p className="text-sm text-muted">
              No curve for this method: a sampler shows its chains once it has draws, and EP and Pathfinder end in a few
              steps.
            </p>
          )}
        </FigurePanel>
        <FigurePanel panels={panels} name="numbers">
          <Table
            rows={
              scores === null || scores === undefined
                ? []
                : [
                    ['test accuracy', fmt(scores['testAccuracy'])],
                    ['held-out log predictive', fmt(scores['testLogPredictive'])],
                    ['train accuracy', fmt(scores['trainAccuracy'])],
                    ['draws used', String(scores['draws'] ?? '-')],
                    ...opt('maxRhat', 3).map(([, v]): [string, string] => ['max R-hat', v]),
                    ...opt('minEss', 0).map(([, v]): [string, string] => ['min ESS', v]),
                    ...opt('divergences', 0).map(([, v]): [string, string] => ['divergences', v]),
                  ]
            }
          />
        </FigurePanel>
      </div>
    </div>
  )
}
