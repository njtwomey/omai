import type { ChainDiagnostics, GroupEstimates, PosteriorPairs, PosteriorRows } from 'omai-engine/bayesian-examples'
import { ChainDiagnosticsView, FigurePanel, PosteriorPairsView, useProbe } from 'omai-render/react'
import type { ReactNode } from 'react'
import { type ExampleView, fmt, Objective, ReferenceCheck, Table, useRecord, Waiting } from './ExamplePage.tsx'
import { GroupEstimatesView } from './GroupEstimatesView.tsx'

/** The panels of a hierarchical example, by name with their titles (the page adds nothing to them but its own wording). */
export const hierarchicalPanels = (estimates: string): Readonly<Record<string, string>> => ({
  estimates,
  pairs: 'The posterior of the hyperparameters, with the divergences',
  diagnostics: 'Chains and convergence',
  numbers: 'The numbers',
  check: 'How we know it works',
})

const text = (v: unknown): string => (typeof v === 'string' ? v : '-')

/**
 * What to observe on a hierarchical example: the group estimates against the raw ones (the shrinkage), the pair plot of
 * the hyperparameters with the sampler's divergent transitions marked, the chains (or the objective of a variational method)
 * and a table of numbers. Each panel asks the session for its probe only while it is on show.
 */
export function HierarchicalObserve(props: {
  readonly view: ExampleView
  /** The probe paths' prefix (`schools`, `radon`). */
  readonly prefix: string
  readonly logX?: boolean
  /** What the estimates figure shows, and what to look for. */
  readonly estimatesNote: ReactNode
  readonly pairsNote: ReactNode
}): ReactNode {
  const { view, prefix, logX, estimatesNote, pairsNote } = props
  const { live, t, panels, done, objective } = view
  const on = (name: string, path: string): string | undefined => (panels.active(name) ? `${prefix}/${path}` : undefined)
  const estimates = useRecord<GroupEstimates>(live, t, on('estimates', 'estimates'), 'groupEstimates')
  const pairs = useRecord<PosteriorPairs>(live, t, on('pairs', 'pairs'), 'posteriorPairs')
  const diagnostics = useRecord<ChainDiagnostics>(live, t, on('diagnostics', 'diagnostics'), 'chainDiagnostics')
  const summary = useProbe(live, t, on('numbers', 'summary')).value as PosteriorRows | null | undefined
  const shrink = useProbe(live, t, on('numbers', 'estimates')).value as GroupEstimates | null | undefined
  const rows: [string, string][] = []
  if (summary !== null && summary !== undefined) {
    rows.push(['draws used', String(summary.draws)])
    if (summary.divergences !== null) rows.push(['divergent transitions', String(summary.divergences)])
    if (summary.maxRhat !== null) rows.push(['largest R-hat', fmt(summary.maxRhat)])
    if (summary.minEss !== null) rows.push(['smallest effective sample size', fmt(summary.minEss, 0)])
    for (const name of ['mu', 'tau', 'sigma', 'beta', 'rho']) {
      const r = summary.rows.find((x) => x.name === name)
      if (r !== undefined) rows.push([`${name} (mean, sd)`, `${fmt(r.mean)}, ${fmt(r.sd)}`])
    }
  }
  if (shrink !== null && shrink !== undefined && Number.isFinite(shrink.meanShrinkage)) {
    rows.push(['mean shrinkage towards the population mean', fmt(shrink.meanShrinkage, 2)])
    rows.push(['shrinkage of the third with the least data', fmt(shrink.shrinkageOfSmallest, 2)])
    rows.push(['shrinkage of the third with the most data', fmt(shrink.shrinkageOfLargest, 2)])
  }
  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
      <FigurePanel panels={panels} name="estimates">
        {estimates === undefined ? (
          <Waiting done={done} />
        ) : (
          <GroupEstimatesView record={estimates} {...(logX === undefined ? {} : { logX })} />
        )}
        <p className="mt-1 text-xs text-muted">{estimatesNote}</p>
      </FigurePanel>
      <FigurePanel panels={panels} name="pairs">
        {pairs === undefined ? <Waiting done={done} /> : <PosteriorPairsView record={pairs} controls={false} />}
        <p className="mt-1 text-xs text-muted">{pairsNote}</p>
      </FigurePanel>
      <FigurePanel panels={panels} name="diagnostics">
        {diagnostics !== undefined ? (
          <ChainDiagnosticsView record={diagnostics} />
        ) : objective !== undefined ? (
          <Objective live={live} path={objective.path} label={objective.label} t={t} />
        ) : (
          <p className="text-sm text-muted">
            No curve for this method: a sampler shows its chains once it has draws, and the other methods have no
            iterations to show.
          </p>
        )}
      </FigurePanel>
      <FigurePanel panels={panels} name="numbers">
        <Table rows={rows} />
      </FigurePanel>
    </div>
  )
}

/** How we know it works on a hierarchical example: the chosen method's posterior against the long reference run. */
export function HierarchicalCheck(props: {
  readonly view: ExampleView
  readonly prefix: string
  readonly explain: ReactNode
}): ReactNode {
  const { view, prefix, explain } = props
  return (
    <ReferenceCheck
      view={view}
      probes={{ summary: { path: `${prefix}/summary` } }}
      checkPath={`${prefix}/check`}
      explain={explain}
      rows={(c) => {
        const x = c as unknown as Record<string, number | string | null>
        const z = typeof x['maxZ'] === 'number' ? `${fmt(x['maxZ'], 1)} (${text(x['worstZ'])})` : 'not defined (no chains)'
        const out: [string, string][] = [
          ['coordinates compared', String(x['compared'])],
          ['largest difference of means, in reference sds', `${fmt(x['maxMeanError'])} (${text(x['worst'])})`],
          ['largest difference, in Monte Carlo errors', z],
          ['median ratio of spreads (method / reference)', fmt(x['medianSdRatio'])],
        ]
        for (const name of ['mu', 'tau', 'sigma', 'beta']) {
          if (typeof x[`${name}Error`] === 'number')
            out.push([`${name}: mean difference in reference sds, spread ratio`, `${fmt(x[`${name}Error`])}, ${fmt(x[`${name}Ratio`])}`])
        }
        out.push(['draws in the reference', String(x['referenceDraws'])])
        return out
      }}
    />
  )
}
