import type { PosteriorComparison } from 'omai-engine/exposition'
import { cartesian, lines } from 'omai-render/charts'
import { figure } from 'omai-render/figure'
import { Chart, FigurePanel, NumberField, PosteriorComparisonView, useColourMode, useProbe } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import { coinExample, coinHiddenKnobs } from '../../../examples/index.ts'
import { ExamplePage, type ExampleView, fmt, ReferenceCheck, Table, Waiting } from '../../../shared/ExamplePage.tsx'
import { coinProducer } from './method.ts'

const PANELS = {
  updating: 'Posterior after the flips seen',
  contraction: 'Posterior as flips accumulate',
  predictive: 'The numbers and the next flip',
  check: 'How we know it works',
}
const FIG = figure({ width: 520, height: 300 })

interface Updating {
  readonly seen: number
  readonly total: number
  readonly successes: number
  readonly truth: number
  readonly comparison: PosteriorComparison
  readonly exact: { mean: number; sd: number; lo: number; hi: number }
  readonly method: { mean: number; sd: number; lo: number; hi: number }
  readonly predictiveLabel: string
  readonly exactPredictive: number
  readonly methodPredictive: number
}

interface Contraction {
  readonly truth: number
  readonly site: string
  readonly points: readonly { n: number; mean: number; lo: number; hi: number; running: number }[]
}

export default function CoinPage(): ReactNode {
  return (
    <ExamplePage
      id="coin-flipping"
      example={coinExample}
      producer={coinProducer}
      panels={PANELS}
      exclude={coinHiddenKnobs}
      observe={(view) => <Outputs view={view} />}
      check={(view) => (
        <ReferenceCheck
          view={view}
          probes={{}}
          checkPath="coin/check"
          explain={
            <>
              The posterior of a Beta prior and Bernoulli flips (or a Gamma prior and Poisson counts) is known in closed
              form: Beta(a + heads, b + tails), or Gamma(a + total, b + n). There is nothing to run for a reference, so
              the chosen method's draws are compared with that exact distribution: the error of its mean in exact
              standard deviations, the ratio of spreads, and the Kolmogorov, Wasserstein and KL distances.
            </>
          }
          rows={(c) => [
            ['exact mean and sd', `${fmt(c['exactMean'], 4)} and ${fmt(c['exactSd'], 4)}`],
            ['method mean and sd', `${fmt(c['methodMean'], 4)} and ${fmt(c['methodSd'], 4)}`],
            ['mean error (in exact sds)', fmt(c['meanError'])],
            ['spread ratio (method / exact)', fmt(c['sdRatio'])],
            ['Kolmogorov distance', fmt(c['ks'])],
            ['Wasserstein distance (in exact sds)', fmt(c['wasserstein'])],
            ['KL (exact || method)', fmt(c['kl'], 4)],
            ['log evidence (exact)', fmt(c['logEvidence'])],
            ['draws', String(c['numDraws'] ?? '-')],
          ]}
          more={(c) => <PosteriorComparisonView record={c['comparison'] as PosteriorComparison} />}
        />
      )}
    />
  )
}

function Outputs(props: { readonly view: ExampleView }): ReactNode {
  const { live, t, panels, done, config } = props.view
  const total = Math.max(1, Math.round(Number(config['flips'])))
  // The flips seen: all of them until the viewer scrubs back; a change of the data brings it back to all.
  const [asked, setAsked] = useState<number | null>(null)
  const seen = Math.max(1, Math.min(total, asked ?? total))
  const updating = useProbe(
    live,
    t,
    panels.active('updating') || panels.active('predictive') ? 'coin/updating' : undefined,
    {
      seen,
    } as never,
  ).value as Updating | null | undefined
  const contraction = useProbe(live, t, panels.active('contraction') ? 'coin/contraction' : undefined).value as
    | Contraction
    | null
    | undefined
  const counts = config['kind'] === 'counts'
  return (
    <div className="flex flex-col gap-4">
      <FigurePanel panels={panels} name="updating">
        <div className="mb-2 max-w-sm">
          <NumberField
            label={counts ? 'counts seen' : 'flips seen'}
            value={seen}
            onChange={(v) => setAsked(Math.max(1, Math.min(total, Math.round(v))))}
            min={1}
            max={total}
            step={1}
            slider
          />
        </div>
        {updating === undefined || updating === null ? (
          <Waiting done={done} />
        ) : (
          <>
            <PosteriorComparisonView record={updating.comparison} />
            <p className="mt-1 text-xs text-muted">
              After the first {updating.seen} {counts ? 'counts' : 'flips'} ({updating.successes}{' '}
              {counts ? 'events in all' : 'heads'}) the chosen method is run again on just those data and drawn over the
              exact posterior. Move the slider to see the posterior narrow as the data accumulate; the true value is{' '}
              {fmt(updating.truth, 2)}.
            </p>
          </>
        )}
      </FigurePanel>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <FigurePanel panels={panels} name="contraction">
          {contraction === undefined || contraction === null ? (
            <Waiting done={done} />
          ) : (
            <Contract data={contraction} seen={seen} />
          )}
        </FigurePanel>
        <FigurePanel panels={panels} name="predictive">
          {updating === undefined || updating === null ? null : (
            <Table
              rows={[
                [counts ? 'rate: exact mean' : 'p: exact mean', fmt(updating.exact.mean, 4)],
                ['exact 95% interval', `${fmt(updating.exact.lo, 3)} to ${fmt(updating.exact.hi, 3)}`],
                ['method mean', fmt(updating.method.mean, 4)],
                ['method 95% interval', `${fmt(updating.method.lo, 3)} to ${fmt(updating.method.hi, 3)}`],
                [`${updating.predictiveLabel}: exact`, fmt(updating.exactPredictive, 4)],
                [`${updating.predictiveLabel}: method`, fmt(updating.methodPredictive, 4)],
                ['true value', fmt(updating.truth, 3)],
              ]}
            />
          )}
        </FigurePanel>
      </div>
    </div>
  )
}

/** The exact posterior mean and 95% interval after each count, the running proportion and the true value. */
function Contract(props: { readonly data: Contraction; readonly seen: number }): ReactNode {
  const { data, seen } = props
  const mode = useColourMode()
  const option = useMemo(() => {
    const n = data.points.map((p) => p.n)
    const last = n.at(-1) ?? 1
    const lo = data.points.map((p) => p.lo)
    const hi = data.points.map((p) => p.hi)
    const top = Math.max(...hi, data.truth)
    return cartesian(
      [
        lines(
          [
            { x: n, y: hi, name: 'upper 95%', width: 1.5, colour: '#4e79a7' },
            { x: n, y: lo, name: 'lower 95%', width: 1.5, colour: '#4e79a7' },
            { x: n, y: data.points.map((p) => p.mean), name: 'exact posterior mean', width: 2.5, colour: '#f28e2b' },
            {
              x: n,
              y: data.points.map((p) => p.running),
              name: 'proportion so far',
              width: 1,
              dash: 'dotted',
              colour: '#9a9a9a',
            },
            {
              x: [0, last],
              y: [data.truth, data.truth],
              name: 'true value',
              width: 1.5,
              dash: 'dashed',
              colour: '#59a14f',
            },
            { x: [seen, seen], y: [0, top], name: 'seen', width: 1, dash: 'dashed', colour: '#bab0ac' },
          ],
          { mode },
        ),
      ],
      {
        mode,
        figure: FIG,
        x: { label: 'observations seen' },
        y: { label: data.site === 'p' ? 'probability of heads' : 'rate' },
      },
    )
  }, [data, seen, mode])
  return (
    <div>
      <Chart option={option} figure={FIG} />
      <p className="mt-1 text-xs text-muted">
        The exact posterior mean and its 95% interval as the observations accumulate: the interval narrows around the
        true value (dashed) as the data speak louder than the prior.
      </p>
    </div>
  )
}
