import { Choice, ControlPanel, ElboRunView } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import { CurveChart } from '../../shared/CurveChart.tsx'
import { RunBar } from '../../shared/RunBar.tsx'
import { useAnytimeRuns } from '../../shared/useAnytimeRuns.ts'
import { GUIDES, type GuideId, guideRuns, targetDraws } from './compute.ts'

const TARGET = targetDraws(500)
const TARGET_DOTS = {
  x: Array.from(TARGET.x0),
  y: Array.from(TARGET.x1),
  name: 'exact draws',
  colour: 'muted',
  size: 4,
  opacity: 0.5,
}

export default function SviGuides(): ReactNode {
  const [budget, setBudget] = useState(15)
  const [guide, setGuide] = useState<GuideId>('full-rank')
  const specs = useMemo(() => guideRuns(), [])
  const runs = useAnytimeRuns(specs, budget * 1000)
  const chosen = runs.runs.find((r) => r.id === guide)?.value
  const dots = useMemo(
    () => [
      TARGET_DOTS,
      ...(chosen === undefined || chosen.x0.length === 0
        ? []
        : [
            {
              x: chosen.x0,
              y: chosen.x1,
              name: GUIDES.find((g) => g.id === guide)?.label ?? guide,
              colour: 'tab1',
              size: 4,
              opacity: 0.7,
            },
          ]),
    ],
    [chosen, guide],
  )
  // The smoothed ELBO of every guide over the steps it has taken, on one chart (the first twentieth of each, where the
  // bound climbs from far below, is left out so the part that tells them apart can be read).
  const race = useMemo(
    () =>
      GUIDES.flatMap((g, i) => {
        const record = runs.runs.find((r) => r.id === g.id)?.value?.record
        const e = record?.smoothed ?? record?.elbo
        if (e === undefined) return []
        const from = Math.floor(e.steps.length / 20)
        return [{ name: g.label, colour: `tab${i}`, width: 2, x: e.steps.slice(from), y: e.values.slice(from) }]
      }),
    [runs.runs],
  )
  const raceX = useMemo(() => ({ label: 'step' }), [])
  const raceY = useMemo(() => ({ label: 'ELBO' }), [])
  return (
    <div className="flex flex-col gap-4">
      <RunBar runs={runs} budget={budget} onBudget={setBudget} budgets={[5, 15, 45]} unit="steps" />
      <div className="grid items-start gap-4 lg:grid-cols-2">
        <div className="flex min-w-0 flex-col gap-2">
          <ControlPanel columns={1}>
            <Choice
              label="guide shown"
              value={guide}
              options={GUIDES.map((g) => ({ value: g.id, label: g.label }))}
              onChange={setGuide}
            />
          </ControlPanel>
          <CurveChart
            curves={[]}
            dots={dots}
            x={{ label: 'x0', limits: [-9, 9] }}
            y={{ label: 'x1', limits: [-6, 20] }}
            aspect={1.3}
          />
        </div>
        <div className="flex min-w-0 flex-col gap-2">
          <h2 className="text-sm font-medium text-fg">The ELBO of each guide (smoothed)</h2>
          <CurveChart curves={race} x={raceX} y={raceY} aspect={1.7} />
        </div>
      </div>
      <details className="rounded-lg border border-grid bg-surface">
        <summary className="cursor-pointer select-none px-4 py-2.5 text-sm font-medium text-fg">
          The run of the guide shown: ELBO, beliefs, final posterior against the truth
        </summary>
        <div className="border-t border-line p-4">
          {chosen === undefined ? (
            <p className="text-sm text-muted">Nothing yet.</p>
          ) : (
            <ElboRunView record={chosen.record} />
          )}
        </div>
      </details>
    </div>
  )
}
