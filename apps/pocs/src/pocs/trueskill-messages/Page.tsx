import { ControlPanel, ModelRunPlayer, NumberField, SeedInput } from 'omai-render/react'
import { type ReactNode, useMemo, useState } from 'react'
import { CurveChart } from '../../shared/CurveChart.tsx'
import { RunBar } from '../../shared/RunBar.tsx'
import { useAnytimeRuns } from '../../shared/useAnytimeRuns.ts'
import { beliefsAlong, MAX_GAMES, messageRun, nutsRun, tournamentOf } from './compute.ts'

const COLOURS = ['tab0', 'tab1', 'tab2', 'tab3']

export default function TrueskillMessages(): ReactNode {
  const [seed, setSeed] = useState(2)
  const [games, setGames] = useState(8)
  const [budget, setBudget] = useState(10)
  const tournament = useMemo(() => tournamentOf(seed), [seed])
  const record = useMemo(() => messageRun(tournament, games), [tournament, games])
  const along = useMemo(() => beliefsAlong(tournament), [tournament])
  const specs = useMemo(() => nutsRun(tournament, games), [tournament, games])
  const runs = useAnytimeRuns(specs, budget * 1000)
  const nuts = runs.runs[0]?.value
  const curves = useMemo(
    () =>
      tournament.players.flatMap((name, p) => {
        const x = along.map((_, j) => j)
        const colour = COLOURS[p] ?? 'tab4'
        const band = (sign: number) => along.map((row) => (row[p]?.mean ?? 0) + sign * (row[p]?.sd ?? 0))
        return [
          { name: `player ${name}`, colour, width: 2.4, x, y: along.map((row) => row[p]?.mean ?? 0) },
          { colour, width: 1, opacity: 0.45, dash: 'dotted' as const, x, y: band(1) },
          { colour, width: 1, opacity: 0.45, dash: 'dotted' as const, x, y: band(-1) },
          {
            colour,
            width: 1.2,
            dash: 'dashed' as const,
            x: [0, MAX_GAMES],
            y: [tournament.skills[p] ?? 0, tournament.skills[p] ?? 0],
          },
        ]
      }),
    [along, tournament],
  )
  const final = along[games]
  return (
    <div className="flex flex-col gap-4">
      <ControlPanel columns={2}>
        <NumberField label="games played" value={games} onChange={setGames} min={1} max={MAX_GAMES} step={1} integer />
        <SeedInput value={seed} onChange={setSeed} />
      </ControlPanel>
      <div className="grid items-start gap-4 lg:grid-cols-2">
        <div className="flex min-w-0 flex-col gap-1">
          <h2 className="text-sm font-medium text-fg">Messages on the factor graph, game by game</h2>
          <ModelRunPlayer record={record} filename="trueskill-messages" />
        </div>
        <div className="flex min-w-0 flex-col gap-1">
          <h2 className="text-sm font-medium text-fg">Each player’s belief as games are added</h2>
          <CurveChart curves={curves} x={{ label: 'games played', limits: [0, MAX_GAMES] }} y={{ label: 'skill' }} />
          <p className="text-xs text-muted">
            Solid: the mean; dotted: one sd either side; dashed: the skill the games were drawn from.
          </p>
        </div>
      </div>
      <RunBar runs={runs} budget={budget} onBudget={setBudget} unit="iterations" />
      <table className="w-full max-w-xl text-left text-sm tabular-nums">
        <thead className="text-xs text-muted">
          <tr>
            <th className="py-1 font-normal">player</th>
            <th className="font-normal">true skill</th>
            <th className="font-normal">EP after {games} games</th>
            <th className="font-normal">NUTS</th>
          </tr>
        </thead>
        <tbody>
          {tournament.players.map((name, p) => (
            <tr key={name} className="border-t border-line">
              <td className="py-1">{name}</td>
              <td>{tournament.skills[p]}</td>
              <td>
                {final?.[p]?.mean.toFixed(2)} ± {final?.[p]?.sd.toFixed(2)}
              </td>
              <td>
                {nuts === undefined || Number.isNaN(nuts[p]?.mean ?? Number.NaN)
                  ? '…'
                  : `${nuts[p]?.mean.toFixed(2)} ± ${nuts[p]?.sd.toFixed(2)}`}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
