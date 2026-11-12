import { type ChainDiagnostics, chainDiagnostics } from 'omai-engine/exposition'
import { posteriorChains } from 'omai-engine/probprog'
import { nutsMethod, temperingMethod } from 'omai-engine/probprog/inference'
import { bimodal } from 'omai-engine/probprog/models'
import { ess, rhat } from 'omai-engine/sampling'
import { Log } from 'omai-engine/trace'
import type { AnytimeRunSpec } from '../../../shared/useAnytimeRuns.ts'

const BOUND = bimodal.bound()
const CHAINS = 4

/** What a run keeps at a snapshot. */
export interface Kept {
  readonly record: ChainDiagnostics | undefined
  /** The fraction of each chain's draws in the upper mode (x[0] above 0); the truth is one half. */
  readonly upper: readonly number[]
  readonly draws: number
  /** Parallel tempering only: the swap rate between each pair of neighbouring temperatures (all chains), and the round trips so far. */
  readonly swaps?: readonly number[]
  readonly roundTrips?: number
}

function keep(chains: ReturnType<typeof posteriorChains>, extra: Pick<Kept, 'swaps' | 'roundTrips'>): Kept {
  const x0 = chains.draws['x[0]']
  const draws = x0?.[0]?.length ?? 0
  if (x0 === undefined || draws < 8) return { record: undefined, upper: [], draws, ...extra }
  const quantities = { 'x[0]': x0, 'x[1]': chains.draws['x[1]'] ?? x0 }
  const summary = Object.entries(quantities).map(([name, rows]) => ({
    site: name,
    name,
    rhat: rhat(rows),
    ess: ess(rows),
  }))
  return {
    record: chainDiagnostics(quantities, { summary, maxDraws: 600 }),
    upper: x0.map((row) => row.filter((v) => v > 0).length / row.length),
    draws,
    ...extra,
  }
}

/** NUTS and parallel tempering over a random walk (six temperatures down to a hundredth), four chains each. */
export function modeRuns(): AnytimeRunSpec<Kept>[] {
  const nuts = nutsMethod(BOUND)
  const pt = temperingMethod(BOUND)
  return [
    {
      id: 'nuts',
      label: 'NUTS',
      method: nuts,
      config: { numChains: CHAINS, numWarmup: 200, numDraws: 1000, inferDiscrete: false },
      derive: (s, x) => {
        const p = nuts.posterior?.(x.state as never, s.config as never)
        return keep(p === undefined ? { draws: {} } : posteriorChains(p, { sites: ['x'] }), {})
      },
    },
    {
      id: 'tempering',
      label: 'parallel tempering',
      method: pt,
      config: {
        kernel: 'rwm',
        numChains: CHAINS,
        numWarmup: 300,
        numDraws: 2000,
        numTemperatures: 6,
        minBeta: 0.02,
        inferDiscrete: false,
      },
      logs: { 'pt/swapRate': Log({ detail: 'full', keep: 'last' }), 'pt/roundTrips': Log({ keep: 'last' }) },
      derive: (s, x) => {
        const p = pt.posterior?.(x.state as never, s.config as never)
        const events = s.run().events as readonly {
          readonly name?: string
          readonly values?: readonly number[]
          readonly value?: number
        }[]
        const rates = events.filter((e) => e.name === 'pt/swapRate').at(-1)?.values
        const trips = events.filter((e) => e.name === 'pt/roundTrips').at(-1)?.value
        // The values are chains x pairs; the rate of a pair is the mean over the chains.
        const pairs = rates === undefined ? 0 : Math.floor(rates.length / CHAINS)
        const swaps =
          rates === undefined
            ? undefined
            : Array.from({ length: pairs }, (_, j) => {
                let sum = 0
                for (let c = 0; c < CHAINS; c++) sum += rates[c * pairs + j] ?? 0
                return sum / CHAINS
              })
        return keep(p === undefined ? { draws: {} } : posteriorChains(p, { sites: ['x'] }), {
          ...(swaps === undefined ? {} : { swaps }),
          ...(trips === undefined ? {} : { roundTrips: trips }),
        })
      },
    },
  ]
}
