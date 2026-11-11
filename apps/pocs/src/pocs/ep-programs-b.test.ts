// The EP programs of the second batch (sparse regression, zero-inflated counts, tournament skills): each is simulated from a
// known truth over a few seeds, run by EP and held to its recovery check (truth recovered within the stated tolerances, EP
// against the exact reference where there is one). A failing row names what EP got wrong.
import { describe, expect, it } from 'vitest'
import type { ProgramCheck } from '../shared/ep/program-check.ts'
import { passes } from '../shared/ep/program-check.ts'
import { checkSparse, defaultSetup, exactSparse, runSparse } from './ep-program-sparse/compute.ts'

interface Program {
  readonly name: string
  readonly seeds: readonly number[]
  readonly check: (seed: number) => ProgramCheck
}

const programs: readonly Program[] = [
  {
    name: 'sparse regression with outliers and unknown noise',
    seeds: [1, 2, 3],
    check: (seed) => {
      const run = runSparse({ ...defaultSetup, seed })
      return checkSparse(run, exactSparse(run))
    },
  },
]

describe.each(programs)('$name', (program) => {
  it.each(program.seeds)(
    'recovers the truth (seed %i)',
    (seed) => {
      const check = program.check(seed)
      const failed = check.rows
        .filter((r) => r.gating !== false && !r.pass)
        .map((r) => `${r.label}: ${r.value} (${r.tolerance})`)
      expect(failed).toEqual([])
      expect(passes(check)).toBe(true)
    },
    120_000,
  )
})
