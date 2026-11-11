// The EP programs of the second batch (sparse regression, zero-inflated counts, tournament skills): each is simulated from a
// known truth over a few seeds, run by EP and held to its recovery check (truth recovered within the stated tolerances, EP
// against the exact reference where there is one). A failing row names what EP got wrong.
import { describe, expect, it } from 'vitest'
import type { ProgramCheck } from '../shared/ep/program-check.ts'
import { passes } from '../shared/ep/program-check.ts'
import * as skills from './ep-program-skills/compute.ts'
import { checkSparse, defaultSetup, exactSparse, runSparse } from './ep-program-sparse/compute.ts'
import * as zip from './ep-program-zero-inflated/compute.ts'

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
  {
    name: 'counts with excess zeros',
    seeds: [1, 2, 3],
    check: (seed) => {
      const run = zip.runZip({ ...zip.defaultSetup, seed })
      return zip.checkZip(run, zip.exactZip(run))
    },
  },
  {
    name: 'who is better: a tournament with unknown skills',
    seeds: [1, 2, 3],
    check: (seed) => {
      const run = skills.runSkills({ ...skills.defaultSetup, seed })
      return skills.checkSkills(run, skills.referenceSkills(run))
    },
  },
]

describe('the tournament skills, pooled over seeds', () => {
  it('EP\u2019s 90% intervals cover at least 80% of the true skills', () => {
    let inside = 0
    let total = 0
    for (const seed of [1, 2, 3]) {
      const run = skills.runSkills({ ...skills.defaultSetup, seed })
      inside += skills.coverageOf(run.data.skills, run.found) * run.data.skills.length
      total += run.data.skills.length
    }
    expect(inside / total).toBeGreaterThanOrEqual(0.8)
  }, 120_000)
})

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
