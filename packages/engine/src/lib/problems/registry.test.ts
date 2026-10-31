import { describe, expect, it } from 'vitest'
import { type Cnf, cdcl, satisfies } from '../logic/index.ts'
import { key } from '../random/index.ts'
import { isHamiltonianCycle, type TspInstance, tourLength } from './index.ts'
import { findProblem, problemRegistry, whereProblems } from './registry.ts'

describe('the problem registry', () => {
  it('makes an instance of every entry at its default size, with a truth that checks out', () => {
    for (const e of problemRegistry.entries) {
      const small = e.make({
        size: e.name === 'plantedKSat' || e.name === 'randomKSat' ? 12 : e.defaultSize,
        key: key(1),
      })
      expect(small.value, e.name).toBeDefined()
      expect(Object.keys(small.truth).length, e.name).toBeGreaterThan(0)
    }
    const tour = findProblem(problemRegistry, 'randomTour')?.make({ size: 8, key: key(2) })
    const t = tour?.value as TspInstance
    expect(tourLength(t.distances, tour?.truth['optimalTour'] as number[])).toBeCloseTo(
      tour?.truth['optimalLength'] as number,
      12,
    )
    const planted = findProblem(problemRegistry, 'plantedKSat')?.make({ size: 25, key: key(3) })
    expect(satisfies(planted?.value as Cnf, planted?.truth['assignment'] as number[])).toBe(true)
    const pigeons = findProblem(problemRegistry, 'pigeonhole')?.make({ size: 4 })
    expect(cdcl(pigeons?.value as Cnf).status).toBe('unsat')
    const ham = findProblem(problemRegistry, 'plantedHamiltonian')?.make({ size: 9, key: key(4) })
    expect(
      isHamiltonianCycle(
        (ham as { value: { adjacency: number[][] } }).value.adjacency,
        ham?.truth['cycle'] as number[],
      ),
    ).toBe(true)
    expect(findProblem(problemRegistry, 'towersOfHanoi')?.make({ size: 3 }).truth).toEqual({ optimalMoves: 7 })
  })

  it('answers queries by kind, solvability and text', () => {
    expect(whereProblems(problemRegistry, { kind: 'tour' })).toHaveLength(4)
    expect(whereProblems(problemRegistry, { kind: 'satisfiability', solvable: 'never' }).map((e) => e.name)).toEqual([
      'pigeonhole',
    ])
    expect(whereProblems(problemRegistry, { text: 'queens' }).map((e) => e.name)).toEqual(['nQueens'])
    expect(whereProblems(problemRegistry).length).toBe(problemRegistry.entries.length)
  })
})
