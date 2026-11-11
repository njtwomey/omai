import { describe, expect, it } from 'vitest'
import { key } from '../../random/index.ts'
import { eightSchoolsDefinition, generateSchoolGroups } from './schools.ts'

describe('generateSchoolGroups', () => {
  it('makes any number of groups with standard errors that differ, and carries its truth', () => {
    const g = generateSchoolGroups(key(4), { groups: 15, mu: 3, tau: 2, se: 8, seSpread: 0.5 })
    const sigma = g.data['sigma'] as number[]
    expect(sigma).toHaveLength(15)
    expect(Math.max(...sigma)).toBeGreaterThan(Math.min(...sigma))
    expect((g.generating['theta'] as number[]).length).toBe(15)
  })

  it('takes the standard errors as given and no spread as the same one', () => {
    expect(generateSchoolGroups(key(1), { groups: 3, mu: 0, tau: 1, standardErrors: [1, 2, 3] }).data['sigma']).toEqual([
      1, 2, 3,
    ])
    expect(generateSchoolGroups(key(1), { groups: 4, mu: 0, tau: 1, se: 7, seSpread: 0 }).data['sigma']).toEqual([
      7, 7, 7, 7,
    ])
  })
})

describe('eightSchoolsDefinition', () => {
  const y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
  const sigma = y.map(() => 5)
  for (const form of ['centred', 'non-centred', 'complete-pooling', 'no-pooling'] as const) {
    it(`${form}: binds to eleven groups`, () => {
      expect(eightSchoolsDefinition(form).bind({ y, sigma }).sizes).toEqual({ school: 11 })
    })
  }
})
