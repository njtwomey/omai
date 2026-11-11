import { describe, expect, it } from 'vitest'
import { key } from '../../random/index.ts'
import { generateIrt, hierarchicalIrtDefinition } from './irt-hierarchical.ts'

describe('generateIrt', () => {
  it('makes a prefix of one table: the first students and questions are the same whatever the size', () => {
    const small = generateIrt(key(3), { students: 5, questions: 3 })
    const large = generateIrt(key(3), { students: 30, questions: 9 })
    expect(small.theta).toEqual(large.theta.slice(0, 5))
    expect(small.b).toEqual(large.b.slice(0, 3))
    expect(small.y).toEqual(large.y.slice(0, 5).map((r) => r.slice(0, 3)))
  })

  it('answers more often right the more able the student', () => {
    const g = generateIrt(key(1), { students: 200, questions: 30, abilitySd: 1.5 })
    const score = g.y.map((r) => r.reduce((a, b) => a + b, 0))
    const order = g.theta.map((_, i) => i).sort((a, b) => (g.theta[a] as number) - (g.theta[b] as number))
    const low = order.slice(0, 50).reduce((a, i) => a + (score[i] as number), 0)
    const high = order.slice(-50).reduce((a, i) => a + (score[i] as number), 0)
    expect(high).toBeGreaterThan(low)
  })
})

describe('hierarchicalIrtDefinition', () => {
  it('binds to answers with the students and questions as plates', () => {
    for (const model of ['rasch', '2pl'] as const)
      for (const form of ['centred', 'non-centred'] as const)
        expect(
          hierarchicalIrtDefinition(model, form).bind({
            y: [
              [1, 0, 1],
              [0, 0, 1],
            ],
          }).sizes,
        ).toEqual({
          student: 2,
          question: 3,
        })
  })
})
