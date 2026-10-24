import { describe, expect, it } from 'vitest'
import { expectMatches, loadFixture } from '../../../testing/golden.ts'
import * as P from '../index.ts'

interface DiagramJson {
  readonly chance: P.BayesNode[]
  readonly decisions: P.DecisionNode[]
  readonly utilities: P.UtilityNode[]
}
const diagramOf = (p: Readonly<Record<string, unknown>>): P.InfluenceDiagram => {
  const d = p['diagram'] as DiagramJson
  return P.influenceDiagram(d.chance, d.decisions, d.utilities)
}

describe('influence diagrams against pyAgrum', () => {
  for (const c of loadFixture('pgm/models/influence').cases) {
    it(c.name, () => {
      const id = diagramOf(c.params)
      if (c.fn === 'meu') {
        const r = P.solveInfluenceDiagram(id)
        expectMatches(r.meu, c.output, 1e-10, 1e-12, c.name)
        // No other deterministic policy does better (checked exhaustively for the first decision's options).
        expect(P.expectedUtility(id, r.policy)).toBeCloseTo(r.meu, 12)
        return
      }
      const v = P.valueOfInformation(id, c.params['decision'] as string, c.params['variables'] as string[])
      expectMatches(v, c.output, 1e-10, 1e-12, c.name)
    })
  }
})

describe('influence diagrams', () => {
  const oil = diagramOf(loadFixture('pgm/models/influence').cases[0]?.params ?? {})

  it('solve the oil wildcatter: test, then drill unless the seismic result is diffuse', () => {
    const r = P.solveInfluenceDiagram(oil)
    expect(r.policy['test']).toEqual([1])
    const rules = r.rules['drill'] ?? []
    const choose = (seismic: string): string | undefined =>
      rules.find((x) => x.when['test'] === 'yes' && x.when['seismic'] === seismic)?.choose
    expect([choose('closed'), choose('open'), choose('diffuse')]).toEqual(['yes', 'yes', 'no'])
    expect(P.informationSet(oil, 'drill')).toEqual(['test', 'seismic'])
    // Every deterministic policy has at most the MEU.
    for (const t of [0, 1]) {
      for (let mask = 0; mask < 16; mask++) {
        const drill = Array.from({ length: 8 }, (_, i) => (mask >> (i % 4)) & 1)
        expect(P.expectedUtility(oil, { test: [t], drill })).toBeLessThanOrEqual(r.meu + 1e-9)
      }
    }
  })

  it('validate structure and policies', () => {
    expect(() =>
      P.influenceDiagram(
        [{ name: 'x', states: ['a'], parents: ['d'], cpt: [1] }],
        [{ name: 'd', states: ['a', 'b'], parents: ['x'] }],
        [],
      ),
    ).toThrow(/cycle/)
    expect(() =>
      P.influenceDiagram(
        [],
        [{ name: 'd', states: ['a'], parents: [] }],
        [{ name: 'u', parents: ['d'], values: [1, 2] }],
      ),
    ).toThrow(/needs 1 values/)
    expect(() => P.expectedUtility(oil, { test: [0, 1] })).toThrow(/needs 1 entries/)
  })
})

describe('plate models', () => {
  const model = P.plateModel([
    { name: 'difficulty', states: ['easy', 'hard'], plates: ['course'], parents: [], cpt: [0.6, 0.4] },
    { name: 'intelligence', states: ['low', 'high'], plates: ['student'], parents: [], cpt: [0.7, 0.3] },
    {
      name: 'grade',
      states: ['A', 'B', 'C'],
      plates: ['student', 'course'],
      parents: ['intelligence', 'difficulty'],
      cpt: [0.3, 0.4, 0.3, 0.05, 0.25, 0.7, 0.9, 0.08, 0.02, 0.5, 0.3, 0.2],
    },
  ])
  const elements = { student: ['ann', 'bo'], course: ['cs1', 'cs2'] }

  it('ground to a Bayes network that shares CPTs, so evidence on one course informs another', () => {
    const bn = P.groundPlateModel(model, elements)
    expect(bn.nodes).toHaveLength(8)
    expect(P.bayesNode(bn, 'grade[bo,cs2]').parents).toEqual(['intelligence[bo]', 'difficulty[cs2]'])
    const evidence = P.groundEvidence(model, [
      { variable: 'grade', index: { student: 'ann', course: 'cs1' }, state: 'A' },
      { variable: 'grade', index: { student: 'bo', course: 'cs1' }, state: 'C' },
    ])
    const prior = P.factorValues(P.variableElimination(bn, ['intelligence[ann]']).posterior)[1] ?? 0
    const post = P.factorValues(P.variableElimination(bn, ['intelligence[ann]'], { evidence }).posterior)[1] ?? 0
    expect(post).toBeGreaterThan(prior)
    // A relation restricts which (student, course) pairs exist.
    const sparse = P.groundPlateModel(model, elements, { relations: { grade: [['ann', 'cs1']] } })
    expect(sparse.nodes.map((n) => n.name)).toContain('grade[ann,cs1]')
    expect(sparse.nodes).toHaveLength(5)
  })

  it('learn shared CPTs by pooling the counts of every instance', () => {
    const obs: P.PlateObservation[] = []
    const grades = [
      ['ann', 'cs1', 'high', 'easy', 'A'],
      ['ann', 'cs2', 'high', 'hard', 'A'],
      ['bo', 'cs1', 'low', 'easy', 'B'],
      ['bo', 'cs2', 'low', 'hard', 'C'],
    ] as const
    for (const [s, c, i, d, g] of grades) {
      obs.push({ variable: 'grade', index: { student: s, course: c }, state: g })
      obs.push({ variable: 'intelligence', index: { student: s }, state: i })
      obs.push({ variable: 'difficulty', index: { course: c }, state: d })
    }
    const fitted = P.fitPlateModel(model, elements, obs)
    expect(fitted.variables.find((v) => v.name === 'intelligence')?.cpt).toEqual([0.5, 0.5])
    // P(grade | low, easy) from bo's cs1 grade alone: B.
    expect(fitted.variables.find((v) => v.name === 'grade')?.cpt.slice(0, 3)).toEqual([0, 1, 0])
    expect(() =>
      P.plateModel([
        { name: 'x', states: ['a'], plates: ['p'], parents: [], cpt: [1] },
        { name: 'y', states: ['a'], plates: [], parents: ['x'], cpt: [1] },
      ]),
    ).toThrow(/plate p/)
  })
})
