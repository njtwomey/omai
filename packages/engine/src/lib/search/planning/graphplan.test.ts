import { describe, expect, it } from 'vitest'
import { policies, recorder, validateAgainstSchema } from '../../trace/index.ts'
import {
  applicable,
  applyAction,
  graphplan,
  graphplanAlgorithm,
  htnPlan,
  htnPlanAlgorithm,
  htnTree,
  linearize,
  mutexPairs,
  partialOrderPlan,
  partialOrderProblem,
  regress,
  strips,
  validatePlan,
} from '../index.ts'

const cake = strips(
  ['have-cake'],
  ['have-cake', 'eaten-cake'],
  [
    { name: 'eat', pre: ['have-cake'], add: ['eaten-cake'], del: ['have-cake'] },
    { name: 'bake', pre: [], add: ['have-cake'], del: [] },
  ],
)

/** The Sussman anomaly in a three-operator blocks world. */
const blocks = ['a', 'b', 'c']
const places = [...blocks, 'table']
const moves = blocks.flatMap((x) =>
  places.flatMap((from) =>
    places
      .filter((to) => to !== x && from !== x && to !== from)
      .map((to) => ({
        name: `move(${x},${from},${to})`,
        pre: [`on(${x},${from})`, `clear(${x})`, ...(to === 'table' ? [] : [`clear(${to})`])],
        add: [`on(${x},${to})`, ...(from === 'table' ? [] : [`clear(${from})`])],
        del: [`on(${x},${from})`, ...(to === 'table' ? [] : [`clear(${to})`])],
      })),
  ),
)
const sussman = strips(['on(c,a)', 'on(a,table)', 'on(b,table)', 'clear(c)', 'clear(b)'], ['on(a,b)', 'on(b,c)'], moves)

describe('STRIPS basics', () => {
  it('applies, regresses and validates', () => {
    const go = { name: 'go', pre: ['home'], add: ['shop'], del: ['home'] }
    expect(applicable(new Set(['home']), go)).toBe(true)
    expect(applyAction(['home'], go)).toEqual(['shop'])
    expect(regress(['home'], go)).toBeNull()
    expect(() => validatePlan(cake, ['fly'])).toThrow()
    expect(() => strips([], [], [go, go])).toThrow()
    expect(() => strips([], [], [{ ...go, cost: -1 }])).toThrow()
  })
})

describe('GraphPlan', () => {
  it('plans the cake problem in two levels and describes its graph', () => {
    const r = graphplan(cake)
    expect(r.steps).toEqual([['eat'], ['bake']])
    expect(r.levels).toBe(2)
    expect(r.graph[1]?.factMutexes).toContainEqual(['eaten-cake', 'have-cake'])
    expect(r.graph[2]?.factMutexes ?? []).not.toContainEqual(['eaten-cake', 'have-cake'])
    expect(r.graph[0]?.actions).toContain('noop have-cake')
  })

  it('solves the Sussman anomaly and detects unsolvable problems', () => {
    const r = graphplan(sussman)
    expect(r.status).toBe('solved')
    expect(validatePlan(sussman, r.plan).valid).toBe(true)
    expect(r.steps.length).toBe(3)
    const stuck = strips(['a'], ['c'], [{ name: 'ab', pre: ['a'], add: ['b'], del: ['a'] }])
    expect(graphplan(stuck).status).toBe('failure')
    const exclusive = strips(
      ['l'],
      ['l', 'r'],
      [
        { name: 'flip', pre: ['l'], add: ['r'], del: ['l'] },
        { name: 'flop', pre: ['r'], add: ['l'], del: ['r'] },
      ],
    )
    expect(graphplan(exclusive).status).toBe('failure')
    expect(mutexPairs(exclusive).mutexes).toEqual([['l', 'r']])
    expect(graphplan(strips(['g'], ['g'], [])).steps).toEqual([])
    expect(graphplan(sussman, { maxLevels: 2 }).status).toBe('budget')
    expect(graphplan(sussman, { maxNodes: 3 }).status).toBe('budget')
  })

  it('traces levels', () => {
    const rec = recorder()
    const algorithm = graphplanAlgorithm(sussman)
    graphplan(sussman, { tracer: rec, policy: policies.teaching })
    expect(validateAgainstSchema(rec.events, algorithm.schema ?? (undefined as never))).toEqual([])
  })
})

describe('partial-order planning', () => {
  it('solves the Sussman anomaly with a consistent ordering', () => {
    const r = partialOrderPlan(sussman)
    expect(r.status).toBe('solved')
    expect(r.linearization).toEqual(['move(c,a,table)', 'move(b,table,c)', 'move(a,table,b)'])
    expect(validatePlan(sussman, r.linearization).valid).toBe(true)
    // Every causal link is protected: no step deleting its fact lies between its ends in the linearisation.
    const plan = r.plan
    expect(plan).not.toBeNull()
    const pos = new Map(linearize(plan ?? partialOrderProblem(sussman).initial).map((name, i) => [name, i]))
    for (const link of plan?.links ?? []) {
      const from = link.from === 0 ? -1 : (pos.get(plan?.steps[link.from]?.name ?? '') ?? -1)
      const to = link.to === 1 ? 99 : (pos.get(plan?.steps[link.to]?.name ?? '') ?? 99)
      for (const [name, i] of pos) {
        const step = plan?.steps.find((s) => s.name === name)
        if (i > from && i < to) expect(step?.del).not.toContain(link.fact)
      }
    }
  })

  it('reports failure on an unsolvable problem within its budget', () => {
    const stuck = strips(['a'], ['c'], [{ name: 'ab', pre: ['a'], add: ['b'], del: ['a'] }])
    expect(partialOrderPlan(stuck).status).toBe('failure')
  })
})

describe('HTN planning', () => {
  const domain = {
    actions: [
      { name: 'pick-a', pre: ['hand-free'], add: ['holding-a'], del: ['hand-free'] },
      { name: 'drop-a', pre: ['holding-a'], add: ['hand-free', 'a-in-box'], del: ['holding-a'] },
      { name: 'open-box', pre: ['box-closed'], add: ['box-open'], del: ['box-closed'] },
      { name: 'pack', pre: ['box-open', 'a-in-box'], add: ['packed'], del: [] },
    ],
    methods: [
      // The first method fails (the box is closed when it packs), so the planner must backtrack.
      { name: 'pack-direct', task: 'ship', subtasks: ['put-a', 'pack'] },
      { name: 'pack-opened', task: 'ship', subtasks: ['open-box', 'put-a', 'pack'] },
      { name: 'carry', task: 'put-a', pre: ['hand-free'], subtasks: ['pick-a', 'drop-a'] },
    ],
  }

  it('decomposes with backtracking and describes the decomposition', () => {
    const r = htnPlan(domain, ['hand-free', 'box-closed'], ['ship'])
    expect(r.status).toBe('solved')
    expect(r.plan).toEqual(['open-box', 'pick-a', 'drop-a', 'pack'])
    expect(r.tree[0]).toEqual({ task: 'ship', method: 'pack-opened', parent: -1 })
    const s = htnTree(r)
    expect(s.nodes.length).toBe(r.tree.length)
    expect(s.edges.length).toBe(r.tree.length - 1)
    expect(r.state).toContain('packed')
  })

  it('fails, stops on budgets, rejects unknown tasks and traces', () => {
    expect(htnPlan(domain, ['box-closed'], ['ship']).status).toBe('failure')
    expect(htnPlan(domain, ['hand-free', 'box-closed'], ['ship'], { maxExpansions: 2 }).status).toBe('budget')
    expect(() => htnPlan(domain, [], ['fly'])).toThrow()
    const rec = recorder()
    const algorithm = htnPlanAlgorithm(domain, ['hand-free', 'box-closed'], ['ship'])
    htnPlan(domain, ['hand-free', 'box-closed'], ['ship'], { tracer: rec, policy: policies.teaching })
    expect(validateAgainstSchema(rec.events, algorithm.schema ?? (undefined as never))).toEqual([])
  })
})
