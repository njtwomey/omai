import { planBlocks, randomBlocks, SUSSMAN } from 'whitebox-engine/methods/gofai'
import { choice, dict, int } from 'whitebox-engine/space'
import { defineEntry, type HubEntry } from '../../../entry/entry.ts'

/** STRIPS planning in the blocks world by forward state-space search. */
export const blocksWorldEntry: HubEntry = defineEntry({
  id: 'classical-ai/planning/blocks-world',
  title: 'Blocks-world planning',
  summary: 'Forward search over STRIPS states with a relaxed-plan heuristic, then the plan carried out by the gripper.',
  collection: 'classical-ai/planning',
  tags: ['planning', 'STRIPS', 'heuristic search', 'blocks world'],
  datasets: [],
  params: dict({
    problem: choice(['Sussman anomaly', 'random'], { default: 'random', label: 'problem' }),
    blocks: int(3, 7, { default: 5, label: 'blocks (random)' }),
    search: choice(['a-star', 'greedy', 'weighted-a-star', 'breadth-first'], { default: 'a-star', label: 'search' }),
    heuristic: choice(['h-ff', 'h-add', 'h-max', 'goal-count', 'blind'], { default: 'h-ff', label: 'heuristic' }),
  }),
  run: ({ params, key }) => {
    const { initial, goal } = params['problem'] === 'random' ? randomBlocks(key, Number(params['blocks'])) : SUSSMAN
    return planBlocks(initial, goal, {
      search: params['search'] as 'a-star',
      heuristic: params['heuristic'] as 'h-ff',
    })
  },
  views: [
    { record: 'boardSteps', title: 'The states the search expands', index: 0 },
    { record: 'boardSteps', title: 'The plan, action by action', index: 1, area: 'below' },
  ],
  methods: [
    { file: 'methods/gofai/planning.ts', name: 'planBlocks' },
    { file: 'lib/search/planning/strips.ts', name: 'forwardPlan' },
  ],
  references: [
    'Fikes and Nilsson (1971), STRIPS: a new approach to the application of theorem proving to problem solving.',
    'Hoffmann and Nebel (2001), The FF planning system.',
  ],
})
