import { renderToString } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'
import { StoryContext } from '../../kit/StoryPage.tsx'
import { storyOf } from '../../registry.ts'
import type { Size } from './compute.ts'

// The page after a run, from a small real result: the figures are not drawn, the prose is.
const SMALL: Size = { grid: 20, cuts: 12, chains: 2, iterations: 600, warmup: 100, thin: 10, curves: 6 }

vi.mock('./Figures.tsx', () => ({
  AverageFigure: () => null,
  CutsFigure: () => null,
  DataFigure: () => null,
  GreedyPanels: () => null,
}))

vi.mock('../../kit/useStoryRun.ts', async () => {
  const { fit } = await import('./compute.ts')
  const { initialRun } = await import('../../kit/runState.ts')
  const result = fit(1, undefined, SMALL)
  return { useStoryRun: () => ({ ...initialRun(1), phase: 'done', result, start: () => {}, again: () => {} }) }
})

/** The page's text with the markup taken out. */
async function pageText(): Promise<string> {
  const { default: Story } = await import('./Story.tsx')
  const entry = storyOf('trees-one-split-at-a-time')
  if (entry === undefined) throw new Error('the story is not registered')
  const html = renderToString(
    <StoryContext.Provider value={{ entry, prev: undefined, next: undefined }}>
      <Story />
    </StoryContext.Provider>,
  )
  return html
    .replace(/<!-- -->/g, '')
    .replace(/<[^>]*>/g, ' ')
    .replace(/\s+/g, ' ')
}

describe('what the trees page says about the trees it weighs', () => {
  it('says the trees are up to two levels deep, with up to three cuts, and not that they have two cuts', async () => {
    const text = await pageText()
    expect(text).toContain('every tree up to two levels deep')
    expect(text).toContain('up to three cuts')
    expect(text).not.toContain('tree of up to two cuts')
    expect(text).not.toContain('at most two of')
  }, 60_000)

  it('gives the shares of no cut, one, two and three cuts, which add to one hundred percent', async () => {
    const text = await pageText()
    const said =
      /gives (\d+)% to the tree with no cut, (\d+)% to trees with one cut, (\d+)% to two cuts and (\d+)% to three/.exec(
        text,
      )
    expect(said).not.toBeNull()
    const total = (said ?? []).slice(1).reduce((a, s) => a + Number(s), 0)
    // each share is rounded to a whole percent, so four of them can be two off
    expect(Math.abs(total - 100)).toBeLessThanOrEqual(2)
  }, 60_000)

  it('quotes a chance of a cut near the greedy cut that is a probability (at most 100%)', async () => {
    const text = await pageText()
    const said = /A cut lies within 0\.3 of .{1,300}? with a chance of (\d+)%/.exec(text)
    expect(said).not.toBeNull()
    expect(Number(said?.[1])).toBeLessThanOrEqual(100)
  }, 60_000)
})
