import { renderToString } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'
import { StoryContext } from '../../kit/StoryPage.tsx'
import { storyOf } from '../../registry.ts'
import type { Size } from './compute.ts'

// The page after a run, from a small real result, once as the method reports it settled and once as it hit its cap.
const SMALL: Size = { rows: 40, priorDraws: 3, grid: 20 }
const held = vi.hoisted(() => ({ settled: true }))

vi.mock('./Figures.tsx', () => ({ NumbersFigure: () => null, PriorFigure: () => null, WeightsFigure: () => null }))

vi.mock('../../kit/useStoryRun.ts', async () => {
  const { fit } = await import('./compute.ts')
  const { initialRun } = await import('../../kit/runState.ts')
  const result = fit(1, undefined, SMALL)
  return {
    useStoryRun: () => ({
      ...initialRun(1),
      phase: 'done',
      result: { ...result, settled: held.settled },
      start: () => {},
      again: () => {},
    }),
  }
})

/** The page's text with the markup taken out. */
async function pageText(settled: boolean): Promise<string> {
  held.settled = settled
  const { default: Story } = await import('./Story.tsx')
  const entry = storyOf('stick-breaking')
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

describe('what the stick-breaking page says about whether the method settled', () => {
  it('says it settled when the run converged', async () => {
    const text = await pageText(true)
    expect(text).toMatch(/The method settled after \d+ sweeps\./)
    expect(text).not.toContain('did not settle')
  }, 60_000)

  it('says plainly that it stopped at its limit when the run did not converge', async () => {
    const text = await pageText(false)
    expect(text).toMatch(/The method did not settle\. It stopped after \d+ sweeps, the most it was allowed\./)
    expect(text).toContain('where it stopped')
    expect(text).not.toContain('The method settled')
  }, 60_000)

  it('quotes the order check from the data file', async () => {
    const text = await pageText(true)
    expect(text).toMatch(/We tried 14 orders\. In 11 of them the model found exactly 4 pieces/)
    expect(text).toMatch(/In the other 3 it found 5 or 6/)
  }, 60_000)
})
