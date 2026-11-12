import { renderToString } from 'react-dom/server'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { StoryContext } from '../../kit/StoryPage.tsx'
import { storyOf } from '../../registry.ts'

// Rendering the page must not start a worker or show a result: only the play button does.
const made: unknown[] = []

beforeEach(() => {
  made.length = 0
  vi.stubGlobal(
    'Worker',
    class {
      constructor(...args: unknown[]) {
        made.push(args)
      }
    },
  )
})
afterEach(() => vi.unstubAllGlobals())

describe('the which-button-wins page', () => {
  it('renders the question, the raw tally and an idle play button, with no worker and no result', async () => {
    const { default: Story } = await import('./Story.tsx')
    const entry = storyOf('which-button-wins')
    if (entry === undefined) throw new Error('the story is not registered')
    const html = renderToString(
      <StoryContext.Provider value={{ entry, prev: undefined, next: undefined }}>
        <Story />
      </StoryContext.Provider>,
    )
    expect(html).toContain('data-phase="idle"')
    expect(html).toContain('Run the test')
    expect(html).toContain('visitors clicked')
    expect(html).toContain('The result will appear here')
    expect(html.indexOf('data-testid="results"')).toBeGreaterThan(html.indexOf('data-testid="play"'))
    expect(html).toContain('data-state="empty"')
    expect(html).not.toContain('The two rates, and the gap between them')
    expect(made).toEqual([])
  }, 60_000)
})
