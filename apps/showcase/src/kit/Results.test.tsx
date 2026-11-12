import { renderToString } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { Results } from './Results.tsx'
import { initialRun } from './runState.ts'
import type { StoryRun } from './useStoryRun.ts'

const noop = (): void => {}
const runOf = (result: number | undefined): StoryRun<number> => ({
  ...initialRun<number>(1),
  phase: result === undefined ? 'idle' : 'done',
  result,
  start: noop,
  again: noop,
})

describe('the results block', () => {
  it('holds one line and builds nothing while there is no result', () => {
    let built = 0
    const html = renderToString(
      <Results run={runOf(undefined)}>
        {() => {
          built += 1
          return <p>the answer</p>
        }}
      </Results>,
    )
    expect(built).toBe(0)
    expect(html).toContain('data-state="empty"')
    expect(html).toContain('Nothing has been computed yet')
    expect(html).not.toContain('the answer')
  })

  it('builds its content from the result once there is one', () => {
    const html = renderToString(<Results run={runOf(42)}>{(n) => <p>the answer is {n}</p>}</Results>)
    expect(html).toContain('data-state="filled"')
    expect(html).toContain('the answer is <!-- -->42')
    expect(html).not.toContain('Nothing has been computed yet')
  })
})
