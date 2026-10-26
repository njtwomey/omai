import { flocking } from 'mdml-engine/methods/complex-systems'
import { describe, expect, it } from 'vitest'
import { agentFrames } from './agents.ts'
import { viewFor } from './registry.ts'

describe('agentFrames', () => {
  it('plays the flock with its polarisation beneath', () => {
    const [record] = flocking({ count: 30, steps: 20, every: 5 }).exposition
    if (record?.kind !== 'agentSteps') throw new Error('expected agentSteps')
    const frames = agentFrames(record)
    expect(frames.count).toBe(5)
    expect(frames.option(2).series).toBeDefined()
    expect(frames.label(0)).toContain('polarisation')
    expect(viewFor('agentSteps').component).toBe('AgentPlayer')
  })
})
