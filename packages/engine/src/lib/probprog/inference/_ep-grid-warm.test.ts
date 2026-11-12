// The warm starts of the EP grid: only finite messages start a node, whatever a serialised state made of an infinite entry.
import { describe, expect, it } from 'vitest'
import type { CompiledGraph } from '../../message-passing/index.ts'
import { nearestSettled, settledMessages, type WarmNode, warmStart } from './_ep-grid-warm.ts'

/** The structure `warmStart` reads of a graph: one Gaussian variable (two numbers a message), one factor with one port on its edge. */
const graph = {
  edges: [{ variable: 0 }],
  variables: [{ family: { flat: [0, 0] } }],
  factors: [{ ports: [{ edge: 0 }] }],
} as unknown as CompiledGraph

describe('warmStart', () => {
  it('starts from the messages of a neighbour when they are finite and fit the graph', () => {
    expect(warmStart(graph, [[1, 2]])).toEqual([{ factor: 0, port: 0, eta: [1, 2] }])
  })

  it('starts cold from a message with a null (an infinite entry that went through JSON), not from a NaN', () => {
    const state = JSON.parse(JSON.stringify({ messages: [[1, Number.POSITIVE_INFINITY]] })) as {
      messages: (number | null)[][]
    }
    expect(state.messages[0]).toEqual([1, null])
    expect(warmStart(graph, state.messages as unknown as number[][])).toEqual([])
    expect(warmStart(graph, [[Number.NaN, 1]])).toEqual([])
    expect(warmStart(graph, [[1, Number.NEGATIVE_INFINITY]])).toEqual([])
  })

  it('starts cold from messages of another structure or none', () => {
    expect(warmStart(graph, undefined)).toEqual([])
    expect(warmStart(graph, [[1, 2, 3]])).toEqual([])
    expect(warmStart(graph, [[1, 2], [3]])).toEqual([])
  })
})

describe('what a node keeps and what the nearest settled node is', () => {
  it('keeps the messages of a settled node with a finite evidence and finite messages only', () => {
    expect(settledMessages(true, -3, [[1, 2]])).toEqual({ messages: [[1, 2]] })
    expect(settledMessages(false, -3, [[1, 2]])).toEqual({})
    expect(settledMessages(true, Number.NEGATIVE_INFINITY, [[1, 2]])).toEqual({})
    expect(settledMessages(true, undefined, [[1, 2]])).toEqual({})
    expect(settledMessages(true, -3, [[1, Number.POSITIVE_INFINITY]])).toEqual({})
  })

  it('skips a node whose messages are not finite (the null of a serialised state), taking the next nearest', () => {
    const node = (i: number, messages: readonly (readonly number[])[] | undefined): WarmNode => ({
      index: [i],
      converged: true,
      logEvidence: -1,
      ...(messages === undefined ? {} : { messages }),
    })
    const broken = JSON.parse('[[1, null]]') as number[][]
    const nodes = [node(5, [[1, 1]]), node(9, broken), node(10, undefined)]
    expect(nearestSettled(nodes, [10])?.index).toEqual([5])
    expect(nearestSettled([node(9, broken)], [10])).toBeUndefined()
  })
})
