import { describe, expect, it } from 'vitest'
import { z } from 'zod'
import {
  type OutcomeWord,
  RunSchema,
  type RunStatus,
  runStatusOf,
  type TensorRole,
  type TraceEvent,
  TraceEventSchema,
} from './events.ts'

// One event of every kind with every field present (optional ones too). `Required<E>` makes the compiler demand each
// field of each interface, so adding one to an interface fails to compile here until it is added to its sample; the
// parse below then fails until the schema knows it (a zod object drops what it does not know), and the key check fails
// for a field the schema has and the interface lacks. The interfaces and the schema cannot drift apart unseen.
const samples: { readonly [E in TraceEvent as E['type']]: Required<E> } = {
  run: {
    type: 'run',
    runId: 'r',
    name: 'n',
    seed: 1,
    version: 'v',
    config: { a: 1 },
    parent: 'p',
    channels: { c: [] },
  },
  metric: { type: 'metric', runId: 'r', step: 1, name: 'm', value: 2 },
  tensor: {
    type: 'tensor',
    runId: 'r',
    step: 1,
    name: 't',
    role: 'grad',
    shape: [2],
    values: [1, 2],
    indices: [0, 1],
    summary: { n: 2, mean: 1, std: 1, min: 0, max: 2, norm: 2, zeros: 0, histogram: { edges: [0, 1], counts: [2] } },
  },
  message: { type: 'message', runId: 'r', step: 1, kind: 'k', from: 'a', to: 'b', payload: { x: 1 } },
  episode: { type: 'episode', runId: 'r', episode: 1, return: 2, length: 3, terminated: true },
  frame: { type: 'frame', runId: 'r', step: 1, frame: [1] },
  checkpoint: { type: 'checkpoint', runId: 'r', step: 1, state: { s: 1 } },
  log: { type: 'log', runId: 'r', step: 1, level: 'warn', text: 'x' },
  end: { type: 'end', runId: 'r', status: 'failed', steps: 3, summary: { loss: 1 } },
}

const statuses: Record<RunStatus, true> = {
  completed: true,
  converged: true,
  diverged: true,
  stopped: true,
  failed: true,
}
const roles: Record<TensorRole, true> = { param: true, grad: true, state: true, activation: true, output: true }

interface ObjectJson {
  readonly properties?: Readonly<Record<string, { readonly const?: string; readonly enum?: readonly string[] }>>
}

const membersOfSchema = (): ObjectJson[] =>
  (z.toJSONSchema(TraceEventSchema, { io: 'input' }) as { oneOf: ObjectJson[] }).oneOf

describe('events: the interfaces and the schema are one vocabulary', () => {
  it('the schema parses every kind of event to itself and knows exactly the interface’s fields', () => {
    for (const event of Object.values(samples)) expect(TraceEventSchema.parse(event)).toEqual(event)
    expect(RunSchema.parse({ schema: 'mdml.run/1', events: Object.values(samples) }).events).toEqual(
      Object.values(samples),
    )
    const members = membersOfSchema()
    const byType = new Map(members.map((m) => [m.properties?.['type']?.const, Object.keys(m.properties ?? {}).sort()]))
    expect([...byType.keys()].sort()).toEqual(Object.keys(samples).sort())
    for (const [type, keys] of byType) expect(keys).toEqual(Object.keys(samples[type as TraceEvent['type']]).sort())
  })

  it('the schema takes exactly the statuses and tensor roles the types name', () => {
    for (const status of Object.keys(statuses))
      expect(TraceEventSchema.safeParse({ ...samples.end, status }).success).toBe(true)
    for (const role of Object.keys(roles))
      expect(TraceEventSchema.safeParse({ ...samples.tensor, role }).success).toBe(true)
    expect(TraceEventSchema.safeParse({ ...samples.end, status: 'unknown' }).success).toBe(false)
    const inJson = (members: ObjectJson[], type: string, field: string): readonly string[] | undefined =>
      members.find((m) => m.properties?.['type']?.const === type)?.properties?.[field]?.enum
    const members = membersOfSchema()
    expect([...(inJson(members, 'end', 'status') ?? [])].sort()).toEqual(Object.keys(statuses).sort())
    expect([...(inJson(members, 'tensor', 'role') ?? [])].sort()).toEqual(Object.keys(roles).sort())
  })

  // The older words of the result types (programming verdicts, root-finder flags) onto the one status: every word has a
  // row (the record is typed by `OutcomeWord`, so a new word fails to compile until it is mapped here).
  it('maps every older outcome word onto one RunStatus', () => {
    const table: Record<OutcomeWord, RunStatus> = {
      optimal: 'converged',
      converged: 'converged',
      'iteration limit': 'completed',
      'node limit': 'completed',
      'maximum iterations': 'completed',
      infeasible: 'failed',
      unbounded: 'failed',
      'numerical difficulties': 'failed',
      'derivative was zero': 'failed',
      'sign error': 'failed',
      success: 'completed',
      event: 'stopped',
      failed: 'failed',
    }
    for (const [word, status] of Object.entries(table)) expect(runStatusOf(word as OutcomeWord), word).toBe(status)
    expect(Object.values(table).every((s) => s in statuses)).toBe(true)
  })
})
