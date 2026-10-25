/**
 * Run summaries as table data: one row per run in a trace (children indented under their parents) with its name,
 * status, steps, seed and the final value of each summary metric, ready for any table component.
 */

import { type Run, type RunNode, runTree } from 'mdml-engine/trace'

/** A table as data: column names and rows of cells, for any table component. */
export interface TableData {
  readonly columns: readonly string[]
  readonly rows: readonly (readonly (string | number | null)[])[]
}

/**
 * A summary table of the runs in a trace: one row per run (children indented under their parents) with its name,
 * status, steps, seed and the final value of each summary metric.
 *
 * @param run - The trace.
 * @returns The table.
 * @example
 * ```ts
 * import { runTable } from 'mdml-render/trace'
 * const run = { schema: 'mdml.run/1' as const, events: [
 *   { type: 'run' as const, runId: 'r', name: 'sgd', seed: 0, version: '1', config: {} },
 *   { type: 'end' as const, runId: 'r', status: 'converged' as const, steps: 12, summary: { loss: 0.1 } },
 * ] }
 * runTable(run) // => { columns: ['run', 'status', 'steps', 'seed', 'loss'], rows: [['sgd', 'converged', 12, 0, 0.1]] }
 * ```
 */
export function runTable(run: Run): TableData {
  const metrics: string[] = []
  const rows: (string | number | null)[][] = []
  const walk = (nodes: readonly RunNode[], depth: number): void => {
    for (const node of nodes) {
      for (const k of Object.keys(node.end?.summary ?? {})) if (!metrics.includes(k)) metrics.push(k)
      rows.push([
        `${'  '.repeat(depth)}${node.run.name}`,
        node.end?.status ?? 'running',
        node.end?.steps ?? null,
        node.run.seed,
        ...metrics.map((m) => node.end?.summary[m] ?? null),
      ])
      walk(node.children, depth + 1)
    }
  }
  walk(runTree(run), 0)
  const width = 4 + metrics.length
  return {
    columns: ['run', 'status', 'steps', 'seed', ...metrics],
    rows: rows.map((r) => [...r, ...new Array<null>(Math.max(0, width - r.length)).fill(null)]),
  }
}
