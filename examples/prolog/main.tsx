/**
 * Example "A logic program, and the proof search behind its answers": a small family database in Prolog, queried by SLD
 * resolution. The search is recorded as it runs (one event for each call, exit, redo and failure) and the figure is the
 * search tree: the answers are the green leaves, and the branches that end in a failure are the places where the
 * search backtracked.
 */
import { consult, type SolveResult, sldTree, solve } from 'omai-engine/logic'
import type { Structure } from 'omai-engine/scene'
import { policies, type Run, recorder } from 'omai-engine/trace'
import { diagram } from 'omai-render/structures'
import { INK } from '../_shared/colours.ts'
import type { ExampleModule, Outcome } from '../_shared/contract.ts'
import { Figure, Panel } from '../_shared/figure.tsx'

export const meta: ExampleModule['meta'] = {
  title: 'A logic program and its proof search',
  summary:
    'A small Prolog family database; the query is answered by SLD resolution and the search tree is drawn from the run.',
  readme: false,
  alt: 'The search tree of the query grandparent(tom, W): the answers, and the branches that fail and backtrack',
}

const program = `
parent(tom, bob).
parent(tom, liz).
parent(tom, sam).
parent(bob, ann).
parent(bob, pat).
parent(liz, joe).
grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
`

const query = 'grandparent(tom, W)'

interface Result {
  readonly tree: Structure
  readonly answers: readonly string[]
  readonly failures: number
}

export function run(): Outcome<Result> {
  const tape = recorder()
  const found: SolveResult = solve(consult(program), query, {
    recordTree: true,
    tracer: tape,
    policy: policies.teaching,
  })
  const tree = sldTree(found)
  const answers = found.answers.map((a) => a.text)
  const failures = (found.tree ?? []).filter((n) => n.kind === 'failure').length
  const run: Run = tape.run()
  const notes = [
    `?- ${query}.  ${answers.join(' ;  ')}`,
    `${found.steps} steps, ${found.inferences} inferences, ${tree.nodes.length} nodes in the tree, ${failures} failed branches`,
  ]
  return { result: { tree, answers, failures }, tapes: { run }, notes }
}

/** The clauses of the program as listed under their predicate and number, as the tree's edges name them. */
function clauseLines(): string[] {
  const counts = new Map<string, number>()
  return program
    .trim()
    .split('\n')
    .map((clause) => {
      const name = /^(\w+)\(([^)]*)\)/.exec(clause)
      const arity = name === null ? 0 : (name[2] ?? '').split(',').length
      const predicate = `${name?.[1] ?? '?'}/${arity}`
      const index = (counts.get(predicate) ?? 0) + 1
      counts.set(predicate, index)
      return `${predicate} #${index}   ${clause}`
    })
}

export function View({ result }: { result: Result }) {
  const scene = diagram(result.tree, { layout: 'layered', direction: 'down', rankSep: 90, nodeSep: 60 })
  const left = 400
  const lines: { text: string; bold?: boolean; gap?: number }[] = [
    { text: 'The program', bold: true },
    ...clauseLines().map((text) => ({ text })),
    { text: 'The query', bold: true, gap: 16 },
    { text: `?- ${query}.` },
    { text: 'The answers', bold: true, gap: 16 },
    { text: result.answers.join('   ') },
    { text: 'How to read the tree', bold: true, gap: 16 },
    { text: 'Each arrow replaces the goal above it with' },
    { text: 'what is left to prove after one clause is used;' },
    { text: 'the arrow names the clause (predicate and number).' },
    { text: '[ ] is an empty goal: it is proved, and the' },
    { text: 'value found for W is an answer.' },
    { text: 'fail: no clause matches, so the search goes' },
    { text: 'back (backtracks) to try the next one.' },
  ]
  let y = 40
  const placed = lines.map((l) => {
    y += l.gap ?? 0
    const at = y
    y += 19
    return { ...l, y: at }
  })
  const height = Math.max(scene.height + 30, y + 20)
  return (
    <Figure width={left + scene.width + 40} height={height} title={meta.alt}>
      <text x={20} y={22} fontFamily="system-ui, sans-serif" fontSize={14} fontWeight={600} fill={INK}>
        A Prolog program and the search that answers one question
      </text>
      {placed.map((l) => (
        <text
          key={l.y}
          x={20}
          y={l.y}
          fontFamily={l.bold ? 'system-ui, sans-serif' : 'ui-monospace, Menlo, monospace'}
          fontSize={l.bold ? 13 : 12}
          fontWeight={l.bold ? 600 : 400}
          fill={INK}
        >
          {l.text}
        </text>
      ))}
      <Panel
        x={left}
        width={scene.width + 40}
        height={scene.height}
        title="the proof search for the query"
        still={scene}
      />
    </Figure>
  )
}
