/**
 * Example "A logic program, and the proof search behind its answers": a small family database in Prolog, queried by SLD
 * resolution. The search is recorded as it runs (one event for each call, exit, redo and failure) and the figure is the
 * search tree: the answers are the green leaves, and the branches that end in a failure are the places where the
 * search backtracked.
 */
import { consult, type SolveResult, sldTree, solve } from 'mdml-engine/logic'
import type { Structure } from 'mdml-engine/scene'
import { policies, type Run, recorder } from 'mdml-engine/trace'
import { diagram } from 'mdml-render/structures'
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

export function View({ result }: { result: Result }) {
  const scene = diagram(result.tree, { layout: 'layered', direction: 'down', rankSep: 90, nodeSep: 60 })
  const title = `?- ${query}.   ${result.answers.join(' ;  ')}   (${result.failures} failed ${result.failures === 1 ? 'branch' : 'branches'})`
  return (
    <Figure width={scene.width + 90} height={scene.height + 30} title={meta.alt}>
      <Panel width={scene.width + 90} height={scene.height} title={title} still={scene} />
    </Figure>
  )
}
