import { type AtomReference, atomReference, epAtom } from 'omai-engine/probprog/ep'
import type { Config } from 'omai-engine/trace'
import { serveRun } from '../runWorker.ts'

/** What the page asks: the atom by name and its knobs (the problem is rebuilt here, from the same seed). */
export interface AtomReferenceRequest {
  readonly name: string
  readonly knobs: Config
}

// The reference of an atom by quadrature is one library call (a grid over the posterior): it cannot report how far it is, so
// the page shows a timer and an unfilled bar, and Stop ends the worker.
serveRun<AtomReferenceRequest, AtomReference>(self, async ({ name, knobs }) => {
  const atom = epAtom(name)
  return atomReference(atom, atom.problem(knobs))
})
