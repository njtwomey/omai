/** The worker that runs the inference: it serves the producer protocol for the methods a name describes (see `spec.ts`). */
import { type MethodRegistry, serveWorker } from 'omai-render/producer'
import { methodOf } from './spec.ts'

const registry: MethodRegistry = {
  get: methodOf,
  need: (name) => {
    const m = methodOf(name)
    if (m === undefined) throw new Error(`no method called '${name.slice(0, 40)}'`)
    return m
  },
  names: () => [],
}

serveWorker(registry)
