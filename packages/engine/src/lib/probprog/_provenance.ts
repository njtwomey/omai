// Provenance tracking for model analysis (NumPyro's `get_dependencies`, which traces provenance through jaxprs): values
// tagged with site names flow through every primitive, and each result carries the union of its inputs' tags. Unlike
// the reverse-mode tape, comparisons and other zero-derivative primitives keep their provenance, so dependencies
// through indexing and control-free comparisons are seen. Concrete values ride along, so code that reads `primalOf`
// works unchanged.
import { bind, isTraced, type Primitive, type Trace, type Traced, type Value } from '../tensor/index.ts'

interface Provenanced extends Traced {
  readonly primal: Value
  readonly sources: ReadonlySet<string>
}

const EMPTY: ReadonlySet<string> = new Set()

/** A trace that tags values with the sites they come from. */
export interface ProvenanceTrace extends Trace {
  tag(value: Value, source: string): Value
  sources(value: unknown): ReadonlySet<string>
}

/** A fresh provenance trace (at a level above any transform a program could be running inside). */
export function provenanceTrace(): ProvenanceTrace {
  const trace: ProvenanceTrace = {
    level: Number.MAX_SAFE_INTEGER,
    lift: (x) => make(x, EMPTY),
    process: <P>(primitive: Primitive<P>, inputs: readonly Traced[], params: P): readonly Value[] => {
      const ins = inputs as readonly Provenanced[]
      const outputs = bind(
        primitive,
        ins.map((x) => x.primal),
        params,
      )
      let sources = EMPTY
      for (const x of ins) {
        if (x.sources.size === 0) continue
        if (sources.size === 0) sources = x.sources
        else if (sources !== x.sources) sources = new Set([...sources, ...x.sources])
      }
      return outputs.map((o) => make(o, sources))
    },
    tag: (value, source) => {
      const prior = isOwn(value) ? value.sources : EMPTY
      return make(isOwn(value) ? value.primal : value, new Set([...prior, source]))
    },
    sources: (value) => (isOwn(value) ? value.sources : EMPTY),
  }
  function isOwn(x: unknown): x is Provenanced {
    return isTraced(x) && x.trace === trace
  }
  function make(primal: Value, sources: ReadonlySet<string>): Provenanced {
    return { kind: 'traced', dtype: primal.dtype, shape: primal.shape, trace, primal, sources }
  }
  return trace
}
