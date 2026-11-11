import { epHooks } from '../../shared/ep/sweeps.ts'
import { serveRun } from '../../shared/runWorker.ts'
import { referenceSkills, type SkillsReference } from './compute.ts'
import type { simulate } from './program.ts'

// The long NUTS run of the tournament, an iteration at a time: it reports each to the page and a stop keeps the draws so far.
serveRun<ReturnType<typeof simulate>, SkillsReference>(self, (data, host) =>
  referenceSkills(data, epHooks(host, 'NUTS reference', 'iteration')),
)
