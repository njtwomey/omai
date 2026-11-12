/** The story's worker: it serves `run` from `compute.ts`, so the computation happens off the page's thread. */

import { serveStory } from '../../kit/protocol.ts'
import { run } from './compute.ts'

serveStory(self, run)
