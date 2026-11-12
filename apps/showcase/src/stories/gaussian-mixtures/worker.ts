import { serveStory } from '../../kit/protocol.ts'
import { type Build, DEFAULT_BUILD, run } from './compute.ts'

/**
 * The worker of 'Gaussian mixtures'. Unlike the other stories it is told what to build: the page posts the reader's choice
 * (`{ type: 'build', build }`) right after making the worker, before the `run` request, and messages are handled in order.
 */
let build: Build = DEFAULT_BUILD

serveStory(self, (seed, say) => run(seed, say, build))

const handleRun = self.onmessage
self.onmessage = (event: MessageEvent): void => {
  const data = event.data as { readonly type?: string; readonly build?: Build }
  if (data.type === 'build' && data.build !== undefined) build = data.build
  else handleRun?.call(self, event)
}
