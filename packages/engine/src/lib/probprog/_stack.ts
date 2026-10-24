// The effect-handler stack of probabilistic programs (NumPyro's `_PYRO_STACK` and `apply_stack`). A handler is pushed
// for the duration of one call of the function it wraps and popped by try/finally, so the stack is empty whenever no
// program is running. Like the transform depth in `autodiff/_level.ts`, this is internal bookkeeping: it never
// changes a result and nothing outside a running program can observe it.
import { WhiteboxError } from '../base/index.ts'
import { permutation } from '../random/index.ts'
import { arange, cast, slice, type Tensor } from '../tensor/index.ts'
import type { Message, Messenger } from './primitives.ts'

const STACK: Messenger[] = []

/** Run `fn` with `handler` on top of the stack (innermost), popping it afterwards whatever happens. */
export function withHandler<T>(handler: Messenger, fn: () => T): T {
  STACK.push(handler)
  try {
    return fn()
  } finally {
    // Remove the handler and anything an interrupted inner call left above it.
    const at = STACK.lastIndexOf(handler)
    if (at >= 0) STACK.length = at
  }
}

/** The number of handlers currently active. */
export function stackDepth(): number {
  return STACK.length
}

function defaultProcess(msg: Message): void {
  if (msg.value !== undefined) return
  switch (msg.type) {
    case 'sample': {
      const fn = msg.fn
      if (fn === undefined) throw new WhiteboxError('value', `sample ${msg.name}: no distribution`)
      if (msg.key === undefined)
        throw new WhiteboxError(
          'value',
          `sample ${msg.name}: no key to draw with; run the program under seed(model, key) or condition the site`,
          { site: msg.name },
        )
      msg.value = fn.hasRsample ? fn.rsample(msg.key, msg.sampleShape) : fn.sample(msg.key, msg.sampleShape)
      return
    }
    case 'param': {
      const init = msg.init
      if (init === undefined) throw new WhiteboxError('value', `param ${msg.name}: no initial value`)
      if (typeof init === 'function') {
        if (msg.key === undefined)
          throw new WhiteboxError('value', `param ${msg.name}: a random initialiser needs seed(model, key)`, {
            site: msg.name,
          })
        msg.value = init(msg.key)
      } else msg.value = init
      return
    }
    case 'plate': {
      const size = msg.size ?? 0
      const sub = msg.subsampleSize ?? size
      if (sub >= size) {
        msg.value = cast(arange(size), 'int32')
        return
      }
      if (msg.key === undefined)
        throw new WhiteboxError('value', `plate ${msg.name}: subsampling needs a key; run under seed(model, key)`)
      msg.value = slice(permutation(msg.key, size), [0], [sub]) as Tensor<'int32'>
      return
    }
    default:
      return
  }
}

/**
 * Send a message through the stack: every handler from the innermost outwards processes it until one sets `stop`,
 * the default behaviour fills a missing value, and the visited handlers postprocess it from the outermost visited one
 * inwards.
 */
export function applyStack(msg: Message): Message {
  let visited = 0
  for (let i = STACK.length - 1; i >= 0; i--) {
    visited += 1
    STACK[i]?.process?.(msg)
    if (msg.stop) break
  }
  defaultProcess(msg)
  for (let i = STACK.length - visited; i < STACK.length; i++) STACK[i]?.postprocess?.(msg)
  return msg
}
