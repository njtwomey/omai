// The depth of nested transformations. A transform takes the next level for the duration of its function call and
// restores it on exit, so a closure that captures an outer transform's value is always processed at the right level.
// This is the one piece of internal bookkeeping state in the tensor machinery: it never changes a result and is always
// balanced by try/finally.
let depth = 0

export function withLevel<T>(fn: (level: number) => T): T {
  depth += 1
  try {
    return fn(depth)
  } finally {
    depth -= 1
  }
}
