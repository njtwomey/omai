/**
 * A trail that fades: the last `length` points of a path drawn as a few line pieces whose opacity rises towards the newest
 * point, so an animated path keeps a short tail instead of drawing everything it has been through.
 */
import type { lines } from 'omai-render/charts'

type LineSpec = Parameters<typeof lines>[0][number]

/** The pieces of a fading trail: one line per piece, the oldest faintest. Each carries the same `name`, so a legend shows it once. */
export function fadingTrail(
  x: readonly number[],
  y: readonly number[],
  o: {
    readonly length: number
    readonly pieces?: number
    readonly name: string
    readonly colour?: string
    readonly width?: number
  },
): LineSpec[] {
  const pieces = o.pieces ?? 8
  const from = Math.max(0, x.length - o.length)
  const size = Math.max(1, Math.ceil((x.length - from - 1) / pieces))
  const out: LineSpec[] = []
  for (let p = 0; p < pieces; p++) {
    // Pieces overlap by one point so the line has no gaps.
    const start = from + p * size
    const end = Math.min(x.length, start + size + 1)
    if (end - start < 2) break
    out.push({
      x: x.slice(start, end),
      y: y.slice(start, end),
      name: p === pieces - 1 ? o.name : `${o.name} ${p}`,
      width: o.width ?? 2,
      opacity: 0.08 + (0.92 * (p + 1)) / pieces,
      ...(o.colour === undefined ? {} : { colour: o.colour }),
    })
  }
  return out
}
