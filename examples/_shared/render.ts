/**
 * What the runner needs from React and the render package, kept here because this folder is the one that depends on
 * them: a view rendered to static SVG markup, and a tape as the library's tape file.
 */

import type { Timeline } from 'mdml-engine/scene'
import { encodeGif, timelineFrames } from 'mdml-render/animation'
import { tapeOf } from 'mdml-render/producer'
import * as React from 'react'
import { createElement, type ReactElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { PAPER } from './colours.ts'
import type { Tape } from './contract.ts'

// The render package's .tsx files are compiled by tsx with the classic JSX transform, which reads a global `React`.
;(globalThis as { React?: unknown }).React = React

/**
 * zrender names the ids and classes of each chart it draws from counters shared by the whole process (`zr20-c0`,
 * `zr20-cls-328`), so the same figure came out differently when an example ran alone or after others. They are
 * renumbered by order of first appearance in the document: instances `zr0`, `zr1`, ... and classes `cls-0`, `cls-1`, ...
 */
export function renumberZrender(svg: string): string {
  const instances = new Map<string, number>()
  const classes = new Map<string, number>()
  const index = (map: Map<string, number>, id: string): number => {
    const known = map.get(id)
    if (known !== undefined) return known
    map.set(id, map.size)
    return map.size - 1
  }
  return svg.replace(/\bzr(\d+)-(?:cls-(\d+))?/g, (_, instance: string, cls: string | undefined) =>
    cls === undefined
      ? `zr${index(instances, instance)}-`
      : `zr${index(instances, instance)}-cls-${index(classes, cls)}`,
  )
}

/** A view rendered to markup (no browser: the view is plain SVG), with the zrender names renumbered per document. */
export function renderView<R>(View: (props: { readonly result: R }) => ReactElement, result: R): string {
  return renumberZrender(renderToStaticMarkup(createElement(View, { result })))
}

/** A tape as JSON: a session as a `mdml.tape/1` file (its method, config and events), a recorder's run as it is. */
export async function tapeJson(name: string, tape: Tape, title: string): Promise<unknown> {
  if (!('advance' in tape)) return tape
  return tapeOf(tape.method.name, [{ label: name, session: tape }], { requests: [], title })
}

/** A timeline of scenes as GIF bytes, in Node: the engine's rasteriser draws the frames (without text) on white. */
export async function animationGif(timeline: Timeline): Promise<Uint8Array> {
  return encodeGif(await timelineFrames(timeline, { background: PAPER }), { fps: timeline.fps })
}
