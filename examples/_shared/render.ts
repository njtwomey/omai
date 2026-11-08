/**
 * What the runner needs from React and the render package, kept here because this folder is the one that depends on
 * them: a view rendered to static SVG markup, and a tape as the library's tape file.
 */

import { tapeOf } from 'mdml-render/producer'
import { createElement, type ReactElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import type { Tape } from './contract.ts'

/** A view rendered to markup (no browser: the view is plain SVG). */
export function renderView<R>(View: (props: { readonly result: R }) => ReactElement, result: R): string {
  return renderToStaticMarkup(createElement(View, { result }))
}

/** A tape as JSON: a session as a `mdml.tape/1` file (its method, config and events), a recorder's run as it is. */
export async function tapeJson(name: string, tape: Tape, title: string): Promise<unknown> {
  if (!('advance' in tape)) return tape
  return tapeOf(tape.method.name, [{ label: name, session: tape }], { requests: [], title })
}
