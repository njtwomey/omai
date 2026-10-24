import { createElement, type ReactNode } from 'react'
import { renderToString } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { type LiveExampleState, useLiveExample } from './live-example.ts'

const seen: LiveExampleState[] = []

function Probe(): ReactNode {
  seen.push(useLiveExample({ title: '', code: '1 + 1 // => 2' }, { loaders: {}, runOnView: false }))
  return null
}

describe('useLiveExample', () => {
  it('starts with the example code, unedited and not yet run', () => {
    renderToString(createElement(Probe))
    expect(seen[0]).toMatchObject({ code: '1 + 1 // => 2', edited: false, outcome: null, running: false })
  })
})
