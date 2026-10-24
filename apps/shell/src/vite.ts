import type { Plugin } from 'vite'
import { noFlashScript } from './colour-script.ts'

/**
 * The shell's Vite plugin: puts the colour-mode script at the top of `<head>`, so the page is marked light or dark
 * before it is first painted.
 */
export function whiteboxShell(): Plugin {
  return {
    name: 'whitebox-shell',
    transformIndexHtml: () => [{ tag: 'script', children: noFlashScript, injectTo: 'head-prepend' }],
  }
}
