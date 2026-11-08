/**
 * The gallery's worker entry: it serves the producer protocol for the toy methods. A page starts it with
 * `new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' })` (see `toys/producer.ts`), and Vite bundles
 * it as its own chunk, so the worker loads only the engine and these methods, never the page's charts or React.
 */

import { serveWorker } from 'omai-render/producer'
import { toyRegistry } from './registry.ts'

serveWorker(toyRegistry)
