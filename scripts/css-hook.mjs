// Node cannot import a stylesheet. The render package's React entry imports one (katex's), which only a browser build
// needs, so for the examples runner a `.css` import is an empty module.
import { register } from 'node:module'

register(
  'data:text/javascript,' +
    encodeURIComponent(
      "export async function load(url, context, next) { if (url.endsWith('.css')) return { format: 'module', source: 'export default {}', shortCircuit: true }; return next(url, context) }",
    ),
)
