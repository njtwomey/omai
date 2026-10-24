declare module 'virtual:whitebox-docs/index' {
  const index: import('./data.ts').DocsIndex
  export default index
}
declare module 'virtual:whitebox-docs/modules' {
  const modules: Readonly<
    Record<string, () => Promise<{ readonly default: readonly import('whitebox-meta/schema').SymbolDoc[] }>>
  >
  export default modules
}
