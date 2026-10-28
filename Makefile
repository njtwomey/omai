# mdml-js: the single interface for routine work. Run `make help` for the list.
SHELL := /bin/bash
.DEFAULT_GOAL := help

MISE := mise exec --
PNPM := $(MISE) pnpm
NODE := $(MISE) node
UV := $(MISE) uv

PACKAGES := engine render meta

.PHONY: help
help: ## List targets
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

.PHONY: setup
setup: ## Install tools (mise), JS dependencies (pnpm) and the Python dev environment (uv)
	mise install
	$(PNPM) install
	$(UV) sync --group dev

.PHONY: barrels
barrels: ## Regenerate index.ts barrels and the module map
	$(NODE) scripts/barrels.ts

.PHONY: barrels-check
barrels-check: ## Fail if any barrel is stale
	$(NODE) scripts/barrels.ts --check

.PHONY: layers
layers: ## Check layering: no import cycles between core domains or methods areas, and the package rules
	$(NODE) scripts/layers.ts

.PHONY: lint
lint: ## Lint TypeScript (Biome), banned globals, and Python (ruff)
	$(PNPM) exec biome check .
	$(NODE) scripts/banned.ts
	$(UV) run --group dev ruff check oracle scripts
	$(UV) run --group dev ruff format --check oracle scripts

.PHONY: format
format: ## Format TypeScript and Python
	$(PNPM) exec biome check --write .
	$(UV) run --group dev ruff format oracle scripts

.PHONY: typecheck
typecheck: ## Type-check every package (tsc) and the oracle (pyright)
	@for p in $(PACKAGES); do echo "tsc $$p"; $(PNPM) --dir packages/$$p exec tsc -p tsconfig.json --noEmit || exit 1; done
	@echo "tsc apps/site"; $(PNPM) --dir apps/site exec tsc -p tsconfig.json --noEmit
	@echo "tsc scripts and config"; $(PNPM) exec tsc -p tsconfig.json
	$(UV) run --group dev --group oracle-core --group oracle-torch --group oracle-prob --group oracle-rl --group oracle-vision pyright oracle

.PHONY: test
test: ## Run unit, property, golden and type tests (Node)
	$(PNPM) exec vitest run

.PHONY: test-watch
test-watch: ## Run tests in watch mode
	$(PNPM) exec vitest

.PHONY: coverage
coverage: ## Run tests with coverage thresholds
	$(PNPM) exec vitest run --coverage

.PHONY: fixtures
fixtures: ## Regenerate golden fixtures from the Python oracle (needs the oracle groups)
	$(UV) run --group oracle-core --group oracle-torch --group oracle-prob --group oracle-rl --group oracle-vision python -m mdml_oracle generate

.PHONY: fixtures-check
fixtures-check: ## Regenerate fixtures to a temporary directory and fail on any difference
	$(UV) run --group oracle-core --group oracle-torch --group oracle-prob --group oracle-rl --group oracle-vision python -m mdml_oracle check

.PHONY: manifest
manifest: ## Rebuild fixtures/manifest.json from the fixture files (for merge conflicts; runs no case module)
	$(UV) run --group oracle-core python -m mdml_oracle manifest

.PHONY: examples
examples: ## Run every @example of the engine core and check its `// =>` lines (ARGS="--only stats --methods" etc.)
	$(NODE) scripts/examples.ts $(ARGS)

.PHONY: build
build: barrels ## Build all packages
	@for p in $(PACKAGES); do $(PNPM) --dir packages/$$p run build || exit 1; done

.PHONY: docs-catalogue
docs-catalogue: ## Precompile the docs catalogue into apps/site/.generated/docs (skipped when newer than the sources)
	$(NODE) apps/site/plugins/build-docs.ts --if-stale

.PHONY: dev
dev: docs-catalogue ## Serve the site (gallery and docs) at http://localhost:5300 with hot reload
	$(PNPM) --dir apps/site exec vite

.PHONY: site
site: docs-catalogue ## Build the site into apps/site/dist
	$(PNPM) --dir apps/site exec vite build

.PHONY: verify
verify: ## Checks scoped to what changed against BASE (default main): the gate for commits and merges
	$(NODE) scripts/verify.ts --base $(or $(BASE),main)

.PHONY: check
check: barrels-check layers lint typecheck test examples ## Everything, every test (periodic full run; `verify` is the per-change gate)

.PHONY: clean
clean: ## Remove build outputs
	$(PNPM) -r exec rm -rf dist
	rm -rf coverage
