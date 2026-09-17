# Changelog

## 0.1.1 (2026-09-17)

- `FathomMiddleware` declares the `fathom` state key in its own state schema, so `on_finding="store"` reaches the caller through a graph that carries a state schema of its own. A deepagents orchestrator dropped the verdict under 0.1.0, measured on the deep_research runs of 16 September.
- `FathomState` and `STATE_KEY` are exported for callers that build their own schema.
- The README records what an orchestrator-level middleware sees when an agent delegates, and points at the free key for the hosted read and repair.
- Tests run a real agent and a real deepagents orchestrator with a fake model, and a control asserts that an undeclared key is dropped.

## 0.1.0 (2026-09-03)

- First release. `FathomMiddleware` maps an agent's tool calls to a committed-state op stream, sends it to the Fathom read after the run, and reports findings through `log`, `raise`, or `store`.
- `supersede` pairs for renames and migrations, `mapping_path` for tool names outside the defaults.
