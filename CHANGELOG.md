# Changelog

## 0.2.0 (2026-09-17)

- `FathomRepairMiddleware` runs the Fathom repair on the actions the model proposes, before the tools run. It answers `proceed`, keeps the agent's own non-contradicting alternatives on `filter`, and on `reground` puts the committed facts back in front of the model and asks again. Unlike `FathomMiddleware`, it changes what the agent does, so it ships as its own class rather than as another `on_finding` value.
- The repair sits in `wrap_model_call` rather than before the model turn, because the actions it evaluates come into existence only once the model has emitted its tool calls.
- Fails open. An unreachable service, a rejected key, or a spent daily limit logs a warning and lets the agent act on its own proposal. `on_error="raise"` turns that into `FathomRepairError`, which is what a measured before-and-after run wants.
- `max_reasks` defaults to 1, matching the DBOS run of 2 September where one re-ask occurred across 45 follow-up steps.
- The key comes from `key=` then `FATHOM_API_KEY`, checked when the middleware is constructed so a run fails at setup rather than partway through. The middleware never requests a key on its own.
- One entry per repaired step appends to agent state under `fathom_repair`, declared in `FathomRepairState` with a concatenating reducer. `FathomRepairError`, `FathomKeyError` and `REPAIR_STATE_KEY` are exported.
- A tool the mapping does not name never becomes a proposal and always passes through, and a model turn with no tool calls costs nothing.
- Requires fathom-read 0.5.0 or later.

## 0.1.1 (2026-09-17)

- `FathomMiddleware` declares the `fathom` state key in its own state schema, so `on_finding="store"` reaches the caller through a graph that carries a state schema of its own. A deepagents orchestrator dropped the verdict under 0.1.0, measured on the deep_research runs of 16 September.
- `FathomState` and `STATE_KEY` are exported for callers that build their own schema.
- The README records what an orchestrator-level middleware sees when an agent delegates, and points at the free key for the hosted read and repair.
- Tests run a real agent and a real deepagents orchestrator with a fake model, and a control asserts that an undeclared key is dropped.

## 0.1.0 (2026-09-03)

- First release. `FathomMiddleware` maps an agent's tool calls to a committed-state op stream, sends it to the Fathom read after the run, and reports findings through `log`, `raise`, or `store`.
- `supersede` pairs for renames and migrations, `mapping_path` for tool names outside the defaults.
