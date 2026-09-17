# langchain-fathom

**A coherence read for LangChain agents, as an agent middleware.**

A long-running agent loses coherence with its own decisions. It renames a field at one step, then writes the old name at a later one. It marks a record done that it never wrote. The run reports success, and the contradiction ships. `langchain-fathom` watches the agent run and, when it finishes, names the step where a later action contradicts an earlier commitment.

The middleware observes the agent and does not change what it does. It maps the tool calls in the message history to a committed-state op stream, sends that stream to the [Fathom read](https://github.com/ERA-Fathom/fathom), and reports the findings. The read is deterministic and needs no model access.

## Install

```
pip install langchain-fathom
```

## Use

```python
from langchain.agents import create_agent
from langchain_fathom import FathomMiddleware

agent = create_agent(
    model="gpt-5.5",
    tools=[...],
    middleware=[FathomMiddleware()],
)

result = agent.invoke({"messages": [...]})
```

By default the middleware logs any findings through the `fathom` logger. Two other modes fit a test suite or a pipeline.

```python
FathomMiddleware(on_finding="raise")   # raise FathomCoherenceError when a run is not coherent
FathomMiddleware(on_finding="store")   # put the verdict on agent state under the "fathom" key
```

Under `store`, the verdict arrives on the result.

```python
result = agent.invoke({"messages": [...]})
verdict = result["fathom"]
verdict["coherent"], verdict["findings"]
```

The middleware declares that key in its own state schema, so a graph that carries a state schema of its own keeps the verdict rather than dropping it. Version 0.1.0 declared nothing, and an orchestrator such as a deepagents agent dropped the verdict before the caller saw it.

If the run is a rename or a migration, tell the read which token replaced which, so it also reports records left on the old value at the end.

```python
FathomMiddleware(supersede=[("guest_id", "customer_id")])
```

## What it finds

| Finding | The agent... |
|---|---|
| `stale_reference` | acts on a record it already removed or renamed away |
| `superseded_value` | writes a value it already replaced |
| `residual` | ends the run with a record still carrying a value it replaced elsewhere |
| `duplicate_commit` | adds an entity a collection already holds |
| `post_commit_mutation` | changes a record after committing it |

A coherent run returns `coherent` and the middleware reports nothing else.

## Agents that delegate

The middleware reads the calls of the agent it rides. An orchestrator that hands work to sub-agents runs each sub-agent as its own agent with its own message history, so an orchestrator-level middleware sees the delegation and the orchestrator's own tools, and not what the sub-agents did. Give each sub-agent its own `FathomMiddleware` to cover the whole run. On one deepagents research run we measured, the orchestrator's middleware alone returned 3 of the 9 findings a trace across every sub-agent returned.

## Your own tool names

The read knows common state-writing tool names. For tools with your own names, map them once and pass the file.

```python
FathomMiddleware(mapping_path="tools.json")
```

```json
{"save_decision": {"op": "set", "kind": "decision", "key": "topic", "value": "text"},
 "confirm_booking": {"op": "commit", "kind": "flight", "key": "booking"}}
```

## The repair

`FathomMiddleware` watches. `FathomRepairMiddleware` acts. It sits between the model and its tools, sends the actions the model just proposed together with what the run has already committed, and gets back one of three decisions.

```python
from langchain_fathom import FathomRepairMiddleware

agent = create_agent(
    model="gpt-5.5",
    tools=[...],
    middleware=[FathomRepairMiddleware()],
)
```

On `proceed` every proposed action is consistent with what the agent already committed and nothing changes. On `filter` the agent keeps its own consistent alternatives and the contradicting calls never reach the tools. On `reground` none of them survive, so the committed facts go back in front of the model as a system-role message and the model is asked again, once by default.

The repair needs a free key. Get one with `fathom key you@example.com` from the [fathom-read](https://github.com/ERA-Fathom/fathom) package and set `FATHOM_API_KEY`, or pass `key=`. The middleware checks for it when you construct it, so a run fails at setup rather than twenty steps in. The two reads stay free without a key.

This middleware changes what your agent does, which is why it carries its own class rather than a flag on the read. Two settings cover the cases where that matters.

```python
FathomRepairMiddleware(on_error="raise")  # stop the run rather than skip the repair
FathomRepairMiddleware(max_reasks=0)      # never re-ask; filter and proceed only
```

`on_error` defaults to `"proceed"`, so an unreachable service or a spent daily limit logs a warning and lets the agent act on its own proposal. A coherence repair should not take down a running agent. Pass `"raise"` when you are measuring a before-and-after, since a silently skipped repair would contaminate the result.

Every repaired step appends an entry to agent state under `fathom_repair`, carrying the decision, the proposals evaluated, the finding kinds of anything dropped, and the re-asks spent. That is the log a run needs to report what the repair did.

```python
result = agent.invoke({"messages": [...]})
for step in result["fathom_repair"]:
    print(step["decision"], step["dropped"])
```

What it costs, in two places. A repair call counts double against the 2,000 calls a day a free key carries. A `reground` decision spends one extra model call, billed by your own provider. A model turn with no tool calls, and a tool the mapping does not name, cost nothing and pass through untouched.

## What we measured

On DBOS's own Hacker News research agent, vendored unchanged and run on gpt-4o-mini through OpenRouter across five topics at ten iterations each, the agent as published repeated 14 of 50 searches and re-read 42 percent of the threads it fetched. With the repair in front of the one step where it proposes its next queries, repeats fell to 0 of 50, re-reads to 12 percent, and distinct threads covered rose 35 percent at the same model and the same iteration count. Across 45 follow-up steps the repair let 23 through untouched, filtered 21, and regrounded 1. Every run, the trace, and the command sit in the [coherence census](https://github.com/ERA-Fathom/coherence-census/tree/main/rows/dbos-hn-agent).

Those numbers come from an agent that hands out its proposed next queries explicitly. Your mileage depends on how much of your agent's state the tool mapping can see.

## Links

- [fathom-read](https://github.com/ERA-Fathom/fathom), the read and its adapters for LangGraph, CrewAI, Letta, OpenInference, DBOS, and coding-agent edit logs
- [Coherence census](https://github.com/ERA-Fathom/coherence-census), every framework the read has run on, with a trace per row
- [Research](https://embeddedriskanalytics.com/research.html) and the paper, [SSRN 6683578](https://doi.org/10.2139/ssrn.6683578)

MIT licensed.
