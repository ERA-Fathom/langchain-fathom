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

## Your own tool names

The read knows common state-writing tool names. For tools with your own names, map them once and pass the file.

```python
FathomMiddleware(mapping_path="tools.json")
```

```json
{"save_decision": {"op": "set", "kind": "decision", "key": "topic", "value": "text"},
 "confirm_booking": {"op": "commit", "kind": "flight", "key": "booking"}}
```

## The read and the repair

The middleware is the read. It tells you where coherence broke. Embedded Risk Analytics also runs a hosted service that acts on what the read finds and re-grounds the agent before the contradiction ships. Write to contact@embeddedriskanalytics.com for a key that lifts the demo rate limit, or to run the read on a workflow of your own.

## Links

- [fathom-read](https://github.com/ERA-Fathom/fathom), the read and its adapters for LangGraph, CrewAI, Letta, OpenInference, DBOS, and coding-agent edit logs
- [Coherence census](https://github.com/ERA-Fathom/coherence-census), every framework the read has run on, with a trace per row
- [Research](https://embeddedriskanalytics.com/research.html) and the paper, [SSRN 6683578](https://doi.org/10.2139/ssrn.6683578)

MIT licensed.
