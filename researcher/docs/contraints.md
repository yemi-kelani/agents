**Human Authored:** Yemi Kelani

Can you do some general research on what established conventions make a good research agent harness. I have settled on python and langgraph 1.x as the language and sdk of choice.

I have several considerations I'd like you to research and report back on:

I am concerned about security, particularly prompt injection. What defensive options do I have here. I aim to make it secure and prompt injection resistant. At the very least, we should be able to confine the search to a whitelist of trusted domains via configuration if extra security is desired for a search.

Also for security should it run in a pod? So I should build an image? I like the idea of sandboxing an agent that contacts the internet but it might be uncessary. The trifecta of danger is when an agent has access to private or sensitive values, access to unverified data, and internet access so it might be enough to contain the operations so the agent can never to all at once or in sucession.

I like the optionality in search providers. I envision being able to dynamically switch the search engine (via configuration not code). I want to support searxgn and tavily at a minimum and, in the future, my own search index with open-search (not a day one a future add on).

I'm concerned about about accurate citations. What verification mechanisms exist today?

I aim to build a harness that supports multiple models e.g. locally hosted models via ollama and cloud models via openai or other protocols.

--- 

Here's my current plan for how the flow works. We research in rounds.

Initialization:

- The user query for research comes in. 
- We should determine if we need more info and ask follow ups with an interrupt.
- Then once we have enough information, we need to decompose the research topics into a set of research areas/encompassing brief.

Then for each research round:

- We can run parallel sequential flows across a subset of topics in our topic list. At most k flows.
- When the research comes back, we can extract claims from them and verfiy them against the text. LLMAJ? There's pros and cons there.
- Accumulate the verified claims.
- Our stopping criteria is an llm sufficiency check and maybe a max round iteration check (would be user configurable). Likely more involved than a simple yes/no check.

I was thinking the ecompassing research brief lives in the state as a list of topics. Each round, we research k topics (or research 1 topic with k agents, though I like this less) in parallel flows. Log researched topics in the state. As we research and verify claims, we track uncertainties that arise as well (also lives in the state as a list). The sufficiency check can add topics from the uncertainty list to the topic list as needed based on the accumulating report. Each flow can also track uncertainties after researching and extracting claims. This can be used by the sufficiency check to determine if new topics should be added. We'll set a max round iterations and max topic

Critique my flow and suggest improvements base on best practice langgraph conventions.

---

What I need from you:
archtecture and components with code snippets

my preferred structure:
```text
architecture.md  # serves as general overview and table of contents for spec sections
components/      # various components numbered according to build sequence
    01-component.md
    02-component.md
    ...
```

`components/` houses the components for each of the necessary components. An alternative structure would be to just have phases instead of topics but I prefer to break it down this way so topic areas are organized.

I plan to implement this piece by piece. I will critique and update each file so make sure they are concise and to the point yet detailed.

Component areas you should design. Feel free to add more if you feel I'm missing something important:

- graph state
- follow-up questioning and query decomposition
- topic creation and research plan creation
- search engines, configs, and swapability
- prompt injection defense and whitelisted/blocked domains/ips
- claim extraction and verification
- stopping criteria / sufficiency check

Look the the graph structure in `researcher-agent.png`