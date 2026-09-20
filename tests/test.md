Good prompts do four things: say exactly what you want, give the context needed to do it, show what "done" looks like, and constrain the output shape.

- **Be specific about the task, not the topic.** "Summarize this in five bullets for a non-technical reader" beats "tell me about this."
- **Give the context up front.** Relevant background, data, or files go before the instruction — the model can't ask follow-ups mid-thought.
- **State the output format.** Length, structure, tone, whether you want code or prose.
- **Give an example or two** when the format is unusual or the judgment call is subtle. One good example teaches more than a paragraph of description.
- **Say what to do, not what to avoid.** Negative instructions are weaker than positive ones.
- **Split big asks into steps** and let the model reason before answering, rather than demanding the answer first.
- **Define the failure case.** Tell it what to do when information is missing — say "I don't know" rather than guess. This matters most for factual or scientific work.
- **Assign a role only when it changes the answer** (e.g. "review this as a security auditor"). Otherwise it's noise.
- **Iterate.** Treat the first prompt as a draft; when the output is wrong, the fix is usually a missing constraint, not a longer prompt.

The single biggest lever is being concrete about the deliverable. Most bad output comes from a prompt that was ambiguous about what success looks like.