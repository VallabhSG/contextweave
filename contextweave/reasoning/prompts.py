"""Prompt templates for the LLM reasoning engine."""

SYSTEM_PROMPT = """You are ContextWeave, a personal memory and context reasoning engine.
You have access to the user's personal context — conversations, notes, browser history,
calendar events, and journal entries — indexed over time.

Your role is to:
1. Answer questions by synthesizing information across multiple context sources
2. Surface non-obvious connections and patterns the user might not see
3. Track how topics, relationships, and thinking evolve over time
4. Always cite specific memories with timestamps when making claims
5. Be honest when context is insufficient — don't fabricate

When referencing context, use the format: [source: timestamp] to cite memories."""


GENERAL_QUERY_PROMPT = """Based on the following personal context memories, answer the user's question.

## Retrieved Context
{context}

## User Question
{query}

Instructions:
- Synthesize information across multiple memories
- Cite specific memories with their timestamps
- If the context doesn't contain enough information, say so clearly
- Surface any non-obvious connections you notice"""


PATTERN_DETECTION_PROMPT = """Analyze the following personal context memories and identify patterns.

## Retrieved Context
{context}

## Focus Area
{query}

Instructions:
- Look for recurring themes, topics, people, or behaviors
- Identify trends over time (increasing/decreasing frequency, shifting sentiment)
- Note any contradictions or tensions between different contexts
- Highlight what's present AND what's notably absent
- Structure your response as: Pattern → Evidence → Implication"""


GAP_ANALYSIS_PROMPT = """Based on the following personal context, identify what the user might be avoiding or overlooking.

## Retrieved Context
{context}

## User Question
{query}

Instructions:
- Look for topics mentioned once then never followed up
- Identify commitments or action items without resolution
- Note people or relationships that seem to have dropped off
- Flag any avoided topics that seem important based on context
- Be thoughtful and non-judgmental in your analysis"""


TEMPORAL_ANALYSIS_PROMPT = """Analyze how the user's thinking or behavior around a topic has evolved over time.

## Retrieved Context (ordered chronologically)
{context}

## Topic
{query}

Instructions:
- Map the evolution chronologically
- Note inflection points where thinking shifted
- Identify what triggered changes
- Compare earliest vs most recent positions
- Highlight any unresolved tensions"""


CROSS_REFERENCE_PROMPT = """Cross-reference information across the user's context to answer this question.

## Retrieved Context
{context}

## Question
{query}

Instructions:
- Connect information from different sources and time periods
- Build a composite picture from fragments
- Note where sources agree and disagree
- Identify information gaps that would strengthen the analysis
- Cite each source with timestamp"""


PRIORITY_SYNTHESIS_PROMPT = """Based on the user's recent context, synthesize what they should prioritize.

## Retrieved Context
{context}

## Timeframe
{query}

Instructions:
- Identify active commitments and deadlines
- Weight by urgency and stated importance
- Note dependencies between tasks/goals
- Flag potential conflicts or overcommitments
- Suggest a prioritized focus list with rationale"""


QUERY_TYPE_PROMPTS = {
    "general": GENERAL_QUERY_PROMPT,
    "patterns": PATTERN_DETECTION_PROMPT,
    "gaps": GAP_ANALYSIS_PROMPT,
    "temporal": TEMPORAL_ANALYSIS_PROMPT,
    "cross_reference": CROSS_REFERENCE_PROMPT,
    "priorities": PRIORITY_SYNTHESIS_PROMPT,
}
