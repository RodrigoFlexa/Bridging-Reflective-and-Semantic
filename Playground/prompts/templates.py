SEMANTIC_TEMPLATE = """Instruction: You are an expert science tutor. Your goal is to answer the target multiple-choice question below.

To assist you, I have retrieved scientific principles - fundamental definitions and laws extracted from similar problems.

Use the Principles to ground your facts and guide your logic.

Structure your response strictly as:
1. **Reasoning:** Explain the step-by-step logic to reach the correct answer.
2. **Principles:** List the scientific concepts applied.
3. **Answer:** State only the correct option letter (A, B, C, or D).

--------------------------------------------------
{similar_facts}
--------------------------------------------------

### TARGET QUESTION:
{question}
"""


HYBRID_TEMPLATE = """Instruction: You are an expert science tutor. Your goal is to answer the target multiple-choice question below.

To assist you, I have retrieved two types of information:
1. **Scientific Principles:** Fundamental definitions and laws (with cognitive memory scores).
2. **Solved Examples:** Similar problems with step-by-step analysis.

Use the Principles to ground your facts and the Solved Examples to guide your logic.

Structure your response strictly as:
1. **Reasoning:** Explain the step-by-step logic to reach the correct answer.
2. **Principles:** List the scientific concepts applied.
3. **Answer:** State only the correct option letter (A, B, C, or D).

--------------------------------------------------
{hybrid_context}
--------------------------------------------------

### TARGET QUESTION:
{question}
"""

SCORE_TEMPLATE = """Task: Evaluate the contribution of the 'Fact' to the agent's 'Reasoning'.
Assign a Score (-1, 0, or 1).

### Input Data:
Question: {question}
Correct Answer: {correct}
Model Choice: {chosen} (Outcome: {outcome_status})
Fact: "{fact}"
Agent's Reasoning: "{reasoning}"

### Scoring Rules:
-1 (Harmful): The Fact led the agent to a failure. 
   (Look for: Did the reasoning cite this fact to justify the wrong answer?)
 0 (Useless/Unused): The Fact was ignored OR it is irrelevant.
   (Look for: Does the reasoning mention this fact? If not, it's 0, even if the fact is true.)
 1 (Useful): The Fact was essential for the success.
   (Look for: Did the reasoning rely on this specific fact to find the correct answer?)

### Output:
Reflection: (1 sentence analyzing the link between Fact and Reasoning)
Score: (Only write -1, 0, or 1)
"""

EXTRACT_FACTS_TEMPLATE = """Extract at most 2 generalizable scientific facts from the response below.

### Response:
{model_response}

### Question:
{question}

### Rules:
- Facts must be universal scientific truths, not specific to this scenario
- Keep facts concise (1 sentence each)
- Output ONLY the facts, one per line, without numbering

Facts:"""

REFLECTION_TEMPLATE = """Instruction: You are an expert science tutor. Your goal is to answer the target multiple-choice question below.

To assist you, I have retrieved solved examples — similar problems with step-by-step analysis and reasoning traces.

Use these examples to guide your logic and identify patterns relevant to the target question.

Structure your response strictly as:
1. **Reasoning:** Explain the step-by-step logic to reach the correct answer.
2. **Principles:** List the scientific concepts applied.
3. **Answer:** State only the correct option letter (A, B, C, or D).

--------------------------------------------------
{reflection_context}
--------------------------------------------------

### TARGET QUESTION:
{question}
"""

REFINE_FACT_TEMPLATE = """### Instruction:
Complete the reasoning to state a single, short scientific fact.

### Context:
Q: {question}
Correct Answer: {correct_answer}
Critique: {reflection}

### Reasoning:
The Old Fact "{old_fact}" was insufficient because {reflection}.
Therefore, to explain that correct answer is "{correct_answer}" and is not {wrong_answer}, the precise scientific principle is that: """
