"""
Templates de prompt para resposta a questões de múltipla escolha.
"""

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


class StructuredAnswer(BaseModel):
    reasoning: str = Field(
        description="A brief description of the reasoning process that led to the choice of the alternative"
    )
    alternative: str = Field(
        description="Only the letter of the chosen alternative (e.g. A, B, C, D, or E — use whichever letter appears in the question)"
    )


answer_json_parser = JsonOutputParser(pydantic_object=StructuredAnswer)
str_parser = StrOutputParser()

# alias mantido para compatibilidade (aponta para o parser de resposta)
json_parser = answer_json_parser

FORMATTED_TEMPLATE = PromptTemplate(
    template="""Instruction: Your goal is to answer the target multiple-choice question below.

Apply relevant scientific principles - fundamental definitions and laws from your knowledge.

Use scientific principles to ground your facts and guide your logic.
### TARGET QUESTION:
{question}

{format_instructions}
""",
    input_variables=["question"],
    partial_variables={"format_instructions": json_parser.get_format_instructions()},
)

SIMPLE_TEMPLATE = PromptTemplate(
    template="""Instruction: Your goal is to answer the target multiple-choice question below.

Apply relevant principles - fundamental definitions and laws from your knowledge.

Structure your response strictly as:
1. **Reasoning:** Explain the step-by-step logic to reach the correct answer.
2. **Answer:** State only the correct option letter (use whichever letter appears in the question, e.g. A, B, C, D, or E).

### TARGET QUESTION:
{question}
""",
    input_variables=["question"],
)


# CRITIQUE #######################################################################################################################################


class FactsUsed(BaseModel):
    fact: int = Field(description="The scientific fact id used to answer the question")
    score: float = Field(
        description="The importance score of the fact to the question. This value should be between -1 and 1, where -1 indicates that the fact is misleading for answering the question, 0 indicates that the fact is irrelevant, and 1 indicates that the fact is crucial for answering the question correctly"
    )


class StructuredCritique(BaseModel):
    critique: str = Field(description="Your critique ")
    scored_facts: list[FactsUsed] = Field(
        description="The list of scientific facts (with their id and importance scores) used to answer the question"
    )


critique_json_parser = JsonOutputParser(pydantic_object=StructuredCritique)


STRUCTURED_CRITIQUE_TEMPLATE = PromptTemplate(
    template="""Instruction: You had answer the question, now your task is to provide a self-critique that identifies the specific scientific misconception in the answer and explains why it is incorrect

    Question: {question}

    Knowledge used to answer the question:
    {semantic_facts}

    Your Answer: {answer} (This answer is {feedback})

    {format_instructions}
    """,
    input_variables=["question", "answer", "feedback", "semantic_facts"],
    partial_variables={
        "format_instructions": critique_json_parser.get_format_instructions()
    },
)

CRITIQUE_TEMPLATE = PromptTemplate(
    template="""Instruction: You had answer the question, now your task is to provide a self-critique that identifies the specific scientific misconception in the answer and explains why it is incorrect

    Question: {question}

    Knowledge used to answer the question: 
    {semantic_facts}
    
    Your Answer: {answer} (This answer is {feedback})
    """,
    input_variables=["question", "answer", "feedback", "semantic_facts"],
)
