import re
from typing import List, Optional


def extract_answer(
    small_llm_model,
    model_text_output: str,
    valid_labels: List[str] = None,
    debug: bool = False,
    question: Optional[str] = None,
) -> str:
    """
    Extração robusta em 3 tiers:
      Tier 1 — padrões de alta confiança (estrutura explícita)
      Tier 2 — padrões de confiança moderada (sem ambiguidade)
      Tier 3 — fallback LLM Judge
    """
    if valid_labels is None:
        valid_labels = ["A", "B", "C", "D", "E"]

    labels_group = "".join(valid_labels)
    pat = f"[{labels_group}]"

    if not model_text_output:
        return "N/A"

    text = model_text_output.strip()

    if "```python" in text or "def solution" in text:
        if debug:
            print("⚠ [CODE DETECTED] Enviando para LLM Judge.")
        return _llm_judge(small_llm_model, text, valid_labels, debug, question)

    # ── TIER 1: Padrões de alta confiança ──────────────────────────────────────
    tier1_patterns = [
        # \boxed{A}  (LaTeX)
        rf"\\boxed\s*\{{\s*({pat})\s*\}}",
        # Final/Correct Answer: (A) | A | [A]
        rf"(?:Final|Correct)\s+Answer\s*[:\-]?\s*(?:is\s+)?(?:Option\s+)?[\(\[]?({pat})[\)\]]?(?=\s|\.|,|!|\?|$)",
        # The (correct) answer/option/choice is (A) | A
        rf"[Tt]he\s+(?:correct\s+)?(?:answer|option|choice)\s+is\s*(?:Option\s+)?[\(\[]?({pat})[\)\]]?(?=\s|\.|,|!|\?|$)",
        # Therefore/Thus/Hence/So … (A)
        rf"(?:Therefore|Thus|Hence|So)[,.]?\s+(?:the\s+(?:correct\s+)?(?:answer|option|choice)\s+is\s*)?[\(\[]?({pat})[\)\]]?(?=\s|\.|,|!|\?|$)",
        # "3. Answer: A" ou "Answer: (A)"
        rf"(?:^|\n)\s*(?:\d+\.\s*)?Answer\s*[:\-]\s*[\(\[]?({pat})[\)\]]?(?=\s|$|\.|,)",
        # "Resposta: A" (português)
        rf"(?:^|\n)\s*(?:\d+\.\s*)?Resposta\s*[:\-]\s*[\(\[]?({pat})[\)\]]?(?=\s|$|\.|,)",
        # Linha isolada com apenas a letra: "D" ou "(D)"
        rf"(?:^|\n)\s*[\(\[]?({pat})[\)\]]?\s*(?:\n|$)",
    ]

    for pattern in tier1_patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))
        if matches:
            candidate = matches[-1].group(1).upper()
            if candidate in valid_labels:
                if debug:
                    print(f"✓ [TIER 1] '{pattern[:60]}...' -> {candidate}")
                return candidate

    # ── TIER 2: Padrões de confiança moderada (só aceita se não-ambíguo) ───────
    tier2_patterns = [
        # "I would choose A" / "I'll go with B"
        rf"(?:I\s+(?:would|will)\s+(?:choose|go\s+with|select|pick)|[Cc]hoosing|[Ss]electing)\s+(?:[Oo]ption\s+)?[\(\[]?({pat})[\)\]]?",
        # "option A is correct" / "alternative B is right"
        rf"(?:[Oo]ption|[Aa]lternative|[Cc]hoice)\s+[\(\[]?({pat})[\)\]]?\s+(?:is\s+)?(?:correct|right|the\s+answer)",
        # **A** em negrito isolado
        rf"\*\*({pat})\*\*",
    ]

    tier2_candidates = []
    for pattern in tier2_patterns:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            c = m.group(1).upper()
            if c in valid_labels:
                tier2_candidates.append(c)

    if tier2_candidates:
        seen, unique = set(), []
        for c in tier2_candidates:
            if c not in seen:
                seen.add(c)
                unique.append(c)

        if len(unique) == 1:
            if debug:
                print(f"✓ [TIER 2] Único candidato -> {unique[0]}")
            return unique[0]

        most_common = max(set(tier2_candidates), key=tier2_candidates.count)
        if tier2_candidates.count(most_common) > 1:
            if debug:
                print(
                    f"~ [TIER 2] Candidato mais frequente entre {unique} -> {most_common}"
                )
            return most_common

        if debug:
            print(f"~ [TIER 2] Ambíguo {unique}, enviando para LLM Judge.")

    # ── TIER 3: LLM Judge ──────────────────────────────────────────────────────
    if debug:
        print("✗ [TIER 1/2] Sem match. Chamando LLM Judge.")
    return _llm_judge(small_llm_model, text, valid_labels, debug, question)


def _llm_judge(
    small_llm_model,
    text: str,
    valid_labels: List[str],
    debug: bool = False,
    question: Optional[str] = None,
) -> str:
    labels_str = ", ".join(valid_labels)
    valid_set = set(valid_labels)

    context_block = ""
    if question:
        context_block = f"""
### CONTEXT (original question with options — use to infer the letter):
{question}
"""

    prompt = f"""You are an Answer Extraction Bot. Your ONLY job is to identify which option letter the following "Model Output" concluded is correct.
{context_block}
### Model Output to Analyze:
{text}

### Instructions:
1. Look for an explicit answer (e.g., "Answer: A", "The answer is B").
2. If no explicit letter is found, match the conclusion of the "Model Output" against the options in the Context and infer the letter.
   - Example: Context has "(A) 5  (B) 10" and Model Output says "The result is 10" → output B.
3. Do NOT solve the problem yourself. Trust only the "Model Output".
4. If the model output is genuinely unclear or refuses to answer, output UNKNOWN.

Output format: A single letter from ({labels_str}), or UNKNOWN. No other text."""

    try:
        response = small_llm_model.invoke(prompt)
        output = response.content if hasattr(response, "content") else str(response)
        clean = output.strip().upper()

        if "UNKNOWN" in clean:
            if debug:
                print("~ [LLM JUIZ] Retornou UNKNOWN.")
            return "N/A"

        letters = [c for c in clean if c in valid_set]

        if len(letters) == 1:
            if debug:
                print(f"✓ [LLM JUIZ] Inferido: {letters[0]}")
            return letters[0]

        if len(letters) > 1:
            if debug:
                print(f"~ [LLM JUIZ] Múltiplas letras {letters}, usando a primeira.")
            return letters[0]

        return "N/A"

    except Exception as e:
        if debug:
            print(f"✗ [ERRO JUIZ] {e}")
        return "N/A"
