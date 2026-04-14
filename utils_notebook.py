"""
Utilities for Cognitive Memory and Semantic Memory Notebooks

This module contains shared functions used by both cognitive_memory.ipynb 
and semantic_cognitive.ipynb notebooks, including:
- Answer extraction
- Text cleaning
- Memory metrics calculation
- Score parsing
- Context formatting
"""

from typing import Optional, List
import ast
import re
import spacy

# Initialize spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

from spacy.matcher import Matcher


class SemanticCleaner:
    def __init__(self):
        self.matcher = Matcher(nlp.vocab)
        self.matcher.add("PASSIVE_CONTINUOUS", [[{"POS": "AUX"}, {"TEXT": "being"}, {"TAG": "VBN"}]])
        
        self.filler_patterns = [
            r"The correct (scientific\s+)?fact (is|states)(\s+that)?", 
            r"The (actual\s+)?scientific truth is that",
            r"A more (accurate|precise) (fact|statement) is that",
            r"To be (scientifically\s+)?accurate,?",
            
            r"Instead of saying.*?, it is (better|more accurate) to say (that)?",
            r"The error in the (old|previous) fact was.*?, and the (new|correct) fact is (that)?",
            
            r"(It should be noted|Note) that\s+",
            r"One must (understand|realize) that\s+",
            r"The (basic|fundamental) idea here is that\s+",
            r"This (is because|occurs because)\s+",

            r"The (scientific\s+)?fact (present\s+in\s+this\s+text|contained\s+herein)\s+is\s+that\s+",
            
            r"This process (follows|is\s+based\s+on)\s+the\s+principle\s+of\s+.*?(where|which)\s+",
            
            r"The (key\s+)?scientific (point|fact) (being\s+made|discussed)\s+is\s+",
            r"The (text|paragraph|author) (explains|describes|states|illustrates) (how|that)\s+",
            r"This (explanation|description) (shows|demonstrates) (how|that)\s+",

            r"The (scientific\s+)?fact (presented\s+is|is|to consider is)(\s+that)?",
            r"The (scientifically\s+)?accurate fact (is|should be)(\s+that)?",
            r"Based on this (critique|analysis|reasoning),?",
            r"The (correct\s+)?fact (that connects|explaining|justifying).*?is(\s+that)?",
            r"In conclusion,?",
            r"This (means|implies|suggests) that",
            r"The (correct\s+)?principle is that",

            r"(Therefore,?\s*)?The correct (answer|option)(\s+to this question)?\s+(is|can be found by understanding)\s*([A-D][\)\.])?",
            r"Option\s+[A-D]\s+is\s+correct\.?",
            r"^\s*[A-D][\)\.]\s+", 
            r"\s+[A-D][\)\.]\s+",  
            
            r"\(\s*option\s*([A-D])?\s*\)?",
            r"\(\s*[A-D]\s*\)", # (A), ( A )
            r"\[\s*[A-D]\s*\]", # [A]
            
            r"The (scientific\s+)?principles? (applied|at play|at work|involved)(\s+in this reasoning)?(\s+here)?\s+(is|are|include)(\s+that)?",
            r"The (scientific\s+)?concept (applied|at play|at work|involved)(\s+in this reasoning)?(\s+here)?\s+(is|are|include)(\s+that)?",
            
            r"In this case,?",
            r"Therefore,?",
            r"It is important to note that",
            r"This phenomenon can be explained by",
            r"As mentioned above,?",
            r"To determine .*?, we need to",
            r"Specifically,?"
        ]

    def remove_fillers(self, text):
        cleaned_text = text
        for pattern in self.filler_patterns:
            # Substitui por espaço único para evitar colagem
            cleaned_text = re.sub(pattern, " ", cleaned_text, flags=re.IGNORECASE)
        return cleaned_text

    def fix_fragments_smart(self, text):
        """
        Conserta fragmentos APENAS se a frase não tiver um verbo principal (ROOT).
        Usa spaCy para análise sintática.
        """
        # Se o texto for muito curto, ignora
        if len(text.split()) < 3: return text

        pattern = r"^([A-Z][a-zA-Z0-9\s\-]+),?\s*(which|that|who)\s+(is|are|refers to)"
        match = re.search(pattern, text)
        
        if match:
            doc = nlp(text)
            
            has_root_verb = False
            for token in doc:
                if token.dep_ == "ROOT" and token.pos_ == "VERB":

                    subjects = [child.text.lower() for child in token.children if child.dep_ == "nsubj"]
                    if "which" in subjects or "that" in subjects or "who" in subjects:

                         pass 
                    else:
                        has_root_verb = True
                        break
            
            if has_root_verb:
                return text
            
            end_pos = match.end()
            new_start = f"{match.group(1)} {match.group(3)}"
            return new_start + text[end_pos:]
            
        return text

    def generalize_entities(self, doc):
        new_tokens = []
        for token in doc:
            if token.ent_type_ == "PERSON":
                new_tokens.append("the subject") 
            else:
                new_tokens.append(token.text)
        return nlp(" ".join(new_tokens))

    def simplify_verbs(self, doc):
        matches = self.matcher(doc)
        if not matches:
            return doc.text
        tokens = [t.text for t in doc]
        for match_id, start, end in reversed(matches):
            span = doc[start:end]
            main_verb = span[2]
            simple_verb = main_verb.lemma_ + "s"
            tokens[start:end] = [simple_verb]
        return " ".join(tokens)

    def polish_text(self, text):
        # Limpeza básica
        text = re.sub(r'\s+([,.!?:;])', r'\1', text) # Espaços antes de pontuação
        text = text.replace(" - ", "-")
        text = text.replace(" 's", "'s")
        text = re.sub(r'\s+', ' ', text).strip()
        
        if text.startswith("Is ") and "?" not in text:
            text = text[3:] # Remove "Is "

        if text: text = text[0].upper() + text[1:]
        text = re.sub(r'([.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
        
        return text

    def clean(self, raw_text):
        if not raw_text: return ""
        
        #  Regex Clean
        step1 = self.remove_fillers(raw_text)
        
        # Fix Fragments Inteligente (spaCy inside)
        step1 = self.fix_fragments_smart(step1.strip())
        
        # NLP Process
        doc = nlp(step1)
        doc = self.generalize_entities(doc)
        step2 = self.simplify_verbs(doc)
        
        # Polimento
        final = self.polish_text(step2)
        
        return final
    
def calcular_metricas_memoria(items):
    """
    Calcula estatísticas básicas de uma lista de memórias recuperadas.
    Retorna: (contagem, média_similaridade, lista_scores_brutos)
    """
    count = len(items)
    
    if count == 0:
        return 0, 0.0, []
    
    # Extrai scores
    sim_scores = [item['similarity'] for item in items]
    avg_sim = sum(sim_scores) / count
    
    return count, avg_sim, sim_scores


def extract_knowledge_components(model_output):
    """Extrai componentes de Reasoning e Principles do output do modelo."""
    # Padrão ajustado para ser tolerante a quebras de linha e variações
    pattern = r"1\.\s*Reasoning:\s*(.*?)\s*2\.\s*Principles:\s*(.*?)\s*3\.\s*Answer:"
    match = re.search(pattern, model_output, re.DOTALL | re.IGNORECASE)
    
    if match:
        return {
            "reasoning": match.group(1).strip(),
            "principles": match.group(2).strip()
        }
    return {"reasoning": "", "principles": ""}


def format_choices(choices_str):
    """Converte a string de dicionário do CSV em um formato legível A) ... B) ..."""
    try:
        choices = ast.literal_eval(choices_str)
        formatted = []
        for label, text in zip(choices['label'], choices['text']):
            formatted.append(f"{label}) {text}")
        return "\n".join(formatted)
    except Exception as e:
        return str(choices_str)


def parse_simple_score(response_text):
    """
    Parseia o output do score_chain para extrair o score (-1, 0, ou 1).
    """
    try:
        # Tenta achar "Score: X" ou apenas o número no final
        reflection = response_text
        if "Score:" in response_text:
            parts = response_text.split("Score:")
            reflection = parts[0].replace("Reflection:", "").strip()
            score_text = parts[1]
        else:
            score_text = response_text

        # Regex busca -1, 0 ou 1
        matches = re.findall(r'(?<!\w)-?\d+', score_text)
        
        if matches:
            score = int(matches[-1]) # Pega o último número encontrado
            # Limites
            if score < -1: score = -1
            if score > 1: score = 1
        else:
            score = 0 # Default neutro (O fato não foi 'usado')
            
        return reflection, score
    except:
        return "Error parsing", 0


def make_question(problemdata,get_answer=False,inline=False):

    question = problemdata['question']
    correct_answer = problemdata['answerKey']
    choices = ast.literal_eval(problemdata['choices'])

    number = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9}
    alternative_correct_text = choices['text'][number[correct_answer]]
    correct_answer_text = f"'({correct_answer})' {alternative_correct_text}"

    text = choices['text']
    labels = choices['label']


    if inline:
        formatted_choices = " ".join([f"({label}) {choice}" for label, choice in zip(labels, text)])
        prompt = f"{question} {formatted_choices}"
    else:
        formatted_choices = "\n".join([f"({label}) {choice}" for label, choice in zip(labels, text)])
        prompt = f"{question}\n{formatted_choices}"

    if get_answer:
        prompt += f"\nCorrect Answer: {correct_answer_text}"
        
    return prompt,correct_answer_text



def extract_facts(llm_model, input_text):
    extraction_prompt = f"""
You are a strict fact extractor. Your task is to extract facts explicitly stated in the input text below.

**Input Text:**
"{input_text}"

**Instructions:**
1. Extract ONLY facts present in the text. Do not infer, hallucinate, or use outside knowledge.
2. Return the answer strictly as a valid JSON object: {{ "facts": ["fact string 1", "fact string 2"], "reasoning": "reasoning_found_in_text" }}
3. You should not confuse what is factual content with what is reasoning content.
4. If the text contains no factual information or assertions, return exactly the word and do not add any punctuation or extra words. 

**Response:**
"""
    
    # Invoca o modelo
    response = llm_model.invoke(extraction_prompt)
    
    # Extrai o conteúdo (lida com objetos de resposta ou strings diretas)
    content = response.content.strip() if hasattr(response, 'content') else str(response).strip()
    
    # Limpeza: Remove formatação de código Markdown se o modelo adicionar (ex: ```json ... ```)
    # if content.startswith("```"):
    #     content = content.replace("```json", "").replace("```", "").strip()

    # Verifica se a resposta é 'False' (caso nenhum fato tenha sido encontrado)
    if content == "False":
        return False

    # Tenta converter a string JSON para um dicionário Python
    try:
        return ast.literal_eval(content)
    except (ValueError, SyntaxError):
        # Caso o modelo falhe em gerar um JSON válido, retorna False por segurança
        return False
