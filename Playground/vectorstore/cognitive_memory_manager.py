import os
import time
from typing import List

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document


def build_metadata_semantic(index, row, doc_id):
    """Metadata para memória semântica (com score)."""
    score_val = 0
    if pd.notna(row.get('quality_score')) and row['quality_score'] != "N/A":
        try:
            score_val = int(row['quality_score'])
        except:
            score_val = 0

    return {
        "chroma_id": doc_id,
        "original_id": str(row['id']),
        "question": str(row['question']),
        "correct_answer": str(row.get('correct_answer', 'None')),
        "score": score_val,
        "frequency": 1,
        "scientific_fact": row['scientific_fact'],
        "origin": row.get('origin', 'train'),
        "type": "semantic_memory",
    }


class CognitiveMemoryManager:
    def __init__(self, db_path, embedding_model, alpha=0.1, decay_lambda=0.99, forget_threshold=-0.6):
        self.db_path = db_path
        self.embedding_model = embedding_model

        # Hiperparâmetros de Memória
        self.ALPHA = alpha
        self.LAMBDA = decay_lambda
        self.THETA_FORGET = forget_threshold

        self.vectorstore = None
        self._collection = None

    def init_from_dataframe(self, df, content_col, id_prefix, metadata_func, reset_db=False):
        if os.path.exists(self.db_path):
            print(f"📂 Carregando memória existente de: {self.db_path}")
            self.vectorstore = Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embedding_model,
                collection_metadata={"hnsw:space": "cosine"}
            )
            if reset_db:
                self.vectorstore.delete_collection()
                print(f"🧹 [Sistema] Memória existente resetada.")
            else:
                print(f"✅ Memória carregada com {self.vectorstore._collection.count()} itens.")
                self._collection = self.vectorstore._collection
                return self

        print(f"🔨 Construindo nova memória cognitiva com {len(df)} itens...")

        docs, ids = [], []
        for index, row in df.iterrows():
            doc_id = f"{id_prefix}_{index}"

            meta = metadata_func(index, row, doc_id)
            initial_score = meta.get('score', 0)
            meta['initial_score'] = initial_score

            if 'chroma_id' not in meta:
                meta['chroma_id'] = doc_id

            docs.append(Document(page_content=str(row[content_col]), metadata=meta))
            ids.append(doc_id)

        self.vectorstore = Chroma.from_documents(
            documents=docs,
            ids=ids,
            embedding=self.embedding_model,
            persist_directory=self.db_path,
            collection_metadata={"hnsw:space": "cosine"}
        )
        self._collection = self.vectorstore._collection
        print(f"✅ Memória inicializada com {self._collection.count()} itens.")
        return self

    def add_new_memories(self, new_facts: List[dict], id_prefix='generated'):
        if not new_facts:
            return

        docs, ids = [], []
        current_count = self._collection.count()

        for i, fact_data in enumerate(new_facts):
            doc_id = f"{id_prefix}_{current_count + i}_{int(time.time())}"

            metadata = fact_data['metadata'].copy()
            metadata['chroma_id'] = doc_id

            docs.append(Document(
                page_content=fact_data['content'],
                metadata=metadata
            ))
            ids.append(doc_id)

        self.vectorstore.add_documents(documents=docs, ids=ids)

    def update_memories_feedback(self, used_ids: list, feedback_scores: list):
        if not used_ids:
            return

        try:
            current_data = self._collection.get(ids=used_ids)
            metadatas = current_data['metadatas']
        except Exception as e:
            print(f"⚠️ [Erro] Não foi possível obter metadados para IDs fornecidos: {e}")
            return

        updates_meta, updates_ids = [], []

        for i, meta in enumerate(metadatas):
            if meta is None:
                continue

            S_t = meta.get('score', 0.0)
            c_t = feedback_scores[i]

            # Fórmula EMA
            S_next = (1 - self.ALPHA) * S_t + self.ALPHA * c_t
            S_next = max(-1.0, min(1.0, S_next))

            meta['score'] = float(S_next)
            meta['frequency'] = meta.get('frequency', 0) + 1

            updates_meta.append(meta)
            updates_ids.append(used_ids[i])

        if updates_ids:
            self._collection.update(ids=updates_ids, metadatas=updates_meta)

    def apply_decay_cycle(self, decay_threshold: float = 0.0):
        """
        Aplica decay exponencial apenas em memórias com score abaixo de um limiar.

        Fórmula de Decay: S(t+1) = lambda * S(t)
        """
        all_data = self._collection.get()
        ids_to_update, metas_to_update = [], []
        ids_to_delete = []

        for doc_id, meta in zip(all_data['ids'], all_data['metadatas']):
            S_t = meta.get('score', 0.0)

            # Aplica decay
            S_next = S_t * self.LAMBDA

            # Verificação de Morte (Esquecimento)
            if S_next < self.THETA_FORGET:
                ids_to_delete.append(doc_id)
            else:
                meta['score'] = float(S_next)
                ids_to_update.append(doc_id)
                metas_to_update.append(meta)

        # Batch Updates
        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)
            print(f"🗑️ Ciclo de Limpeza: {len(ids_to_delete)} memórias esquecidas (Score < {self.THETA_FORGET}).")

    def retrieve_ranked_memories(self, query: str, k: int = 3, threshold: float = 0.24,
                                 semantic_weight: float = 0.7, show_scores: bool = False):
        initial_fetch_k = k * 5
        results = self.vectorstore._similarity_search_with_relevance_scores(query, k=initial_fetch_k)

        ranked = []
        for doc, cos_sim in results:
            if cos_sim < threshold:
                continue

            memory_strength = doc.metadata.get("score", 0.0)
            final_score = (cos_sim * semantic_weight) + (memory_strength * (1 - semantic_weight))

            ranked.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "doc_id": doc.metadata.get("chroma_id"),
                "similarity": cos_sim,
                "memory_strength": memory_strength,
                "final_score": final_score
            })

        ranked.sort(key=lambda x: x["final_score"], reverse=True)
        final = ranked[:k]

        if not final:
            formatted_context = "No relevant scientific facts were found."
        else:
            parts = ['Here is a list of scientific facts that might help you answer the question:']
            for i, m in enumerate(final):
                if show_scores:
                    parts.append(
                        f"[{i+1}] (Sim: {m['similarity']:.2f}, Mem: {m['memory_strength']:.2f}, "
                        f"Final: {m['final_score']:.2f}) {m['content']}"
                    )
                else:
                    parts.append(f"{i+1}. {m['content']}")
            formatted_context = "\n".join(parts)

        return final, formatted_context
