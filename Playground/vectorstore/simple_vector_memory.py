import os

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document


def build_metadata_semantic(index, row, doc_id):
    """Metadata para memória semântica (fatos científicos)."""
    return {
        "chroma_id": doc_id,
        "original_id": str(row['id']),
        "question": str(row['question']),
        "correct_answer": str(row['correct_answer']) if pd.notna(row.get('correct_answer')) else "None",
        "scientific_fact": row['scientific_fact'],
        "origin": row.get('origin', 'train'),
        "type": "semantic_memory",
    }


def build_metadata_reflection(index, row, doc_id):
    """Metadata para memória reflexiva (sem score)."""
    return {
        "chroma_id": doc_id,
        "original_id": str(row['id']),
        "question": str(row['question']),
        "correct_answer": str(row.get('correct_answer', 'None')),
        "reflection": row['clean_reasoning'],
        "origin": row.get('origin', 'train'),
        "type": "reflection_memory",
    }


class SimpleVectorMemory:
    def __init__(self, db_path, embedding_model):
        self.db_path = db_path
        self.embedding_model = embedding_model
        self.vectorstore = None
        self._collection = None

    def init_from_dataframe(self, df, content_col, id_prefix, metadata_func, reset_db=False):
        """
        Cria ou carrega o banco de dados a partir de um DataFrame.
        """
        if reset_db and os.path.exists(self.db_path):
            print(f"🧹 [Sistema] Reset total solicitado. Apagando {self.db_path}...")
            try:
                if self.vectorstore:
                    self.vectorstore.delete_collection()
                    self.vectorstore = None
            except Exception as e:
                print(f"⚠️ Aviso ao limpar diretório: {e}")

        if os.path.exists(self.db_path):
            print(f"📂 Carregando base de reflexão de: {self.db_path}")
            self.vectorstore = Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embedding_model,
                collection_metadata={"hnsw:space": "cosine"}
            )
            print(f"✅ Base carregada com {self.vectorstore._collection.count()} itens.")

        else:
            print(f"🔨 Construindo novo índice de reflexão (COSINE) com {len(df)} itens...")

            docs, ids = [], []
            for index, row in df.iterrows():
                doc_id = f"{id_prefix}_{index}"
                meta = metadata_func(index, row, doc_id)

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
        return self

    def retrieve_memories(self, query: str, k: int = 3, threshold: float = 0.0):
        results = self.vectorstore.similarity_search_with_score(query, k=k)

        final_results = []
        for doc, distance in results:
            similarity = 1.0 - distance

            if similarity < threshold:
                continue

            final_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "doc_id": doc.metadata.get("chroma_id"),
                "similarity": similarity
            })
        return final_results
