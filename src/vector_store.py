import os
from dotenv import load_dotenv
import sys
import sqlite3
import tempfile

# 環境変数のロード
load_dotenv()

# SQLiteのバージョン確認
print(f"Using SQLite version: {sqlite3.sqlite_version}")

# chromadbのインポート
import chromadb
from langchain_openai import OpenAIEmbeddings

# 固定のコレクション名
COLLECTION_NAME = "ask_the_doc_collection"

class VectorStore:
    def __init__(self):
        """ChromaDBのベクトルストアを初期化"""
        try:
            # インメモリモードでクライアントを初期化
            self.client = chromadb.Client(
                settings=chromadb.Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                    is_persistent=False
                )
            )
            
            # コレクションの作成または取得
            try:
                self.collection = self.client.get_collection(name=COLLECTION_NAME)
                print(f"Collection '{COLLECTION_NAME}' already exists")
            except Exception:
                print(f"Creating new collection '{COLLECTION_NAME}'")
                self.collection = self.client.create_collection(
                    name=COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"}
                )
                
            # 埋め込みモデルの設定
            self.embeddings = OpenAIEmbeddings()
            print("VectorStore initialization completed successfully")
            
        except Exception as e:
            print(f"Error initializing VectorStore: {e}")
            raise

    def add_documents(self, documents):
        """ドキュメントを追加"""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [f"doc_{i}" for i in range(len(documents))]
        
        embeddings = self.embeddings.embed_documents(texts)
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Added {len(texts)} documents to collection")

    def update_documents(self, documents):
        """ドキュメントを更新"""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [f"doc_{i}" for i in range(len(documents))]
        
        embeddings = self.embeddings.embed_documents(texts)
        self.collection.update(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Updated {len(texts)} documents in collection")

    def upsert_documents(self, documents, ids=None):
        """ドキュメントを追加または更新"""
        try:
            if self.collection is None:
                print("Collection is not available")
                return
            
            # Documentオブジェクトの場合とプレーンテキストの場合の両方に対応
            if hasattr(documents[0], 'page_content'):
                # Documentオブジェクトの場合
                texts = [doc.page_content for doc in documents]
                metadatas = [doc.metadata for doc in documents]
            else:
                # プレーンテキストの場合
                texts = documents
                metadatas = [{} for _ in documents]
            
            # IDsの生成または使用
            if ids is None:
                ids = [f"doc_{i}" for i in range(len(documents))]
            
            # 埋め込みベクトルの生成
            print(f"Generating embeddings for {len(texts)} documents...")
            try:
                embeddings = self.embeddings.embed_documents(texts)
                if not embeddings or len(embeddings) != len(texts):
                    print(f"Warning: Generated {len(embeddings) if embeddings else 0} embeddings for {len(texts)} texts")
                    
                # ドキュメントごとに処理して確実に追加
                for i, (text, metadata, doc_id, embedding) in enumerate(zip(texts, metadatas, ids, embeddings)):
                    try:
                        self.collection.upsert(
                            embeddings=[embedding],
                            documents=[text],
                            metadatas=[metadata],
                            ids=[doc_id]
                        )
                    except Exception as e:
                        print(f"Error upserting document {i} (ID: {doc_id}): {e}")
                
                print(f"Successfully upserted {len(texts)} documents to collection '{COLLECTION_NAME}'")
            except Exception as e:
                print(f"Error generating embeddings: {e}")
                raise
        except Exception as e:
            print(f"Error in upsert_documents: {e}")
            raise

    def delete_documents(self, ids):
        """ドキュメントを削除"""
        try:
            if not ids:
                print("No IDs provided for deletion")
                return
                
            self.collection.delete(ids=ids)
            print(f"Deleted {len(ids)} documents from collection")
        except Exception as e:
            print(f"Error deleting documents: {e}")
            raise

    def get_documents(self, ids=None):
        """ドキュメントを取得"""
        try:
            if self.collection is None:
                print("Collection is not available")
                return {"ids": [], "documents": [], "metadatas": []}
            
            # idsが指定されていない場合はすべてのドキュメントを取得
            if ids is None:
                result = self.collection.get()
                print(f"Retrieved {len(result.get('ids', []))} documents from collection")
                return result
                
            result = self.collection.get(ids=ids)
            print(f"Retrieved {len(result.get('ids', []))} documents by IDs from collection")
            return result
        except Exception as e:
            print(f"Error getting documents: {e}")
            return {"ids": [], "documents": [], "metadatas": []}

    def search(self, query, n_results=5, filter_conditions=None):
        """
        クエリに基づいてドキュメントを検索
        
        引数:
            query: 検索クエリ
            n_results: 返す結果の数
            filter_conditions: メタデータによるフィルタリング条件の辞書 {"field": "value"}
        """
        try:
            # クエリの埋め込みを生成
            query_embedding = self.embeddings.embed_query(query)
            
            # フィルタリング条件を作成（指定されている場合）
            where = None
            where_document = None
            
            if filter_conditions:
                # ChromaDBのwhere句はAND条件で結合される
                where = {}
                
                for key, value in filter_conditions.items():
                    # 完全一致ではなく、部分一致で検索するためには複雑なクエリ構造が必要
                    # ChromaDBの制限により完全一致になる
                    if key and value:
                        where[key] = value
                
                print(f"Applying filter conditions: {where}")
            
            # 検索を実行
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                where_document=where_document
            )
            
            n_results = len(results.get('ids', [[]])[0])
            print(f"Search query '{query}' returned {n_results} results with filters: {filter_conditions}")
            
            return results
        except Exception as e:
            print(f"Error searching documents: {e}")
            return {"ids": [[]], "documents": [[]], "distances": [[]], "metadatas": [[]]}

    def count(self):
        """ドキュメント数を取得"""
        try:
            count = self.collection.count()
            print(f"Collection contains {count} documents")
            return count
        except Exception as e:
            print(f"Error counting documents: {e}")
            return 0 