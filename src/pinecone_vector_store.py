import os
from dotenv import load_dotenv
import sys
import tempfile
import streamlit as st
import json
import traceback
import time
import uuid
from datetime import datetime

# 環境変数のロード
load_dotenv()

# 必要なライブラリのインポート
from langchain_openai import OpenAIEmbeddings

# 固定のコレクション名
PINECONE_NAMESPACE = "ask_the_doc_collection"

class PineconeVectorStore:
    def __init__(self):
        """PineconeベースのベクトルストアをStreamlit上で初期化"""
        try:
            print("Pineconeベクトルストアの初期化を開始します...")
            
            # Pineconeクライアントが既にセッションにあれば再利用
            if 'pinecone_client' in st.session_state and st.session_state.pinecone_client:
                self.pinecone_client = st.session_state.pinecone_client
                print("セッション状態からPineconeクライアントを取得しました")
            else:
                # Pineconeクライアントの初期化
                from components.pinecone_client import PineconeClient
                self.pinecone_client = PineconeClient()
                st.session_state.pinecone_client = self.pinecone_client
                print("Pineconeクライアントを新規に初期化しました")
            
            # クライアントの接続状態を確認
            self.available = getattr(self.pinecone_client, 'available', False)
            
            # 名前空間設定
            self.namespace = PINECONE_NAMESPACE
            self.pinecone_client.namespace = self.namespace

            # 埋め込みモデルの設定
            self.embeddings = OpenAIEmbeddings()
            print("PineconeVectorStoreの初期化が完了しました")
            
        except Exception as e:
            print(f"PineconeVectorStoreの初期化中にエラーが発生しました: {e}")
            print(traceback.format_exc())
            self.available = False
            raise

    def add_documents(self, documents):
        """ドキュメントを追加"""
        return self.upsert_documents(documents)

    def update_documents(self, documents):
        """ドキュメントを更新"""
        return self.upsert_documents(documents)

    def upsert_documents(self, documents, ids=None):
        """ドキュメントを追加または更新"""
        if not self.available:
            print("Pineconeが利用できないため、ドキュメントをアップロードできません")
            return False
        
        try:
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
                ids = [f"doc_{i}_{uuid.uuid4()}" for i in range(len(documents))]
            
            # 埋め込みベクトルの生成
            print(f"{len(texts)}件のドキュメントの埋め込みベクトルを生成中...")
            embeddings = self.embeddings.embed_documents(texts)
            
            # Pineconeに登録するベクトル配列を作成
            vectors = []
            for i, (id_, text, metadata, embedding) in enumerate(zip(ids, texts, metadatas, embeddings)):
                # メタデータにテキスト内容を含める
                metadata['text'] = text
                
                # ベクトルエントリを作成
                vector_entry = {
                    "id": id_,
                    "values": embedding,
                    "metadata": metadata
                }
                vectors.append(vector_entry)
                
                # 100件ごとにバッチアップサート（Pineconeの制限対応）
                if (i + 1) % 100 == 0 or i == len(ids) - 1:
                    batch_vectors = vectors[:100]
                    vectors = vectors[100:]
                    
                    # REST APIまたはSDKでアップサート
                    self._upsert_batch(batch_vectors)
                    print(f"{i+1}/{len(ids)}件のベクトル登録完了")
            
            print(f"合計{len(ids)}件のドキュメントを正常にPineconeにアップロードしました")
            return True
        except Exception as e:
            print(f"ドキュメントのアップロード中にエラーが発生しました: {e}")
            print(traceback.format_exc())
            return False

    def _upsert_batch(self, vectors):
        """ベクトルのバッチをPineconeにアップロード"""
        try:
            # 公式SDKがある場合はSDKを使用
            if hasattr(self.pinecone_client, 'index'):
                self.pinecone_client.index.upsert(
                    vectors=vectors,
                    namespace=self.namespace
                )
                return True
                
            # REST APIでアップサート
            api_url = f"https://api.pinecone.io/vectors/upsert/{self.pinecone_client.index_name}"
            data = {
                "vectors": vectors,
                "namespace": self.namespace
            }
            
            response = self.pinecone_client._make_request(
                method="POST",
                url=api_url,
                json_data=data
            )
            
            if response and response.status_code in [200, 201, 202]:
                return True
            else:
                print(f"バッチアップサートエラー: {getattr(response, 'status_code', 'N/A')} - {getattr(response, 'text', 'No response')}")
                return False
        except Exception as e:
            print(f"バッチアップサート中のエラー: {e}")
            print(traceback.format_exc())
            return False

    def delete_documents(self, ids):
        """ドキュメントを削除"""
        if not self.available or not ids:
            return False
            
        try:
            # 公式SDKがある場合はSDKを使用
            if hasattr(self.pinecone_client, 'index'):
                self.pinecone_client.index.delete(
                    ids=ids,
                    namespace=self.namespace
                )
                print(f"{len(ids)}件のドキュメントを削除しました")
                return True
                
            # REST APIで削除
            api_url = f"https://api.pinecone.io/vectors/delete/{self.pinecone_client.index_name}"
            data = {
                "ids": ids,
                "namespace": self.namespace
            }
            
            response = self.pinecone_client._make_request(
                method="POST",
                url=api_url,
                json_data=data
            )
            
            if response and response.status_code in [200, 201, 202]:
                print(f"{len(ids)}件のドキュメントを削除しました")
                return True
            else:
                print(f"ドキュメント削除エラー: {getattr(response, 'status_code', 'N/A')} - {getattr(response, 'text', 'No response')}")
                return False
        except Exception as e:
            print(f"ドキュメント削除中のエラー: {e}")
            print(traceback.format_exc())
            return False

    def get_documents(self, ids=None):
        """ドキュメントを取得"""
        if not self.available:
            return {"ids": [], "documents": [], "metadatas": []}
            
        try:
            if ids is None:
                # 全てのドキュメントを取得（実際にはPineconeで全取得は難しいので、フェッチはせず空のリストを返す）
                print("Pineconeでは全てのドキュメントを一度に取得することはできません")
                return {"ids": [], "documents": [], "metadatas": []}
            
            # IDsが指定されている場合は、それらを取得
            # 公式SDKがある場合はSDKを使用
            results = {"ids": [], "documents": [], "metadatas": []}
            
            if hasattr(self.pinecone_client, 'index'):
                fetch_response = self.pinecone_client.index.fetch(
                    ids=ids,
                    namespace=self.namespace
                )
                
                if fetch_response and fetch_response.vectors:
                    for id_, vector in fetch_response.vectors.items():
                        results["ids"].append(id_)
                        results["documents"].append(vector.metadata.get("text", ""))
                        results["metadatas"].append({k: v for k, v in vector.metadata.items() if k != "text"})
                
                return results
                
            # REST APIで取得
            api_url = f"https://api.pinecone.io/vectors/fetch/{self.pinecone_client.index_name}"
            params = {
                "ids": ids,
                "namespace": self.namespace
            }
            
            response = self.pinecone_client._make_request(
                method="GET",
                url=api_url,
                params=params
            )
            
            if response and response.status_code == 200:
                try:
                    data = response.json()
                    if "vectors" in data:
                        for id_, vector in data["vectors"].items():
                            results["ids"].append(id_)
                            results["documents"].append(vector.get("metadata", {}).get("text", ""))
                            
                            # テキスト以外のメタデータを取得
                            metadata = {k: v for k, v in vector.get("metadata", {}).items() if k != "text"}
                            results["metadatas"].append(metadata)
                
                    return results
                except Exception as e:
                    print(f"レスポンスのJSON解析エラー: {e}")
            
            return results
        except Exception as e:
            print(f"ドキュメント取得中のエラー: {e}")
            print(traceback.format_exc())
            return {"ids": [], "documents": [], "metadatas": []}

    def search(self, query, n_results=5, filter_conditions=None):
        """クエリに基づいてドキュメントを検索"""
        if not self.available:
            return {"ids": [[]], "documents": [[]], "distances": [[]], "metadatas": [[]]}
            
        try:
            # クエリの埋め込みを生成
            query_embedding = self.embeddings.embed_query(query)
            
            # フィルター条件の処理
            filter_dict = {}
            if filter_conditions:
                for key, value in filter_conditions.items():
                    if key and value:
                        filter_dict[key] = {"$eq": value}
            
            # 公式SDKがある場合はSDKを使用
            if hasattr(self.pinecone_client, 'index'):
                query_response = self.pinecone_client.index.query(
                    vector=query_embedding,
                    top_k=n_results,
                    include_metadata=True,
                    filter=filter_dict if filter_dict else None,
                    namespace=self.namespace
                )
                
                # ChromaDB形式の結果に変換
                results = {
                    "ids": [[]],
                    "documents": [[]],
                    "distances": [[]],
                    "metadatas": [[]]
                }
                
                if query_response and query_response.matches:
                    for match in query_response.matches:
                        results["ids"][0].append(match.id)
                        results["documents"][0].append(match.metadata.get("text", ""))
                        results["distances"][0].append(1.0 - match.score)  # cosine類似度を距離に変換
                        
                        # テキスト以外のメタデータを取得
                        metadata = {k: v for k, v in match.metadata.items() if k != "text"}
                        results["metadatas"][0].append(metadata)
                
                return results
                
            # REST APIで検索
            api_url = f"https://api.pinecone.io/query/{self.pinecone_client.index_name}"
            data = {
                "vector": query_embedding,
                "topK": n_results,
                "includeMetadata": True,
                "filter": filter_dict if filter_dict else None,
                "namespace": self.namespace
            }
            
            response = self.pinecone_client._make_request(
                method="POST",
                url=api_url,
                json_data=data
            )
            
            # ChromaDB形式の結果に変換
            results = {
                "ids": [[]],
                "documents": [[]],
                "distances": [[]],
                "metadatas": [[]]
            }
            
            if response and response.status_code == 200:
                try:
                    data = response.json()
                    if "matches" in data:
                        for match in data["matches"]:
                            results["ids"][0].append(match.get("id", ""))
                            results["documents"][0].append(match.get("metadata", {}).get("text", ""))
                            results["distances"][0].append(1.0 - match.get("score", 0))  # cosine類似度を距離に変換
                            
                            # テキスト以外のメタデータを取得
                            metadata = {k: v for k, v in match.get("metadata", {}).items() if k != "text"}
                            results["metadatas"][0].append(metadata)
                except Exception as e:
                    print(f"検索結果の処理中にエラーが発生しました: {e}")
            
            return results
        except Exception as e:
            print(f"検索中にエラーが発生しました: {e}")
            print(traceback.format_exc())
            return {"ids": [[]], "documents": [[]], "distances": [[]], "metadatas": [[]]}

    def count(self):
        """ドキュメント数を取得"""
        if not self.available:
            return 0
            
        try:
            # 公式SDKがある場合はSDKを使用
            if hasattr(self.pinecone_client, 'index'):
                stats = self.pinecone_client.index.describe_index_stats()
                if stats and hasattr(stats, "namespaces") and self.namespace in stats.namespaces:
                    count = stats.namespaces[self.namespace].vector_count
                    print(f"Pineconeコレクションには{count}件のドキュメントが登録されています")
                    return count
                return 0
                
            # REST APIで統計を取得
            api_url = f"https://api.pinecone.io/describe_index_stats/{self.pinecone_client.index_name}"
            
            response = self.pinecone_client._make_request(
                method="GET",
                url=api_url
            )
            
            if response and response.status_code == 200:
                try:
                    data = response.json()
                    if "namespaces" in data and self.namespace in data["namespaces"]:
                        count = data["namespaces"][self.namespace].get("vector_count", 0)
                        print(f"Pineconeコレクションには{count}件のドキュメントが登録されています")
                        return count
                except Exception as e:
                    print(f"レスポンスの解析中にエラーが発生しました: {e}")
            
            return 0
        except Exception as e:
            print(f"ドキュメント数の取得中にエラーが発生しました: {e}")
            print(traceback.format_exc())
            return 0 