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
import logging
import requests

# 環境変数のロード
load_dotenv()

# 必要なライブラリのインポート
from langchain_openai import OpenAIEmbeddings

# 固定のコレクション名
PINECONE_NAMESPACE = ""  # デフォルトの名前空間を使用

# ロガーの設定
logger = logging.getLogger('app.pinecone_vector_store')

class PineconeVectorStore:
    def __init__(self):
        """PineconeベースのベクトルストアをStreamlit上で初期化"""
        try:
            logger.info("Pineconeベクトルストアの初期化を開始します...")
            
            # APIキーとベースURLの設定
            # Streamlit Cloudのシークレットから読み込む
            self.api_key = st.secrets.get("PINECONE_API_KEY", os.environ.get('PINECONE_API_KEY'))
            self.base_url = "https://api.pinecone.io"
            self.index_name = st.secrets.get("PINECONE_INDEX", os.environ.get('PINECONE_INDEX'))
            
            if not self.api_key or not self.index_name:
                raise ValueError("PINECONE_API_KEY または PINECONE_INDEX が設定されていません")
            
            logger.info(f"環境変数: PINECONE_API_KEY={'設定済み' if self.api_key else '未設定'}")
            logger.info(f"環境変数: PINECONE_ENVIRONMENT={st.secrets.get('PINECONE_ENVIRONMENT', os.environ.get('PINECONE_ENVIRONMENT'))}")
            logger.info(f"環境変数: PINECONE_INDEX={self.index_name}")
            
            # Pineconeクライアントが既にセッションにあれば再利用
            if 'pinecone_client' in st.session_state and st.session_state.pinecone_client:
                self.pinecone_client = st.session_state.pinecone_client
                logger.info("セッション状態からPineconeクライアントを取得しました")
            else:
                # Pineconeクライアントの初期化
                from components.pinecone_client import PineconeClient
                self.pinecone_client = PineconeClient()
                st.session_state.pinecone_client = self.pinecone_client
                logger.info("Pineconeクライアントを新規に初期化しました")
            
            # クライアントの接続状態を確認
            self.available = getattr(self.pinecone_client, 'available', False)
            logger.info(f"Pineconeクライアント接続状態: {'利用可能' if self.available else '利用不可'}")
            
            # 接続状態の詳細を表示
            if not self.available:
                logger.info("\n接続状態の詳細:")
                logger.info(f"- vector_store_available: {self.available}")
                if hasattr(self.pinecone_client, 'available'):
                    logger.info(f"- pinecone_client_available: {self.pinecone_client.available}")
                if hasattr(self.pinecone_client, 'initialization_error'):
                    logger.info(f"- initialization_error: {self.pinecone_client.initialization_error}")
                if hasattr(self.pinecone_client, 'temporary_failure'):
                    logger.info(f"- temporary_failure: {self.pinecone_client.temporary_failure}")
                if hasattr(self.pinecone_client, 'failed_attempts'):
                    logger.info(f"- failed_attempts: {self.pinecone_client.failed_attempts}")
                if hasattr(self.pinecone_client, 'is_streamlit_cloud'):
                    logger.info(f"- is_streamlit_cloud: {self.pinecone_client.is_streamlit_cloud}")
            
            # REST API接続が成功している場合も確認
            if not self.available:
                # REST APIメソッドを直接呼び出して確認
                api_available = self._check_rest_api_connection()
                if api_available:
                    logger.info("REST API経由でのPinecone接続が確認できました。VectorStoreを使用可能にします。")
                    self.available = True
                    self.pinecone_client.available = True
                else:
                    logger.error("REST API経由での接続も失敗しました。")
                    logger.error("接続状態の詳細:")
                    logger.error(f"- vector_store_available: {self.available}")
                    logger.error(f"- pinecone_client_available: {getattr(self.pinecone_client, 'available', False)}")
                    logger.error(f"- temporary_failure: {getattr(self.pinecone_client, 'temporary_failure', False)}")
                    logger.error(f"- is_streamlit_cloud: {getattr(self.pinecone_client, 'is_streamlit_cloud', False)}")
            
            # Pineconeが利用可能でない場合は早期リターン
            if not self.available:
                logger.error("Pineconeクライアントが利用できません")
                error_msg = "Pineconeクライアントが利用できません。\n\n"
                error_msg += "デバッグ情報:\n"
                error_msg += f"- vector_store_available: {self.available}\n"
                error_msg += f"- pinecone_client_available: {getattr(self.pinecone_client, 'available', False)}\n"
                error_msg += f"- initialization_error: {getattr(self.pinecone_client, 'initialization_error', 'なし')}\n"
                error_msg += f"- temporary_failure: {getattr(self.pinecone_client, 'temporary_failure', False)}\n"
                error_msg += f"- failed_attempts: {getattr(self.pinecone_client, 'failed_attempts', 0)}\n"
                error_msg += f"- is_streamlit_cloud: {getattr(self.pinecone_client, 'is_streamlit_cloud', False)}\n\n"
                error_msg += "接続問題を解決するオプション:\n"
                error_msg += "1. インターネット接続が安定しているか確認してください\n"
                error_msg += "2. Pinecone APIキーが正しく設定されているか確認してください\n"
                error_msg += "3. インデックスが存在し、アクセス可能か確認してください\n"
                error_msg += "4. 問題が解決しない場合は「緊急オフラインモード」を使用すると、一時的にメモリ内ストレージでアプリを使用できます\n"
                raise ValueError(error_msg)
            
            # 名前空間設定（デフォルトの名前空間を使用）
            self.namespace = ""
            logger.info(f"名前空間: {self.namespace if self.namespace else 'デフォルト'}")
            self.pinecone_client.namespace = self.namespace

            # 埋め込みモデルの設定
            try:
                from components.llm import oai_embeddings
                self.embeddings = oai_embeddings
                logger.info("OpenAI埋め込みモデルの初期化が完了しました")
            except Exception as e:
                logger.error(f"OpenAI埋め込みモデルの初期化エラー: {e}")
                logger.error(traceback.format_exc())
                raise
                
            # 一時的な障害モードかどうか
            self.temporary_failure = getattr(self.pinecone_client, 'temporary_failure', False)
            self.is_streamlit_cloud = getattr(self.pinecone_client, 'is_streamlit_cloud', False)
            
            # 緊急オフラインモード用のメモリ内ストレージ
            self.offline_storage = {
                "vectors": [],
                "metadata": [],
                "ids": []
            }
            
            logger.info("PineconeVectorStoreの初期化が完了しました")
            
        except Exception as e:
            logger.error(f"PineconeVectorStoreの初期化中にエラーが発生しました: {e}")
            logger.error(traceback.format_exc())
            self.available = False
            self.temporary_failure = False
            raise

    def _check_rest_api_connection(self):
        """REST API経由でPineconeの接続を確認する"""
        try:
            if not hasattr(self.pinecone_client, '_make_request'):
                logger.error("Pineconeクライアントに_make_requestメソッドがありません")
                return False
                
            # インデックス一覧を取得
            api_url = "https://api.pinecone.io/indexes"
            response = self.pinecone_client._make_request(
                method="GET", 
                url=api_url, 
                max_retries=3, 
                timeout=30
            )
            
            if response and response.status_code == 200:
                logger.info("REST API経由でPineconeに接続できました")
                return True
            else:
                logger.error(f"REST API経由でのPinecone接続テストに失敗: {getattr(response, 'status_code', 'N/A')}")
                return False
        except Exception as e:
            logger.error(f"REST API接続確認中のエラー: {e}")
            logger.error(traceback.format_exc())
            return False

    def add_documents(self, documents):
        """ドキュメントを追加"""
        return self.upsert_documents(documents)

    def update_documents(self, documents):
        """ドキュメントを更新"""
        return self.upsert_documents(documents)

    def upsert_documents(self, texts, metadatas=None):
        """
        ドキュメントをPineconeにアップロードする
        """
        try:
            # 埋め込みベクトルの生成
            logger.info(f"{len(texts)}件のドキュメントの埋め込みベクトルを生成中...")
            embeddings = self.embeddings.embed_documents(texts)
            
            # ベクトルIDの生成
            ids = [f"doc_{i}_{uuid.uuid4()}" for i in range(len(texts))]
            
            # メタデータの準備と検証
            if metadatas is None:
                metadatas = [{} for _ in texts]
            
            # メタデータの検証と正規化
            normalized_metadatas = []
            for metadata in metadatas:
                normalized = {}
                for key, value in metadata.items():
                    # 空の文字列はNoneに変換
                    if value == "":
                        value = None
                    # 文字列の場合は前後の空白を削除
                    elif isinstance(value, str):
                        value = value.strip()
                    normalized[key] = value
                normalized_metadatas.append(normalized)
            
            # ベクトルデータの準備
            vectors = []
            for i, (text, embedding, metadata) in enumerate(zip(texts, embeddings, normalized_metadatas)):
                vector = {
                    "id": ids[i],
                    "values": embedding,
                    "metadata": {
                        "text": text,
                        **{k: v for k, v in metadata.items() if v is not None}  # Noneの値は除外
                    }
                }
                vectors.append(vector)
            
            # バッチサイズの設定
            BATCH_SIZE = 100
            total_batches = (len(vectors) + BATCH_SIZE - 1) // BATCH_SIZE
            
            # バッチ処理
            for batch_idx in range(total_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = min((batch_idx + 1) * BATCH_SIZE, len(vectors))
                current_batch = vectors[start_idx:end_idx]
                
                # リトライロジックの改善
                max_retries = 5
                base_delay = 2
                last_error = None
                
                for attempt in range(max_retries):
                    try:
                        logger.info(f"REST APIを使用してベクトルをアップロード中... (バッチ {batch_idx+1}/{total_batches}, 試行 {attempt+1}/{max_retries})")
                        
                        # リクエストヘッダーの設定
                        headers = {
                            "Api-Key": self.api_key,
                            "Accept": "application/json",
                            "Content-Type": "application/json"
                        }
                        
                        # リクエストデータの準備（名前空間は空文字列の場合は省略）
                        data = {
                            "vectors": current_batch
                        }
                        if self.namespace:  # 名前空間が指定されている場合のみ追加
                            data["namespace"] = self.namespace
                        
                        # リクエストの送信
                        response = requests.post(
                            f"{self.base_url}/vectors/upsert/{self.index_name}",
                            headers=headers,
                            json=data,
                            timeout=30 * (attempt + 1)  # タイムアウトを徐々に増加
                        )
                        
                        # レスポンスの確認
                        if response.status_code == 200:
                            logger.info(f"バッチ {batch_idx+1}/{total_batches} のアップロード成功")
                            break
                        elif response.status_code == 429:  # Rate limit
                            delay = base_delay * (2 ** attempt)  # 指数バックオフ
                            logger.warning(f"レート制限に達しました。{delay}秒後に再試行します...")
                            time.sleep(delay)
                        elif response.status_code == 500:  # Server error
                            delay = base_delay * (2 ** attempt)
                            logger.warning(f"サーバーエラー (500): {delay}秒後に再試行します...")
                            # エラーレスポンスの詳細をログに記録
                            try:
                                error_detail = response.json()
                                logger.warning(f"エラー詳細: {error_detail}")
                            except:
                                logger.warning(f"エラーレスポンス: {response.text[:200]}")
                            time.sleep(delay)
                        else:
                            error_msg = f"予期せぬエラー: {response.status_code} - {response.text}"
                            logger.error(error_msg)
                            last_error = error_msg
                            
                    except requests.exceptions.RequestException as e:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"リクエストエラー: {str(e)}。{delay}秒後に再試行します...")
                        time.sleep(delay)
                        last_error = str(e)
                
                # すべての試行が失敗した場合
                if attempt == max_retries - 1:
                    error_msg = f"バッチ {batch_idx+1}/{total_batches} のアップロードに失敗しました。"
                    error_msg += f"\n最後のエラー: {last_error}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            
            logger.info("すべてのバッチのアップロードが完了しました")
            return True
            
        except Exception as e:
            logger.error(f"ドキュメントのアップロード中にエラーが発生しました: {str(e)}")
            logger.error(traceback.format_exc())
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
                logger.info(f"{len(ids)}件のドキュメントを削除しました")
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
                logger.info(f"{len(ids)}件のドキュメントを削除しました")
                return True
            else:
                logger.error(f"ドキュメント削除エラー: {getattr(response, 'status_code', 'N/A')} - {getattr(response, 'text', 'No response')}")
                return False
        except Exception as e:
            logger.error(f"ドキュメント削除中のエラー: {e}")
            logger.error(traceback.format_exc())
            return False

    def get_documents(self, ids=None):
        """ドキュメントを取得"""
        if not self.available:
            return {"ids": [], "documents": [], "metadatas": []}
            
        try:
            if ids is None:
                # 全てのドキュメントを取得（実際にはPineconeで全取得は難しいので、フェッチはせず空のリストを返す）
                logger.info("Pineconeでは全てのドキュメントを一度に取得することはできません")
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
                    logger.error(f"レスポンスのJSON解析エラー: {e}")
            
            return results
        except Exception as e:
            logger.error(f"ドキュメント取得中のエラー: {e}")
            logger.error(traceback.format_exc())
            return {"ids": [], "documents": [], "metadatas": []}

    def search(self, query, n_results=5, filter_conditions=None):
        """クエリに基づいてドキュメントを検索"""
        # 緊急モード検出
        emergency_mode = self.is_streamlit_cloud and self.temporary_failure
        if emergency_mode and len(self.offline_storage["vectors"]) > 0:
            logger.info("緊急オフラインモード: メモリ内ストレージを検索します")
            
            try:
                # クエリの埋め込みを生成
                query_embedding = self.embeddings.embed_query(query)
                
                # コサイン類似度を計算
                cos_scores = []
                for vec in self.offline_storage["vectors"]:
                    # 簡易的なコサイン類似度計算
                    dot_product = sum(a*b for a, b in zip(query_embedding, vec))
                    magnitude1 = sum(a*a for a in query_embedding) ** 0.5
                    magnitude2 = sum(b*b for b in vec) ** 0.5
                    similarity = dot_product / (magnitude1 * magnitude2) if magnitude1 * magnitude2 > 0 else 0
                    cos_scores.append(similarity)
                
                # フィルター条件の適用（簡易的）
                filtered_indices = list(range(len(cos_scores)))
                if filter_conditions:
                    filtered_indices = []
                    for i, metadata in enumerate(self.offline_storage["metadata"]):
                        match = True
                        for key, value in filter_conditions.items():
                            if key in metadata and metadata[key] != value:
                                match = False
                                break
                        if match:
                            filtered_indices.append(i)
                
                # スコアでソートして上位n_results件を取得
                sorted_indices = sorted(filtered_indices, key=lambda i: cos_scores[i], reverse=True)[:n_results]
                
                # 結果を構築
                results = {
                    "ids": [[self.offline_storage["ids"][i] for i in sorted_indices]],
                    "documents": [[self.offline_storage["metadata"][i].get("text", "") for i in sorted_indices]],
                    "distances": [[1.0 - cos_scores[i] for i in sorted_indices]],
                    "metadatas": [[{k: v for k, v in self.offline_storage["metadata"][i].items() if k != "text"} for i in sorted_indices]]
                }
                
                logger.info(f"緊急モード: {len(sorted_indices)}件の結果をメモリ内ストレージから検索しました")
                return results
            except Exception as e:
                logger.error(f"緊急モードでの検索中にエラー: {e}")
                logger.error(traceback.format_exc())
        
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
                    logger.error(f"検索結果の処理中にエラーが発生しました: {e}")
            
            return results
        except Exception as e:
            logger.error(f"検索中にエラーが発生しました: {e}")
            logger.error(traceback.format_exc())
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
                    logger.info(f"Pineconeコレクションには{count}件のドキュメントが登録されています")
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
                        logger.info(f"Pineconeコレクションには{count}件のドキュメントが登録されています")
                        return count
                except Exception as e:
                    logger.error(f"レスポンスの解析中にエラーが発生しました: {e}")
            
            return 0
        except Exception as e:
            logger.error(f"ドキュメント数の取得中にエラーが発生しました: {e}")
            logger.error(traceback.format_exc())
            return 0 