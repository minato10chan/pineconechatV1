import os
import json
import uuid
from datetime import datetime
import time
import requests
import traceback
import streamlit as st

# Pineconeのインポートを試みる
try:
    import pinecone
    PINECONE_AVAILABLE = True
    print("Pineconeモジュールのインポートに成功しました")
except Exception as e:
    print(f"Pineconeのインポートエラー: {e}")
    PINECONE_AVAILABLE = False

class PineconeClient:
    def __init__(self):
        # 環境変数から直接取得
        self.api_key = os.environ.get("PINECONE_API_KEY")
        self.environment = os.environ.get("PINECONE_ENVIRONMENT", "us-east-1")
        self.index_name = os.environ.get("PINECONE_INDEX", "langchain-index")
        
        # Streamlit Secretsを試す (環境変数が設定されていない場合)
        if not self.api_key:
            try:
                self.api_key = st.secrets.get("PINECONE_API_KEY")
                if not self.environment:
                    self.environment = st.secrets.get("PINECONE_ENVIRONMENT", "us-east-1")
                if not self.index_name:
                    self.index_name = st.secrets.get("PINECONE_INDEX", "langchain-index")
                print("Streamlit Secretsから認証情報を取得しました")
            except Exception as e:
                print(f"Streamlit Secretsからの取得に失敗: {e}")
        
        # デバッグ情報
        if self.api_key:
            masked_key = f"{self.api_key[:8]}...{self.api_key[-4:]}"
            print(f"Pinecone設定 - 環境: {self.environment}, インデックス: {self.index_name}, APIキー: {masked_key}")
        else:
            print("ERROR: PINECONE_API_KEY環境変数が設定されていません")
            self.available = False
            return
        
        # インターネット接続確認
        if not self._check_internet_connection():
            print("ERROR: インターネット接続に問題があります。ローカルモードで動作します。")
            self.available = False
            return
            
        # REST APIでの接続テストを最初に試行
        self.headers = {
            "Api-Key": self.api_key,
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        
        # REST APIでの接続テスト
        if self._test_api_connection_rest():
            print("REST APIでPineconeに接続しました")
            self.available = True
            # インデックスを確認・作成
            self._check_index_rest()
            # 名前空間設定
            self.namespace = "chat-history"
            print("REST API接続でPineconeクライアントの初期化完了")
            return
        
        # REST APIが失敗した場合、公式SDKを試みる
        if PINECONE_AVAILABLE:
            try:
                print("公式SDKでPineconeに接続を試みます...")
                pinecone.init(api_key=self.api_key, environment=self.environment)
                try:
                    # インデックスリストを取得
                    indexes = pinecone.list_indexes()
                    print(f"利用可能なPineconeインデックス: {indexes}")
                    
                    if self.index_name not in indexes:
                        print(f"警告: インデックス '{self.index_name}' が見つかりません")
                        print("新しいインデックスの作成を試みます")
                        try:
                            # インデックスの作成を試みる（既存のインデックスがない場合）
                            pinecone.create_index(
                                name=self.index_name,
                                dimension=1536,  # OpenAI embedding size
                                metric="cosine"
                            )
                            print(f"インデックス '{self.index_name}' を作成しました")
                        except Exception as e:
                            print(f"インデックス作成エラー: {e}")
                    
                    self.index = pinecone.Index(self.index_name)
                    print(f"Pineconeインデックス '{self.index_name}' に接続成功！")
                    self.available = True
                    self.namespace = "chat-history"
                    print("SDK接続でPineconeクライアントの初期化完了")
                    return
                except Exception as e:
                    print(f"インデックス操作中のエラー: {e}")
                    print(traceback.format_exc())
            except Exception as e:
                print(f"公式SDKでの接続に失敗: {e}")
                print(traceback.format_exc())
        else:
            print("Pinecone SDKが利用できないため、SDK接続は試行しません")
            
        # 両方の接続方法が失敗した場合
        if not getattr(self, 'available', False):
            print("PineconeへのすべてのAPI接続が失敗しました。ローカルモードで動作します。")
            self.available = False
    
    def _make_request(self, method, url, json_data=None, params=None, max_retries=3, timeout=30):
        """REST APIリクエストを実行する共通メソッド"""
        retries = 0
        while retries < max_retries:
            try:
                print(f"HTTP {method} リクエスト: {url}")
                if method.upper() == "GET":
                    response = requests.get(
                        url, 
                        headers=self.headers,
                        params=params,
                        timeout=timeout
                    )
                elif method.upper() == "POST":
                    print(f"POSTデータ: {json.dumps(json_data)[:100]}..." if json_data else "POSTデータなし")
                    response = requests.post(
                        url, 
                        headers=self.headers,
                        json=json_data,
                        params=params,
                        timeout=timeout
                    )
                elif method.upper() == "DELETE":
                    response = requests.delete(
                        url, 
                        headers=self.headers,
                        json=json_data,
                        params=params,
                        timeout=timeout
                    )
                else:
                    print(f"サポートされていないHTTPメソッド: {method}")
                    return None
                
                print(f"レスポンス: ステータスコード {response.status_code}")
                # エラーの場合は応答本文を表示
                if response.status_code >= 400:
                    print(f"エラーレスポンス本文: {response.text}")
                
                # レートリミットエラーの場合は再試行
                if response.status_code == 429:
                    retries += 1
                    wait_time = 2 ** retries  # 指数バックオフ
                    print(f"レートリミットに達しました。{wait_time}秒後に再試行します...")
                    time.sleep(wait_time)
                    continue
                
                # エラーはログに記録して応答を返す
                if response.status_code >= 400:
                    print(f"API呼び出しエラー: {method} {url} - ステータス: {response.status_code}")
                    try:
                        print(f"レスポンス: {response.json()}")
                    except:
                        print(f"レスポンス: {response.text}")
                
                return response
                
            except requests.exceptions.ConnectionError as e:
                print(f"接続エラー: {e}")
                retries += 1
                if retries < max_retries:
                    wait_time = 2 ** retries
                    print(f"{wait_time}秒後に再試行します...")
                    time.sleep(wait_time)
                else:
                    print("最大再試行回数に達しました")
                    return None
            except requests.exceptions.Timeout as e:
                print(f"タイムアウトエラー: {e}")
                retries += 1
                if retries < max_retries:
                    wait_time = 2 ** retries
                    print(f"タイムアウト - {wait_time}秒後に再試行します...")
                    time.sleep(wait_time)
                else:
                    print("最大再試行回数に達しました")
                    return None
            except Exception as e:
                print(f"APIリクエスト中のエラー: {e}")
                print(traceback.format_exc())
                return None
        
        return None
    
    def _test_api_connection_rest(self):
        """REST APIを使用してPineconeに接続テスト"""
        try:
            # REST APIでインデックス一覧を取得
            api_url = "https://api.pinecone.io/indexes"
            
            print(f"Pinecone REST APIで接続テスト中... URL: {api_url}")
            print(f"ヘッダー: {self.headers} (APIキーは一部マスク)")
            
            # タイムアウトを長くして再試行回数を増やす
            response = self._make_request(method="GET", url=api_url, max_retries=5, timeout=30)
            
            if response and response.status_code == 200:
                print(f"Pinecone REST API接続成功: {response.status_code}")
                try:
                    index_list = response.json()
                    print(f"利用可能なインデックス: {index_list}")
                    # インデックスが存在するかチェック
                    if self.index_name in index_list:
                        print(f"インデックス '{self.index_name}' が存在します")
                        return True
                    else:
                        print(f"インデックス '{self.index_name}' が見つかりません。作成を試みます。")
                        # インデックス作成コードは_check_index_restメソッドに移動
                except Exception as e:
                    print(f"レスポンスのJSON解析エラー: {e}")
                    print(f"レスポンス内容: {response.text}")
                return True  # 接続自体は成功
            else:
                print(f"Pinecone REST API接続エラー: {getattr(response, 'status_code', 'N/A')} - {getattr(response, 'text', 'No response')}")
                return False
        except Exception as e:
            print(f"Pinecone REST API接続テスト中の例外: {e}")
            print(traceback.format_exc())
            return False
    
    def _check_index_rest(self):
        """REST APIを使用してインデックスの存在確認と作成"""
        try:
            # インデックス情報を取得
            api_url = f"https://api.pinecone.io/indexes/{self.index_name}"
            print(f"インデックス '{self.index_name}' の存在確認中... URL: {api_url}")
            
            response = self._make_request(method="GET", url=api_url)
            
            if response and response.status_code == 200:
                print(f"インデックス '{self.index_name}' が存在します")
                return True
            elif response and response.status_code == 404:
                print(f"インデックス '{self.index_name}' が見つかりません。作成を試みます...")
                
                # インデックス作成リクエスト
                create_url = "https://api.pinecone.io/indexes"
                create_data = {
                    "name": self.index_name,
                    "dimension": 1536,  # OpenAI embeddings default
                    "metric": "cosine"
                }
                
                create_response = self._make_request(
                    method="POST",
                    url=create_url,
                    json_data=create_data
                )
                
                if create_response and create_response.status_code in [200, 201, 202]:
                    print(f"インデックス '{self.index_name}' の作成リクエストが受け付けられました")
                    # インデックス作成は非同期なので、完了を待つ
                    time.sleep(5)
                    return True
                else:
                    print(f"インデックス作成エラー: {getattr(create_response, 'status_code', 'N/A')} - {getattr(create_response, 'text', 'No response')}")
                    return False
            else:
                print(f"インデックス確認中のエラー: {getattr(response, 'status_code', 'N/A')} - {getattr(response, 'text', 'No response')}")
                return False
        except Exception as e:
            print(f"インデックス確認中の例外: {e}")
            print(traceback.format_exc())
            return False
    
    def _check_internet_connection(self):
        """インターネット接続を確認する簡易的な関数"""
        try:
            # Googleの公開DNSサーバーに接続を試みる
            response = requests.get("https://8.8.8.8", timeout=3)
            print("インターネット接続: OK")
            return True
        except requests.RequestException:
            try:
                # バックアップとしてCloudflareのDNSにも試す
                response = requests.get("https://1.1.1.1", timeout=3)
                print("インターネット接続: OK (Cloudflare)")
                return True
            except requests.RequestException:
                print("インターネット接続: 失敗 - ネットワーク接続を確認してください")
                return False

    def save_chat_history(self, chat_history):
        """会話履歴をPineconeに保存"""
        if not chat_history or not self.available:
            return None
        
        # 会話履歴をJSONに変換
        chat_json = json.dumps(chat_history)
        
        # タイムスタンプとユーザーIDを含むIDを生成
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        session_id = os.environ.get("STREAMLIT_SESSION_ID", str(uuid.uuid4()))
        vector_id = f"chat_{session_id}_{timestamp}"
        
        # メタデータ作成
        metadata = {
            "timestamp": timestamp,
            "session_id": session_id,
            "type": "chat_history",
            "chat_data": chat_json
        }
        
        try:
            # 公式SDKが利用可能ならそれを使う
            if hasattr(self, 'index') and isinstance(self.index, pinecone.Index):
                self.index.upsert(
                    vectors=[
                        {
                            "id": vector_id,
                            "values": [0.0],  # ダミーベクトル値
                            "metadata": metadata
                        }
                    ],
                    namespace=self.namespace
                )
                print(f"会話履歴をPineconeに保存しました (SDK): {vector_id}")
                return vector_id
            
            # REST APIでベクトルをアップサート
            api_url = f"https://api.pinecone.io/vectors/upsert/{self.index_name}"
            
            data = {
                "vectors": [
                    {
                        "id": vector_id,
                        "values": [0.0],  # ダミーベクトル値
                        "metadata": metadata
                    }
                ],
                "namespace": self.namespace
            }
            
            response = self._make_request(
                method="POST",
                url=api_url,
                json_data=data
            )
            
            if response and response.status_code in [200, 201, 202]:
                print(f"会話履歴をPineconeに保存しました (REST): {vector_id}")
                return vector_id
            else:
                print(f"会話履歴の保存エラー: {getattr(response, 'status_code', 'N/A')} - {getattr(response, 'text', 'No response')}")
                return None
        except Exception as e:
            print(f"Pineconeへの保存エラー: {e}")
            print(traceback.format_exc())
            return None
    
    def load_chat_history(self, session_id=None):
        """Pineconeから会話履歴を読み込む"""
        if not self.available:
            return None
            
        # セッションIDが指定されていなければ環境変数から取得
        if not session_id:
            session_id = os.environ.get("STREAMLIT_SESSION_ID", None)
        
        if not session_id:
            print("セッションIDが指定されていないため、会話履歴を読み込めません")
            return None
        
        try:
            print(f"セッションID '{session_id}' の会話履歴を検索中...")
            
            # 公式SDKが利用可能ならそれを使う
            if hasattr(self, 'index') and isinstance(self.index, pinecone.Index):
                filter_dict = {
                    "session_id": {"$eq": session_id},
                    "type": {"$eq": "chat_history"}
                }
                
                results = self.index.query(
                    vector=[0.0],
                    filter=filter_dict,
                    top_k=1,
                    include_metadata=True,
                    namespace=self.namespace
                )
                
                if results.matches and len(results.matches) > 0:
                    chat_data = results.matches[0].metadata.get("chat_data")
                    if chat_data:
                        print(f"Pineconeから会話履歴を読み込みました (SDK): {session_id}")
                        return json.loads(chat_data)
                    else:
                        print("会話履歴データが見つかりません (SDK)")
                else:
                    print(f"セッションID '{session_id}' の会話履歴が見つかりませんでした (SDK)")
                return None
            
            # REST APIでクエリ実行
            api_url = f"https://api.pinecone.io/query/{self.index_name}"
            
            data = {
                "vector": [0.0],  # ダミークエリベクトル
                "filter": {
                    "session_id": {"$eq": session_id},
                    "type": {"$eq": "chat_history"}
                },
                "top_k": 1,
                "include_metadata": True,
                "namespace": self.namespace
            }
            
            response = self._make_request(
                method="POST",
                url=api_url,
                json_data=data
            )
            
            if response and response.status_code == 200:
                result = response.json()
                if result.get("matches") and len(result["matches"]) > 0:
                    # 最新の会話履歴を取得
                    chat_data = result["matches"][0]["metadata"].get("chat_data")
                    if chat_data:
                        print(f"Pineconeから会話履歴を読み込みました (REST): {session_id}")
                        return json.loads(chat_data)
                    else:
                        print("会話履歴データが見つかりません (REST)")
                else:
                    print(f"セッションID '{session_id}' の会話履歴が見つかりませんでした (REST)")
            else:
                print(f"会話履歴の検索エラー: {getattr(response, 'status_code', 'N/A')} - {getattr(response, 'text', 'No response')}")
        except Exception as e:
            print(f"Pineconeからの読み込みエラー: {e}")
            print(traceback.format_exc())
        
        return None 