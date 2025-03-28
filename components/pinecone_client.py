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
except Exception as e:
    print(f"Pineconeのインポートエラー: {e}")
    PINECONE_AVAILABLE = False

class PineconeClient:
    def __init__(self):
        # Streamlit Secretsから設定を取得
        try:
            self.api_key = st.secrets.get("PINECONE_API_KEY")
            self.environment = st.secrets.get("PINECONE_ENVIRONMENT", "us-east-1")
            self.index_name = st.secrets.get("PINECONE_INDEX", "langchain-index")
        except Exception as e:
            # セクレットが見つからない場合は環境変数から取得
            print(f"Streamlit Secretsからの取得に失敗したため環境変数を使用: {e}")
            self.api_key = os.environ.get("PINECONE_API_KEY")
            self.environment = os.environ.get("PINECONE_ENVIRONMENT", "us-east-1")
            self.index_name = os.environ.get("PINECONE_INDEX", "langchain-index")
        
        # デバッグ情報（APIキーは安全のため一部マスク）
        masked_key = "**********" if self.api_key else "未設定"
        print(f"Pinecone設定 - 環境: {self.environment}, インデックス: {self.index_name}, APIキー: {masked_key}")
        
        if not self.api_key:
            print("PINECONE_API_KEY環境変数が設定されていません")
            self.available = False
            return
        
        # インターネット接続確認
        if not self._check_internet_connection():
            print("インターネット接続に問題があります。ローカルモードで動作します。")
            self.available = False
            return
            
        # Pineconeへの接続を初期化（公式SDK方式）
        try:
            print("公式SDKでPineconeに接続中...")
            pinecone.init(api_key=self.api_key, environment=self.environment)
            self.index = pinecone.Index(self.index_name)
            print(f"Pineconeインデックス '{self.index_name}' に接続成功！")
            self.available = True
            self.namespace = "chat-history"
            print("Pineconeクライアントの初期化完了")
            return
        except Exception as e:
            print(f"公式SDKでの接続に失敗: {e}")
            print(f"詳細: {traceback.format_exc()}")
            print("代替方法でPineconeに接続を試みます...")
        
        # 代替方法: REST APIを使用
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
        else:
            print("PineconeへのREST API接続に失敗しました。ローカルモードで動作します。")
            self.available = False
            
        # 名前空間設定
        self.namespace = "chat-history"
        print("Pineconeクライアントの初期化完了")
    
    def __del__(self):
        """デストラクタ：クライアント終了時の処理"""
        try:
            if PINECONE_AVAILABLE:
                pinecone.deinit()
                print("Pinecone接続を終了しました")
        except Exception as e:
            print(f"Pinecone終了処理中のエラー: {e}")
    
    def _test_api_connection_rest(self):
        """REST APIを使用してPineconeに接続テスト"""
        try:
            # REST APIでインデックス一覧を取得
            api_url = "https://api.pinecone.io/indexes"
            
            print(f"Pinecone REST APIで接続テスト中... URL: {api_url}")
            response = requests.get(api_url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                print(f"Pinecone REST API接続成功: {response.status_code}")
                try:
                    index_list = response.json()
                    print(f"利用可能なインデックス: {index_list}")
                except Exception as e:
                    print(f"レスポンスのJSON解析エラー: {e}")
                    print(f"レスポンス内容: {response.text}")
                return True
            else:
                print(f"Pinecone REST API接続エラー: {response.status_code} - {response.text}")
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
            
            response = requests.get(api_url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                print(f"インデックス '{self.index_name}' が存在します")
                return True
            elif response.status_code == 404:
                print(f"インデックス '{self.index_name}' が見つかりません。作成を試みます...")
                
                # インデックス作成リクエスト
                create_url = "https://api.pinecone.io/indexes"
                create_data = {
                    "name": self.index_name,
                    "dimension": 1,
                    "metric": "cosine"
                }
                
                create_response = requests.post(
                    create_url, 
                    headers=self.headers, 
                    json=create_data,
                    timeout=30
                )
                
                if create_response.status_code in [200, 201, 202]:
                    print(f"インデックス '{self.index_name}' の作成リクエストが受け付けられました")
                    # インデックス作成は非同期なので、完了を待つ
                    time.sleep(5)
                    return True
                else:
                    print(f"インデックス作成エラー: {create_response.status_code} - {create_response.text}")
                    return False
            else:
                print(f"インデックス確認中のエラー: {response.status_code} - {response.text}")
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
            
            response = requests.post(api_url, headers=self.headers, json=data, timeout=30)
            
            if response.status_code in [200, 201, 202]:
                print(f"会話履歴をPineconeに保存しました (REST): {vector_id}")
                return vector_id
            else:
                print(f"会話履歴の保存エラー: {response.status_code} - {response.text}")
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
            
            response = requests.post(api_url, headers=self.headers, json=data, timeout=30)
            
            if response.status_code == 200:
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
                print(f"会話履歴の検索エラー: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Pineconeからの読み込みエラー: {e}")
            print(traceback.format_exc())
        
        return None 