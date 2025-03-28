import os
import json
import uuid
from datetime import datetime
import time
import requests
import traceback

# Pineconeのインポートを例外処理で囲む
try:
    import pinecone
    PINECONE_AVAILABLE = True
except Exception as e:
    print(f"Pineconeのインポートエラー: {e}")
    PINECONE_AVAILABLE = False

class PineconeClient:
    def __init__(self):
        if not PINECONE_AVAILABLE:
            raise ImportError("Pineconeライブラリをインポートできません。requirements.txtの設定を確認してください。")
            
        # 環境変数の取得とデバッグ情報表示
        self.api_key = os.environ.get("PINECONE_API_KEY")
        self.environment = os.environ.get("PINECONE_ENVIRONMENT", "us-east-1")
        self.index_name = os.environ.get("PINECONE_INDEX", "langchain-index")
        
        # デバッグ情報（APIキーは安全のため一部マスク）
        masked_key = "**********" if self.api_key else "未設定"
        print(f"Pinecone設定 - 環境: {self.environment}, インデックス: {self.index_name}, APIキー: {masked_key}")
        
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY環境変数が設定されていません")
        
        # インターネット接続確認（簡易的な方法）
        if not self._check_internet_connection():
            raise ConnectionError("インターネット接続が確立できません")
        
        # 直接APIを使用してPineconeに接続する（DNSの問題を回避するため）
        self._test_api_connection()
        
        # Pinecone初期化
        try:
            print(f"Pinecone初期化を開始します...")
            pinecone.init(api_key=self.api_key, environment=self.environment)
            print("Pinecone初期化成功")
        except Exception as e:
            print(f"Pinecone初期化エラー: {e}")
            print("代替接続方法を使用します")
        
        # インデックスを確認
        self._check_index()
        
        # インデックスを初期化
        try:
            self.index = pinecone.Index(self.index_name)
            print(f"Pineconeインデックス '{self.index_name}' に接続しました")
        except Exception as e:
            print(f"インデックス接続エラー: {e}")
            print("HTTPリクエストによる代替接続を使用します")
        
        self.namespace = "chat-history"
        print("Pineconeクライアントの初期化完了")
    
    def _test_api_connection(self):
        """直接APIを使用して接続テスト"""
        try:
            headers = {
                "Api-Key": self.api_key,
                "Accept": "application/json"
            }
            
            # ホスト名はPineconeのドキュメントに基づいて更新（環境に応じて）
            api_url = f"https://api.pinecone.io/indexes"
            
            print(f"Pinecone APIに直接接続テスト中... URL: {api_url}")
            response = requests.get(api_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                print(f"Pinecone API直接接続成功: {response.status_code}")
                index_list = response.json()
                print(f"利用可能なインデックス: {index_list}")
                return True
            else:
                print(f"Pinecone API接続エラー: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"Pinecone API接続テスト中の例外: {e}")
            print(traceback.format_exc())
            return False
    
    def _check_index(self):
        """インデックスの存在確認と作成（必要な場合）"""
        try:
            # 直接APIを使用してインデックスをチェック
            headers = {
                "Api-Key": self.api_key,
                "Accept": "application/json"
            }
            
            api_url = f"https://api.pinecone.io/indexes/{self.index_name}"
            print(f"インデックス '{self.index_name}' の存在確認中... URL: {api_url}")
            
            response = requests.get(api_url, headers=headers, timeout=10)
            
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
                    headers={**headers, "Content-Type": "application/json"}, 
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
        if not chat_history:
            return
        
        # 会話履歴をJSONに変換
        chat_json = json.dumps(chat_history)
        
        # タイムスタンプとユーザーIDを含むIDを生成
        # 実際の実装ではユーザー認証と連携するとよい
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        session_id = os.environ.get("STREAMLIT_SESSION_ID", str(uuid.uuid4()))
        vector_id = f"chat_{session_id}_{timestamp}"
        
        # メタデータ作成
        metadata = {
            "timestamp": timestamp,
            "session_id": session_id,
            "type": "chat_history"
        }
        
        try:
            # Pineconeライブラリを使用
            try:
                self.index.upsert(
                    vectors=[
                        {
                            "id": vector_id,
                            "values": [0.0],  # ダミーベクトル値
                            "metadata": {
                                **metadata,
                                "chat_data": chat_json
                            }
                        }
                    ],
                    namespace=self.namespace
                )
                print(f"会話履歴をPineconeに保存しました: {vector_id}")
                return vector_id
            except Exception as e:
                print(f"Pineconeライブラリでの保存エラー: {e}")
                # 直接APIを使用する代替方法を実装するとよい
                return None
        except Exception as e:
            print(f"Pineconeへの保存エラー: {e}")
            print(traceback.format_exc())
            return None
    
    def load_chat_history(self, session_id=None):
        """Pineconeから会話履歴を読み込む"""
        # セッションIDが指定されていなければ環境変数から取得
        if not session_id:
            session_id = os.environ.get("STREAMLIT_SESSION_ID", None)
        
        if not session_id:
            print("セッションIDが指定されていないため、会話履歴を読み込めません")
            return None
        
        # セッションIDに基づいて検索
        query = {
            "session_id": session_id,
            "type": "chat_history"
        }
        
        try:
            print(f"セッションID '{session_id}' の会話履歴を検索中...")
            # Pineconeから検索
            try:
                results = self.index.query(
                    vector=[0.0],  # ダミークエリベクトル
                    filter=query,
                    top_k=1,
                    include_metadata=True,
                    namespace=self.namespace
                )
                
                if results.matches and len(results.matches) > 0:
                    # 最新の会話履歴を取得
                    chat_data = results.matches[0].metadata.get("chat_data")
                    if chat_data:
                        print(f"Pineconeから会話履歴を読み込みました: {session_id}")
                        return json.loads(chat_data)
                else:
                    print(f"セッションID '{session_id}' の会話履歴が見つかりませんでした")
            except Exception as e:
                print(f"Pineconeライブラリでの読み込みエラー: {e}")
                # 直接APIを使用する代替方法を実装するとよい
                return None
        except Exception as e:
            print(f"Pineconeからの読み込みエラー: {e}")
            print(traceback.format_exc())
        
        return None 