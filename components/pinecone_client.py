import os
import json
import uuid
from datetime import datetime
import time

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
        api_key = os.environ.get("PINECONE_API_KEY")
        environment = os.environ.get("PINECONE_ENVIRONMENT", "us-west1-gcp")
        index_name = os.environ.get("PINECONE_INDEX", "chat-history")
        
        # デバッグ情報（APIキーは安全のため一部マスク）
        masked_key = "**********" if api_key else "未設定"
        print(f"Pinecone設定 - 環境: {environment}, インデックス: {index_name}, APIキー: {masked_key}")
        
        if not api_key:
            raise ValueError("PINECONE_API_KEY環境変数が設定されていません")
        
        # インターネット接続確認（簡易的な方法）
        self._check_internet_connection()
        
        # リトライロジックでPinecone初期化
        max_retries = 3
        retry_delay = 2  # 秒
        
        for attempt in range(max_retries):
            try:
                print(f"Pinecone初期化を試行中... (試行 {attempt+1}/{max_retries})")
                # Pinecone初期化
                pinecone.init(api_key=api_key, environment=environment)
                print("Pinecone初期化成功")
                break
            except Exception as e:
                print(f"Pinecone初期化中のエラー (試行 {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    print(f"{retry_delay}秒後に再試行します...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # 指数バックオフ
                else:
                    print("Pinecone初期化の最大試行回数に達しました")
                    raise
        
        # インデックスの存在確認と作成も同様にリトライロジック
        for attempt in range(max_retries):
            try:
                print(f"Pineconeインデックスの確認中... (試行 {attempt+1}/{max_retries})")
                existing_indexes = pinecone.list_indexes()
                print(f"利用可能なインデックス: {existing_indexes}")
                
                if index_name not in existing_indexes:
                    try:
                        print(f"インデックス '{index_name}' を作成しています...")
                        pinecone.create_index(
                            name=index_name,
                            dimension=1,  # 会話履歴用なのでベクトル次元は必要最小限
                            metric="cosine"
                        )
                        print(f"Pineconeインデックス '{index_name}' を作成しました")
                    except Exception as e:
                        print(f"インデックス作成エラー: {e}")
                        # インデックス作成に失敗した場合でも、既存のインデックスがあれば続行
                        if index_name not in pinecone.list_indexes():
                            raise
                break
            except Exception as e:
                print(f"インデックス確認中のエラー (試行 {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    print(f"{retry_delay}秒後に再試行します...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print("インデックス確認の最大試行回数に達しました")
                    raise
        
        print(f"Pineconeインデックス '{index_name}' に接続しています...")
        self.index = pinecone.Index(index_name)
        self.namespace = "chat-history"
        print("Pineconeクライアントの初期化完了")
    
    def _check_internet_connection(self):
        """インターネット接続を確認する簡易的な関数"""
        import socket
        try:
            # Googleの公開DNSサーバーに接続を試みる
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            print("インターネット接続: OK")
            return True
        except OSError:
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
            # Pineconeに保存 (vectorは会話データ用なので[0]だけ)
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
            print(f"Pineconeへの保存エラー: {e}")
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
            print(f"Pineconeからの読み込みエラー: {e}")
        
        return None 