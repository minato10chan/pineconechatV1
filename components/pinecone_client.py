import os
import json
import uuid
from datetime import datetime

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
            
        api_key = os.environ.get("PINECONE_API_KEY")
        environment = os.environ.get("PINECONE_ENVIRONMENT", "us-west1-gcp")
        index_name = os.environ.get("PINECONE_INDEX", "chat-history")
        
        if not api_key:
            raise ValueError("PINECONE_API_KEY環境変数が設定されていません")
        
        # Pinecone初期化
        pinecone.init(api_key=api_key, environment=environment)
        
        # インデックスが存在するか確認し、なければ作成
        existing_indexes = pinecone.list_indexes()
        if index_name not in existing_indexes:
            try:
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
        
        self.index = pinecone.Index(index_name)
        self.namespace = "chat-history"
    
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
            return None
        
        # セッションIDに基づいて検索
        query = {
            "session_id": session_id,
            "type": "chat_history"
        }
        
        try:
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
        except Exception as e:
            print(f"Pineconeからの読み込みエラー: {e}")
        
        return None 