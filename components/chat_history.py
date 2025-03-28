import streamlit as st
from typing import List, Dict, Any
import pandas as pd
import io
import os
import time

# Pineconeクライアントをインポート - try-exceptで囲む
try:
    from components.pinecone_client import PineconeClient
    PINECONE_IMPORT_SUCCESS = True
except ImportError as e:
    print(f"Pineconeクライアントのインポートエラー: {e}")
    PINECONE_IMPORT_SUCCESS = False

class ChatHistory:
    def __init__(self):
        # Pineconeクライアントの初期化
        self.pinecone_available = False
        if PINECONE_IMPORT_SUCCESS:
            try:
                self.pinecone_client = PineconeClient()
                self.pinecone_available = True
                print("Pineconeクライアントを初期化しました")
            except Exception as e:
                print(f"Pineconeの初期化エラー: {e}")
                self.pinecone_available = False
        
        # セッション状態に会話履歴が存在しない場合は初期化
        if 'chat_history' not in st.session_state:
            # Pineconeから履歴をロード
            if self.pinecone_available:
                try:
                    loaded_history = self.pinecone_client.load_chat_history()
                    if loaded_history:
                        st.session_state.chat_history = loaded_history
                        print("Pineconeから会話履歴を復元しました")
                    else:
                        st.session_state.chat_history = []
                        print("Pineconeに復元可能な会話履歴がありませんでした")
                except Exception as e:
                    print(f"会話履歴のロードエラー: {e}")
                    st.session_state.chat_history = []
            else:
                st.session_state.chat_history = []
                print("Pineconeが利用できないため、ローカルのみで会話履歴を管理します")
        
        if 'current_context' not in st.session_state:
            st.session_state.current_context = []
        
        # 前回の保存時刻を記録
        if 'last_save_time' not in st.session_state:
            st.session_state.last_save_time = time.time()
    
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None):
        """メッセージを会話履歴に追加"""
        message = {
            'role': role,
            'content': content,
            'metadata': metadata or {}
        }
        st.session_state.chat_history.append(message)
        
        # Pineconeに保存（メッセージが追加されるたびに保存すると負荷が高いため、
        # 最後の保存から一定時間経過している場合のみ保存）
        self._save_to_pinecone_if_needed()
    
    def add_context(self, context: str):
        """コンテキストを追加"""
        st.session_state.current_context.append(context)
    
    def clear_context(self):
        """コンテキストをクリア"""
        st.session_state.current_context = []
    
    def get_history(self) -> List[Dict[str, Any]]:
        """会話履歴を取得"""
        return st.session_state.chat_history
    
    def get_context(self) -> List[str]:
        """現在のコンテキストを取得"""
        return st.session_state.current_context
    
    def clear_history(self):
        """会話履歴をクリア"""
        st.session_state.chat_history = []
        st.session_state.current_context = []
        
        # Pineconeに空の履歴を保存（履歴クリアを同期）
        if self.pinecone_available:
            try:
                self.pinecone_client.save_chat_history([])
                st.session_state.last_save_time = time.time()
                print("Pineconeの会話履歴をクリアしました")
            except Exception as e:
                print(f"会話履歴のクリア中にエラー: {e}")
    
    def get_formatted_history(self) -> str:
        """会話履歴を文字列形式で取得"""
        formatted = ""
        for msg in self.get_history():
            role = "ユーザー" if msg['role'] == 'user' else "アシスタント"
            formatted += f"{role}: {msg['content']}\n\n"
        return formatted
        
    def get_csv_export(self) -> bytes:
        """会話履歴をCSV形式でエクスポート"""
        if not self.get_history():
            return None
            
        data = []
        for msg in self.get_history():
            role = "ユーザー" if msg['role'] == 'user' else "アシスタント"
            data.append({
                "役割": role,
                "内容": msg['content']
            })
            
        df = pd.DataFrame(data)
        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')  # BOM付きUTF-8でExcelでも文字化けしないように
        return csv_buffer.getvalue()
    
    def _save_to_pinecone_if_needed(self):
        """必要に応じてPineconeに会話履歴を保存"""
        # Pineconeが利用可能で、最後の保存から一定時間（30秒）経過している場合に保存
        current_time = time.time()
        if (self.pinecone_available and 
            (current_time - st.session_state.last_save_time > 30)):
            try:
                result = self.pinecone_client.save_chat_history(st.session_state.chat_history)
                if result:
                    st.session_state.last_save_time = current_time
                    print("会話履歴をPineconeに保存しました")
            except Exception as e:
                print(f"会話履歴の保存中にエラー: {e}")
    
    def force_save(self):
        """会話履歴を強制的にPineconeに保存"""
        if self.pinecone_available:
            try:
                result = self.pinecone_client.save_chat_history(st.session_state.chat_history)
                if result:
                    st.session_state.last_save_time = time.time()
                    print("会話履歴を強制的にPineconeに保存しました")
                    return True
                return False
            except Exception as e:
                print(f"会話履歴の強制保存中にエラー: {e}")
                return False
        return False 