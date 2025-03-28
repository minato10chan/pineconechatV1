import streamlit as st
from typing import List, Dict, Any
import pandas as pd
import io

class ChatHistory:
    def __init__(self):
        # セッション状態に会話履歴が存在しない場合は初期化
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'current_context' not in st.session_state:
            st.session_state.current_context = []
    
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None):
        """メッセージを会話履歴に追加"""
        message = {
            'role': role,
            'content': content,
            'metadata': metadata or {}
        }
        st.session_state.chat_history.append(message)
    
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