# pysqlite3コードを削除
import streamlit as st
import datetime
import os
from dotenv import load_dotenv
import traceback
import time
import requests
import sys
import platform

# 環境変数を確実にロード
load_dotenv(override=True)

# グローバル変数の初期化
vector_store = None
vector_store_available = False

# Pinecone APIキーが環境変数から正しく読み込まれているか確認
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_env = os.environ.get("PINECONE_ENVIRONMENT")
pinecone_index = os.environ.get("PINECONE_INDEX")

print(f"環境変数: PINECONE_API_KEY={'設定済み' if pinecone_api_key else '未設定'}")
print(f"環境変数: PINECONE_ENVIRONMENT={pinecone_env}")
print(f"環境変数: PINECONE_INDEX={pinecone_index}")

# Pinecone SDK接続テストを削除（REST APIのみ使用）

# 最初のStreamlitコマンドとしてページ設定を行う
st.set_page_config(page_title='🦜🔗 Ask the Doc App', layout="wide")

# 必要なインポート
from langchain_openai import OpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import pandas as pd
import io

# カスタムモジュールのインポート
from components.llm import llm, oai_embeddings
from components.categories import MAJOR_CATEGORIES, MEDIUM_CATEGORIES
from components.prompts import RAG_PROMPT_TEMPLATE
from components.chat_history import ChatHistory

# セッション状態の初期化
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'custom_prompts' not in st.session_state:
    st.session_state.custom_prompts = [
        {
            'name': 'デフォルト',
            'content': RAG_PROMPT_TEMPLATE
        }
    ]
if 'selected_prompt' not in st.session_state:
    st.session_state.selected_prompt = 'デフォルト'

# チャット履歴の初期化
chat_history = ChatHistory()

# VectorStoreのインスタンスを初期化する関数
def initialize_vector_store():
    global vector_store, vector_store_available
    
    try:
        if vector_store is None:
            print("最初のベクトルストア初期化を実行します...")
            from src.pinecone_vector_store import PineconeVectorStore
            vector_store = PineconeVectorStore()
            
            # 使用可能かどうかを確認
            vector_store_available = getattr(vector_store, 'available', False)
            print(f"PineconeベースのVectorStoreを初期化しました。状態: {'利用可能' if vector_store_available else '利用不可'}")
            
            # 接続状態の詳細を表示
            if not vector_store_available:
                print("\n接続状態の詳細:")
                print(f"- vector_store_available: {vector_store_available}")
                if hasattr(vector_store, 'pinecone_client'):
                    client = vector_store.pinecone_client
                    print(f"- pinecone_client_available: {getattr(client, 'available', False)}")
                    print(f"- initialization_error: {getattr(client, 'initialization_error', 'なし')}")
                    print(f"- temporary_failure: {getattr(client, 'temporary_failure', False)}")
                    print(f"- failed_attempts: {getattr(client, 'failed_attempts', 0)}")
                    print(f"- is_streamlit_cloud: {getattr(client, 'is_streamlit_cloud', False)}")
                
                # エラーメッセージを構築
                error_msg = "ベクトルデータベースの接続でエラーが発生しました。\n\n"
                error_msg += "デバッグ情報:\n"
                error_msg += f"- vector_store_available: {vector_store_available}\n"
                
                if hasattr(vector_store, 'pinecone_client'):
                    client = vector_store.pinecone_client
                    error_msg += f"- pinecone_client_available: {getattr(client, 'available', False)}\n"
                    error_msg += f"- initialization_error: {getattr(client, 'initialization_error', 'なし')}\n"
                    error_msg += f"- temporary_failure: {getattr(client, 'temporary_failure', False)}\n"
                    error_msg += f"- failed_attempts: {getattr(client, 'failed_attempts', 0)}\n"
                    error_msg += f"- is_streamlit_cloud: {getattr(client, 'is_streamlit_cloud', False)}\n"
                
                error_msg += "\n接続問題を解決するオプション:\n"
                error_msg += "1. インターネット接続が安定しているか確認してください\n"
                error_msg += "2. Pinecone APIキーが正しく設定されているか確認してください\n"
                error_msg += "3. インデックスが存在し、アクセス可能か確認してください\n"
                error_msg += "4. 問題が解決しない場合は「緊急オフラインモード」を使用すると、一時的にメモリ内ストレージでアプリを使用できます\n"
                
                st.error(error_msg)
                return False
            
            print("VectorStore initialization completed. Status: Available")
            print(f"グローバル変数の最終状態: vector_store_available = {vector_store_available}")
            print(f"vector_store.available = {vector_store.available}")
            return True
            
    except Exception as e:
        print(f"VectorStoreの初期化中にエラー: {e}")
        print(traceback.format_exc())
        vector_store_available = False
        vector_store = None
        
        # エラーメッセージを構築
        error_msg = "ベクトルデータベースの初期化中にエラーが発生しました。\n\n"
        error_msg += f"エラーの詳細: {str(e)}\n\n"
        error_msg += "デバッグ情報:\n"
        error_msg += f"- vector_store_available: {vector_store_available}\n"
        error_msg += f"- error_type: {type(e).__name__}\n"
        error_msg += f"- error_message: {str(e)}\n\n"
        error_msg += "接続問題を解決するオプション:\n"
        error_msg += "1. インターネット接続が安定しているか確認してください\n"
        error_msg += "2. Pinecone APIキーが正しく設定されているか確認してください\n"
        error_msg += "3. インデックスが存在し、アクセス可能か確認してください\n"
        error_msg += "4. 問題が解決しない場合は「緊急オフラインモード」を使用すると、一時的にメモリ内ストレージでアプリを使用できます\n"
        
        st.error(error_msg)
        return False

# 最初の1回だけ初期化を試みる (アプリケーション起動時)
if 'vector_store_initialized' not in st.session_state:
    print("最初のベクトルストア初期化を実行します...")
    
    # デフォルトでは接続チェックを高速にするためのフラグ
    st.session_state.connection_check_completed = False
    
    try:
        # Streamlitのサイドバーにデバッグ情報とモード選択を追加
        with st.sidebar:
            # 緊急モードの自動有効化オプション
            auto_emergency_mode = st.checkbox("緊急オフラインモードで起動", value=False, key="auto_emergency_mode")
            if auto_emergency_mode:
                st.warning("緊急オフラインモードが有効です。Pinecone接続を使用せず、メモリ内ストレージで動作します。")
            
            # アプリ実行中に常にデバッグパネルを表示
            debug_mode = st.checkbox("デバッグモードを有効化", value=True)
            if debug_mode:
                st.write("### アップロード状態")
                upload_status = st.empty()  # このコンポーネントを利用して状態を更新
        
        # ベクトルストア初期化
        vector_store = initialize_vector_store()
        
        # 接続の状態を記録
        st.session_state.connection_check_completed = True
        
        # デバッグ情報を表示
        if 'debug_mode' in st.session_state and st.session_state.debug_mode:
            with st.sidebar:
                st.write(f"初期化結果: vector_store_available = {vector_store_available}")
                if vector_store:
                    st.write(f"vector_store.available = {getattr(vector_store, 'available', 'undefined')}")
                    st.write(f"緊急モード: {getattr(vector_store, 'temporary_failure', False)}")
                    if hasattr(vector_store, 'pinecone_client'):
                        client = vector_store.pinecone_client
                        st.write(f"client.available = {getattr(client, 'available', 'undefined')}")
                        st.write(f"client.temporary_failure = {getattr(client, 'temporary_failure', False)}")
    except Exception as e:
        print(f"初期化中にエラーが発生: {e}")
        print(traceback.format_exc())
        if 'debug_mode' in st.session_state and st.session_state.debug_mode:
            with st.sidebar:
                st.error(f"初期化エラー: {str(e)}")
                st.code(traceback.format_exc(), language="python")
    
    st.session_state.vector_store_initialized = True

def register_document(uploaded_file, additional_metadata=None):
    """
    アップロードされたファイルをベクトルデータベースに登録する関数。
    additional_metadata: 追加のメタデータ辞書
    """
    # グローバル変数の宣言を最初に移動
    global vector_store, vector_store_available
    
    # ファイルアップロード処理のログ追加
    print(f"==== ファイルアップロード処理開始: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')} ====")
    if uploaded_file:
        print(f"ファイル名: {uploaded_file.name}, サイズ: {uploaded_file.size}バイト")
    
    # タイムアウト設定
    upload_timeout = 120
    start_time = time.time()
    print(f"処理タイムアウト設定: {upload_timeout}秒")
    
    # デバッグコンテナ
    debug_container = st.expander("デバッグ情報", expanded=False)
    
    # 接続状態のログ
    print(f"ベクトルDB接続状態: {'有効' if vector_store_available else '無効'}")
    if vector_store:
        print(f"vector_store.available: {getattr(vector_store, 'available', 'undefined')}")
        print(f"緊急モード: {getattr(vector_store, 'temporary_failure', False)}")
        
        if hasattr(vector_store, 'pinecone_client'):
            client = vector_store.pinecone_client
            print(f"Pineconeクライアント状態: {getattr(client, 'available', 'undefined')}")
            print(f"一時的障害モード: {getattr(client, 'temporary_failure', False)}")
    
    # ファイル処理の各ステップでログ出力
    try:
        if uploaded_file:
            print(f"ファイル読み込み開始: {time.time() - start_time:.2f}秒経過")
            # ファイル読み込み処理...
            
            # バッチ処理の開始前にログ
            print(f"ベクトルDB登録処理開始: {time.time() - start_time:.2f}秒経過")
            
            # 各バッチ処理でログ
            for batch_idx in range(バッチ数):
                print(f"バッチ {batch_idx+1} 処理開始: {time.time() - start_time:.2f}秒経過")
                # バッチ処理...
                print(f"バッチ {batch_idx+1} 処理完了: {time.time() - start_time:.2f}秒経過, 結果: {'成功' if 成功 else '失敗'}")
            
            print(f"==== ファイル処理完了: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}, 合計処理時間: {time.time() - start_time:.2f}秒 ====")
    except Exception as e:
        print(f"ファイル処理エラー: {str(e)}")
        print(traceback.format_exc())
        
        # エラーメッセージを構築
        error_msg = "ベクトルデータベースの接続でエラーが発生しました。\n\n"
        error_msg += "デバッグ情報:\n"
        error_msg += f"- vector_store_available: {vector_store_available}\n"
        error_msg += f"- error_type: {type(e).__name__}\n"
        error_msg += f"- error_message: {str(e)}\n\n"
        
        if vector_store:
            error_msg += "VectorStore情報:\n"
            error_msg += f"- available: {getattr(vector_store, 'available', 'undefined')}\n"
            error_msg += f"- temporary_failure: {getattr(vector_store, 'temporary_failure', False)}\n"
            error_msg += f"- is_streamlit_cloud: {getattr(vector_store, 'is_streamlit_cloud', False)}\n"
            
            if hasattr(vector_store, 'pinecone_client'):
                client = vector_store.pinecone_client
                error_msg += "\nPineconeクライアント情報:\n"
                error_msg += f"- available: {getattr(client, 'available', 'undefined')}\n"
                error_msg += f"- temporary_failure: {getattr(client, 'temporary_failure', False)}\n"
                error_msg += f"- failed_attempts: {getattr(client, 'failed_attempts', 0)}\n"
                error_msg += f"- initialization_error: {getattr(client, 'initialization_error', 'なし')}\n"
        
        error_msg += "\n接続問題を解決するオプション:\n"
        error_msg += "1. インターネット接続が安定しているか確認してください\n"
        error_msg += "2. Pinecone APIキーが正しく設定されているか確認してください\n"
        error_msg += "3. インデックスが存在し、アクセス可能か確認してください\n"
        error_msg += "4. 問題が解決しない場合は「緊急オフラインモード」を使用すると、一時的にメモリ内ストレージでアプリを使用できます\n"
        
        st.error(error_msg)
        return False

def manage_db():
    """
    ベクトルデータベースを管理するページの関数。
    """
    # グローバル変数の宣言を最初に移動
    global vector_store, vector_store_available

    st.header("ベクトルデータベース管理")

    # ページロード時に接続状態を確認
    try:
        if vector_store and hasattr(vector_store, 'pinecone_client'):
            client = vector_store.pinecone_client
            if hasattr(client, '_check_rest_api_connection'):
                with st.spinner("Pinecone接続状態を確認中..."):
                    api_test_result = client._check_rest_api_connection()
                    print(f"Pinecone REST API接続テスト結果: {api_test_result}")
    except Exception as e:
        print(f"ページロード時の接続確認エラー: {e}")

    # デバッグモードの表示
    if 'debug_mode' in st.session_state and st.session_state.debug_mode:
        st.write("### 現在のデバッグ情報")
        st.write(f"vector_store_available = {vector_store_available}")
        if vector_store:
            st.write(f"vector_store.available = {getattr(vector_store, 'available', 'undefined')}")
            st.write(f"vector_store.temporary_failure = {getattr(vector_store, 'temporary_failure', False)}")
            if hasattr(vector_store, 'pinecone_client'):
                client = vector_store.pinecone_client
                st.write(f"client.available = {getattr(client, 'available', 'undefined')}")
                st.write(f"client.temporary_failure = {getattr(client, 'temporary_failure', False)}")
                st.write(f"client.is_streamlit_cloud = {getattr(client, 'is_streamlit_cloud', False)}")
                st.write(f"client.failed_attempts = {getattr(client, 'failed_attempts', 0)}")
                
                # APIリクエスト結果を表示
                if hasattr(client, '_make_request'):
                    try:
                        with st.expander("Pinecone API接続テスト", expanded=False):
                            with st.spinner("API接続テスト実行中..."):
                                api_url = "https://api.pinecone.io/indexes"
                                response = client._make_request(
                                    method="GET", 
                                    url=api_url, 
                                    max_retries=1, 
                                    timeout=5
                                )
                                if response:
                                    st.success(f"API応答: ステータスコード {response.status_code}")
                                    if response.status_code == 200:
                                        try:
                                            st.json(response.json())
                                        except:
                                            st.text(response.text[:500])
                                else:
                                    st.error("API接続テスト失敗: レスポンスなし")
                    except Exception as e:
                        st.error(f"API接続テストエラー: {str(e)}")

    # 緊急モードの検出
    emergency_mode = False
    if vector_store:
        emergency_mode = getattr(vector_store, 'temporary_failure', False) and getattr(vector_store, 'is_streamlit_cloud', False)
        if emergency_mode:
            st.warning("⚠️ 緊急オフラインモードで動作中です。Pineconeへの接続は一時的に無効化されています。メモリ内ストレージを使用します。")
            # 緊急モード時は制限された機能を表示
            offline_storage = getattr(vector_store, 'offline_storage', None)
            if offline_storage:
                item_count = len(offline_storage.get("ids", []))
                st.info(f"メモリ内ストレージには現在 {item_count} 件のベクトルが保存されています")
                
                # 緊急モードを解除するボタン
                if st.button("緊急モードを解除"):
                    try:
                        vector_store.temporary_failure = False
                        if hasattr(vector_store, 'pinecone_client'):
                            vector_store.pinecone_client.temporary_failure = False
                            vector_store.pinecone_client.failed_attempts = 0
                        st.success("緊急モードを解除しました。ページを再読み込みします。")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"緊急モード解除中にエラーが発生しました: {e}")
            
            # 緊急モードでもアップロード機能を提供する
            st.subheader("ドキュメントを緊急モードで登録")
            # （以下、アップロード用のUIコードをシンプルに提供）
            uploaded_file = st.file_uploader('テキストをアップロードしてください', type='txt', key="emergency_uploader")
            if uploaded_file and st.button("緊急モードで登録"):
                with st.spinner('メモリ内ストレージに登録中...'):
                    register_document(uploaded_file, additional_metadata={"emergency_mode": True})

    if not vector_store_available and not emergency_mode:
        error_message = "ベクトルデータベースの接続でエラーが発生しました。現在、ベクトルデータベースは使用できません。"
        
        # より詳細なエラー情報を表示
        if hasattr(vector_store, 'pinecone_client') and vector_store.pinecone_client:
            client = vector_store.pinecone_client
            error_message += "\n\n接続状態の詳細:"
            
            # クライアントの利用可能性
            client_available = getattr(client, 'available', False)
            error_message += f"\n- Pineconeクライアント: {'利用可能' if client_available else '利用不可'}"
            
            # 初期化エラー
            if hasattr(client, 'initialization_error'):
                error_message += f"\n- 初期化エラー: {client.initialization_error}"
            
            # REST API接続状態
            if hasattr(client, '_check_rest_api_connection'):
                api_available = client._check_rest_api_connection()
                error_message += f"\n- REST API接続: {'成功' if api_available else '失敗'}"
            
            # インデックス情報
            if hasattr(client, 'index_name'):
                error_message += f"\n- インデックス名: {client.index_name}"
        
        st.error(error_message)
        
        # デバッグ情報を追加
        st.write("## デバッグ情報")
        st.write(f"vector_store_available: {vector_store_available}")
        if vector_store:
            st.write("vector_storeオブジェクト情報:")
            st.write(f"- 型: {type(vector_store)}")
            st.write(f"- 利用可能: {getattr(vector_store, 'available', 'undefined')}")
            if hasattr(vector_store, 'pinecone_client'):
                client = vector_store.pinecone_client
                st.write("Pineconeクライアント情報:")
                st.write(f"- 型: {type(client)}")
                st.write(f"- 利用可能: {getattr(client, 'available', 'undefined')}")
                st.write(f"- REST API接続: {hasattr(client, '_check_rest_api_connection')}")
                # REST API接続をテストして結果を表示
                if hasattr(client, '_check_rest_api_connection'):
                    try:
                        rest_api_status = client._check_rest_api_connection()
                        st.write(f"- REST API接続テスト結果: {rest_api_status}")
                    except Exception as e:
                        st.write(f"- REST API接続テスト中にエラー: {str(e)}")
                if hasattr(client, 'initialization_error'):
                    st.write(f"- 初期化エラー: {client.initialization_error}")
        
        # 緊急モードへの切り替えボタン
        st.subheader("接続問題を解決するオプション")
        col1, col2 = st.columns(2)
        
        with col1:
            if vector_store and st.button("緊急オフラインモードに切り替え"):
                try:
                    if hasattr(vector_store, 'pinecone_client'):
                        vector_store.temporary_failure = True
                        vector_store.is_streamlit_cloud = True
                        
                        if hasattr(vector_store.pinecone_client, 'temporary_failure'):
                            vector_store.pinecone_client.temporary_failure = True
                            vector_store.pinecone_client.is_streamlit_cloud = True
                            vector_store.pinecone_client.failed_attempts = 3
                        
                        st.success("緊急オフラインモードに切り替えました。ページを再読み込みします。")
                        time.sleep(1)
                        st.rerun()
                except Exception as e:
                    st.error(f"緊急モードへの切り替え中にエラーが発生しました: {e}")
        
        # リトライボタンを提供
        with col2:
            if st.button("接続を再試行"):
                with st.spinner("Pineconeへの接続を再試行しています..."):
                    try:
                        # セッション状態をクリア
                        if 'vector_store' in st.session_state:
                            del st.session_state.vector_store
                        if 'vector_store_initialized' in st.session_state:
                            del st.session_state.vector_store_initialized
                        
                        # 再初期化
                        initialize_vector_store()
                        
                        # 接続状態を確認
                        if vector_store_available:
                            st.success("接続に成功しました！ページを再読み込みします。")
                            st.rerun()
                        else:
                            st.error("接続の再試行は完了しましたが、まだ使用できない状態です。")
                            st.write("デバッグ情報:")
                            st.write(f"vector_store_available: {vector_store_available}")
                            if vector_store:
                                st.write(f"vector_store.available: {getattr(vector_store, 'available', 'undefined')}")
                    except Exception as e:
                        st.error(f"接続の再試行に失敗しました: {e}")
                        st.error("エラーの詳細:")
                        st.exception(e)
        
        # 代わりにREST API経由での解決方法を提案
        st.info("""
        注: Pineconeのコントローラーサーバーに接続できない場合は、REST API経由での接続は成功している可能性があります。
        アプリケーションはREST APIを自動的に使用します。
        
        以下の点を確認してください：
        1. インターネット接続が安定しているか
        2. Pinecone APIキーが正しく設定されているか
        3. インデックスが存在し、アクセス可能か
        
        問題が解決しない場合は「緊急オフラインモード」を使用すると、一時的にメモリ内ストレージでアプリを使用できます。
        """)
        return
    
    # 1.ドキュメント登録
    st.subheader("ドキュメントをデータベースに登録")
    
    # ファイルアップロード
    uploaded_file = st.file_uploader('テキストをアップロードしてください', type='txt')
    
    if uploaded_file:
        # メタデータ入力フォーム
        with st.expander("メタデータ入力", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                municipality = st.text_input("市区町村名", "")
                major_category = st.selectbox(
                    "大カテゴリ",
                    MAJOR_CATEGORIES
                )
                medium_category = st.selectbox(
                    "中カテゴリ",
                    MEDIUM_CATEGORIES.get(major_category, [])
                )
            
            with col2:
                source = st.text_input("ソース元", "")
                date_time = st.date_input("登録日時", value=datetime.date.today())
                publication_date = st.date_input("データ公開日", value=None)
                latitude = st.text_input("緯度", "")
                longitude = st.text_input("経度", "")
        
        # 登録ボタン
        if st.button("登録する"):
            with st.spinner('登録中...'):
                # メタデータの作成
                metadata = {
                    "municipality": municipality,
                    "major_category": major_category,
                    "medium_category": medium_category,
                    "source": source,
                    "registration_date": str(date_time) if date_time else "",
                    "publication_date": str(publication_date) if publication_date else "",
                    "latitude": latitude,
                    "longitude": longitude,
                }
                
                # ドキュメント登録関数を呼び出し
                register_document(uploaded_file, additional_metadata=metadata)

    st.markdown("---")

    # 2.登録状況確認
    st.subheader("ベクトルデータベース 登録状況確認")
    
    # 検索フィルター
    with st.expander("検索フィルター", expanded=False):
        filter_municipality = st.text_input("市区町村名で絞り込み", "")
        filter_category = st.text_input("カテゴリで絞り込み", "")
    
    # 表示ボタン
    if st.button("登録済みドキュメントを表示"):
        with st.spinner('取得中...'):
            try:
                # ドキュメント数を取得
                count = vector_store.count()
                st.info(f"データベースには{count}件のドキュメントが登録されています")
                
                # 注意: Pineconeは全件取得に対応していないため、検索結果のみ表示
                st.warning("Pineconeでは全件表示ができません。検索フォームを使ってドキュメントを検索してください。")
                
            except Exception as e:
                st.error(f"ドキュメント取得中にエラーが発生しました: {e}")
                st.exception(e)
    
    # 3.データベース操作（メンテナンス機能）
    with st.expander("データベースメンテナンス", expanded=False):
        st.warning("⚠️ 以下の操作は慎重に行ってください")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 特定IDのドキュメント削除
            delete_id = st.text_input("削除するドキュメントID", "")
            if st.button("ドキュメントを削除") and delete_id:
                with st.spinner('削除中...'):
                    try:
                        result = vector_store.delete_documents([delete_id])
                        if result:
                            st.success(f"ドキュメント {delete_id} を削除しました")
                        else:
                            st.error(f"ドキュメント {delete_id} の削除に失敗しました")
                    except Exception as e:
                        st.error(f"削除中にエラーが発生しました: {e}")
                        st.exception(e)

# ページ関数の定義 - チャットインターフェースの実装
def chat_interface():
    # タイトル表示
    st.header("ドキュメントに質問する")
    
    # リセットボタン
    if st.sidebar.button("会話をリセット"):
        chat_history.clear_history()
        st.sidebar.success("会話履歴をリセットしました")
        st.rerun()
    
    # プロンプトの選択
    selected_prompt = st.sidebar.selectbox(
        "プロンプトを選択",
        options=[p['name'] for p in st.session_state.custom_prompts],
        index=0
    )
    st.session_state.selected_prompt = selected_prompt
    
    # 選択されたプロンプトのテンプレートを取得
    prompt_template = next(
        (p['content'] for p in st.session_state.custom_prompts if p['name'] == selected_prompt),
        RAG_PROMPT_TEMPLATE
    )
    
    # 会話履歴の表示
    for message in chat_history.get_history():
        role = message["role"]
        content = message["content"]
        with st.chat_message(role):
            st.markdown(content)
    
    if not vector_store_available:
        st.error("ベクトルデータベースが利用できないため、質問応答機能は制限されます。")
        st.info("Pineconeを設定するか、ローカル環境で実行してください。")
    
    # 質問入力
    if question := st.chat_input("質問を入力してください"):
        # ユーザーの質問をチャット履歴に追加
        with st.chat_message("user"):
            st.markdown(question)
        chat_history.add_message("user", question)
        
        # 回答を生成
        with st.chat_message("assistant"):
            with st.spinner("回答を考え中..."):
                try:
                    # 質問をベクトル化して関連ドキュメントを検索
                    if vector_store_available:
                        filter_conditions = {}  # 必要に応じてフィルター条件を追加
                        search_results = vector_store.search(question, n_results=5, filter_conditions=filter_conditions)
                        contexts = []
                        
                        if search_results and len(search_results["documents"][0]) > 0:
                            for i, (doc, metadata) in enumerate(zip(search_results["documents"][0], search_results["metadatas"][0])):
                                context = f"出典: {metadata.get('source', 'unknown')}\n内容: {doc}"
                                contexts.append(context)
                        
                        # コンテキストが見つからない場合
                        if not contexts:
                            answer = "申し訳ありませんが、その質問に答えるための関連情報が見つかりませんでした。別の質問をしてみるか、より多くの文書を登録してください。"
                            st.markdown(answer)
                            chat_history.add_message("assistant", answer)
                            return
                    else:
                        # ベクトルストアが利用できない場合は一般的な回答
                        contexts = ["ベクトルデータベースが使用できないため、登録済みドキュメントにアクセスできません。一般的な応答のみを提供します。"]
                    
                    # コンテキストを使ってLLMで回答を生成
                    prompt = ChatPromptTemplate.from_template(prompt_template)
                    
                    chain = (
                        {"context": lambda _: "\n\n".join(contexts), "question": lambda x: x}
                        | prompt
                        | llm
                        | StrOutputParser()
                    )
                    
                    answer = chain.invoke(question)
                    st.markdown(answer)
                    
                    # 回答を履歴に追加
                    chat_history.add_message("assistant", answer)
                    
                except Exception as e:
                    error_message = f"回答の生成中にエラーが発生しました: {e}"
                    st.error(error_message)
                    chat_history.add_message("assistant", error_message)

# ページ関数の定義 - プロンプト管理
def prompt_management():
    st.header("プロンプト管理")
    
    # 現在のプロンプト一覧を表示
    st.subheader("登録済みプロンプト")
    
    # プロンプト選択用のセレクトボックス
    prompt_names = [p['name'] for p in st.session_state.custom_prompts]
    selected_index = prompt_names.index(st.session_state.selected_prompt) if st.session_state.selected_prompt in prompt_names else 0
    
    selected_prompt_name = st.selectbox(
        "編集するプロンプトを選択",
        options=prompt_names,
        index=selected_index
    )
    
    # 選択されたプロンプトの内容を取得
    selected_prompt = next((p for p in st.session_state.custom_prompts if p['name'] == selected_prompt_name), None)
    
    if selected_prompt:
        # プロンプト編集フォーム
        with st.form(key="edit_prompt_form"):
            prompt_name = st.text_input("プロンプト名", value=selected_prompt['name'])
            prompt_content = st.text_area("プロンプト内容", value=selected_prompt['content'], height=300)
            
            col1, col2 = st.columns(2)
            submit_button = col1.form_submit_button("更新")
            delete_button = col2.form_submit_button("削除", type="secondary")
            
            if submit_button and prompt_name and prompt_content:
                # 同じ名前のプロンプトを更新
                for i, p in enumerate(st.session_state.custom_prompts):
                    if p['name'] == selected_prompt_name:
                        st.session_state.custom_prompts[i] = {
                            'name': prompt_name,
                            'content': prompt_content
                        }
                        break
                
                # 選択されているプロンプト名も更新
                if st.session_state.selected_prompt == selected_prompt_name:
                    st.session_state.selected_prompt = prompt_name
                
                st.success(f"プロンプト '{prompt_name}' を更新しました")
                st.rerun()
            
            if delete_button and len(st.session_state.custom_prompts) > 1:
                # プロンプトを削除（デフォルトは削除不可）
                if selected_prompt_name == "デフォルト":
                    st.error("デフォルトプロンプトは削除できません")
                else:
                    st.session_state.custom_prompts = [p for p in st.session_state.custom_prompts if p['name'] != selected_prompt_name]
                    
                    # 選択されているプロンプトが削除された場合はデフォルトに戻す
                    if st.session_state.selected_prompt == selected_prompt_name:
                        st.session_state.selected_prompt = "デフォルト"
                    
                    st.success(f"プロンプト '{selected_prompt_name}' を削除しました")
                    st.rerun()
    
    # 新規プロンプト追加
    st.subheader("新規プロンプト追加")
    
    with st.form(key="add_prompt_form"):
        new_prompt_name = st.text_input("新規プロンプト名")
        new_prompt_content = st.text_area("新規プロンプト内容", value=RAG_PROMPT_TEMPLATE, height=300)
        
        submit_button = st.form_submit_button("追加")
        
        if submit_button and new_prompt_name and new_prompt_content:
            # 同名のプロンプトがないか確認
            if any(p['name'] == new_prompt_name for p in st.session_state.custom_prompts):
                st.error(f"プロンプト名 '{new_prompt_name}' は既に使用されています。別の名前を選択してください。")
            else:
                # 新規プロンプトを追加
                st.session_state.custom_prompts.append({
                    'name': new_prompt_name,
                    'content': new_prompt_content
                })
                st.success(f"新規プロンプト '{new_prompt_name}' を追加しました")
                st.rerun()

# ダッシュボード表示
def dashboard():
    st.header("ダッシュボード")
    
    # システム情報
    st.subheader("システム情報")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("会話数", len(chat_history.get_history()) // 2)
        
    with col2:
        if vector_store_available:
            doc_count = vector_store.count()
            st.metric("登録ドキュメント数", doc_count)
        else:
            st.metric("登録ドキュメント数", "N/A")
            st.info("ベクトルデータベースが接続されていません")
    
    # 会話ログのエクスポート
    st.subheader("会話ログのエクスポート")
    
    if st.button("会話ログをCSVでダウンロード"):
        csv_data = chat_history.get_csv_export()
        if csv_data:
            # CSVデータをダウンロード可能にする
            filename = f"chat_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            st.download_button(
                label="ダウンロード",
                data=csv_data,
                file_name=filename,
                mime="text/csv",
            )
        else:
            st.info("エクスポートする会話履歴がありません")
    
    # 環境変数の確認
    with st.expander("環境変数", expanded=False):
        if "OPENAI_API_KEY" in os.environ:
            st.success("OPENAI_API_KEY: 設定済み")
        else:
            st.error("OPENAI_API_KEY: 未設定")
            
        if "PINECONE_API_KEY" in os.environ:
            st.success("PINECONE_API_KEY: 設定済み")
        else:
            st.error("PINECONE_API_KEY: 未設定")
            
        if "PINECONE_ENVIRONMENT" in os.environ:
            st.success(f"PINECONE_ENVIRONMENT: {os.environ.get('PINECONE_ENVIRONMENT')}")
        else:
            st.warning("PINECONE_ENVIRONMENT: 未設定 (デフォルト値使用)")

# サイドバーメニュー
st.sidebar.title("メニュー")
page = st.sidebar.radio(
    "ページを選択",
    ["ドキュメントに質問する", "ベクトルDB管理", "プロンプト管理", "ダッシュボード"]
)

# セッション終了時に会話履歴を保存
try:
    if chat_history.pinecone_available:
        saved = chat_history.force_save()
        if saved:
            print("会話履歴を保存しました")
except Exception as e:
    print(f"会話履歴の保存中にエラー: {e}")

# ページに応じた表示
if page == "ドキュメントに質問する":
    chat_interface()
elif page == "ベクトルDB管理":
    manage_db()
elif page == "プロンプト管理":
    prompt_management()
elif page == "ダッシュボード":
    dashboard()

# アプリケーション起動時のセッション状態初期化
import streamlit as st
import traceback

# セッション状態のリセット
if 'force_reset' not in st.session_state:
    print("セッション状態の初期化を実行中...")
    for key in ['vector_store', 'vector_store_initialized']:
        if key in st.session_state:
            print(f"セッション状態から {key} を削除")
            del st.session_state[key]
    st.session_state.force_reset = True
    print("セッション状態の初期化完了")

# 設定とパフォーマンスの確認
import os
# Pinecone接続タイムアウトの設定
os.environ['PINECONE_REQUEST_TIMEOUT'] = '60'  # 秒単位でタイムアウトを設定
print(f"Pineconeリクエストタイムアウトを {os.environ.get('PINECONE_REQUEST_TIMEOUT', '設定なし')} 秒に設定")

# デバッグ情報をログに出力
print("システム情報:")
print(f"- Python バージョン: {platform.python_version()}")
print(f"- プラットフォーム: {platform.platform()}")
