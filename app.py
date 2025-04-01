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
import logging
from logging.handlers import RotatingFileHandler

# ログ出力の設定
def setup_logging():
    """ロガーの設定"""
    # 既存のハンドラをクリア
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # ロガーの作成
    logger = logging.getLogger('app')
    logger.setLevel(logging.INFO)
    
    # 既存のハンドラをクリア
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # フォーマッターの設定
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # ファイルハンドラの設定
    file_handler = logging.FileHandler('app.log', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # コンソールハンドラの設定
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # ログの伝播を防止
    logger.propagate = False
    
    return logger

# ロガーの初期化
logger = setup_logging()

# 環境変数を確実にロード
load_dotenv(override=True)

# グローバル変数の初期化
vector_store = None
vector_store_available = False

# Pinecone APIキーが環境変数から正しく読み込まれているか確認
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_env = os.environ.get("PINECONE_ENVIRONMENT")
pinecone_index = os.environ.get("PINECONE_INDEX")

# 環境変数の確認（1回だけ出力）
logger.info("環境変数の確認:")
logger.info(f"- PINECONE_API_KEY: {'設定済み' if pinecone_api_key else '未設定'}")
logger.info(f"- PINECONE_ENVIRONMENT: {pinecone_env}")
logger.info(f"- PINECONE_INDEX: {pinecone_index}")

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

# VectorStoreのインスタンスを取得する関数
def get_vector_store():
    """ベクトルストアのインスタンスを取得する"""
    global vector_store, vector_store_available
    
    try:
        # ベクトルストアが初期化されていない場合は初期化を試みる
        if vector_store is None:
            logger.info("ベクトルストアが未初期化のため、初期化を試みます")
            initialize_vector_store()
        
        # ベクトルストアの状態を確認
        if vector_store is None:
            logger.error("ベクトルストアの初期化に失敗しました")
            return None
            
        # 利用可能かどうかを確認
        vector_store_available = getattr(vector_store, 'available', False)
        logger.info(f"ベクトルストアの状態: {'利用可能' if vector_store_available else '利用不可'}")
        
        return vector_store
        
    except Exception as e:
        logger.error(f"ベクトルストアの取得中にエラー: {e}")
        logger.error(traceback.format_exc())
        return None

def register_document(uploaded_file):
    """アップロードされたファイルをドキュメントとして登録"""
    try:
        logger.info("ファイルアップロード処理開始: %s", datetime.datetime.now())
        logger.info("ファイル情報:")
        logger.info("- ファイル名: %s", uploaded_file.name)
        logger.info("- ファイルタイプ: %s", uploaded_file.type)
        logger.info("- ファイルサイズ: %s bytes", uploaded_file.size)
        logger.info("- アップロード時刻: %s", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        logger.info("=" * 50)

        # 処理タイムアウトの設定
        timeout = 120  # 秒
        logger.info("処理タイムアウト設定: %d秒", timeout)

        # ベクトルDBの接続状態を確認
        vector_store = get_vector_store()
        if vector_store is None:
            logger.error("ベクトルストアの取得に失敗しました")
            return False
            
        logger.info("ベクトルDB接続状態: %s", "有効" if vector_store.available else "無効")
        logger.info("vector_store.available: %s", vector_store.available)
        logger.info("緊急モード: %s", vector_store.temporary_failure)
        logger.info("Pineconeクライアント状態: %s", getattr(vector_store.pinecone_client, 'available', False))
        logger.info("一時的障害モード: %s", vector_store.temporary_failure)

        # ファイル読み込み開始時刻を記録
        start_time = time.time()
        logger.info("ファイル読み込み開始: %.2f秒経過", time.time() - start_time)

        # ファイルの内容を読み込む（複数のエンコーディングを試行）
        encodings = ['utf-8', 'shift-jis', 'cp932', 'euc-jp', 'iso-2022-jp']
        content = None
        
        for encoding in encodings:
            try:
                # ファイルポインタを先頭に戻す
                uploaded_file.seek(0)
                content = uploaded_file.read().decode(encoding)
                logger.info("ファイルを %s エンコーディングで読み込み成功", encoding)
                break
            except UnicodeDecodeError:
                logger.warning("%s エンコーディングでの読み込みに失敗", encoding)
                continue
        
        if content is None:
            raise ValueError("ファイルの文字エンコーディングを特定できませんでした")

        # ファイル読み込み完了時刻を記録
        logger.info("ファイル読み込み完了: %.2f秒経過", time.time() - start_time)

        # ドキュメントの登録
        if vector_store.available:
            success = vector_store.upsert_documents([content])
            if success:
                logger.info("ドキュメントの登録が完了しました")
                return True
            else:
                logger.error("ドキュメントの登録に失敗しました")
                return False
        else:
            logger.error("ベクトルDBが利用できません")
            return False

    except Exception as e:
        logger.error("ファイル処理エラー: %s", str(e))
        logger.error(traceback.format_exc())
        logger.error("ドキュメント登録に失敗しました")
        return False

def manage_db():
    """
    ベクトルデータベースを管理するページの関数。
    """
    # グローバル変数の宣言を最初に移動
    global vector_store, vector_store_available

    logger.info("="*50)
    logger.info(f"ベクトルDB管理ページを開く: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ベクトルストアの状態を再確認
    if not vector_store or not vector_store_available:
        logger.warning("ベクトルストアが利用できないため、再初期化を試みます")
        initialize_vector_store()
    
    logger.info(f"現在の状態:")
    logger.info(f"- vector_store_available: {vector_store_available}")
    if vector_store:
        logger.info(f"- vector_store.available: {getattr(vector_store, 'available', 'undefined')}")
        logger.info(f"- 緊急モード: {getattr(vector_store, 'temporary_failure', False)}")
        if hasattr(vector_store, 'pinecone_client'):
            client = vector_store.pinecone_client
            logger.info(f"- Pineconeクライアント状態: {getattr(client, 'available', 'undefined')}")
            logger.info(f"- 一時的障害モード: {getattr(client, 'temporary_failure', False)}")
    logger.info("="*50)

    st.header("ベクトルデータベース管理")

    # ページロード時に接続状態を確認
    try:
        if vector_store and hasattr(vector_store, 'pinecone_client'):
            client = vector_store.pinecone_client
            if hasattr(client, '_check_rest_api_connection'):
                logger.info("Pinecone接続状態を確認中...")
                with st.spinner("Pinecone接続状態を確認中..."):
                    api_test_result = client._check_rest_api_connection()
                    logger.info(f"Pinecone REST API接続テスト結果: {api_test_result}")
                    if not api_test_result:
                        logger.warning("Pinecone接続テストに失敗しました")
                        st.warning("Pineconeへの接続に問題があります。一部の機能が制限される可能性があります。")
    except Exception as e:
        logger.error(f"ページロード時の接続確認エラー: {e}")
        logger.error(traceback.format_exc())
        st.error("接続状態の確認中にエラーが発生しました")

    # デバッグモードの表示
    if 'debug_mode' in st.session_state and st.session_state.debug_mode:
        logger.info("デバッグモードが有効です")
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

    # 1.ドキュメント登録
    st.subheader("ドキュメントをデータベースに登録")
    
    # ファイルアップロード
    uploaded_file = st.file_uploader('テキストをアップロードしてください', type='txt')
    
    if uploaded_file:
        logger.info(f"ファイルアップロード検知: {uploaded_file.name}")
        logger.info(f"ファイル情報: タイプ={uploaded_file.type}, サイズ={uploaded_file.size:,} bytes")
        
        # ベクトルストアの状態を再確認
        if not vector_store or not vector_store_available:
            logger.warning("ファイルアップロード時にベクトルストアが利用できないため、再初期化を試みます")
            initialize_vector_store()
            if not vector_store_available:
                st.error("ベクトルストアの接続に問題があります。緊急オフラインモードを使用するか、接続を確認してください。")
                return
        
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
            logger.info("ドキュメント登録処理を開始")
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
                logger.info(f"メタデータ: {metadata}")
                
                # ドキュメント登録関数を呼び出し
                if register_document(uploaded_file):
                    logger.info("ドキュメント登録が完了しました")
                    st.success("ドキュメントの登録が完了しました！")
                else:
                    logger.error("ドキュメント登録に失敗しました")
                    st.error("ドキュメントの登録に失敗しました。")

    st.markdown("---")

    # 2.登録状況確認
    st.subheader("ベクトルデータベース 登録状況確認")
    
    # 検索フィルター
    with st.expander("検索フィルター", expanded=False):
        filter_municipality = st.text_input("市区町村名で絞り込み", "")
        filter_category = st.text_input("カテゴリで絞り込み", "")
    
    # 表示ボタン
    if st.button("登録済みドキュメントを表示"):
        logger.info("登録済みドキュメントの表示を開始")
        with st.spinner('取得中...'):
            try:
                # ドキュメント数を取得
                count = vector_store.count()
                logger.info(f"データベース内のドキュメント数: {count}")
                st.info(f"データベースには{count}件のドキュメントが登録されています")
                
                # 注意: Pineconeは全件取得に対応していないため、検索結果のみ表示
                st.warning("Pineconeでは全件表示ができません。検索フォームを使ってドキュメントを検索してください。")
                
            except Exception as e:
                logger.error(f"ドキュメント取得中にエラー: {e}")
                logger.error(traceback.format_exc())
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
                logger.info(f"ドキュメント削除処理を開始: ID={delete_id}")
                with st.spinner('削除中...'):
                    try:
                        result = vector_store.delete_documents([delete_id])
                        if result:
                            logger.info(f"ドキュメント {delete_id} の削除に成功")
                            st.success(f"ドキュメント {delete_id} を削除しました")
                        else:
                            logger.error(f"ドキュメント {delete_id} の削除に失敗")
                            st.error(f"ドキュメント {delete_id} の削除に失敗しました")
                    except Exception as e:
                        logger.error(f"削除中にエラー: {e}")
                        logger.error(traceback.format_exc())
                        st.error(f"削除中にエラーが発生しました: {e}")
                        st.exception(e)

    logger.info("="*50)
    logger.info(f"ベクトルDB管理ページの処理を完了: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*50)

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

# ログ表示用のエクスパンダーを追加
with st.sidebar.expander("アプリケーションログ", expanded=True):
    # ログファイルの内容を読み込んで表示
    log_dir = "logs"
    if os.path.exists(log_dir):
        log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
        if log_files:
            # 最新のログファイルを選択
            latest_log = max(log_files, key=lambda x: os.path.getctime(os.path.join(log_dir, x)))
            log_path = os.path.join(log_dir, latest_log)
            
            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    # 最新の100行のみを表示
                    log_lines = f.readlines()[-100:]
                    for line in log_lines:
                        # エラーログは赤色で表示
                        if "ERROR" in line:
                            st.error(line.strip())
                        # 警告ログは黄色で表示
                        elif "WARNING" in line:
                            st.warning(line.strip())
                        # その他のログは通常表示
                        else:
                            st.text(line.strip())
            except Exception as e:
                st.error(f"ログファイルの読み込み中にエラーが発生しました: {str(e)}")
        else:
            st.info("ログファイルが見つかりません")
    else:
        st.info("ログディレクトリが存在しません")

# メニュー選択
page = st.sidebar.radio(
    "ページを選択",
    ["ドキュメントに質問する", "ベクトルDB管理", "プロンプト管理", "ダッシュボード"]
)

# セッション終了時に会話履歴を保存
try:
    if chat_history.pinecone_available:
        saved = chat_history.force_save()
        if saved:
            logger.info("会話履歴を保存しました")
except Exception as e:
    logger.error(f"会話履歴の保存中にエラー: {e}")

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
    logger.info("セッション状態の初期化を実行中...")
    for key in ['vector_store', 'vector_store_initialized']:
        if key in st.session_state:
            logger.info(f"セッション状態から {key} を削除")
            del st.session_state[key]
    st.session_state.force_reset = True
    logger.info("セッション状態の初期化完了")

# 設定とパフォーマンスの確認
import os
# Pinecone接続タイムアウトの設定
os.environ['PINECONE_REQUEST_TIMEOUT'] = '60'  # 秒単位でタイムアウトを設定
logger.info(f"Pineconeリクエストタイムアウトを {os.environ.get('PINECONE_REQUEST_TIMEOUT', '設定なし')} 秒に設定")

# デバッグ情報をログに出力
logger.info("システム情報:")
logger.info(f"- Python バージョン: {platform.python_version()}")
logger.info(f"- プラットフォーム: {platform.platform()}")

def process_uploaded_file(uploaded_file):
    """アップロードされたファイルを処理する"""
    try:
        logger.info("="*50)
        logger.info(f"ファイルアップロード処理開始: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"ファイル情報:")
        logger.info(f"- ファイル名: {uploaded_file.name}")
        logger.info(f"- ファイルタイプ: {uploaded_file.type}")
        logger.info(f"- ファイルサイズ: {uploaded_file.size:,} bytes")
        logger.info(f"- アップロード時刻: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*50)

        # ファイルの内容を読み込む
        logger.info("ファイル内容の読み込みを開始...")
        content = uploaded_file.read()
        logger.info(f"ファイル内容の読み込み完了: {len(content):,} bytes")
        logger.info(f"文字数: {len(content):,} 文字")

        # テキストに変換
        logger.info("テキストへの変換を開始...")
        text = content.decode('utf-8')
        logger.info(f"テキスト変換完了: {len(text):,} 文字")
        logger.info(f"最初の100文字: {text[:100]}...")

        # チャンクに分割
        logger.info("テキストの分割を開始...")
        chunks = split_text(text)
        logger.info(f"テキスト分割完了: {len(chunks):,} チャンク")
        logger.info(f"チャンクサイズ情報:")
        logger.info(f"- 最小チャンクサイズ: {min(len(chunk) for chunk in chunks):,} 文字")
        logger.info(f"- 最大チャンクサイズ: {max(len(chunk) for chunk in chunks):,} 文字")
        logger.info(f"- 平均チャンクサイズ: {sum(len(chunk) for chunk in chunks) / len(chunks):,.1f} 文字")

        # ベクトルストアに保存
        if vector_store and vector_store.available:
            logger.info("ベクトルストアへの保存を開始...")
            logger.info(f"保存先: {vector_store.__class__.__name__}")
            start_time = time.time()
            
            for i, chunk in enumerate(chunks, 1):
                try:
                    chunk_start_time = time.time()
                    vector_store.add_texts([chunk])
                    chunk_end_time = time.time()
                    logger.info(f"チャンク {i}/{len(chunks)} の保存完了:")
                    logger.info(f"- 処理時間: {chunk_end_time - chunk_start_time:.2f}秒")
                    logger.info(f"- チャンクサイズ: {len(chunk):,} 文字")
                except Exception as e:
                    logger.error(f"チャンク {i}/{len(chunks)} の保存中にエラー:")
                    logger.error(f"- エラータイプ: {type(e).__name__}")
                    logger.error(f"- エラーメッセージ: {str(e)}")
                    logger.error(f"- スタックトレース: {traceback.format_exc()}")
                    raise
            
            end_time = time.time()
            logger.info(f"すべてのチャンクの保存が完了:")
            logger.info(f"- 総処理時間: {end_time - start_time:.2f}秒")
            logger.info(f"- 平均処理時間/チャンク: {(end_time - start_time) / len(chunks):.2f}秒")
        else:
            logger.warning("ベクトルストアが利用できないため、メモリに保存")
            if 'memory_store' not in st.session_state:
                st.session_state.memory_store = []
            st.session_state.memory_store.extend(chunks)
            logger.info(f"メモリに {len(chunks):,} チャンクを保存")
            logger.info(f"現在のメモリ内チャンク総数: {len(st.session_state.memory_store):,}")

        logger.info("="*50)
        logger.info(f"ファイル処理完了: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*50)
        return True
    except Exception as e:
        logger.error("="*50)
        logger.error(f"ファイル処理中にエラーが発生: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.error(f"エラーの詳細:")
        logger.error(f"- エラータイプ: {type(e).__name__}")
        logger.error(f"- エラーメッセージ: {str(e)}")
        logger.error(f"- スタックトレース: {traceback.format_exc()}")
        logger.error("="*50)
        return False

def main():
    """メインアプリケーション"""
    try:
        # ログ出力の開始
        logger.info("アプリケーション起動")
        logger.info("システム情報:")
        logger.info(f"- Python バージョン: {sys.version}")
        logger.info(f"- プラットフォーム: {platform.platform()}")

        # 初期化処理
        initialize_session_state()
        logger.info("セッション状態の初期化完了")

        # ベクトルストアの初期化
        if not st.session_state.get('vector_store_initialized'):
            logger.info("最初のベクトルストア初期化を実行します...")
            initialize_vector_store()
            logger.info("ベクトルストアの初期化完了")

        # タイトルと説明
        st.title("📚 ドキュメント検索チャット")
        st.markdown("""
        ### 使い方
        1. ドキュメントをアップロード
        2. 質問を入力
        3. 関連する情報を基に回答を生成
        """)

        # ファイルアップロード
        uploaded_file = st.file_uploader("ドキュメントをアップロード", type=['txt', 'pdf', 'doc', 'docx'])
        if uploaded_file:
            logger.info(f"ファイルアップロード検知: {uploaded_file.name}")
            if process_uploaded_file(uploaded_file):
                st.success("ファイルの処理が完了しました！")
                logger.info("ファイル処理が正常に完了")
            else:
                st.error("ファイルの処理中にエラーが発生しました。")
                logger.error("ファイル処理が失敗")

        # チャットインターフェース
        if "messages" not in st.session_state:
            st.session_state.messages = []
            logger.info("チャット履歴を初期化")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                logger.info(f"メッセージ表示: {message['role']}")

        if prompt := st.chat_input("質問を入力してください"):
            logger.info(f"新しい質問を受信: {prompt}")
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                try:
                    response = get_response(prompt)
                    st.markdown(response)
                    logger.info("アシスタントの応答を生成")
                except Exception as e:
                    error_message = f"エラーが発生しました: {str(e)}"
                    st.error(error_message)
                    logger.error(f"応答生成中にエラー: {str(e)}")
                    logger.error(f"エラーの詳細: {traceback.format_exc()}")

            st.session_state.messages.append({"role": "assistant", "content": response})
            logger.info("チャット履歴を更新")

    except Exception as e:
        logger.error(f"アプリケーション実行中にエラー: {str(e)}")
        logger.error(f"エラーの詳細: {traceback.format_exc()}")
        st.error("アプリケーションでエラーが発生しました。")
