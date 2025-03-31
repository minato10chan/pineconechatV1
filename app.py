# pysqlite3コードを削除
import streamlit as st
import datetime
import os
from dotenv import load_dotenv
import traceback

# 環境変数を確実にロード
load_dotenv(override=True)

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

# グローバル変数の初期化
vector_store = None
vector_store_available = False
chat_history = ChatHistory()

# VectorStoreのインスタンスを初期化する関数
def initialize_vector_store():
    global vector_store, vector_store_available
    
    # 既に初期化済みの場合は再初期化しない
    if vector_store is not None:
        print("VectorStoreは既に初期化されています。再初期化をスキップします。")
        # 既にvector_storeはあるが、使用可能かどうかを確認
        vector_store_available = getattr(vector_store, 'available', False)
        # クライアントの状態も確認
        if hasattr(vector_store, 'pinecone_client'):
            client_available = getattr(vector_store.pinecone_client, 'available', False)
            vector_store_available = vector_store_available or client_available
        print(f"ベクトルストアの状態: {'利用可能' if vector_store_available else '利用不可'}")
        return vector_store
    
    # セッション状態に保存されている場合はそれを使用
    if 'vector_store' in st.session_state and st.session_state.vector_store is not None:
        vector_store = st.session_state.vector_store
        # 使用可能かどうかを確認
        vector_store_available = getattr(vector_store, 'available', False)
        # クライアントの状態も確認
        if hasattr(vector_store, 'pinecone_client'):
            client_available = getattr(vector_store.pinecone_client, 'available', False)
            vector_store_available = vector_store_available or client_available
        print(f"セッション状態からVectorStoreを復元しました。状態: {'利用可能' if vector_store_available else '利用不可'}")
        return vector_store
        
    try:
        print("VectorStoreの初期化を開始します...")
        
        # Pineconeベースのベクトルストアの初期化
        try:
            from src.pinecone_vector_store import PineconeVectorStore
            vector_store = PineconeVectorStore()
            # 使用可能かどうかを確認
            vector_store_available = getattr(vector_store, 'available', False)
            print(f"PineconeベースのVectorStoreを初期化しました。状態: {'利用可能' if vector_store_available else '利用不可'}")
            
            # REST API経由での接続を再確認
            if not vector_store_available and hasattr(vector_store, 'pinecone_client'):
                # _check_rest_api_connectionメソッドが実装されている場合は使用
                if hasattr(vector_store, '_check_rest_api_connection'):
                    api_available = vector_store._check_rest_api_connection()
                    if api_available:
                        print("REST API経由でPineconeに接続できました。VectorStoreを使用可能にします。")
                        vector_store_available = True
                        vector_store.available = True
                else:
                    # 従来の方法で確認
                    client_available = getattr(vector_store.pinecone_client, 'available', False)
                    if client_available:
                        print("REST API経由でPineconeに接続できています。VectorStoreを使用可能にします。")
                        vector_store_available = True
                        vector_store.available = True
        except Exception as e:
            print(f"PineconeVectorStoreの初期化中にエラー: {e}")
            vector_store_available = False
            vector_store = None
            raise
        
        # セッション状態に保存
        st.session_state.vector_store = vector_store
        print(f"VectorStore initialization completed. Status: {'Available' if vector_store_available else 'Unavailable'}")
        return vector_store
    except Exception as e:
        vector_store_available = False
        print(f"Error initializing VectorStore: {e}")
        return None

# 最初の1回だけ初期化を試みる
if 'vector_store_initialized' not in st.session_state:
    initialize_vector_store()
    st.session_state.vector_store_initialized = True

def register_document(uploaded_file, additional_metadata=None):
    """
    アップロードされたファイルをベクトルデータベースに登録する関数。
    additional_metadata: 追加のメタデータ辞書
    """
    if not vector_store_available:
        st.error("データベース接続でエラーが発生しました。ベクトルデータベースが使用できません。")
        return
    
    if uploaded_file is not None:
        try:
            # ファイルの内容を読み込み - 複数のエンコーディングを試す
            content = None
            encodings_to_try = ['utf-8', 'shift_jis', 'cp932', 'euc_jp', 'iso2022_jp']
            
            file_bytes = uploaded_file.getvalue()
            
            # 異なるエンコーディングを試す
            for encoding in encodings_to_try:
                try:
                    content = file_bytes.decode(encoding)
                    st.success(f"ファイルを {encoding} エンコーディングで読み込みました")
                    break
                except UnicodeDecodeError:
                    continue
            
            # どのエンコーディングでも読み込めなかった場合
            if content is None:
                st.error("ファイルのエンコーディングを検出できませんでした。UTF-8, Shift-JIS, EUC-JP, ISO-2022-JPのいずれかで保存されたファイルをお試しください。")
                return
            
            # メモリ内でテキストを分割
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=512,
                chunk_overlap=10,
                add_start_index=True,
                separators=["\n\n", "\n", "。", ".", " ", ""],
            )
            
            # 基本メタデータの作成
            base_metadata = {'source': uploaded_file.name}
            
            # 追加メタデータが指定されている場合は統合
            if additional_metadata:
                base_metadata.update(additional_metadata)
            
            # ドキュメントを作成
            from langchain_core.documents import Document
            raw_document = Document(
                page_content=content,
                metadata=base_metadata
            )
            
            # ドキュメントを分割
            documents = text_splitter.split_documents([raw_document])

            # セッション状態にドキュメントを保存
            st.session_state.documents.extend(documents)

            # IDsの作成
            original_ids = []
            for i, doc in enumerate(documents):
                source_ = os.path.splitext(uploaded_file.name)[0]  # 拡張子を除く
                start_ = doc.metadata.get('start_index', i)
                id_str = f"{source_}_{start_:08}" #0パディングして8桁に
                original_ids.append(id_str)

            # グローバルのVectorStoreインスタンスを使用
            global vector_store
            
            # ドキュメントの追加（UPSERT）
            result = vector_store.upsert_documents(documents=documents, ids=original_ids)
            
            if result:
                st.success(f"{uploaded_file.name} をデータベースに登録しました。")
                st.info(f"{len(documents)}件のチャンクに分割されました")
            else:
                st.warning(f"{uploaded_file.name} の登録に問題がありました。詳細はログを確認してください。")
            
        except Exception as e:
            st.error(f"ドキュメントの登録中にエラーが発生しました: {e}")
            st.error("エラーの詳細:")
            st.exception(e)

def manage_db():
    """
    ベクトルデータベースを管理するページの関数。
    """
    st.header("ベクトルデータベース管理")

    if not vector_store_available:
        error_message = "ベクトルデータベースの接続でエラーが発生しました。現在、ベクトルデータベースは使用できません。"
        
        # より詳細なエラー情報を表示
        if hasattr(vector_store, 'pinecone_client') and vector_store.pinecone_client:
            if hasattr(vector_store.pinecone_client, 'initialization_error'):
                error_message += f"\n\nエラーの詳細: {vector_store.pinecone_client.initialization_error}"
        
        st.error(error_message)
        
        # リトライボタンを提供
        if st.button("接続を再試行"):
            with st.spinner("Pineconeへの接続を再試行しています..."):
                # vector_storeを初期化し直す
                global vector_store
                try:
                    # セッション状態をクリア
                    if 'vector_store' in st.session_state:
                        del st.session_state.vector_store
                    if 'vector_store_initialized' in st.session_state:
                        del st.session_state.vector_store_initialized
                    
                    # 再初期化
                    initialize_vector_store()
                    st.success("接続に成功しました！ページを再読み込みします。")
                    st.rerun()
                except Exception as e:
                    st.error(f"接続の再試行に失敗しました: {e}")
        
        # 代わりにREST API経由での解決方法を提案
        st.info("注: Pineconeのコントローラーサーバーに接続できない場合は、REST API経由での接続は成功している可能性があります。アプリケーションはREST APIを自動的に使用します。")
        return

    # グローバルのvector_storeを使用
    global vector_store

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
