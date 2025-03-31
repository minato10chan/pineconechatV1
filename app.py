# pysqlite3コードを削除
import streamlit as st
import datetime
import os
from dotenv import load_dotenv
import traceback
import time
import requests

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
    
    print("VectorStoreの初期化を開始します...")
    
    # 既に初期化済みの場合は再初期化しない
    if vector_store is not None and vector_store_available:
        print("VectorStoreは既に初期化済みで利用可能です。再初期化をスキップします。")
        return vector_store
    
    # セッション状態に保存されている場合はそれを使用
    if 'vector_store' in st.session_state and st.session_state.vector_store is not None:
        print("セッション状態からVectorStoreを復元します...")
        vector_store = st.session_state.vector_store
        vector_store_available = getattr(vector_store, 'available', False)
        if hasattr(vector_store, 'pinecone_client'):
            client_available = getattr(vector_store.pinecone_client, 'available', False)
            vector_store_available = vector_store_available or client_available
        print(f"セッション状態からVectorStoreを復元しました。状態: {'利用可能' if vector_store_available else '利用不可'}")
        
        # REST API接続を確認して状態を更新
        if not vector_store_available and hasattr(vector_store, '_check_rest_api_connection'):
            try:
                if vector_store._check_rest_api_connection():
                    print("セッション状態復元後: REST API接続が確認できました。利用可能に設定します。")
                    vector_store_available = True
                    vector_store.available = True
                    if hasattr(vector_store, 'pinecone_client') and hasattr(vector_store.pinecone_client, 'available'):
                        vector_store.pinecone_client.available = True
                    st.session_state.vector_store = vector_store  # 更新した状態を保存
            except Exception as e:
                print(f"REST API接続確認中にエラー: {e}")
        
        if vector_store_available:
            return vector_store
        
    try:
        print("Pineconeベースのベクトルストアの初期化を開始します...")
        
        # Pineconeベースのベクトルストアの初期化
        try:
            from src.pinecone_vector_store import PineconeVectorStore
            vector_store = PineconeVectorStore()
            
            # 使用可能かどうかを確認
            vector_store_available = getattr(vector_store, 'available', False)
            print(f"PineconeベースのVectorStoreを初期化しました。状態: {'利用可能' if vector_store_available else '利用不可'}")
            
            # REST API経由での接続を再確認
            if not vector_store_available and hasattr(vector_store, 'pinecone_client'):
                if hasattr(vector_store, '_check_rest_api_connection'):
                    api_available = vector_store._check_rest_api_connection()
                    if api_available:
                        print("REST API経由でPineconeに接続できました。VectorStoreを使用可能にします。")
                        vector_store_available = True
                        vector_store.available = True
                        if hasattr(vector_store.pinecone_client, 'available'):
                            vector_store.pinecone_client.available = True
                else:
                    client_available = getattr(vector_store.pinecone_client, 'available', False)
                    if client_available:
                        print("REST API経由でPineconeに接続できています。VectorStoreを使用可能にします。")
                        vector_store_available = True
                        vector_store.available = True
            
            # 最終的な接続状態を確認
            if not vector_store_available:
                print("警告: VectorStoreの初期化は完了しましたが、使用可能な状態ではありません。")
                if hasattr(vector_store, 'pinecone_client'):
                    print(f"Pineconeクライアントの状態: {'利用可能' if getattr(vector_store.pinecone_client, 'available', False) else '利用不可'}")
                    if hasattr(vector_store.pinecone_client, 'initialization_error'):
                        print(f"初期化エラー: {vector_store.pinecone_client.initialization_error}")
            
        except Exception as e:
            print(f"PineconeVectorStoreの初期化中にエラー: {e}")
            print(f"エラーの詳細: {traceback.format_exc()}")
            vector_store_available = False
            vector_store = None
            raise
        
        # 最終チェックとして、REST API接続が成功していれば、強制的に利用可能に設定
        if not vector_store_available and vector_store and hasattr(vector_store, 'pinecone_client'):
            client = vector_store.pinecone_client
            print("REST API接続の最終チェックを実行中...")
            try:
                if hasattr(client, '_check_rest_api_connection'):
                    if client._check_rest_api_connection():
                        print("最終チェック: REST API接続が確認できました。強制的に利用可能に設定します。")
                        vector_store_available = True
                        vector_store.available = True
                        if hasattr(client, 'available'):
                            client.available = True
                        print(f"接続状態の強制更新後: vector_store_available = {vector_store_available}")
            except Exception as e:
                print(f"REST API接続の最終チェック中にエラー: {e}")

        # セッション状態に保存
        st.session_state.vector_store = vector_store
        print(f"VectorStore initialization completed. Status: {'Available' if vector_store_available else 'Unavailable'}")
        
        # グローバル変数の状態を明示的に確認して出力
        print(f"グローバル変数の最終状態: vector_store_available = {vector_store_available}")
        if vector_store:
            print(f"vector_store.available = {getattr(vector_store, 'available', 'undefined')}")
            
        return vector_store
    except Exception as e:
        vector_store_available = False
        print(f"Error initializing VectorStore: {e}")
        print(f"Error traceback: {traceback.format_exc()}")
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
    # グローバル変数の宣言を最初に移動
    global vector_store, vector_store_available
    
    # タイムアウト設定 - ファイルアップロード処理の最大時間（秒）
    upload_timeout = 120  # 2分
    start_time = time.time()
    
    # デバッグモードを有効化
    debug_container = st.expander("デバッグ情報", expanded=False)
    with debug_container:
        st.write("### アップロード処理のデバッグ情報")
        st.write(f"処理開始時刻: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.write(f"タイムアウト設定: {upload_timeout}秒")
        
        # 接続状態のログ
        st.write("#### 初期接続状態")
        st.write(f"vector_store_available: {vector_store_available}")
        if vector_store:
            st.write(f"vector_store.available: {getattr(vector_store, 'available', 'undefined')}")
            st.write(f"vector_store.temporary_failure: {getattr(vector_store, 'temporary_failure', False)}")
            if hasattr(vector_store, 'pinecone_client'):
                client = vector_store.pinecone_client
                st.write(f"Pineconeクライアント状態: {getattr(client, 'available', 'undefined')}")
                st.write(f"Pineconeクライアント一時的障害モード: {getattr(client, 'temporary_failure', False)}")
    
    # ベクトルデータベース接続状態を再確認
    if not vector_store_available and vector_store and hasattr(vector_store, 'pinecone_client'):
        client = vector_store.pinecone_client
        try:
            # REST API接続を試行
            if hasattr(client, '_check_rest_api_connection'):
                with debug_container:
                    st.write("#### REST API接続テスト")
                
                connection_result = client._check_rest_api_connection()
                
                with debug_container:
                    st.write(f"REST API接続テスト結果: {connection_result}")
                
                if connection_result:
                    print("ファイルアップロード前: REST API接続が確認できました。強制的に利用可能に設定します。")
                    vector_store_available = True
                    vector_store.available = True
                    if hasattr(client, 'available'):
                        client.available = True
                    print(f"接続状態の強制更新後: vector_store_available = {vector_store_available}")
                    
                    with debug_container:
                        st.write("#### 接続状態更新")
                        st.write(f"接続成功により状態を更新: vector_store_available = {vector_store_available}")
        except Exception as e:
            print(f"REST API接続の確認中にエラー: {e}")
            print(f"エラーの詳細: {traceback.format_exc()}")
            
            with debug_container:
                st.write("#### REST API接続エラー")
                st.write(f"エラー: {str(e)}")
                st.code(traceback.format_exc(), language="python")
    
    # 接続状態を確認して詳細な情報を出力
    print(f"ファイルアップロード処理開始: vector_store_available = {vector_store_available}")
    if vector_store:
        print(f"vector_store.available = {getattr(vector_store, 'available', 'undefined')}")
        if hasattr(vector_store, 'pinecone_client'):
            client = vector_store.pinecone_client
            print(f"Pineconeクライアント状態: {getattr(client, 'available', 'undefined')}")

    if not vector_store_available:
        st.error("データベース接続でエラーが発生しました。ベクトルデータベースが使用できません。")
        
        # 接続状態の詳細を表示
        st.write("## 接続状態詳細")
        st.write(f"vector_store_available: {vector_store_available}")
        if vector_store:
            # vector_storeの属性を出力
            st.write(f"vector_store.available: {getattr(vector_store, 'available', 'undefined')}")
            
            # Pineconeクライアントの詳細情報を表示
            if hasattr(vector_store, 'pinecone_client'):
                client = vector_store.pinecone_client
                st.write("### Pineconeクライアント情報")
                st.write(f"API環境: {getattr(client, 'environment', 'unknown')}")
                st.write(f"インデックス名: {getattr(client, 'index_name', 'unknown')}")
                st.write(f"Streamlit Cloud環境: {getattr(client, 'is_streamlit_cloud', False)}")
                st.write(f"一時的障害モード: {getattr(client, 'temporary_failure', False)}")
                st.write(f"失敗試行回数: {getattr(client, 'failed_attempts', 0)}")
                
                if hasattr(client, 'initialization_error'):
                    st.write("### 初期化エラー")
                    st.error(client.initialization_error)
            
            # 接続テスト
            try:
                if hasattr(vector_store, 'pinecone_client') and hasattr(vector_store.pinecone_client, '_check_rest_api_connection'):
                    with st.spinner("REST API接続をテスト中..."):
                        rest_result = vector_store.pinecone_client._check_rest_api_connection()
                    st.write(f"REST API接続テスト: {'成功' if rest_result else '失敗'}")
                    
                    # レスポンスの詳細を表示
                    try:
                        # インデックス一覧を取得
                        api_url = "https://api.pinecone.io/indexes"
                        with st.spinner("インデックス情報を取得中..."):
                            response = vector_store.pinecone_client._make_request(
                                method="GET", 
                                url=api_url, 
                                max_retries=1, 
                                timeout=10
                            )
                        
                        if response:
                            st.write(f"API応答ステータス: {response.status_code}")
                            if response.status_code == 200:
                                st.success("インデックス一覧の取得に成功しました")
                                try:
                                    index_list = response.json()
                                    st.write(f"利用可能なインデックス: {index_list}")
                                except Exception as e:
                                    st.write(f"JSONデコードエラー: {str(e)}")
                            else:
                                st.error(f"インデックス一覧の取得に失敗しました: {response.status_code}")
                                st.code(response.text, language="text")
                    except Exception as e:
                        st.write(f"インデックス情報の取得中にエラー: {str(e)}")
                    
                    # 接続が成功している場合は再試行ボタンを表示
                    if rest_result and st.button("REST APIで再試行"):
                        vector_store_available = True
                        vector_store.available = True
                        if hasattr(vector_store.pinecone_client, 'available'):
                            vector_store.pinecone_client.available = True
                        st.success("REST API接続を有効化しました。続行します。")
                        time.sleep(1)  # 少し待機してUIを更新
                        st.rerun()
            except Exception as e:
                st.write(f"REST API接続テスト中にエラー: {str(e)}")
                st.code(traceback.format_exc(), language="python")

        # ネットワーク接続のテスト
        try:
            with st.spinner("インターネット接続をテスト中..."):
                internet_test_urls = ["https://8.8.8.8", "https://1.1.1.1", "https://www.google.com"]
                for url in internet_test_urls:
                    try:
                        response = requests.get(url, timeout=5)
                        st.success(f"{url} に接続成功: {response.status_code}")
                        break
                    except Exception as e:
                        st.error(f"{url} への接続に失敗: {str(e)}")
        except Exception as e:
            st.error(f"インターネット接続テスト中にエラー: {str(e)}")
                
        # 緊急モードへの切り替えボタン
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
                st.code(traceback.format_exc(), language="python")
        
        # リトライボタンを提供
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
        
        return
    
    if uploaded_file is not None:
        # プログレスバーを表示
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("ファイルを処理中...")
        
        try:
            # ファイルの内容を読み込み - 複数のエンコーディングを試す
            content = None
            encodings_to_try = ['utf-8', 'shift_jis', 'cp932', 'euc_jp', 'iso2022_jp']
            
            file_bytes = uploaded_file.getvalue()
            progress_bar.progress(10)
            status_text.text("ファイルのエンコーディングを検出中...")
            
            with debug_container:
                st.write("#### ファイル情報")
                st.write(f"ファイル名: {uploaded_file.name}")
                st.write(f"ファイルサイズ: {len(file_bytes)} バイト")
            
            # 異なるエンコーディングを試す
            for encoding in encodings_to_try:
                try:
                    content = file_bytes.decode(encoding)
                    st.success(f"ファイルを {encoding} エンコーディングで読み込みました")
                    
                    with debug_container:
                        st.write(f"エンコーディング: {encoding}")
                        st.write(f"テキスト長: {len(content)} 文字")
                    
                    break
                except UnicodeDecodeError:
                    continue
            
            # どのエンコーディングでも読み込めなかった場合
            if content is None:
                st.error("ファイルのエンコーディングを検出できませんでした。UTF-8, Shift-JIS, EUC-JP, ISO-2022-JPのいずれかで保存されたファイルをお試しください。")
                progress_bar.empty()
                status_text.empty()
                return
            
            progress_bar.progress(20)
            status_text.text("テキストを分割中...")
            
            # タイムアウトチェック
            if time.time() - start_time > upload_timeout:
                st.error(f"処理がタイムアウトしました（{upload_timeout}秒）。ファイルサイズを小さくするか、後でもう一度お試しください。")
                progress_bar.empty()
                status_text.empty()
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
            
            progress_bar.progress(30)
            status_text.text("ドキュメントを作成中...")
            
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

            with debug_container:
                st.write("#### 分割情報")
                st.write(f"分割されたドキュメント数: {len(documents)}")
                st.write(f"平均チャンク長: {sum([len(doc.page_content) for doc in documents]) / len(documents) if documents else 0:.1f} 文字")

            progress_bar.progress(40)
            status_text.text("ベクトルIDを作成中...")
            
            # タイムアウトチェック
            if time.time() - start_time > upload_timeout:
                st.error(f"処理がタイムアウトしました（{upload_timeout}秒）。ファイルサイズを小さくするか、後でもう一度お試しください。")
                progress_bar.empty()
                status_text.empty()
                return

            # IDsの作成
            original_ids = []
            for i, doc in enumerate(documents):
                source_ = os.path.splitext(uploaded_file.name)[0]  # 拡張子を除く
                start_ = doc.metadata.get('start_index', i)
                id_str = f"{source_}_{start_:08}" #0パディングして8桁に
                original_ids.append(id_str)

            progress_bar.progress(50)
            status_text.text(f"ドキュメントをデータベースに登録中... (0/{len(documents)}件)")
            
            # ドキュメントの追加（UPSERT）を小さなバッチに分割して実行
            batch_size = 5  # 一度に処理するドキュメント数を減らす
            success_count = 0
            
            # デバッグ情報に進捗を表示
            with debug_container:
                st.write("#### ベクトルデータベース処理")
                upload_status = st.empty()
                upload_details = st.empty()
                upload_status.text(f"処理開始: 合計 {len(documents)} 件を {batch_size} 件ずつバッチ処理")
            
            for batch_start in range(0, len(documents), batch_size):
                # タイムアウトチェック
                if time.time() - start_time > upload_timeout:
                    st.warning(f"処理がタイムアウトしました（{upload_timeout}秒）。{success_count}件のチャンクが登録されました。")
                    break
                
                batch_end = min(batch_start + batch_size, len(documents))
                batch_docs = documents[batch_start:batch_end]
                batch_ids = original_ids[batch_start:batch_end]
                
                # デバッグ情報にバッチ詳細を表示
                with debug_container:
                    upload_details.text(f"バッチ {batch_start//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}: {batch_start+1}～{batch_end}/{len(documents)}件を処理中")
                
                try:
                    # バッチをアップロード
                    vector_store_method = None
                    if hasattr(vector_store, 'upsert_documents'):
                        vector_store_method = "upsert_documents"
                    elif hasattr(vector_store, 'add_documents'):
                        vector_store_method = "add_documents"
                    
                    with debug_container:
                        st.write(f"使用メソッド: {vector_store_method}")
                        current_time = time.time()
                        st.write(f"バッチ処理開始時刻: {datetime.datetime.fromtimestamp(current_time).strftime('%H:%M:%S.%f')[:-3]}")
                    
                    # ベクトルストアの使用可能状態を再確認
                    if not vector_store_available and vector_store:
                        with debug_container:
                            st.warning("ベクトルストアが使用不可の状態でアップロードを試行")
                            st.write(f"vector_store.available: {getattr(vector_store, 'available', 'undefined')}")
                            if hasattr(vector_store, 'pinecone_client'):
                                st.write(f"client.available: {getattr(vector_store.pinecone_client, 'available', 'undefined')}")
                    
                    # バッチをアップロード（例外をキャッチして詳細表示）
                    result = False
                    try:
                        if vector_store_method == "upsert_documents":
                            result = vector_store.upsert_documents(documents=batch_docs, ids=batch_ids)
                        elif vector_store_method == "add_documents":
                            result = vector_store.add_documents(documents=batch_docs, ids=batch_ids)
                    except Exception as upload_error:
                        with debug_container:
                            st.error(f"アップロードエラー: {str(upload_error)}")
                            st.code(traceback.format_exc(), language="python")
                        raise
                    
                    with debug_container:
                        end_time = time.time()
                        st.write(f"バッチ処理終了時刻: {datetime.datetime.fromtimestamp(end_time).strftime('%H:%M:%S.%f')[:-3]}")
                        st.write(f"処理時間: {end_time - current_time:.2f}秒")
                        st.write(f"結果: {result}")
                    
                    if result:
                        success_count += len(batch_docs)
                        # プログレスバーを更新（50%～90%）
                        progress_percentage = 50 + int(40 * success_count / len(documents))
                        progress_bar.progress(progress_percentage)
                        status_text.text(f"ドキュメントをデータベースに登録中... ({success_count}/{len(documents)}件)")
                    else:
                        with debug_container:
                            st.warning(f"バッチ {batch_start//batch_size + 1} の登録に失敗しました（falseが返されました）")
                        st.warning(f"バッチ {batch_start//batch_size + 1} の登録に問題がありました。")
                except Exception as batch_error:
                    st.error(f"バッチ {batch_start//batch_size + 1} の登録中にエラーが発生しました: {str(batch_error)}")
                    print(f"Batch upsert error: {str(batch_error)}")
                    print(f"Error type: {type(batch_error)}")
                    print(f"Error traceback: {traceback.format_exc()}")
                    
                    # より詳細なデバッグ情報
                    with debug_container:
                        st.write("#### バッチ処理エラー")
                        st.error(f"エラータイプ: {type(batch_error).__name__}")
                        st.error(f"エラーメッセージ: {str(batch_error)}")
                        st.code(traceback.format_exc(), language="python")
                        
                        # ベクトルストアの状態を確認
                        st.write("#### エラー後のベクトルストア状態")
                        st.write(f"vector_store_available: {vector_store_available}")
                        if vector_store:
                            st.write(f"vector_store.available: {getattr(vector_store, 'available', 'undefined')}")
                            if hasattr(vector_store, 'pinecone_client'):
                                client = vector_store.pinecone_client
                                st.write(f"client.available: {getattr(client, 'available', 'undefined')}")
                        
                        # REST API接続を再テスト
                        try:
                            if vector_store and hasattr(vector_store, 'pinecone_client') and \
                               hasattr(vector_store.pinecone_client, '_check_rest_api_connection'):
                                st.write("REST API接続を再テスト中...")
                                api_status = vector_store.pinecone_client._check_rest_api_connection()
                                st.write(f"REST API接続テスト結果: {api_status}")
                        except Exception as test_error:
                            st.write(f"REST API接続テスト中にエラー: {str(test_error)}")
            
            # 最終進捗状況の更新
            with debug_container:
                upload_status.text(f"処理完了: {success_count}/{len(documents)}件が成功")
            
            progress_bar.progress(100)
            
            if success_count > 0:
                st.success(f"{uploaded_file.name} をデータベースに登録しました。")
                st.info(f"{success_count}/{len(documents)}件のチャンクが正常に登録されました")
            else:
                st.error(f"{uploaded_file.name} の登録に失敗しました。詳細はデバッグ情報を確認してください。")
            
            # クリーンアップ
            progress_bar.empty()
            status_text.empty()
            
        except Exception as e:
            st.error(f"ドキュメントの処理中にエラーが発生しました: {str(e)}")
            st.error("エラーの詳細:")
            st.exception(e)
            # エラーの詳細をログに出力
            print(f"Document processing error details: {str(e)}")
            print(f"Error type: {type(e)}")
            print(f"Error traceback: {traceback.format_exc()}")
            
            # デバッグ情報に詳細を追加
            with debug_container:
                st.write("#### 致命的なエラー")
                st.error(f"エラータイプ: {type(e).__name__}")
                st.error(f"エラーメッセージ: {str(e)}")
                st.code(traceback.format_exc(), language="python")
            
            # クリーンアップ
            if 'progress_bar' in locals():
                progress_bar.empty()
            if 'status_text' in locals():
                status_text.empty()

def manage_db():
    """
    ベクトルデータベースを管理するページの関数。
    """
    # グローバル変数の宣言を最初に移動
    global vector_store, vector_store_available

    st.header("ベクトルデータベース管理")
    
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
import platform
print(f"- Python バージョン: {platform.python_version()}")
print(f"- プラットフォーム: {platform.platform()}")
