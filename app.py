# SQLiteをpysqlite3で上書き
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("Successfully overrode sqlite3 with pysqlite3")
except ImportError:
    print("Failed to override sqlite3 with pysqlite3")

import streamlit as st
import datetime

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
import os
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
        return vector_store
    
    # セッション状態に保存されている場合はそれを使用
    if 'vector_store' in st.session_state and st.session_state.vector_store is not None:
        vector_store = st.session_state.vector_store
        vector_store_available = True
        print("セッション状態からVectorStoreを復元しました")
        return vector_store
        
    try:
        print("VectorStoreの初期化を開始します...")
        from src.vector_store import VectorStore
        vector_store = VectorStore()
        vector_store_available = True
        # セッション状態に保存
        st.session_state.vector_store = vector_store
        print("VectorStore successfully initialized")
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
    アップロードされたファイルをChromaDBに登録する関数。
    additional_metadata: 追加のメタデータ辞書
    """
    if not vector_store_available:
        st.error("データベース接続でエラーが発生しました。ChromaDBが使用できません。")
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
            vector_store.upsert_documents(documents=documents, ids=original_ids)

            st.success(f"{uploaded_file.name} をデータベースに登録しました。")
            st.info(f"{len(documents)}件のチャンクに分割されました")
        except Exception as e:
            st.error(f"ドキュメントの登録中にエラーが発生しました: {e}")
            st.error("エラーの詳細:")
            st.exception(e)

def manage_chromadb():
    """
    ChromaDBを管理するページの関数。
    """
    st.header("ChromaDB 管理")

    if not vector_store_available:
        st.error("ChromaDBの接続でエラーが発生しました。現在、ベクトルデータベースは使用できません。")
        st.warning("これはSQLiteのバージョンの非互換性によるものです。Streamlit Cloudでの実行には制限があります。")
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
    st.subheader("ChromaDB 登録状況確認")
    
    # 検索フィルター
    with st.expander("検索フィルター", expanded=False):
        filter_municipality = st.text_input("市区町村名で絞り込み", "")
        filter_category = st.text_input("カテゴリで絞り込み", "")
    
    # 表示ボタン
    if st.button("登録済みドキュメントを表示"):
        with st.spinner('取得中...'):
            try:
                # グローバルのVectorStoreインスタンスを使用
                dict_data = vector_store.get_documents(ids=None)
                
                if dict_data and len(dict_data.get('ids', [])) > 0:
                    # フィルタリング
                    filtered_indices = range(len(dict_data['ids']))
                    
                    if filter_municipality or filter_category:
                        filtered_indices = []
                        for i, metadata in enumerate(dict_data['metadatas']):
                            municipality_match = True
                            category_match = True
                            
                            if filter_municipality and metadata.get('municipality'):
                                municipality_match = filter_municipality.lower() in metadata['municipality'].lower()
                            
                            if filter_category:
                                major_match = metadata.get('major_category') and filter_category.lower() in metadata['major_category'].lower()
                                medium_match = metadata.get('medium_category') and filter_category.lower() in metadata['medium_category'].lower()
                                category_match = major_match or medium_match
                                
                            if municipality_match and category_match:
                                filtered_indices.append(i)
                    
                    # フィルタリングされたデータでDataFrameを作成
                    filtered_ids = [dict_data['ids'][i] for i in filtered_indices]
                    filtered_docs = [dict_data['documents'][i] for i in filtered_indices]
                    filtered_metas = [dict_data['metadatas'][i] for i in filtered_indices]
                    
                    tmp_df = pd.DataFrame({
                        "IDs": filtered_ids,
                        "Documents": filtered_docs,
                        "市区町村": [m.get('municipality', '') for m in filtered_metas],
                        "大カテゴリ": [m.get('major_category', '') for m in filtered_metas],
                        "中カテゴリ": [m.get('medium_category', '') for m in filtered_metas],
                        "ソース元": [m.get('source', '') for m in filtered_metas],
                        "登録日時": [m.get('registration_date', '') for m in filtered_metas],
                        "データ公開日": [m.get('publication_date', '') for m in filtered_metas],
                        "緯度経度": [f"{m.get('latitude', '')}, {m.get('longitude', '')}" for m in filtered_metas]
                    })
                    
                    st.dataframe(tmp_df)
                    st.success(f"合計 {len(filtered_ids)} 件のドキュメントが表示されています（全 {len(dict_data['ids'])} 件中）")
                else:
                    st.info("データベースに登録されたデータはありません。")
            except Exception as e:
                st.error(f"データの取得中にエラーが発生しました: {e}")
                st.error("エラーの詳細:")
                st.exception(e)

    st.markdown("---")

    # 3.全データ削除
    st.subheader("ChromaDB 登録データ全削除")
    if st.button("全データを削除する"):
        with st.spinner('削除中...'):
            try:
                # グローバルのVectorStoreインスタンスを使用
                current_ids = vector_store.get_documents(ids=None).get('ids', [])
                if current_ids:
                    vector_store.delete_documents(ids=current_ids)
                    st.success(f"データベースから {len(current_ids)} 件のドキュメントが削除されました")
                else:
                    st.info("削除するデータがありません。")
            except Exception as e:
                st.error(f"データの削除中にエラーが発生しました: {e}")
                st.error("エラーの詳細:")
                st.exception(e)

def manage_prompts():
    """
    プロンプトを管理するページの関数。
    """
    st.header("プロンプト管理")

    # プロンプトの追加
    st.subheader("新しいプロンプトの追加")
    new_prompt_name = st.text_input("プロンプト名", "")
    new_prompt_content = st.text_area("プロンプト内容", height=300)
    
    if st.button("プロンプトを追加") and new_prompt_name and new_prompt_content:
        if len(st.session_state.custom_prompts) >= 3:
            st.error("プロンプトは最大3つまで登録できます。")
        else:
            st.session_state.custom_prompts.append({
                'name': new_prompt_name,
                'content': new_prompt_content
            })
            st.success(f"プロンプト '{new_prompt_name}' を追加しました。")

    # プロンプトの一覧表示と編集
    st.subheader("登録済みプロンプト")
    for i, prompt in enumerate(st.session_state.custom_prompts):
        with st.expander(f"プロンプト: {prompt['name']}", expanded=False):
            edited_name = st.text_input("プロンプト名", prompt['name'], key=f"name_{i}")
            edited_content = st.text_area("プロンプト内容", prompt['content'], height=300, key=f"content_{i}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("更新", key=f"update_{i}"):
                    st.session_state.custom_prompts[i]['name'] = edited_name
                    st.session_state.custom_prompts[i]['content'] = edited_content
                    st.success(f"プロンプト '{edited_name}' を更新しました。")
            with col2:
                if st.button("削除", key=f"delete_{i}") and len(st.session_state.custom_prompts) > 1:
                    st.session_state.custom_prompts.pop(i)
                    if st.session_state.selected_prompt == prompt['name']:
                        st.session_state.selected_prompt = st.session_state.custom_prompts[0]['name']
                    st.success(f"プロンプト '{prompt['name']}' を削除しました。")

    # プロンプトの選択
    st.subheader("使用するプロンプトの選択")
    prompt_names = [p['name'] for p in st.session_state.custom_prompts]
    selected_prompt = st.selectbox(
        "プロンプトを選択してください",
        prompt_names,
        index=prompt_names.index(st.session_state.selected_prompt)
    )
    st.session_state.selected_prompt = selected_prompt

# RAGを使ったLLM回答生成
def generate_response(query_text, filter_conditions=None):
    """
    質問に対する回答を生成する関数。
    filter_conditions: メタデータによるフィルタリング条件
    """
    if not vector_store_available:
        return "申し訳ありません。現在、ベクトルデータベースに接続できないため、質問に回答できません。"
    
    if query_text:
        try:
            # グローバルのVectorStoreインスタンスを使用
            global vector_store

            # 選択されたプロンプトを取得
            selected_prompt = next(p for p in st.session_state.custom_prompts if p['name'] == st.session_state.selected_prompt)
            prompt = ChatPromptTemplate.from_template(selected_prompt['content'])

            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            # 検索結果を取得（フィルタリング条件があれば適用）
            search_results = vector_store.search(query_text, n_results=5, filter_conditions=filter_conditions)
            
            # 検索結果がない場合
            if not search_results or not search_results.get('documents', [[]])[0]:
                return "申し訳ありません。指定された条件に一致するドキュメントが見つかりませんでした。検索条件を変更してお試しください。"
            
            # 検索結果をドキュメント形式に変換
            from langchain_core.documents import Document
            docs = []
            for i, doc_text in enumerate(search_results['documents'][0]):
                # メタデータの取得（利用可能な場合）
                metadata = {}
                if search_results.get('metadatas') and len(search_results['metadatas']) > 0:
                    metadata = search_results['metadatas'][0][i] if i < len(search_results['metadatas'][0]) else {}
                
                # ドキュメントの作成
                doc = Document(
                    page_content=doc_text,
                    metadata=metadata
                )
                docs.append(doc)

            # 使用するメタデータの情報を表示
            st.markdown("#### 検索結果")
            meta_info = []
            for i, doc in enumerate(docs):
                meta_str = ""
                if doc.metadata.get('municipality'):
                    meta_str += f"【市区町村】{doc.metadata['municipality']} "
                if doc.metadata.get('major_category'):
                    meta_str += f"【大カテゴリ】{doc.metadata['major_category']} "
                if doc.metadata.get('medium_category'):
                    meta_str += f"【中カテゴリ】{doc.metadata['medium_category']} "
                if doc.metadata.get('source'):
                    meta_str += f"【ソース元】{doc.metadata['source']}"
                
                meta_info.append(f"{i+1}. {meta_str}")
            
            st.markdown("\n".join(meta_info))

            # 使用するプロンプトを表示
            st.markdown("#### 使用するプロンプト")
            st.markdown(f"**プロンプト名**: {selected_prompt['name']}")
            with st.expander("プロンプト内容を表示"):
                st.text(selected_prompt['content'])

            # 会話履歴を取得
            chat_history_text = chat_history.get_formatted_history()

            qa_chain = (
                {
                    "context": lambda x: format_docs(docs),
                    "chat_history": lambda x: chat_history_text,
                    "question": RunnablePassthrough(),
                }
                | prompt
                | llm
                | StrOutputParser()
            )
            return qa_chain.invoke(query_text)
        except Exception as e:
            st.error(f"質問の処理中にエラーが発生しました: {e}")
            st.error("エラーの詳細:")
            st.exception(e)
            return None

def ask_question():
    """
    質問するページの関数。
    """
    st.header("ドキュメントに質問する")

    if not vector_store_available:
        st.error("ChromaDBの接続でエラーが発生しました。現在、ベクトルデータベースは使用できません。")
        st.warning("これはSQLiteのバージョンの非互換性によるものです。Streamlit Cloudでの実行には制限があります。")
        st.info("ローカル環境での実行をお試しください。")
        return

    # フィルタリング条件の設定
    with st.expander("検索範囲の絞り込み", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            filter_municipality = st.text_input("市区町村名", "")
            filter_major_category = st.selectbox(
                "大カテゴリ",
                [""] + MAJOR_CATEGORIES
            )
        
        with col2:
            filter_medium_category = st.selectbox(
                "中カテゴリ",
                [""] + (MEDIUM_CATEGORIES.get(filter_major_category, []) if filter_major_category else [])
            )
            filter_source = st.text_input("ソース元", "")

    # 会話履歴の表示
    st.markdown("### 会話履歴")
    
    # 会話履歴の表示と操作ボタンを分けて配置
    for message in chat_history.get_history():
        role = "ユーザー" if message['role'] == 'user' else "アシスタント"
        st.markdown(f"**{role}**: {message['content']}")
    
    # 履歴操作ボタンを横に配置
    col1, col2 = st.columns(2)
    
    with col1:
        # 会話履歴をクリアするボタン
        if st.button('会話履歴をクリア'):
            chat_history.clear_history()
            st.success("会話履歴をクリアしました。")
            st.rerun()
    
    with col2:
        # 会話履歴をCSVでダウンロードするボタン
        if chat_history.get_history():
            csv_data = chat_history.get_csv_export()
            if csv_data:
                current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label="CSVでダウンロード",
                    data=csv_data,
                    file_name=f"会話履歴_{current_time}.csv",
                    mime="text/csv",
                )

    # Query text
    query_text = st.text_input('質問を入力:', 
                               placeholder='簡単な概要を記入してください')

    # 質問送信ボタン
    if st.button('Submit') and query_text:
        with st.spinner('回答を生成中...'):
            # フィルタリング条件の作成
            filter_conditions = {}
            if filter_municipality:
                filter_conditions["municipality"] = filter_municipality
            if filter_major_category:
                filter_conditions["major_category"] = filter_major_category
            if filter_medium_category:
                filter_conditions["medium_category"] = filter_medium_category
            if filter_source:
                filter_conditions["source"] = filter_source
                
            # ユーザーの質問を会話履歴に追加
            chat_history.add_message('user', query_text)
            
            response = generate_response(query_text, filter_conditions)
            if response:
                # アシスタントの回答を会話履歴に追加
                chat_history.add_message('assistant', response)
                st.success("回答:")
                st.info(response)
            else:
                st.error("回答の生成に失敗しました。")

def fallback_mode():
    """
    ChromaDBが使用できない場合のフォールバックモード
    """
    st.header("ChromaDBが使用できません")
    st.error("SQLiteのバージョンの問題により、ChromaDBを使用できません。")
    st.info("このアプリは、SQLite 3.35.0以上が必要です。Streamlit Cloudでは現在、SQLite 3.34.1が使用されています。")
    
    st.markdown("""
    ## 解決策
    
    1. **ローカル環境での実行**: 
       - このアプリをローカル環境でクローンして実行してください
       - 最新のSQLiteがインストールされていることを確認してください
    
    2. **代替のベクトルデータベース**:
       - ChromaDBの代わりに、他のベクトルデータベース（FAISS、Milvusなど）を使用することも検討できます
    
    3. **インメモリモードでの使用**:
       - 現在、DuckDB+Parquetバックエンドでの実行を試みていますが、これも失敗しています
       - 詳細については、ログを確認してください
    """)
    
    # 技術的な詳細
    with st.expander("技術的な詳細"):
        st.code("""
# エラーの原因
ChromaDBは内部でSQLite 3.35.0以上を必要としていますが、
Streamlit Cloudでは現在、SQLite 3.34.1が使用されています。

# 試みた解決策
1. pysqlite3-binaryのインストール
2. SQLiteのソースからのビルド
3. DuckDB+Parquetバックエンドの使用
4. モンキーパッチの適用

いずれも環境制限により成功していません。
        """)

def main():
    """
    アプリケーションのメイン関数。
    """
    # タイトルを表示
    st.title('🦜🔗 Ask the Doc App')

    # サイドバーにデバッグ情報表示ボタンを追加
    with st.sidebar:
        st.title("メニュー")
        
        # 開発者向けデバッグ情報
        with st.expander("デバッグ情報", expanded=False):
            if st.button("環境変数をチェック"):
                # 環境変数の確認（APIキーは安全のためマスク）
                env_vars = {
                    "OPENAI_API_KEY": "設定済み" if os.environ.get("OPENAI_API_KEY") else "未設定",
                    "PINECONE_API_KEY": "設定済み" if os.environ.get("PINECONE_API_KEY") else "未設定",
                    "PINECONE_ENVIRONMENT": os.environ.get("PINECONE_ENVIRONMENT", "未設定"),
                    "PINECONE_INDEX": os.environ.get("PINECONE_INDEX", "未設定"),
                    "STREAMLIT_SESSION_ID": os.environ.get("STREAMLIT_SESSION_ID", "自動生成")
                }
                st.json(env_vars)
                
                # Pineconeの接続状態
                st.write("#### Pineconeの状態")
                pinecone_status = {
                    "利用可能": chat_history.pinecone_available,
                    "初期化済み": st.session_state.get("pinecone_initialized", False)
                }
                st.json(pinecone_status)
                
                # VectorStoreの状態
                st.write("#### VectorStoreの状態")
                vs_status = {
                    "利用可能": vector_store_available,
                    "初期化済み": st.session_state.get("vector_store_initialized", False)
                }
                st.json(vs_status)
                
            if st.button("セッション状態表示"):
                # セッション状態の表示（センシティブな情報は除外）
                safe_session = {k: v for k, v in st.session_state.items() 
                              if k not in ['pinecone_client', 'vector_store']}
                st.json(safe_session)

    # ChromaDBが使用できない場合はフォールバックモード
    if not vector_store_available:
        fallback_mode()
        return

    # ページ選択
    page = st.sidebar.radio("ページを選択してください", ["ChromaDB 管理", "質問する", "プロンプト管理"])

    # 各ページへ移動
    if page == "質問する":
        ask_question()
    elif page == "ChromaDB 管理":
        manage_chromadb()
    elif page == "プロンプト管理":
        manage_prompts()
    
    # ページが変更されるたびにPineconeに会話履歴を保存
    try:
        chat_history.force_save()
    except Exception as e:
        print(f"会話履歴の保存エラー: {e}")
    
    # アプリケーション終了時に会話履歴を保存するための関数を登録
    def save_on_exit():
        try:
            chat_history.force_save()
            print("アプリケーション終了時に会話履歴を保存しました")
        except Exception as e:
            print(f"終了時の会話履歴保存エラー: {e}")
    
    import atexit
    atexit.register(save_on_exit)

if __name__ == "__main__":
    main()
