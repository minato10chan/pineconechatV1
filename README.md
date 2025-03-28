# PineconeChat

ChromaDBとPineconeを使用した会話履歴管理機能を持つStreamlitチャットアプリケーション

## 機能

- ChromaDBを使用したRAG (Retrieval Augmented Generation) 質問応答
- 会話履歴のPineconeへの永続化
- 会話ログのCSV出力
- カテゴリごとのフィルタリング

## セットアップ

### 必要条件

- Python 3.9以上
- pip

### インストール

1. リポジトリをクローン:
```bash
git clone https://github.com/yourusername/pineconechatV1.git
cd pineconechatV1
```

2. 必要なパッケージをインストール:
```bash
pip install -r requirements.txt
```

3. 環境変数の設定:
`.env.example`ファイルを`.env`にコピーして必要な情報を入力:
```
OPENAI_API_KEY=your-openai-api-key
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENVIRONMENT=your-pinecone-environment
PINECONE_INDEX=your-pinecone-index
```

## 実行方法

ローカルで実行:
```bash
streamlit run app.py
```

## Streamlit Cloudへのデプロイ方法

1. GitHubのリポジトリにコードをプッシュします。

2. [Streamlit Cloud](https://streamlit.io/cloud)でアカウントを作成し、サインインします。

3. 「New app」ボタンをクリックし、GitHubリポジトリ、ブランチ、メインPythonファイル（app.py）を指定します。

4. Secretsの設定:
   - アプリのデプロイ後、「App settings」→「Secrets」メニューを開きます
   - 以下の形式で必要な設定を入力します:
   ```
   OPENAI_API_KEY = "your-openai-api-key"
   PINECONE_API_KEY = "your-pinecone-api-key"
   PINECONE_ENVIRONMENT = "us-east-1"
   PINECONE_INDEX = "langchain-index"
   ```

5. 「Save」をクリックして設定を保存し、アプリを再起動します。

## トラブルシューティング

### Pinecone接続エラー

Streamlit CloudからPineconeへの接続に問題がある場合:

1. Streamlit Secretsの設定を確認してください
2. REST API接続モードを使用することでDNS解決の問題を回避します
3. アプリケーションのデバッグ情報パネルでエラー詳細を確認できます

### ChromaDB非対応エラー

Streamlit CloudではSQLiteのバージョンの問題によりChromaDBが使用できないことがあります。この場合:

1. ローカルモードで会話履歴管理だけを使用できます
2. 完全な機能を使用するにはローカル環境でアプリを実行してください

## ライセンス

[MIT License](LICENSE)

## 未実装

- 会話ログの出力機能（テキストやｃSV？）
- ベクトルDBの参照正確度

- langchainの他機能（チェーンとかエージェント？）
- ベクトルDBメタデータ修正
- geminiAPI
- テキストファイル以外の読み込み（PDF、CSV）

