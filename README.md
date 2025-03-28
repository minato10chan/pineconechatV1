# Ask the Doc App

ドキュメントに対して質問できるAIアプリケーションです。

## セットアップ方法

1. リポジトリをクローン
```bash
git clone [your-repository-url]
cd [repository-name]
```

2. SQLite のセットアップ
```bash
# Linux/Mac の場合
chmod +x build_sqlite.sh
./build_sqlite.sh

# Windows の場合
build_sqlite.bat
```

3. 必要なパッケージをインストール
```bash
pip install -r requirements.txt
```

4. 環境変数の設定
- `.env.example`ファイルを`.env`にコピー
```bash
cp .env.example .env
```
- `.env`ファイルを開き、必要なAPIキーを設定
  - `OPENAI_API_KEY`: OpenAI APIキー
  - その他のAPIキー（必要な場合）

## 使用方法

1. アプリケーションの起動
```bash
streamlit run app.py
```

2. ブラウザで`http://localhost:8501`にアクセス

3. ドキュメントのアップロードと質問
- 「ChromaDB 管理」ページでドキュメントをアップロード
- 「質問する」ページでドキュメントについて質問

## 注意事項

- APIキーは`.env`ファイルで管理され、GitHubにはアップロードされません
- `.env`ファイルは必ず`.gitignore`に含まれており、誤ってコミットされないようになっています
- 本番環境では、適切なセキュリティ対策を行ってください
- SQLite のセットアップには、Visual Studio と CMake が必要です（Windows の場合）
- このアプリケーションはChromaDBをインメモリモードで使用しています。そのため、アプリを再起動するとデータは失われます。
- より永続的なデータ保存が必要な場合は、ChromaDBの設定を変更する必要があります。

## ライセンス

MIT License

## 未実装

- 会話ログの出力機能（テキストやｃSV？）
- ベクトルDBの参照正確度

- langchainの他機能（チェーンとかエージェント？）
- ベクトルDBメタデータ修正
- geminiAPI
- テキストファイル以外の読み込み（PDF、CSV）

