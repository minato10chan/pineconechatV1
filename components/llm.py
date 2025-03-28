import os
from dotenv import load_dotenv
load_dotenv() # .envファイルは親ディレクトリ方向に探索される
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

# ChatOpenAI
llm = ChatOpenAI(
    model="gpt-4o-mini",  # または "gpt-3.5-turbo" を使用
    temperature=0,
    api_key=OPENAI_API_KEY
)

# Embedding モデル
oai_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)

# 動作確認
if __name__ == "__main__":
    # LLM試験
    res = llm.invoke("こんにちは")
    print(res)
    # Embeddings試験
    # single_vector = embeddings.embed_query(text)
    # two_vectors = embeddings.embed_documents([text, text2])
    documents=["こんにちは","こんばんは"]
    embeddings_doc = oai_embeddings.embed_documents(documents)
    # 長いので最初のテキストの最初の５つのみ
    print(embeddings_doc[0][:5])

