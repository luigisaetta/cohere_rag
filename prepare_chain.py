#
# Factory Methods
#

from langchain.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate

from langchain_community.llms import Cohere
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

import weaviate
from weaviate.auth import AuthApiKey
from langchain_community.embeddings import CohereEmbeddings

from config_private import COHERE_KEY, WV_API_KEY

# number of docs returned from retriever
TOP_K = 10
# number of docs returned from reranker
TOP_N = 6

# To adapt to Weaviate this one must be used even for English
EMBED_MODEL = "embed-multilingual-v3.0"

# the URL of the WV cluster
WEAVIATE_URL = "https://orals-cluster1-j7b4goli.weaviate.network"

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

#
# the Factory Method
#
def prepare_chain(index_name="Document3"):

    auth_client_secret = AuthApiKey(api_key=WV_API_KEY)

    # stick to v3 to use hybrid search
    wv_client = weaviate.Client(
        url=WEAVIATE_URL,
        auth_client_secret=auth_client_secret,
        additional_headers={"X-Cohere-Api-Key": COHERE_KEY}
    )

    # create the HB retriever
    retriever_hb = WeaviateHybridSearchRetriever(client=wv_client,
                                                 # here we should specify the name of the index
                                                 index_name=index_name, 
                                                 text_key="text")

    retriever_hb.k = TOP_K

    # add on top add a reranker    
    compressor = CohereRerank(cohere_api_key=COHERE_KEY, top_n=TOP_N)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever_hb
    )

    prompt_template = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
    ---------------------
    Context: {context}
    ---------------------
    Question: {question} 
    Answer:
    """

    prompt = PromptTemplate.from_template(prompt_template)

    llm = Cohere(model="command", max_tokens=512, temperature=0.1, cohere_api_key=COHERE_KEY)

    #
    # a little more complex because we want metadata and references
    #
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": compression_retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    return rag_chain_with_source



