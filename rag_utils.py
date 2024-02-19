import transformers
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from torch import bfloat16


def build_db():
    # Loading all the PDFs from the documents folder
    print("Loading documents...")
    loader = DirectoryLoader(path="documents", glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()

    # Getting all the texts in chunks of 500 characters with 50 character overlap
    print("Splitting texts...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Embeddings model
    embeddings_model = make_embeddings_model()

    # Load the texts into the vector database using the embeddings model
    print("Loading vector db...")
    vector_db = FAISS.from_documents(texts, embeddings_model)
    vector_db.save_local(folder_path="data")


def make_embeddings_model():
    print("Instantiating embeddings model...")
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


def build_retrieval_qa_pipeline():
    # LLM: LLama2 7B Chat
    # Local CTransformers model that runs on CPU
    # Download it to the model/ folder from the link below:
    # https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q8_0.bin
    print("Instantiating LLM...")
    llm = CTransformers(
        model="model/llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        config={
            "max_new_tokens": 256,
            "temperature": 0.001,
        }
    )

    # If you want to run it on GPU, use the code below instead:
    # llm = get_llm_on_gpu()

    # Embeddings model
    embeddings_model = make_embeddings_model()

    # Vector Database (FAISS)
    # First param is the path where the data will be stored
    print("Instantiating vector db...")
    vector_db = FAISS.load_local(folder_path="data", embeddings=embeddings_model)

    # Setting up prompt
    print("Setting up prompt...")
    qa_template = """Use the following pieces of information to answer the user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}
        Question: {question}

        Only return the helpful answer below and nothing else.
        Helpful answer:
        """
    prompt_template = PromptTemplate(
        template=qa_template,
        input_variables=["context", "question"],
    )

    # Building langchain Retrieval QA
    print("Building retrieval qa pipeline...")
    retrieval_qa_pipeline = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )

    return retrieval_qa_pipeline


def get_llm_on_gpu():
    model_id = "meta-llama/Llama-2-7b-chat-hf"

    # begin initializing HF items, you need an access token
    hf_auth = "<add your access token here>"
    model_config = transformers.AutoConfig.from_pretrained(
        model_id,
        use_auth_token=hf_auth
    )
    # set quantization configuration to load large model with less GPU memory
    # this requires the `bitsandbytes` library
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16,
    )
    # Pulling the model from Huggingface
    llm = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map="auto",
        use_auth_token=hf_auth,
    )

    return llm
