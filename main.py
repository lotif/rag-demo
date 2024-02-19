import datetime

from rag_utils import build_retrieval_qa_pipeline


def ask(question):
    retrieval_qa_pipeline = build_retrieval_qa_pipeline()

    start = datetime.datetime.now()
    print("Asking question...")
    response = retrieval_qa_pipeline({"query": question})
    end = datetime.datetime.now()

    print("\n================================================================================")
    print(f"Answer: {response['result']}")
    print(f"Retrieved in {(end - start).seconds}s")

    print("================================================================================")
    source_docs = response["source_documents"]
    for i, doc in enumerate(source_docs):
        print(f"Source Document {i + 1}")
        print(f"Document Name: {doc.metadata['source']}")
        print(f"Page Number: {doc.metadata['page']}")
        print(f"Source Text: {doc.page_content}")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


# ask("Which telescope is the biggest NASA has ever made?")
# ask("How does the James Webb telescope digitizes the analog signals from the near-IR detectors?")