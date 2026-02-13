from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings,ChatHuggingFace,HuggingFaceEndpoint
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import os 
from dotenv import load_dotenv
load_dotenv()
hf_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")

class RAGPipeline:
    def __init__(self,video_id):
        self.video_id=video_id
        self.parser=StrOutputParser()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.embeddings = self.getting_embeddings()

        if not os.path.exists("vectorstore"):
            print("Creating vectorstore...")
            transcript = self.get_transcript()
            docs = self.text_split(transcript)
            self.store_embeddings(docs, self.embeddings)
        self.chain=self.form_chain(self.setting_retriever())
    
    def get_session_history(self, session_id):
        return self.memory.chat_memory

    def get_transcript(self):
        print(f"Fetching transcript for video ID: {self.video_id}")
        transcript_list=YouTubeTranscriptApi().fetch(self.video_id,languages=["en"])
        print(f"Transcript fetched with {len(transcript_list)} entries.")
        return " ".join([f"[{i.start:.2f}s]{i.text}" for i in transcript_list])

    def text_split(self,text):
        print("Splitting transcript into chunks...")
        splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        print(f"Transcript split into {len(splitter.create_documents([text]))} chunks.")
        return splitter.create_documents([text])

    def getting_embeddings(self):
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    def store_embeddings(self,docs, embeddings):
        vectorstore=FAISS.from_documents(docs, embeddings)
        vectorstore.save_local("vectorstore")

    def load_embeddings(self):
        print("Loading vectorstore...")
        embeddings=self.getting_embeddings()
        print("Vectorstore loaded.")
        return FAISS.load_local("vectorstore", embeddings)

    def setting_retriever(self):
        print("Setting up retriever...")
        vectorstore = FAISS.load_local(
            "vectorstore",
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "lambda_mult": 0.7}
        )
        print("Retriever set up.")
        return retriever


    def format_docs(self,retrieved_docs):
        print(f"Formatting {len(retrieved_docs)} retrieved documents...")
        context_text = "".join(doc.page_content for doc in retrieved_docs)
        print("Documents formatted into context.")
        return context_text
    
    def form_chain(self,retriever):
        print("Forming RAG chain...")
        prompt = PromptTemplate(
        template = """
        You are a YouTube video QA assistant.

        Use ONLY the transcript context below to answer.
        Provide the timestamp in the format [xx.xx]s.
        If multiple timestamps exist, list all of them.
        If the answer is not explicitly stated, say:
        "I could not find this in the video transcript."

        Respond in this exact format:

        <final answer sentence>
        Provide ONLY ONE most relevant timestamp in format:
        Timestamps: [xx.xx]s

        Transcript Context:
        {context}

        Question: {question}
        """,
            input_variables = ['context', 'question']
        )
        llm=HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",task="conversational", max_new_tokens=1024,temperature=0.5)
        chat_model = ChatHuggingFace(llm=llm,)
        parallel_chain = RunnableParallel({
            'context': RunnableLambda(lambda x: x["question"]) |retriever | RunnableLambda(self.format_docs),
            'question': RunnablePassthrough()
        })
        main_chain = parallel_chain | prompt | chat_model | self.parser
        chain_with_memory = RunnableWithMessageHistory( main_chain, self.get_session_history, input_messages_key="question", history_messages_key="history")
        print("RAG chain formed.")
        return chain_with_memory
    
    def run(self, question):
        print(f"Running RAG pipeline for question: {question}")
        answer = self.chain.invoke(
            {"question": question},
            config={"configurable": {"session_id": "default"}}
        )
        return answer
