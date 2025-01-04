from typing import Optional, List  # Add this line
from langchain_core.callbacks import CallbackManager
from langchain.llms.base import LLM
from ollama import chat, ChatResponse
from pydantic import BaseModel  # Pydantic's BaseModel for field definitions

from langchain_core.output_parsers import StrOutputParser

class OllamaLLM(LLM, BaseModel):
    model_name: str
    verbose: bool = False

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if self.verbose:
            print(f"Sending prompt to {self.model_name}: {prompt}")
        response: ChatResponse = chat(
            model=self.model_name,
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response.message.content

    @property
    def _llm_type(self) -> str:
        return "ollama"

# Replace LlamaCpp with OllamaLLM
model = OllamaLLM(model_name="llama3.2", verbose=True)

# Define a ChatPromptTemplate (no changes required here)
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.prompts import PromptTemplate

template = """
    {context}
    """

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model | StrOutputParser()

print(chain.invoke({"context": "بازی فوتبال تراکتور و گل گهر چند چند شد؟"}))


prompt = ChatPromptTemplate(
    input_variables=["context", "question"],
    messages=[
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["context", "question"],
                template=(
                    "You are an assistant for question-answering tasks. "
                    "Use the following pieces of retrieved context to answer the question. "
                    "If you don't know the answer, just say that you don't know. "
                    "Use three sentences maximum and keep the answer concise.\n"
                    "Question: {question}\n"
                    "Context: {context}\n"
                    "Answer:"
                )
            )
        )
    ]
)

# Integration with the custom model
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(doc for doc in docs)

# Define the RAG chain
rag_chain = (
    {"context": format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)


from langchain_core.runnables import RunnableMap

# Example input
inputs = {
    "context": "",
    "question": "where is perspolis football team located?"
}

# Run the RAG chain
output = rag_chain.invoke(inputs)
print("Generated Answer:", output)
