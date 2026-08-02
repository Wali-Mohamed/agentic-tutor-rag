import numpy as np
from fastembed.rerank.cross_encoder import TextCrossEncoder
import minsearch

# ==========================================================
# 1. System Prompts & Templates
# ==========================================================

INSTRUCTIONS = """
Your task is to act as a helpful virtual assistant for Wali, a GCSE and KS3 Math tutor. 
You will answer questions from parents and students based strictly on the provided context.

Use the context to find relevant information and provide polite, clear, and accurate answers. 
If the answer is not found in the provided context, do not guess or make up information. 
Instead, respond with: "I don't have that information. Please contact Wali directly via Phone or WhatsApp at 07737889846."
""".strip()

PROMPT_TEMPLATE = """
QUESTION: {question}

CONTEXT:
{context}
""".strip()


# ==========================================================
# 2. Base RAG Class (For Keyword / Minsearch Search)
# ==========================================================


class RAGBase:

    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        model="gpt-4o-mini",
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.model = model

    def search(self, query, num_results=5, filter_dict=None):
        # Boost user_query since it closely matches how a user will ask the question
        boost_dict = {"user_query": 3.0, "mapped_context": 1.0}

        if filter_dict is None:
            filter_dict = {}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict,
        )

    def build_context(self, search_results):
        lines = []
        for doc in search_results:
            lines.append(
                f"Category: {doc.get('category', 'General')} > {doc.get('subcategory', 'General')}"
            )
            lines.append("Q: " + doc.get("user_query", ""))
            lines.append("A: " + doc.get("expected_answer", ""))
            lines.append("")

        return "\n".join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(question=query, context=context)

    def llm(self, prompt):
        input_messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt},
        ]

        response = self.llm_client.responses.create(
            model=self.model, input=input_messages
        )

        return response.output_text

    def rag(self, query, filter_dict=None):
        search_results = self.search(query, filter_dict=filter_dict)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)
        return answer


# ==========================================================
# 3. Vector Search Index (ONNX / FastEmbed Retriever)
# ==========================================================


class VectorIndex:

    def __init__(self, embedder):
        self.embedder = embedder
        self.documents = []
        self.matrix = None

    def fit(self, documents, text_to_embed):
        """Stores documents and generates the vector matrix.

        text_to_embed: List of strings to encode into vectors.
        """
        self.documents = documents
        vectors = list(self.embedder.embed(text_to_embed))
        self.matrix = np.vstack(vectors)

    def search(self, query, num_results=5, filter_dict=None):
        # 1. Embed the query using the ONNX embedder
        query_vector = next(self.embedder.embed([query]))

        # 2. Hard metadata filtering
        valid_indices = []
        for i, doc in enumerate(self.documents):
            match = True
            if filter_dict:
                for key, value in filter_dict.items():
                    if doc.get(key) != value:
                        match = False
                        break
            if match:
                valid_indices.append(i)

        if not valid_indices:
            return []

        # 3. Vector dot-product similarity on filtered subset
        valid_matrix = self.matrix[valid_indices]
        scores = valid_matrix.dot(query_vector)

        # 4. Sort and return matching document dictionaries
        top_local_indices = np.argsort(-scores)[:num_results]
        return [self.documents[valid_indices[idx]] for idx in top_local_indices]


# ==========================================================
# 4. Vector RAG Class (Inherits from RAGBase)
# ==========================================================


class AdvancedRAG:
    def __init__(self, vector_index, keyword_index, llm_client, prompt_template):
        self.vector_index = vector_index
        self.keyword_index = keyword_index
        self.llm_client = llm_client
        self.prompt_template = prompt_template
        # Initialize FastEmbed's lightweight Cross-Encoder for Re-ranking
        self.reranker = TextCrossEncoder(model_name="BAAI/bge-reranker-base")
        

    def rewrite_query(self, user_query):
        """1. USER QUERY REWRITING"""
        prompt = f"""
        Rewrite the following user question into a clean, concise search query optimized for a math tutor knowledge base. 
        Remove conversational filler. Return ONLY the search query.
        User Question: "{user_query}"
        """
        response = self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()

    def hybrid_search(self, query, top_k=5):
        """2. HYBRID SEARCH (Vector + Keyword)"""
        # A. Vector Search
        vector_results = self.vector_index.search(query, num_results=top_k)
        
        # B. Keyword Search
        keyword_results = self.keyword_index.search(
            query=query, 
            filter_dict={}, 
            boost_dict={"question": 3.0, "mapped_context": 1.0}, # Give more weight to matching FAQ questions
            num_results=top_k
        )
        
        # C. Combine and Deduplicate (using 'context' as the unique key)
        combined_results = {}
        for doc in vector_results + keyword_results:
            combined_results[doc['mapped_context']] = doc
            
        return list(combined_results.values())
    def agentic_query_planner(self, user_query):
            """Replaces standard rewriting with an Agentic JSON planner."""
            prompt = f"""
            You are an expert search agent for a math tutor knowledge base.
            Analyze the user's question and create a search plan.
            
            You must reply in strict JSON format with exactly two keys:
            1. "vector_query": A clean, descriptive sentence capturing the semantic meaning of the question.
            2. "keywords": A list of 2-4 highly specific technical terms or nouns from the query. (Do not alter or summarize these words).
            
            User Question: "{user_query}"
            """
            
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"}, 
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            import json
            return json.loads(response.choices[0].message.content)
    
    def agentic_hybrid_search(self, search_plan, top_k=5):
        """Uses the Agent's plan to execute a precision search."""
        # 1. Vector Engine uses the semantic sentence
        vector_results = self.vector_index.search(
            search_plan["vector_query"], 
            num_results=top_k
        )
        
        # 2. Keyword Engine uses the exact technical terms
        keyword_string = " ".join(search_plan["keywords"])
        keyword_results = self.keyword_index.search(
            query=keyword_string, 
            filter_dict={}, 
            boost_dict={"user_query": 3.0, "mapped_context": 1.0}, 
            num_results=top_k
        )
        
        # 3. Combine and Deduplicate safely using chunk_id
        combined_results = {}
        for doc in vector_results + keyword_results:
            combined_results[doc['chunk_id']] = doc
            
        return list(combined_results.values())
    

    def rerank_documents(self, query, documents, top_n=3):
        """3. DOCUMENT RE-RANKING"""
        if not documents:
            return []
            
        # Extract just the text context for the Cross-Encoder to score
        contexts = [doc['mapped_context'] for doc in documents]
        
        # Cross-Encoder compares the query against each document context and scores relevance
        scores_generator = self.reranker.rerank(query, contexts)
        scores = list(scores_generator)
        
        # Attach scores to documents and sort them descending
        for i, doc in enumerate(documents):
            doc['rerank_score'] = scores[i]
            
        ranked_docs = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
        return ranked_docs[:top_n]

    def rag(self, user_query):
        """The Complete Pipeline"""
        # Step 1: Rewrite
        optimized_query = self.rewrite_query(user_query)
        
        # Step 2: Hybrid Retrieve (Get Top 10 combined)
        retrieved_docs = self.hybrid_search(optimized_query, top_k=5)
        
        # Step 3: Re-rank (Keep Top 3 most relevant)
        top_docs = self.rerank_documents(optimized_query, retrieved_docs, top_n=3)
        
        # Step 4: Generate
        context_str = "\n\n".join([doc['context'] for doc in top_docs])
        final_prompt = self.prompt_template.format(question=user_query, context=context_str)
        
        response = self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": final_prompt}]
        )
        return response.choices[0].message.content
    def agentic_rag(self, query, num_results=5):
        """End-to-end Agentic RAG pipeline for the Chatbot App."""
        # 1. Agent plans the search
        search_plan = self.agentic_query_planner(query)
        
        # 2. Fetch a wide pool of documents (Top 10)
        retrieved_docs = self.agentic_hybrid_search(search_plan, top_k=10)
        
        # 3. Reranker picks the absolute best (Top 5)
        top_docs = self.rerank_documents(query, retrieved_docs, top_n=num_results)
        
        # 4. Format the context for the LLM
        # (Make sure these keys match your actual document keys!)
        context_text = ""
        for doc in top_docs:
            context_text += f"Context: {doc['mapped_context']}\nAnswer: {doc['expected_answer']}\n\n"
            
        # 5. Generate the final human-friendly response
        # Assuming you saved a prompt_template in __init__
        final_prompt = self.prompt_template.format(question=query, context=context_text)
        
        response = self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": final_prompt}]
        )
        
        return response.choices[0].message.content
   
class VectorRAG(RAGBase):
    """Subclass of RAGBase optimized for vector search indices."""

    def search(self, query, num_results=5, filter_dict=None):
        return self.index.search(
            query, num_results=num_results, filter_dict=filter_dict
        )