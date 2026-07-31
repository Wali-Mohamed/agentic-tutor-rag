import numpy as np

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
        boost_dict = {"user_query": 2.0, "mapped_context": 1.0}

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


class VectorRAG(RAGBase):
    """Subclass of RAGBase optimized for vector search indices."""

    def search(self, query, num_results=5, filter_dict=None):
        return self.index.search(
            query, num_results=num_results, filter_dict=filter_dict
        )