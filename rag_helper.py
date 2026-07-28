INSTRUCTIONS = '''
Your task is to act as a helpful virtual assistant for Wali, a GCSE and KS3 Math tutor. 
You will answer questions from parents and students based strictly on the provided context.

Use the context to find relevant information and provide polite, clear, and accurate answers. 
If the answer is not found in the provided context, do not guess or make up information. 
Instead, respond with: "I don't have that information. Please contact Wali directly via Phone or WhatsApp at 07737889846."
'''

PROMPT_TEMPLATE = '''
QUESTION: {question}

CONTEXT:
{context}
'''.strip()


class RAGBase:

    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        model='gpt-5.4-mini'
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.model = model

    def search(self, query, num_results=5,filter_dict=None):
        # Boost user_query since it closely matches how a user will ask the question
        boost_dict = {'user_query': 2.0, 'mapped_context': 1.0}
        # We default to no filter, but allow the agent to pass one (like {'category': 'pricing'})
        if filter_dict is None:
            filter_dict = {}      

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict
        )

    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            # Build the context string using the keys from your tutor dataset
            lines.append(f"Category: {doc.get('category', 'General')} > {doc.get('subcategory', 'General')}")
            lines.append('Q: ' + doc['user_query'])
            lines.append('A: ' + doc['expected_answer'])
            lines.append('')

        return '\n'.join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(
            question=query, context=context
        )

    def llm(self, prompt):
        input_messages = [
            {'role': 'developer', 'content': self.instructions},
            {'role': 'user', 'content': prompt}
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages
        )

        return response.output_text

    def rag(self, query, filter_dict=None):
        # Pass the filter_dict down to the search method
        search_results = self.search(query, filter_dict=filter_dict)
        
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)
        
        return answer
