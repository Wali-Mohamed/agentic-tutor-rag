# agentic-tutor-rag
An Agentic RAG virtual assistant for GCSE math tutoring, featuring tool calling, evaluation, and monitoring. Built for LLM Zoomcamp 2026.


## Problem Statement

Parents and students looking for GCSE Mathematics tuition often have many questions before deciding whether to book lessons, such as:

- Which exam boards are supported?
- How much do lessons cost?
- What lesson times are available?
- How are lessons delivered?
- Can the tutor help struggling students?
- What are the payment and cancellation policies?

Although this information exists, it is often scattered across websites, emails, WhatsApp conversations, or personal notes. As a result, prospective students must wait for responses, while tutors spend valuable time repeatedly answering the same questions instead of focusing on teaching.

This project addresses that problem by developing an **AI-powered Retrieval-Augmented Generation (RAG) assistant** for a GCSE Mathematics tutoring service. Rather than relying solely on a Large Language Model (LLM), the system first retrieves the most relevant information from a structured knowledge base containing tutor information, pricing, scheduling, teaching philosophy, lesson delivery, business policies, and booking instructions. The retrieved context is then provided to the LLM to generate accurate, natural-language responses.

Using a retrieval-based approach ensures that responses are grounded in verified information from the knowledge base, reducing hallucinations while improving consistency and reliability. The result is a virtual assistant that provides instant, trustworthy answers to frequently asked questions, improving the experience for both prospective students and the tutor.