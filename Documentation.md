Of course. You're ready to tackle a project that truly pushes the boundaries of what a portfolio piece can be. Let's design a general-purpose system that combines fine-tuning with a sophisticated, multi-agent architecture.

Here is the complete documentation for **"Synapse," an Autonomous Research Agent**. This project is designed to be a language-agnostic, powerful tool that showcases the full spectrum of advanced GenAI engineering skills.

* * * * *

### **Project Documentation: "Synapse" - The Autonomous Research Agent**

#### **1\. Project Vision & Goal**

-   **Problem Statement**: High-quality research is a time-consuming, multi-step process involving query planning, web browsing, document analysis, and synthesis. Standard RAG systems can fetch facts from a predefined knowledge base, but they cannot autonomously explore new topics, validate information across multiple sources, or create novel, structured reports from scratch.

-   **Project Goal**: To engineer an autonomous, multi-agent AI system named "Synapse" that can conduct comprehensive research on any user-provided topic. Synapse will dynamically create a research plan, execute it by browsing the web and analyzing user-provided documents, and synthesize the findings into a well-structured, cited report. The core of this system will be a **fine-tuned LLM specialized in the skill of synthesis and summarization**, making it a powerful, general-purpose research tool.

-   **Success Metric**: The final application will accept a complex research topic, produce a detailed report with verifiable citations, and its core synthesis module (the fine-tuned model) will quantitatively outperform its base model on a custom evaluation benchmark for summarization quality.

#### **2\. System Architecture: A Planner-Executor Multi-Agent System**

The architecture is modeled after a human research workflow, with a "Planner" agent directing a team of "Executor" agents.

**Architectural Flow Diagram:**

```
+--------------+     +-----------------+     +-----------------+
|  User Topic  |---->| FastAPI Backend |---->|  Planner Agent  |
| (e.g., "...")|     +-----------------+     |  (LangChain)    |
+--------------+                             +--------+--------+
      ^                                                | (Creates Research Plan)
      |                                                v
      |                                    +-----------+-----------+
      |                                    |   Task Queue (e.g.,   |
      |                                    |  ["Search for X",     |
      |                                    |   "Analyze PDF Y"]    |
      |                                    +-----------+-----------+
      |                                                |
      | (Final Report)                     +-----------+-----------+
      |                                    |                       |
      v                                    v                       v
+--------------+     +-------------------+   +---------------------+   +--------------------------+
|  Synthesizer |<----|  Aggregated Data  |<--| Web Research Agent  |   | Document Analysis Agent  |
| Agent (Fine- |     |  (Structured JSON)|   | (Executes Search    |   | (Executes PDF/TXT        |
| Tuned Model) |     +-------------------+   | & Scrape Tasks)     |   | Analysis Tasks)          |
+--------------+                             +---------------------+   +--------------------------+

```

#### **3\. Technology Stack**

| Component | Technology | Rationale & Key Features |
| --- | --- | --- |
| **Model Fine-Tuning** | Google AI Studio / Vertex AI | Provides an accessible platform to fine-tune Gemini models for the specialized task of high-quality synthesis. |
| **Base Model** | Gemini 1.5 Flash | Excellent for its balance of performance, speed, and cost, making it ideal for the synthesis fine-tuning task. |
| **Core LLM** | Gemini 2.5 Pro | The primary reasoning engine for the Planner and Executor agents due to its superior instruction-following and planning capabilities. |
| **Agent Framework** | LangGraph (part of LangChain) | The perfect choice for creating cyclical, stateful multi-agent systems like the Planner-Executor model. It provides more control than standard Agent Executors. |
| **Data Indexing** | LlamaIndex & FAISS | LlamaIndex for robust document ingestion; FAISS for high-speed vector storage and retrieval for the Document Analysis Agent. |
| **Web Search Tool** | `google-search-results` (SerpAPI) | A reliable tool for providing the Web Research Agent with real-time access to Google search results. |
| **Frontend** | Streamlit or Gradio | For building a polished, interactive web UI where users can submit topics, upload files, and view the final research report. |
| **API Backend** | FastAPI | For creating a scalable, asynchronous, and robust backend to serve the agent's capabilities. |
| **Containerization** | Docker & Docker Compose | To ensure a reproducible environment and simplify the deployment of the entire multi-service application. |
| **Evaluation** | RAGAS & Custom Scripts | RAGAS for evaluating the document analysis agent's retrieval quality, and custom scripts for evaluating the summarization/synthesis quality of the fine-tuned model. |

* * * * *

#### **4\. Detailed Implementation Plan**

##### **Phase 1: Dataset Creation & Fine-Tuning (Est. 10 hours)**

1.  **Generate the Synthesis Fine-Tuning Dataset (`data/synthesis_dataset.jsonl`)**:

    -   **Goal**: Create a dataset to teach a model how to synthesize messy, multi-source text into a clean, structured summary.

    -   **Format**: Use the instruction-following JSONL format: `{"text": "<s>[INST] {prompt} [/INST] {output} </s>"}`.

    -   **Process**:

        1.  **Gather Raw Data**: Programmatically collect sets of 3-5 related but distinct text snippets on various topics (e.g., search results for "CRISPR gene editing," sections from a Wikipedia article, etc.).

        2.  **Create Prompts**: The prompt for each example will be a structured input like:

            JSON

            ```
            "Summarize the following documents into a coherent, well-structured report with an introduction, key findings, and a conclusion. Cite each piece of information with its corresponding source number.\n\n[Source 1]: '...raw text...'\n[Source 2]: '...raw text...'\n[Source 3]: '...raw text...'"

            ```

        3.  **Generate High-Quality Outputs**: Manually write the first 50-100 "golden" outputs. The output should be a perfectly formatted markdown report. Then, use Gemini 2.5 Pro to help you scale this process and generate hundreds more examples.

2.  **Fine-Tune Gemini 1.5 Flash**:

    -   Use Google AI Studio to upload your dataset and train the model.

    -   Thoroughly test the fine-tuned model in the playground to confirm it has learned the skill of synthesis and citation. Note its API endpoint.

##### **Phase 2: Building the Multi-Agent System (Est. 9 hours)**

1.  **Tool Definition (`tools/research_tools.py`)**:

    -   **Web Search Tool**: Create a tool using the `google-search-results` library that takes a search query and returns a list of snippets and URLs.

    -   **Web Scraper Tool**: Create a tool that takes a URL and returns the clean text content of the page (use a library like `BeautifulSoup`).

    -   **Document Indexing & Query Tool**: This tool will be used by the Document Analysis Agent. It will need two functions: one to take an uploaded file, process it with LlamaIndex, and add it to a FAISS index, and another to query that index.

2.  **Agent Creation with LangGraph**:

    -   **Define Agent Nodes**: In LangGraph, each agent is a "node." You'll define nodes for the `Planner`, `Web_Researcher`, `Doc_Analyzer`, and `Synthesizer`.

    -   **Planner Agent**: This agent receives the user's topic and creates a JSON object representing a step-by-step research plan. Example plan: `{"steps": [{"agent": "Web_Researcher", "task": "Find the latest news on nuclear fusion breakthroughs"}, {"agent": "Doc_Analyzer", "task": "Summarize the key findings of the uploaded 'ITER_report.pdf'"}, ...]}`

    -   **Executor Agents**: The `Web_Researcher` and `Doc_Analyzer` agents will receive tasks from the Planner and use their tools to execute them. They will return structured data (e.g., JSON with "content" and "source" keys).

    -   **Define Edges**: Use LangGraph's conditional "edges" to route the flow of control. After the Planner creates the plan, the graph will loop through the tasks, calling the appropriate agent for each step. Once all research tasks are complete, the final edge will route the aggregated data to the `Synthesizer` node.

    -   **Synthesizer Agent**: This final node will call your **fine-tuned Gemini model** to perform the synthesis task and generate the final report.

##### **Phase 3: Backend, Frontend, and Deployment (Est. 8 hours)**

1.  **FastAPI Backend**:

    -   Implement an endpoint `/research` that accepts a topic and optionally a list of uploaded files.

    -   This endpoint will trigger your LangGraph agent system.

    -   Because research can be slow, implement this using FastAPI's `BackgroundTasks` or a more robust task queue like Celery, allowing you to immediately return a "job ID" to the user, who can then poll another endpoint for the final result.

2.  **Streamlit Frontend**:

    -   Design a UI that allows a user to enter a research topic and upload one or more documents.

    -   When the "Start Research" button is clicked, it will call the `/research` endpoint.

    -   The UI should then display the agent's progress in real-time by polling a status endpoint on the backend, showing which task is currently being executed.

    -   Once complete, it will display the final, beautifully formatted markdown report.

3.  **Dockerization**:

    -   Create a `docker-compose.yml` file to manage the `backend` and `frontend` services, ensuring they can communicate and that all necessary environment variables and data volumes (for FAISS indexes) are correctly configured.

##### **Phase 4: Evaluation and Documentation (Est. 3 hours)**

1.  **Summarization Quality Evaluation (`evaluation/evaluate_synthesis.py`)**:

    -   Create a hold-out test set of 20-30 multi-source documents.

    -   Generate a summary for each set using both the **base Gemini 1.5 Flash model** and your **fine-tuned model**.

    -   Use another powerful LLM (like Gemini 2.5 Pro) as an evaluator. Prompt it to rate both summaries on a scale of 1-10 for coherence, accuracy, and proper citation.

    -   Report the average score improvement of your fine-tuned model.

2.  **Finalize `README.md`**:

    -   Document the entire project, including the architecture diagram, setup instructions using Docker Compose, and a detailed "Evaluation" section that presents your quantitative results, proving the value of your fine-tuning.

This project is ambitious, but it directly maps to the skills of a top-tier GenAI engineer. By completing it, you will have a powerful story to tell about building complex, reliable, and specialized AI systems from the ground up.