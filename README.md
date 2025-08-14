# Offline Oracle

Offline Oracle is a powerful, universal toolkit designed to transform any offline ZIM file archive (like Wikipedia, Stack Exchange, or medical manuals) into a private, verifiable, and conversational knowledge base.

The core purpose is to allow users to have intelligent, cited conversations with their own data. By forcing a Large Language Model (LLM) to answer questions using only facts retrieved from your trusted ZIM files, Offline Oracle creates a system that provides reliable, fact-based answers, effectively eliminating model hallucination.

This project was initially developed with significant AI assistance for a specific use case and is now being released for the community. While it is feature-complete, there may be bugs or edge cases. We welcome contributions and issue reports via GitHub!

## The Vision: A Conversation with Your Archive

Imagine having an entire copy of Wikipedia offline and being able to ask it complex questions, not just search for keywords. Instead of reading through dozens of articles, you can have a conversation with an AI expert that synthesizes information and provides citations for every fact. That is the goal of Offline Oracle.

## Features

-   **Universal ZIM Ingestion:** Process any ZIM file into a searchable knowledge base.
-   **Robust Build Process:** The indexer (`build.py`) is designed to be interruptible and can be resumed at any time, saving progress automatically. It supports both CPU and GPU for embedding creation.
-   **Flexible LLM Support:** The server acts as a smart proxy, compatible with:
    -   **Offline Local Models:** `openai-compatible` servers like LM Studio, GPT4All, and Jan.
    -   **Offline Ollama Models:** Native support for local Ollama instances.
    -   **Online API Providers:** Full support for `OpenAI`, `Anthropic (Claude)`, and `Google (Gemini)`.
-   **Verifiable & Cited Answers:** The system prompt is engineered to force the LLM to cite the source article for every piece of information it provides.

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Viceman256/Offline-Oracle.git
    cd offline-oracle
    ```

2.  **Run the Interactive Setup:**
    This script will guide you through creating a virtual environment and installing all required packages. It will automatically ask if you want to install CPU or GPU (NVIDIA) versions of the dependencies.

    -   On **Windows**, double-click `1-Setup_Oracle.bat`.
    -   On **Linux or macOS**, run: `bash 1-Setup_Oracle.sh`

## Usage

1.  **Configure the Oracle:**
    -   The setup script automatically creates a `config.ini` file for you.
    -   Open `config.ini` and edit the `[Paths]` section to point to your ZIM files and desired output location.
    -   In the `[LLM]` section, set your desired `provider`. If using a local model, ensure the `backend_url` is correct. If using an API, fill in your `api_key` and desired `model`.
    -   Review other settings in the `[RAG]`, `[Server]`, and `[Builder]` sections to fine-tune performance.

2.  **Build the Knowledge Base:**
    This step processes your ZIM files and creates the searchable database. It can take a very long time, especially on a CPU. You can safely stop the process (`Ctrl+C`) and run the script again to resume where it left off.

    -   On **Windows**, double-click `2-Build_Oracle.bat`.
    -   On **Linux or macOS**, run: `bash 2-Build_Oracle.sh`

3.  **Run the Server:**
    This starts the main server that connects your knowledge base to your chosen LLM.

    -   On **Windows**, double-click `3-Run_Oracle.bat`.
    -   On **Linux or macOS**, run: `bash 3-Run_Oracle.sh`

4.  **Connect Your UI:**
    -   Open your chat UI (e.g., OpenWebUI).
    -   In the connection settings, point it to the Offline Oracle server address (by default: `http://localhost:1255`).
    -   You should now see your models listed with an "Oracle-" prefix. Select one and start your conversation!

## Advanced Configuration & Troubleshooting

#### Optimizing `num_workers` for the Build Process

In `config.ini`, the `num_workers` setting controls how many parallel processes are used to parse ZIM files.
-   By default (`num_workers = 0`), the script tries to pick a safe number. It detects your computer's **logical cores** (e.g., a 32-core CPU with SMT/Hyper-Threading has 64 logical cores) but caps the number of workers at 16.
-   This cap is a safety measure to prevent systems with many cores but limited RAM from crashing. Each worker uses a significant amount of memory.
-   **For powerful servers, you should manually set this value.** A good starting point is the number of **physical cores** your CPU has. 

#### Fixing `ReadTimeout` Errors with Local Models

If you are using a large local model on a CPU, you might see a `ReadTimeout` error in the server console. This means the model took too long to start generating its response.
-   **Solution:** In `config.ini`, increase the `request_timeout` value in the `[Server]` section. The default is `300.0` (5 minutes), which should be enough for most models, but you can increase it further if needed.

#### Understanding Model Behavior: "Lazy" vs. Literal Answers

You may notice that different LLMs produce very different answers even with the same information.
-   **Local Models (e.g., Mistral via LM Studio):** These models tend to be very literal. They will follow the system prompt's instructions precisely, resulting in perfectly synthesized and cited answers built *only* from the text you provide.
-   **Advanced API Models (e.g., GPT-4o, Claude 3):** These models are heavily tuned for "helpfulness" and safety. Sometimes, they will recognize a topic (like "heart failure treatment") and generate a high-quality, generic answer from their internal knowledge, only "cherry-picking" one or two citations from your provided text.
-   This is not a bug in Offline Oracle, but an inherent behavioral trait of the LLM. If you need strictly verifiable answers, a local model may perform more reliably than a powerful-but-creative API model.

## A Note on API vs. Offline Use

While support for online API providers like OpenAI, Anthropic, and Google has been fully implemented for flexibility, **Offline Oracle was fundamentally designed for offline use.**

The primary vision is a self-contained, private ecosystem where your personal data archives can be intelligently queried without ever sending information to an external server. The best and most private experience is achieved by using the `openai-compatible` or `ollama` providers to connect to a powerful local model running on your own machine.