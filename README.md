
# 🔬 AI Research Paper Finder

*Powered by LlamaIndex + OpenRouter*
<img width="1667" height="227" alt="image" src="https://github.com/user-attachments/assets/635c6d91-3bdd-4efb-9142-fbe7d4dead83" />


An intelligent research paper discovery tool that uses **semantic search** and **AI-based ranking** to help researchers quickly find the most relevant academic papers.
It combines **ArXiv’s research paper database** with **LlamaIndex’s vector embeddings** and **OpenRouter’s LLM capabilities** to provide:

* 📑 Curated paper recommendations
* 🧠 AI-generated relevance explanations
* ⚡ Intelligent ranking beyond keyword matching
* 🔗 Direct links to ArXiv papers

---

# Demo 

if this link is not working, it might be due to limited API usage. But you can checkout the demo video below
https://huggingface.co/spaces/ajnx014/research-paper-finder

### Video
https://github.com/user-attachments/assets/60517ab0-0022-4fa6-a74c-e805088e906a

## 🚀 Features

* **Semantic Search** – Uses LlamaIndex with HuggingFace embeddings for contextual paper matching.
* **Intelligent Ranking** – AI prioritizes relevance, quality, and methodological alignment.
* **Contextual Summaries** – LLM explains *why* a paper is relevant to your query.
* **ArXiv Integration** – Access to **2M+ papers** directly from ArXiv’s API.
* **User-Friendly Interface** – Built with Gradio for simple input/output interaction.

---

## 🛠️ Tech Stack  

| Component     | Technology |
|---------------|------------|
| 🔍 Search     | [LlamaIndex](https://www.llamaindex.ai/) |
| 🤖 LLM        | [OpenRouter](https://openrouter.ai/) (Llama 3.1) |
| 📚 Dataset    | [ArXiv API](https://arxiv.org/help/api) |
| 📝 Embeddings | [HuggingFace](https://huggingface.co/sentence-transformers) |
| 🎨 UI         | [Gradio](https://gradio.app/) |

---

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/ArjunJagdale/AI-Research-Paper-Finder.git
cd AI-Research-Paper-Finder
pip install -r requirements.txt
```

Make sure you have an **OpenRouter API key**:

```bash
export OPENROUTER_API_KEY="your_api_key_here"
```

---

## ▶️ Usage

Run the app locally:

```bash
python app.py
```

By default, the app will be available at:
👉 `http://localhost:7860`

### Example Queries

* **Graph Neural Networks for Drug Discovery** – Find GNN applications in molecular property prediction.
* **Few-shot Learning for Medical Image Segmentation** – Papers on meta-learning for medical imaging.
* **Multimodal Large Language Models** – Studies combining vision + language.
* **Federated Learning with Differential Privacy** – Research on privacy-preserving federated learning.

---

## 📊 How It Works

1. **User Input**: Provide a research topic + description of requirements.
2. **Paper Retrieval**: ArXiv API fetches matching papers.
3. **Vector Indexing**: LlamaIndex builds embeddings for semantic ranking.
4. **AI Ranking & Summaries**: OpenRouter’s LLM selects top papers and explains relevance.
5. **Output**: Top 3 curated papers with links, abstracts, and AI-generated relevance notes.

---

## 📝 Example Output

```markdown
# 🔬 AI-Curated Research Papers for: Few-shot Learning for Medical Image Segmentation
*Powered by LlamaIndex + OpenRouter*

**Paper 1: XYZ Title**  
**Authors:** Author1, Author2 et al.  
**Published:** 2024-03-15  
**Link:** https://arxiv.org/abs/1234.5678  

**Why this paper is relevant (AI Analysis):**  
This paper proposes meta-learning approaches that align with the challenge of limited labeled data in medical imaging. The methodology directly supports segmentation tasks for MRI and CT scans.  

**Abstract Preview:**  
In this paper, we propose a few-shot segmentation method for medical images...  
```

---

## 🤝 Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

---

## 📜 License

This project is licensed under the MIT License.
