import os
import gradio as gr
import requests
import json
from typing import List, Dict
import time
from urllib.parse import quote
import re

# LlamaIndex imports
from llama_index.core import Settings, Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openrouter import OpenRouter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

class ResearchPaperFinder:
    def __init__(self):
        # Initialize LlamaIndex components
        self.setup_llamaindex()
        
    def setup_llamaindex(self):
        """Initialize LlamaIndex with OpenRouter LLM and embedding model"""
        try:
            # Configure OpenRouter LLM
            self.llm = OpenRouter(
                api_key=OPENROUTER_API_KEY,
                model="meta-llama/llama-3.1-8b-instruct:free",
                temperature=0.3
            )
            
            # Configure embedding model (using local HuggingFace model)
            self.embed_model = HuggingFaceEmbedding(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Configure LlamaIndex settings
            Settings.llm = self.llm
            Settings.embed_model = self.embed_model
            Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
            
            print("LlamaIndex initialized successfully")
            
        except Exception as e:
            print(f"LlamaIndex initialization error: {str(e)}")
            self.llm = None
            self.embed_model = None
    
    def search_arxiv_papers(self, query: str, max_results: int = 15) -> List[Dict]:
        """Search ArXiv for papers using their API"""
        try:
            # Clean and format query for ArXiv
            clean_query = re.sub(r'[^\w\s]', ' ', query)
            clean_query = ' '.join(clean_query.split())
            search_query = quote(f"all:{clean_query}")
            
            url = f"http://export.arxiv.org/api/query?search_query={search_query}&start=0&max_results={max_results}&sortBy=relevance&sortOrder=descending"
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            papers = self.parse_arxiv_response(response.text)
            return papers
            
        except Exception as e:
            print(f"ArXiv search error: {str(e)}")
            return []
    
    def parse_arxiv_response(self, xml_content: str) -> List[Dict]:
        """Parse ArXiv XML response"""
        import xml.etree.ElementTree as ET
        
        papers = []
        try:
            root = ET.fromstring(xml_content)
            
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                title_elem = entry.find('{http://www.w3.org/2005/Atom}title')
                summary_elem = entry.find('{http://www.w3.org/2005/Atom}summary')
                id_elem = entry.find('{http://www.w3.org/2005/Atom}id')
                published_elem = entry.find('{http://www.w3.org/2005/Atom}published')
                
                # Get authors
                authors = []
                for author in entry.findall('{http://www.w3.org/2005/Atom}author'):
                    name_elem = author.find('{http://www.w3.org/2005/Atom}name')
                    if name_elem is not None:
                        authors.append(name_elem.text)
                
                if title_elem is not None and summary_elem is not None and id_elem is not None:
                    paper = {
                        'title': title_elem.text.strip().replace('\n', ' '),
                        'abstract': summary_elem.text.strip().replace('\n', ' '),
                        'url': id_elem.text.strip(),
                        'authors': ', '.join(authors[:3]) + (' et al.' if len(authors) > 3 else ''),
                        'published': published_elem.text[:10] if published_elem is not None else 'N/A'
                    }
                    papers.append(paper)
                    
        except Exception as e:
            print(f"XML parsing error: {str(e)}")
            
        return papers
    
    def create_paper_index(self, papers: List[Dict]) -> VectorStoreIndex:
        """Create LlamaIndex vector store from papers"""
        try:
            documents = []
            for i, paper in enumerate(papers):
                # Create comprehensive document content
                content = f"""Title: {paper['title']}
Authors: {paper['authors']}
Published: {paper['published']}
Abstract: {paper['abstract']}
URL: {paper['url']}"""
                
                # Create LlamaIndex document with metadata
                doc = Document(
                    text=content,
                    metadata={
                        'paper_id': i,
                        'title': paper['title'],
                        'authors': paper['authors'],
                        'url': paper['url'],
                        'published': paper['published']
                    }
                )
                documents.append(doc)
            
            # Create vector store index
            index = VectorStoreIndex.from_documents(documents)
            return index
            
        except Exception as e:
            print(f"Index creation error: {str(e)}")
            return None
    
    def rank_papers_with_llamaindex(self, papers: List[Dict], topic: str, description: str) -> List[Dict]:
        """Use LlamaIndex to intelligently rank and retrieve papers"""
        if not papers or not self.llm:
            return papers[:3]
        
        try:
            # Create vector index of papers
            index = self.create_paper_index(papers)
            if not index:
                return papers[:3]
            
            # Create query engine
            retriever = VectorIndexRetriever(index=index, similarity_top_k=len(papers))
            query_engine = RetrieverQueryEngine(retriever=retriever)
            
            # Create comprehensive query
            query = f"""Research Topic: {topic}
Research Description: {description}

Find and rank the most relevant papers for this research. Consider:
1. Direct relevance to the research topic
2. Methodological alignment with research goals
3. Potential contribution to the research objectives
4. Quality and impact of the work

Return the top 3 most relevant papers with brief explanations."""
            
            # Query the index
            response = query_engine.query(query)
            
            # Extract paper IDs from retrieved nodes
            retrieved_papers = []
            seen_ids = set()
            
            if hasattr(response, 'source_nodes'):
                for node in response.source_nodes[:3]:  # Top 3
                    paper_id = node.metadata.get('paper_id')
                    if paper_id is not None and paper_id not in seen_ids:
                        if paper_id < len(papers):
                            retrieved_papers.append(papers[paper_id])
                            seen_ids.add(paper_id)
            
            # If we don't have enough papers, fill with remaining ones
            while len(retrieved_papers) < 3 and len(retrieved_papers) < len(papers):
                for i, paper in enumerate(papers):
                    if i not in seen_ids:
                        retrieved_papers.append(paper)
                        seen_ids.add(i)
                        if len(retrieved_papers) >= 3:
                            break
            
            return retrieved_papers[:3]
            
        except Exception as e:
            print(f"LlamaIndex ranking error: {str(e)}")
            return papers[:3]  # Fallback to first 3
    
    def generate_relevance_summary(self, paper: Dict, topic: str, description: str) -> str:
        """Use LlamaIndex LLM to generate relevance summary"""
        if not self.llm:
            return f"This paper addresses {topic} and presents relevant methodologies for your research."
        
        try:
            prompt = f"""Research Topic: {topic}
Research Description: {description}

Paper Details:
Title: {paper['title']}
Authors: {paper['authors']}
Abstract: {paper['abstract'][:800]}

Analyze why this specific paper is relevant to the research topic and description. Write a concise 2-3 sentence explanation focusing on:
1. Key contributions that align with the research goals
2. Methodologies or findings that could be useful
3. How it advances understanding in this research area

Be specific about the connections between the paper and the research needs."""

            response = self.llm.complete(prompt)
            return response.text.strip()
            
        except Exception as e:
            print(f"Summary generation error: {str(e)}")
            return f"This paper on {topic} presents relevant research findings and methodologies that align with your research objectives."
    
    def find_papers(self, topic: str, description: str) -> str:
        """Main function using LlamaIndex for intelligent paper discovery"""
        if not topic.strip():
            return "Please provide a research topic."
        
        if not OPENROUTER_API_KEY:
            return "Error: OpenRouter API key not configured. Please set the OPENROUTER_API_KEY environment variable."
        
        if not self.llm:
            return "Error: LlamaIndex not properly initialized. Please check your API configuration."
        
        try:
            # Search for papers
            search_query = f"{topic} {description}".strip()
            print(f"Searching ArXiv for: {search_query}")
            
            papers = self.search_arxiv_papers(search_query, max_results=20)
            
            if not papers:
                return f"No papers found for the topic: {topic}. Try using different keywords or a broader topic."
            
            print(f"Found {len(papers)} papers. Using LlamaIndex for intelligent ranking...")
            
            # Use LlamaIndex to rank papers intelligently
            top_papers = self.rank_papers_with_llamaindex(papers, topic, description)
            
            if not top_papers:
                return "Error occurred during paper analysis. Please try again."
            
            # Generate results with LlamaIndex LLM
            results = []
            for i, paper in enumerate(top_papers, 1):
                print(f"Generating summary for paper {i}...")
                relevance_summary = self.generate_relevance_summary(paper, topic, description)
                
                result = f"""**Paper {i}: {paper['title']}**
**Authors:** {paper['authors']}
**Published:** {paper['published']}
**Link:** {paper['url']}

**Why this paper is relevant (AI Analysis):**
{relevance_summary}

**Abstract Preview:**
{paper['abstract'][:300]}...

---
"""
                results.append(result)
                time.sleep(0.5)  # Rate limiting
            
            header = f"# 🔬 AI-Curated Research Papers for: {topic}\n*Powered by LlamaIndex + OpenRouter*\n\n"
            final_result = header + "\n".join(results)
            
            if len(top_papers) < 3:
                final_result += f"\n*Note: Found {len(top_papers)} highly relevant papers using AI analysis.*"
            
            return final_result
            
        except Exception as e:
            return f"An error occurred: {str(e)}. Please try again with different keywords."

# Initialize the finder
finder = ResearchPaperFinder()

def search_papers(topic, description):
    """Gradio wrapper function"""
    return finder.find_papers(topic, description)

# Create Gradio interface
with gr.Blocks(
    title="🔬 AI Research Paper Finder (LlamaIndex + OpenRouter)",
    theme=gr.themes.Soft(),
    css="""
    .container { max-width: 1200px; margin: auto; }
    .header { text-align: center; margin-bottom: 2rem; }
    .tech-stack { background: #e3f2fd; padding: 1rem; border-radius: 8px; margin: 1rem 0; }
    """
) as app:
    
    gr.Markdown("""
    # 🔬 AI Research Paper Finder
    ### *Powered by LlamaIndex + OpenRouter*
    
    Intelligent research paper discovery using advanced AI ranking and semantic search.
    This tool combines ArXiv search with LlamaIndex's vector database and OpenRouter's LLM capabilities.
    """, elem_classes="header")
    
    gr.Markdown("""
    **🚀 Technology Stack:**
    - **LlamaIndex**: Vector database and semantic search for intelligent paper ranking
    - **OpenRouter**: Advanced LLM (Llama 3.1) for relevance analysis and summaries  
    - **ArXiv API**: Access to 2M+ research papers
    - **HuggingFace Embeddings**: Semantic understanding of paper content
    """, elem_classes="tech-stack")
    
    with gr.Row():
        with gr.Column(scale=1):
            topic_input = gr.Textbox(
                label="🎯 Research Topic",
                placeholder="e.g., 'transformer models for protein folding prediction'",
                lines=2,
                max_lines=3
            )
            
            description_input = gr.Textbox(
                label="📝 Research Description & Requirements",
                placeholder="Describe your specific research focus, preferred methodologies, datasets you're working with, or particular aspects you want to explore...",
                lines=4,
                max_lines=8
            )
            
            search_btn = gr.Button("🔍 Find Papers with AI", variant="primary", size="lg")
            
            gr.Markdown("""
            ### 💡 How the AI Analysis Works:
            1. **Semantic Search**: LlamaIndex creates vector embeddings of paper content
            2. **Intelligent Ranking**: AI analyzes relevance beyond keyword matching
            3. **Contextual Summaries**: LLM explains why each paper matters for your research
            4. **Quality Focus**: Returns only the most relevant papers, not quantity
            """)
        
        with gr.Column(scale=2):
            output = gr.Markdown(
                value="""**Ready to find papers!** 

Enter your research topic and description, then click "Find Papers with AI" to:
- Search ArXiv's database of research papers
- Use LlamaIndex for intelligent semantic ranking
- Get AI-generated relevance explanations
- Receive direct links to the most suitable papers

*First search may take 30-60 seconds as the AI models initialize.*""",
                elem_classes="tech-stack"
            )
    
    # Connect search function
    search_btn.click(
        fn=search_papers,
        inputs=[topic_input, description_input],
        outputs=output,
        show_progress=True
    )
    
    # Enhanced examples showing LlamaIndex capabilities
    gr.Examples(
        examples=[
            [
                "Graph Neural Networks for Drug Discovery", 
                "Looking for GNN applications in molecular property prediction, drug-target interaction, and chemical synthesis planning. Prefer recent methods with benchmark comparisons."
            ],
            [
                "Few-shot Learning for Medical Image Segmentation", 
                "Need papers on meta-learning approaches for medical imaging with limited labeled data. Focus on methods that work with MRI, CT, or ultrasound images."
            ],
            [
                "Multimodal Large Language Models", 
                "Research on models that combine vision and language understanding. Interested in architecture designs, training strategies, and evaluation benchmarks."
            ],
            [
                "Federated Learning with Differential Privacy", 
                "Papers combining FL with privacy-preserving techniques. Looking for theoretical analysis and practical implementations in healthcare or finance."
            ]
        ],
        inputs=[topic_input, description_input],
        label="🎯 Example Research Queries (Try These!)"
    )
    
    gr.Markdown("""
    ---
    **🔧 Powered by:** LlamaIndex (Vector Search) + OpenRouter (LLM Analysis) + ArXiv (Paper Database)
    """)

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )