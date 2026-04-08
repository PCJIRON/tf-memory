import os
import json
import sqlite3
import networkx as nx
import asyncio
from typing import Any
import math
from collections import defaultdict

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

# ==========================================
# GraphRAG Memory Server Setup
# ==========================================
db_dir = os.path.join(os.path.dirname(__file__), ".memory_db")
os.makedirs(db_dir, exist_ok=True)

# SQLite + NetworkX for Relationships (Graph Space)
sqlite_conn = sqlite3.connect(os.path.join(db_dir, "graph.db"))
cursor = sqlite_conn.cursor()

cursor.executescript("""
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    type TEXT,
    content TEXT
);
CREATE TABLE IF NOT EXISTS edges (
    source TEXT,
    target TEXT,
    relation TEXT,
    UNIQUE(source, target, relation)
);
""")
sqlite_conn.commit()

# Load DB into NetworkX for fast traversal
G = nx.DiGraph()

for row in cursor.execute("SELECT id, type, content FROM nodes"):
    G.add_node(row[0], type=row[1], content=row[2])
for row in cursor.execute("SELECT source, target, relation FROM edges"):
    G.add_edge(row[0], row[1], relation=row[2])

server = Server("graph-memory-mcp")

# ==========================================
# Simple TF-IDF Vector Search (No ChromaDB needed)
# ==========================================
class SimpleVectorSearch:
    def __init__(self):
        self.documents = {}
        self.vocabulary = set()
        self.idf = defaultdict(float)
    
    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenizer: lowercase, split by whitespace, remove punctuation"""
        text = text.lower()
        tokens = text.split()
        return [t.strip(".,!?;:'\"()[]{}") for t in tokens if t.strip()]
    
    def _compute_tf(self, tokens: list[str]) -> dict[str, float]:
        """Compute term frequency"""
        tf = defaultdict(float)
        for token in tokens:
            tf[token] += 1
        total = len(tokens) if tokens else 1
        return {token: count / total for token, count in tf.items()}
    
    def _cosine_similarity(self, vec1: dict[str, float], vec2: dict[str, float]) -> float:
        """Compute cosine similarity between two sparse vectors"""
        all_keys = set(vec1.keys()) | set(vec2.keys())
        dot_product = sum(vec1.get(k, 0) * vec2.get(k, 0) for k in all_keys)
        norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)
    
    def add_document(self, doc_id: str, text: str):
        """Add document to the index"""
        tokens = self._tokenize(text)
        self.documents[doc_id] = tokens
        self.vocabulary.update(tokens)
        self._update_idf()
    
    def _update_idf(self):
        """Update inverse document frequency"""
        n_docs = len(self.documents)
        doc_freq = defaultdict(int)
        for doc_id, tokens in self.documents.items():
            for token in set(tokens):
                doc_freq[token] += 1
        self.idf = {
            token: math.log(n_docs / (1 + freq)) + 1
            for token, freq in doc_freq.items()
        }
    
    def query(self, query_text: str, n_results: int = 1) -> dict:
        """Search for most similar documents"""
        if not self.documents:
            return {"ids": [], "documents": [], "metadatas": [], "distances": []}

        query_tokens = self._tokenize(query_text)
        query_vec = self._compute_tf(query_tokens)

        # Weight query by IDF
        query_vec = {
            token: tf * self.idf.get(token, 1.0)
            for token, tf in query_vec.items()
        }

        # Compute similarity with all documents
        similarities = {}
        for doc_id, doc_tokens in self.documents.items():
            doc_vec = self._compute_tf(doc_tokens)
            doc_vec = {token: tf * self.idf.get(token, 1.0) for token, tf in doc_vec.items()}
            sim = self._cosine_similarity(query_vec, doc_vec)
            similarities[doc_id] = sim

        # Sort by similarity
        sorted_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        top_results = sorted_results[:n_results]

        # Return flat lists instead of nested lists to avoid serialization issues
        return {
            "ids": [r[0] for r in top_results],
            "documents": [self.documents.get(r[0], [""])[0] if self.documents.get(r[0]) else "" for r in top_results],
            "metadatas": [{} for _ in top_results],
            "distances": [1 - r[1] for r in top_results]
        }

# Initialize vector search
vector_search = SimpleVectorSearch()

# Load existing documents into vector search
for node_id, data in G.nodes(data=True):
    if data.get("type") in ["chat", "decision"]:
        vector_search.add_document(node_id, data.get("content", ""))

# ==========================================
# Helpers
# ==========================================
def add_node(node_id: str, n_type: str, content: str):
    cursor.execute("INSERT OR REPLACE INTO nodes (id, type, content) VALUES (?, ?, ?)", (node_id, n_type, content))
    sqlite_conn.commit()
    G.add_node(node_id, type=n_type, content=content)

    # Only vectorize chat and decision nodes
    if n_type in ["chat", "decision"]:
        vector_search.add_document(node_id, content)

def add_edge(src: str, tgt: str, relation: str):
    cursor.execute("INSERT OR IGNORE INTO edges (source, target, relation) VALUES (?, ?, ?)", (src, tgt, relation))
    sqlite_conn.commit()
    G.add_edge(src, tgt, relation=relation)


# ==========================================
# MCP Tools
# ==========================================
@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="store_interaction",
            description="Store chat history and its relationships (modified files, phase) into the GraphRAG memory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "chat_id": {"type": "string", "description": "Unique ID for this interaction (e.g. 'chat-001')"},
                    "content": {"type": "string", "description": "Summary of the conversation or AI response"},
                    "phase": {"type": "string", "description": "The current phase or TODO this belongs to (empty if none)"},
                    "modified_files": {"type": "string", "description": "Comma-separated list of files edited in this interaction (e.g. 'src/App.tsx,src/pages/Login.tsx')"}
                },
                "required": ["chat_id", "content"]
            }
        ),
        types.Tool(
            name="retrieve_context",
            description="Use semantic query to find past chats, then traverse the graph to return related files and phases (Blast Radius).",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for in memory"}
                },
                "required": ["query"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict[str, Any] | None) -> list[types.TextContent]:
    if not arguments:
        raise ValueError("Missing arguments")

    if name == "store_interaction":
        chat_id = arguments["chat_id"]
        content = arguments["content"]
        phase = arguments.get("phase", "")
        modified_files_raw = arguments.get("modified_files", "")
        
        # Parse comma-separated string to list
        files = [f.strip() for f in modified_files_raw.split(",") if f.strip()] if modified_files_raw else []

        # 1. Create central Chat node & Vectorize it
        add_node(chat_id, "chat", content)

        # 2. Link Phase
        if phase:
            phase_id = f"phase_{phase.replace(' ', '_').lower()}"
            add_node(phase_id, "phase", phase)
            add_edge(phase_id, chat_id, "CONTAINS")

        # 3. Link Files (Edges: Chat -> modifies -> File)
        for f in files:
            file_id = f"file_{f.split('/')[-1]}"
            add_node(file_id, "file", f)
            add_edge(chat_id, file_id, "MODIFIES")

        return [types.TextContent(type="text", text=f"Graph Memory updated natively. Nodes and edges attached to '{chat_id}'.")]

    elif name == "retrieve_context":
        query = arguments["query"]
        if not vector_search.documents:
            return [types.TextContent(type="text", text="Graph Memory is empty.")]

        # Vector Search: Find closest Chat node
        results = vector_search.query(query_text=query, n_results=1)

        if not results.get("ids") or len(results["ids"]) == 0:
            return [types.TextContent(type="text", text="No related context found.")]

        root_node = results["ids"][0]

        # Graph Traversal: Find 1-hop connections (Blast Radius)
        connected_nodes = list(nx.single_source_shortest_path_length(G.to_undirected(), root_node, cutoff=1).keys())

        compiled_context = []
        compiled_context.append(f"**Primary Match [{root_node}]**: {G.nodes[root_node]['content']}")

        related_files = []
        related_phases = []

        for n in connected_nodes:
            if n == root_node: continue
            data = G.nodes[n]
            if data["type"] == "file":
                related_files.append(data["content"])
            elif data["type"] == "phase":
                related_phases.append(data["content"])

        if related_files:
            compiled_context.append(f"**Associated Files**: {', '.join(related_files)}")
        if related_phases:
            compiled_context.append(f"**Associated Phases**: {', '.join(related_phases)}")

        final_text = "GraphRAG Retrieved Context:\n" + "\n".join(compiled_context)
        return [types.TextContent(type="text", text=final_text)]

    raise ValueError(f"Unknown tool: {name}")

async def main():
    # Create proper ServerCapabilities with tools enabled
    capabilities = types.ServerCapabilities(
        tools=types.ToolsCapability(listChanged=True)
    )
    
    options = InitializationOptions(
        server_name="graph-memory-mcp",
        server_version="1.0.0",
        capabilities=capabilities
    )
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, options)

if __name__ == "__main__":
    asyncio.run(main())
