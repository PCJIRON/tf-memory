# GraphRAG Memory Server

A powerful MCP (Model Context Protocol) server that implements a **Graph-based Retrieval-Augmented Generation (GraphRAG)** memory system. This server enables AI assistants to maintain persistent memory across sessions by storing chat history, file relationships, and project phases in a graph database with semantic search capabilities.

## 🌟 Features

- **🧠 Persistent Memory**: Store and retrieve chat interactions across sessions
- **🔗 Graph-Based Relationships**: Link chats, files, and project phases in a connected knowledge graph
- **🔍 Semantic Search**: TF-IDF vector search to find relevant past interactions
- **💥 Blast Radius Analysis**: Traverse the graph to find all related files and phases for any interaction
- **💾 Lightweight Storage**: Uses SQLite for graph storage with NetworkX for fast traversal
- **⚡ Zero External Dependencies**: No need for ChromaDB or other vector databases - comes with built-in TF-IDF search

## 📋 Prerequisites

- Python 3.10 or higher
- pip package manager

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/PCJIRON/tf-memory.git
   cd tf-memory/local-vector-mcp
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📖 How It Works

This MCP server provides two main tools:

### 1. `store_interaction`
Stores chat history and relationships in the graph database.

**Parameters**:
- `chat_id` (required): Unique ID for the interaction (e.g., 'chat-001')
- `content` (required): Summary of the conversation or AI response
- `phase` (optional): The current project phase or TODO this belongs to
- `modified_files` (optional): Comma-separated list of files edited (e.g., 'src/App.tsx,src/pages/Login.tsx')

**What it does**:
- Creates a chat node in the graph
- Links to phase node (if provided)
- Links to file nodes (if provided)
- Vectorizes the content for semantic search

### 2. `retrieve_context`
Retrieves relevant past interactions using semantic search and graph traversal.

**Parameters**:
- `query` (required): What to search for in memory

**What it does**:
- Uses TF-IDF to find the most similar past chat
- Traverses the graph (1-hop) to find related files and phases
- Returns compiled context with primary match and associations

## 🎯 Usage

### Running the Server

Start the MCP server:

```bash
python memory_server.py
```

The server runs on stdio (standard input/output) as per the MCP protocol specification.

### Integrating with AI Tools

This server is designed to work with MCP-compatible AI tools and frameworks. Configure your AI client to use this server as an MCP endpoint.

#### Example: Using with Claude Desktop or MCP-compatible clients

Add this to your MCP configuration:

```json
{
  "mcpServers": {
    "graph-memory": {
      "command": "python",
      "args": ["path/to/memory_server.py"],
      "cwd": "path/to/local-vector-mcp"
    }
  }
}
```

### Example Workflow

#### 1. Store an Interaction

```json
{
  "tool": "store_interaction",
  "arguments": {
    "chat_id": "chat-001",
    "content": "Implemented user authentication with JWT tokens",
    "phase": "authentication",
    "modified_files": "src/auth.py,src/models/user.py,tests/test_auth.py"
  }
}
```

**Result**: Creates nodes for the chat, phase, and files, with edges linking them together.

#### 2. Retrieve Related Context

```json
{
  "tool": "retrieve_context",
  "arguments": {
    "query": "How did we handle authentication?"
  }
}
```

**Result**: Returns the most relevant past interaction along with associated files and phases.

## 🏗️ Architecture

### Storage Layer

- **SQLite Database**: Stores nodes and edges in `.memory_db/graph.db`
  - `nodes` table: Stores entity ID, type (chat/phase/file), and content
  - `edges` table: Stores relationships between nodes with relation types

- **NetworkX Graph**: In-memory graph loaded from SQLite for fast traversal and analysis

### Search Engine

- **Custom TF-IDF Implementation**: Lightweight vector search without external dependencies
  - Tokenization and term frequency computation
  - Inverse document frequency (IDF) calculation
  - Cosine similarity for document matching
  - Automatically indexes chat and decision nodes

### Graph Structure

```
phase_authentication
        │
        └─ CONTAINS ─► chat-001: "Implemented user authentication..."
                              │
                              └─ MODIFIES ─► file_auth.py
                              │
                              └─ MODIFIES ─► file_user.py
```

## 📁 Project Structure

```
local-vector-mcp/
├── memory_server.py          # Main MCP server implementation
├── requirements.txt          # Python dependencies
├── README.md                # This file
└── .memory_db/              # Auto-generated database directory
    ├── graph.db             # SQLite graph database
    └── chroma/              # (Reserved for future ChromaDB integration)
```

## 🔧 Advanced Usage

### Manual Database Inspection

You can inspect the SQLite database using any SQLite browser or the command line:

```bash
sqlite3 .memory_db/graph.db "SELECT * FROM nodes;"
sqlite3 .memory_db/graph.db "SELECT * FROM edges;"
```

### Extending the Server

You can extend the server by:
1. Adding new tool definitions in `handle_list_tools()`
2. Implementing new tool handlers in `handle_call_tool()`
3. Adding new node/edge types for your use case
4. Enhancing the TF-IDF implementation with stemming or better tokenization

## 🎨 Use Cases

- **AI Coding Assistants**: Maintain context across multiple coding sessions
- **Project Documentation**: Automatically build knowledge graphs of project evolution
- **Decision Tracking**: Store and retrieve architectural decisions with their context
- **File Impact Analysis**: Understand which files are related to specific features or phases
- **Context Retrieal**: Quickly find relevant past work when tackling similar problems

## 🔮 Future Enhancements

- [ ] Support for multi-hop graph traversal in retrieve_context
- [ ] Advanced NLP (stemming, lemmatization) for better search
- [ ] Optional ChromaDB integration for embedding-based search
- [ ] Export/import functionality for knowledge sharing
- [ ] REST API wrapper for web-based integrations
- [ ] Visualization tools for the knowledge graph

## 🐛 Troubleshooting

### Server won't start
- Ensure Python 3.10+ is installed
- Verify all dependencies are installed: `pip install -r requirements.txt`

### No results from retrieve_context
- The memory might be empty. Use `store_interaction` first to populate it
- Check that your query is semantically similar to stored content

### Database corruption
- The `.memory_db` directory contains the database files
- You can safely delete it to start fresh (warning: this erases all memory)

## 📄 License

This project is open source. Feel free to use, modify, and distribute.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📧 Contact

For questions or support, reach out via the repository's issue tracker.

---

**Built with ❤️ for better AI memory management**
