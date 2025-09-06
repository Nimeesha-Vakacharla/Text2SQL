# Text2SQL
# Full-Stack Agentic Text2SQL

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

## Overview

This repository contains the implementation of **Full-Stack Agentic Text2SQL**, a comprehensive web application designed to translate natural language queries into executable SQL statements. Built for non-technical users, it democratizes database access by allowing users to interact with relational databases using everyday English, without requiring SQL expertise. The system leverages the LLaMA 3.1 8B large language model (LLM), fine-tuned on the Spider dataset, and incorporates an innovative agentic layer for ambiguity resolution, query optimization, and schema-aware validation.

The project addresses key challenges in Text-to-SQL translation, such as ambiguous inputs, complex database schemas, and accuracy in query generation. It provides a user-friendly Streamlit-based frontend for input and visualization, a robust backend with SQLite databases, and local LLM inference via Ollama for efficient, cost-effective deployment.

This project was developed as part of the DATA 266 Group Project (May 12, 2025) by Team 2.

### Key Objectives
- Enable non-technical users (e.g., managers, analysts, students) to query databases in natural language.
- Bridge the gap between human language and structured SQL, handling complex queries like multi-table joins, aggregations, and nested subqueries.
- Incorporate an agentic framework for interactive feedback, error handling, and explainability.
- Achieve high performance through fine-tuning on the Spider dataset, with metrics showing significant improvements over baselines.

### Why This Project?
Traditional Text-to-SQL tools (e.g., SQLNet, RAT-SQL, or commercial LLMs like GPT-3/Codex) often lack interactivity, schema validation, or robust ambiguity handling. Our system stands out by:
- Using a modular agent (OllamaText2SQLAgent) that generates multiple SQL candidates, scores them, and selects the optimal one.
- Providing runtime diagnostics (e.g., alerts for Cartesian products or missing LIMIT clauses).
- Offering explainable outputs with query history, schema views, and visualizations.
- Running entirely locally with Ollama for privacy and low latency.

## Features

- **Natural Language Query Input**: Users enter queries like "Which students failed math?" or "Show sales trends for Q1".
- **SQL Generation and Execution**: Powered by fine-tuned LLaMA 3.1 8B, generates and runs SQL against SQLite databases from the Spider dataset.
- **Agentic Ambiguity Resolution**: Detects vague queries (e.g., "Show student records"), prompts for clarification via the UI, and refines the query contextually.
- **Schema Introspection**: Uses SQLAlchemy to extract table/column metadata, primary/foreign keys, and constraints for accurate generation.
- **Query Validation and Optimization**: Scores candidates based on schema compliance, coverage, and conditions; checks for errors before execution.
- **Interactive UI**: Streamlit frontend with:
  - Text input for queries.
  - Display of generated SQL, results in Pandas DataFrames.
  - Schema viewer for tables/columns.
  - Dynamic visualizations (bar charts, pie charts, scatter plots) using Plotly.
  - Query history for tracking past interactions.
- **Performance Metrics**:
  - Pre-trained LLaMA 3.1 8B: BLEU: 0.3925, Exact Match: 0.0700, SQL Accuracy: 0.0700.
  - Fine-tuned Model: BLEU: 0.5158, Exact Match: 0.2400, SQL Accuracy: 0.2400.
  - Improvements: +31.5% BLEU, +240% Exact Match/SQL Accuracy.
- **Dataset Support**: Trained and evaluated on the Spider dataset (200+ cross-domain schemas from education, aviation, etc.).
- **Local Deployment**: Uses Ollama for CPU-based LLM inference, ensuring accessibility on consumer hardware.

## Architecture

The system follows a full-stack architecture with a clear separation of concerns: frontend for user interaction, backend for processing, and an agentic layer for intelligent query handling.

### High-Level Components
1. **Frontend (Streamlit UI)**: Handles user input, displays results, schema, and visualizations.
2. **Backend (Python Services)**:
   - Database: SQLite with Spider schemas.
   - Schema Analyzer: Custom `DatabaseAnalyzer` class using SQLAlchemy for introspection.
   - LLM Pipeline: LLaMA 3.1 8B via Ollama for SQL generation.
3. **Agentic Layer (OllamaText2SQLAgent)**: Core intelligence module that:
   - Parses schema and builds prompts.
   - Generates multiple SQL candidates.
   - Resolves ambiguity with user prompts.
   - Validates and executes queries.
   - Provides feedback and error handling.
4. **Data Pipeline**: Loads Spider dataset, preprocesses schemas, fine-tunes model with LoRA.
<img width="1047" height="495" alt="image" src="https://github.com/user-attachments/assets/ff07a0e4-d8a8-456e-81e7-90cec2211c5f" />


## Usage

1. Launch the Streamlit app as above.
2. In the UI:
   - View schema in the sidebar.
   - Enter a query (e.g., "List students with major 101").
   - If ambiguous, respond to follow-up prompts (e.g., "Include activity participation?").
   - View generated SQL, results table, and charts.
3. Query History: Stored in-session; persists across interactions.
4. Error Handling: If a query fails, the agent provides feedback (e.g., "Invalid join detected").

### Example Queries
- Simple: "Which students failed math?" → `SELECT * FROM student WHERE grade < 60;`
- Complex: "Show activities with more than 10 participants" → Involves JOINs and aggregations.
- Ambiguous: "Show student records" → Prompts: "Which table: students or faculty?" → Refined SQL.

For handling ambiguous queries, the agent uses LLaMA's contextual understanding and UI prompts to refine inputs.

## Demo

A live demo is available at `http://localhost:8503` after setup. Screenshots are in `docs/images/` (e.g., image1.png for UI, image2.png for results).

## Performance and Results

- **Evaluation on Spider Dataset**:
  | Metric          | Pre-trained | Fine-tuned | Improvement |
  |-----------------|-------------|------------|-------------|
  | BLEU Score     | 0.3925     | 0.5158    | +31.5%     |
  | Exact Match    | 0.0700     | 0.2400    | +240%      |
  | SQL Accuracy   | 0.0700     | 0.2400    | +240%      |

- Sample Output: Query "Count all singers" → SQL: `SELECT count(*) FROM singer` (Exact Match).

### Model Comparisons
| Model       | Strengths                          | Limitations                          | Performance on Spider |
|-------------|------------------------------------|--------------------------------------|-----------------------|
| SQLNet (2017) | Avoids RL, reduces syntax errors  | Single-table only, no joins         | Low generalization   |
| RAT-SQL (2020)| Handles complex schemas as graphs | Static, no validation/interactivity | State-of-the-art     |
| GPT-3/Codex| Zero-shot, infers schema          | Opaque, no error handling           | Competitive but unreliable |
| Our System | Agentic, explainable, interactive | Local inference limits scale        | BLEU: 0.5158         |

Justification for LLaMA 3.1 8B: Instruction-tuned, efficient (8B params), excels in structured prompts and ambiguity handling.

## Model Training Details

- **Pre-training**: LLaMA 3.1 8B-Instruct from Hugging Face, loaded with 4-bit quantization (BitsAndBytesConfig) for efficiency.
- **Fine-tuning**: LoRA adapter on Spider (train/validation split), 3 epochs, LR=2e-4, cosine scheduler. Only 0.6744% parameters trained.
- **Outcome**: Improved schema understanding and SQL syntax.

Scripts in `scripts/` for training/evaluation.

## Future Work

- **Enhanced Disambiguation**: Add multi-turn dialogue and contextual embeddings.
- **Automated Feedback**: Implement RLHF from user interactions.
- **Multimodal Support**: Handle image-based schemas via OCR; extend to NoSQL databases.
- **Scalability**: GPU support, cloud deployment (e.g., AWS), larger models like LLaMA 70B.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please fork the repo and submit a PR. For issues, open a ticket on GitHub.

## Acknowledgments

- Spider Dataset: Yu et al. (2018).
- LLaMA 3.1: Meta AI (via Hugging Face).
- Tools: Streamlit, Ollama, SQLAlchemy, Plotly.

For questions, contact the team via GitHub issues. Thank you!
