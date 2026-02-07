# LLM-Assisted Regulatory / Policy Mapping Assistant (Prototype)

## Overview
This project is a prototype AI system designed to demonstrate how artificial intelligence can be used to **extract, understand, and map regulatory or policy rules to structured outputs**.

The system simulates a regulatory reporting assistant that:
- Accepts a natural language scenario
- Retrieves relevant regulatory rules using semantic search
- Interprets those rules
- Maps them to predefined policy/reporting fields
- Produces structured, explainable outputs with an audit trail

Although a simplified COREP-style reporting example is used, the architecture is **domain-agnostic** and applicable to any rule or policy interpretation task.

---

## Problem Statement
Regulatory and policy documents are typically unstructured, complex, and difficult to interpret manually. Analysts must read dense rules and determine how they affect specific policy or reporting fields, which is time-consuming and error-prone.

This prototype demonstrates how an AI-based architecture can:
- Learn and understand regulatory rules
- Extract relevant information from unstructured text
- Map rules to structured policy outputs
- Maintain transparency and traceability in decisions

---

## System Architecture

The system follows a **Retrieval-Augmented Reasoning (RAG-style)** pipeline:

1. **Document Ingestion**
   - Regulatory rules and instructions are loaded from text files.
   - Documents are split into semantically meaningful chunks.

2. **Semantic Embedding & Search**
   - Chunks are embedded using a sentence-transformer model.
   - FAISS is used for fast similarity-based retrieval.
   - Relevant rules are retrieved based on the user’s natural language query.

3. **Reasoning & Policy Mapping**
   - Retrieved rules are interpreted using a controlled reasoning layer.
   - Rules are mapped to predefined policy/reporting fields.
   - Impacts are inferred (e.g., increase or decrease).

4. **Structured Output & Auditability**
   - Results are produced in a structured format.
   - An audit log records which regulatory rules informed each decision.

---

## Example Use Case

**User Input**


**AI Output**
- CET1 Capital → Increase
- Tier 1 Capital → Increase
- Total Capital → Increase

**Audit Log**
- PRA Own Funds Rule: CET1 includes ordinary share capital

This mirrors how a human analyst would reason, but in a faster, consistent, and explainable way.

---

## Technologies Used

- Python
- Sentence-Transformers (for embeddings)
- FAISS (for vector similarity search)
- NumPy
- Rule-based reasoning layer (LLM-style controlled reasoning)

---

## Key Skills Demonstrated

- Artificial Intelligence system design
- Information extraction from unstructured text
- Semantic search and embeddings
- Policy and rule interpretation
- Prompt-style constrained reasoning
- Explainability and auditability
- Clean, modular architecture

---

## Why This Project Matters

This prototype demonstrates an **AI architecture mindset**, not just model usage.  
It focuses on:
- Trustworthy AI behavior
- Structured outputs instead of free text
- Separation of retrieval, reasoning, and presentation
- Extensibility to real-world regulatory or policy domains

---

## Limitations and Future Work

- The rule set is intentionally small for demonstration.
- The reasoning layer is rule-based; it can be extended using an LLM.
- Numeric calculations and full regulatory coverage are out of scope for this prototype.

Future extensions could include:
- Integration with a full LLM for deeper reasoning
- Support for multiple policy templates
- Enhanced validation and consistency checks
- Web-based user interface

---

## Conclusion

This project serves as a proof-of-concept for building AI systems that can **understand rules, map them to policy, and explain their decisions**, aligning closely with real-world regulatory and compliance use cases.
