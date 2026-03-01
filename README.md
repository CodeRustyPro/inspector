# Cat Inspect AI — "Institutional Memory" for Field Operations

> **The Problem:** Cat Inspect captures 6 million inspections a year, but visual data remains trapped in PDFs. Cat AI Assistant answers text-based questions but doesn't "look" at the machine's history.

**Cat Inspect AI** is the bridge. It transforms every inspection photo into a searchable data point, building a collective "fleet memory" that evolves with every snap of a camera.

---

## 🚀 Why It's Different

- **Institutional Memory:** Unlike standard vision AI, this system doesn't just see a defect; it remembers when that same defect occurred on other machines in your fleet and tells you the outcome.
- **Regulatory Awareness:** Directly maps findings to **MSHA 30 CFR** regulations using a live-indexed regulatory database.
- **Predictive Escalation:** Analyzes historically similar visual patterns (e.g., small hydraulic weeps) to predict time-to-failure and cost-of-neglect.
- **True Multimodal:** Seamlessly fuses visual assessment, voice-to-text notes, and equipment-specific structured data (Cat SIS).

---

## 🛠️ Key Capabilities

- **Intelligent Inspection:** Snap a photo, speak a note, and get a severity-rated assessment informed by fleet history.
- **Visual Parts Identification:** Identify complex components and retrieve part numbers directly from photos.
- **Automated Reporting:** Generate professional, data-driven inspection reports with one click.
- **Context-Aware OCR:** Reads nameplates and serial numbers to cross-reference with service history.

---

## 🤖 AI Disclosure & Methodology

This project utilizes advanced AI to automate the heavy lifting of equipment inspection:

- **Visual Reasoning:** Uses **Google Gemini 2.0 Flash** for state-of-the-art vision analysis, defect reasoning, and regulatory mapping.
- **Visual Similarity:** Implements **CLIP (ViT-B/32)** embeddings to enable cross-modal search across the fleet's historical image database.
- **Vector Intelligence:** Powered by **Actian VectorAI DB** to store and retrieve high-dimensional "memory" of past inspections and eCFR regulations.

---

## 🏁 Quick Start

1. Start the backend: `docker compose up -d`
2. Configure your environment: `export GEMINI_API_KEY='your-key'`
3. Launch the app: `run.sh`
4. Access the portal at `http://localhost:8000`

---
*Built for HackIllinois 2026 — Caterpillar Challenge Track.*
