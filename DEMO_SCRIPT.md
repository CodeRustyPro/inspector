# Cat Inspect AI — Video Script & Showcase Strategy

## Prizes You're Eligible For

Submit to ALL of these:
1. **Caterpillar Track** (primary — this IS the challenge)
2. **Best Use of Actian VectorAI DB** (fleet memory + hybrid search)
3. **Best Use of Gemini API** (vision analysis, OCR, report gen, parts ID)
4. **Best UI/UX** (optional category submission)
5. **Most Creative** (optional category submission)

---

## 3-Minute Video Script

The video should be screen-recorded with voiceover. No slides — show the app working.

### [0:00–0:20] THE HOOK — The Problem (20 sec)

**Show**: Google image of Cat Inspect current app (the simple checklist UI)

> *"Cat Inspect captures 6 million inspections a year. Photos, ratings, checklists. But here's the problem — not a single one of those photos gets analyzed by AI. An operator photographs a burst hydraulic hose, tags it red, and that data goes into a PDF. No computer vision. No defect detection. No fleet learning.*
>
> *Cat AI Assistant, launched this year, is conversational — it answers text questions. But it doesn't look at photos. There's a gap.*
>
> *We built what fills it."*

### [0:20–0:35] THE PRODUCT — First Impression (15 sec)

**Show**: App loading on screen. Set CAT 336, 3,200 hours. The PM chip appears.

> *"Cat Inspect AI. Upload a photo, speak a voice note, and the system thinks — not just about what it sees, but about what it remembers.*
>
> *Equipment model, service hours, fleet history — all context for the analysis."*

### [0:35–1:15] DEMO 1 — Hydraulic Hose (40 sec)

**Show**: Upload the dirty/oily hose image. Wait for result. Scroll through.

> *"Dirty hydraulic hoses. The AI sees oil saturation — not just dust. Rates it YELLOW. But watch what happens next.*
>
> *[scroll to escalation]* *Escalation risk — the fleet data shows that on a CAT 320, an identical oil weep escalated to a catastrophic burst in 300 hours. $2,800 repair plus $1,200 environmental cleanup.*
>
> *[scroll to similar findings]* *Similar findings — four matching fleet records. GREEN to YELLOW to RED. You're looking at the progression before it happens. That's not AI guessing — that's institutional memory from 56 previous inspections."*

### [1:15–2:10] DEMO 2 — Fire Extinguisher ⭐ THE SHOWSTOPPER (55 sec)

**Show**: Upload expired fire extinguisher. Wait for result. This is the money shot.

> *"Now I upload an expired fire extinguisher.*
>
> *[result loads]* *RED. Severity 9 out of 10. But look at what fires —*
>
> *[scroll to compliance]* *Regulatory compliance. 30 CFR 56.4230 — that's the exact MSHA regulation requiring fire extinguishers on self-propelled equipment. This citation isn't hallucinated — it's pulled live from the Electronic Code of Federal Regulations database, stored in our Actian VectorAI DB.*
>
> *[scroll to similar findings]* *Fleet history. Five fire extinguisher records across the fleet — expired tags, missing units, borderline gauges. An escalation timeline from GREEN to RED.*
>
> *A $150 extinguisher on a half-million-dollar machine. One engine fire and it's total loss. The system caught it, cited the regulation, and told you what happened on every other machine in the fleet.*
>
> *Every inspection makes the next one smarter."*

### [2:10–2:35] TECHNICAL DEPTH — Architecture (25 sec)

**Show**: Quick flip through the Parts tab (snap a photo → identify part), OCR tab (read nameplate), and the Report tab.

> *"Beyond inspection, the same AI identifies parts from photos with part numbers and fitment, reads serial plates via OCR, and generates dealer-quality reports with one click.*
>
> *The stack: Gemini 2.0 Flash for vision and reasoning. CLIP embeddings for visual similarity. Actian VectorAI DB for fleet memory — hybrid search fusing image and text embeddings. Real MSHA regulations from eCFR. Cat SIS equipment specs. FastAPI backend, voice-to-text input, and a single-page responsive frontend."*

### [2:35–2:55] IMPACT — Why It Matters (20 sec)

> *"Cat equipment runs in mining, construction, infrastructure — industries where a missed inspection kills people. Contaminated hydraulic fluid causes 75% of hydraulic failures. Proactive maintenance cuts ownership costs by 25% and extends equipment life by 30 to 60%.*
>
> *This isn't a wrapper around an API. It's institutional memory. The 56th inspection knows what the 1st one didn't. That's the future Cat asked us to build."*

### [2:55–3:00] CLOSE

> *"Cat Inspect AI. If Cat Inspect and Cat AI Assistant had a baby."*

---

## 5-Minute In-Person Showcase Strategy

You have 3 min to present + 2 min Q&A. Here's how to structure it:

### Minute 1: Problem + Positioning (talking, no demo)

> "Cat Inspect captures 6 million inspections a year, but zero AI analysis on photos. Cat AI Assistant answers text questions but doesn't look at images. We built the bridge — multimodal AI inspection with fleet memory."

**Key phrase for judges**: *"If Cat Inspect and Cat AI Assistant had a baby."* — this is literally the track description. Use it.

### Minute 2: Live Demo — Fire Extinguisher

Upload the expired fire extinguisher live. While it's thinking, explain:
- "Gemini looks at the image, identifies the component, rates severity"
- "Actian VectorAI searches fleet history for similar findings"
- "eCFR regulations are matched to the detected component"

When the result loads, scroll to:
1. RED rating + compliance citation
2. Similar findings with fleet progression
3. Cost estimate

### Minute 3: Technical Depth + Impact

Show the architecture briefly:
- "Hybrid search: CLIP image embeddings + text embeddings, fused in VectorAI"
- "Not a ChatGPT wrapper — we built the pipeline: embed → search → context assembly → Gemini → store"
- "Every inspection gets vectorized and stored — the system literally learns"

Close with: *"A $150 fix on a $500K machine. The AI catches it, cites the regulation, and shows fleet history. That's the future of field operations."*

### Q&A Preparation — Likely Questions

| Question | Answer |
|----------|--------|
| "How is this different from just uploading to ChatGPT?" | "ChatGPT has no fleet memory. It can't say 'the last CAT 320 with this issue burst in 300 hours.' We have a vector database of fleet history that grows with every inspection. Plus real regulatory citations from eCFR, not hallucinations." |
| "How do you handle false positives?" | "Conservative by design — the prompt explicitly tells Gemini 'dust is dust, not a seal failure.' We tested with clean hoses and it rates them GREEN. It doesn't cry wolf." |
| "What's the VectorAI DB doing?" | "Two things: fleet memory (hybrid search over 56+ inspection embeddings) and regulatory storage (eCFR regulations). When you upload a fire extinguisher, it searches both — finding similar fleet cases AND the exact MSHA citation." |
| "Could this work with real Cat data?" | "The architecture is data-agnostic — real Cat Inspect records would drop right in. The vector embeddings work the same whether it's our seed data or 6 million real inspections." |
| "What AI tools did you use?" | "Gemini 2.0 Flash for vision analysis, reasoning, OCR, parts ID, and report generation. OpenAI CLIP for image/text embeddings. Actian VectorAI DB for vector storage and hybrid search. All API-based, no fine-tuning." |

---

## README Section — AI Disclosure (Required)

Add this to your README:

```markdown
## AI Tools Used

- **Google Gemini 2.0 Flash**: Vision analysis, component detection, severity rating, OCR,
  parts identification, report generation. Used via API with custom system prompts.
- **OpenAI CLIP (ViT-B/32)**: Image and text embedding generation for hybrid vector search.
- **Actian VectorAI DB**: Vector database for storing and searching fleet inspection history
  and regulatory text using cosine similarity.
- **Gemini** was also used for code assistance during development.

### What AI Generated vs. What We Built
- **We built**: The inspection pipeline (embed → hybrid search → context assembly → prompt
  construction → result parsing → storage), the VectorAI integration layer, the regulatory
  retrieval system (eCFR), the equipment specs cross-referencing, the escalation detection
  algorithm, the fleet comparison logic, the frontend UI, and the complete seed dataset.
- **AI generated**: The visual analysis text, severity ratings, and cost estimates within each
  inspection result are generated by Gemini based on our structured prompt with fleet context.
```

---

## Devpost Submission Checklist

- [ ] Public GitHub repo linked
- [ ] Video demo ≤ 3 minutes (unlisted YouTube)
- [ ] Description addresses Caterpillar track challenge
- [ ] All team members added to Devpost submission
- [ ] AI disclosure in README
- [ ] Submit to tracks: Caterpillar, Best Use of Actian VectorAI DB, Best Use of Gemini API
- [ ] Optional: Best UI/UX, Most Creative
