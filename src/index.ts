import 'dotenv/config';
import express from 'express';
import fs from 'fs';
import path from 'path';
import cors from 'cors';
import Groq from 'groq-sdk';
import { pipeline } from "@xenova/transformers";
import mongoose from 'mongoose';

const app = express();
app.use(cors({ origin: "*" }));

app.use(express.json());

// -------------------- CONFIG --------------------
const GROQ_KEY = process.env.GROQ_API_KEY;
const MONGO_URI = process.env.MONGO_URI;
const PORT = process.env.PORT || 8080;

if (!GROQ_KEY) {
  console.error('❌ Missing GROQ_API_KEY in .env');
  process.exit(1);
}
if (!MONGO_URI) {
  console.error('❌ Missing MONGO_URI in .env');
  process.exit(1);
}

const client = new Groq({ apiKey: GROQ_KEY });

// -------------------- MONGOOSE MODEL --------------------
mongoose.connect(MONGO_URI, { dbName: 'clause-sense' })
  .then(() => console.log('✅ Connected to MongoDB Atlas'))
  .catch(err => {
    console.error('❌ MongoDB connection error:', err);
    process.exit(1);
  });

const QueryLogSchema = new mongoose.Schema({
  query: String,
  parsed_query: Object,
  decision: Object,
  clauses_used: Array,
  created_at: { type: Date, default: Date.now }
});

const QueryLog = mongoose.model('QueryLog', QueryLogSchema);

// -------------------- LOAD EMBEDDINGS --------------------
const EMB_FILE = path.join(__dirname, '..', 'data', 'embeddings.json');
let embeddings: any[] = [];
if (fs.existsSync(EMB_FILE)) {
  try {
    embeddings = JSON.parse(fs.readFileSync(EMB_FILE, 'utf-8'));
    console.log(`✅ Loaded ${embeddings.length} embeddings`);
  } catch {
    console.warn('⚠ Could not parse embeddings.json — re-run indexer.');
    embeddings = [];
  }
} else {
  console.warn('⚠ Embeddings not found. Run: npm run index-pdfs');
}

// -------------------- EMBEDDING MODEL --------------------
let embedder: any = null;
async function loadEmbedder() {
  if (!embedder) {
    console.log('⏳ Loading local embedding model...');
    embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    console.log('✅ Embedding model ready.');
  }
  return embedder;
}
async function embedText(text: string): Promise<number[]> {
  const model = await loadEmbedder();
  const out = await model(text, { pooling: 'mean', normalize: true });
  return Array.from(out.data) as number[];
}
function cosine(a: number[], b: number[]) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] ** 2;
    nb += b[i] ** 2;
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-10);
}

const MIN_SIMILARITY = 0.25;
const TOP_K = 6;

// -------------------- SEARCH ROUTE --------------------
app.post('/api/search', async (req, res) => {
  try {
    const { query } = req.body;
    if (!query || typeof query !== 'string') {
      return res.status(400).json({ error: 'query (string) is required' });
    }

    // --- 1) Parse query ---
    const parsePrompt = `
You are a strict JSON extractor.
Given an insurance claim or policy query, extract:
- age (number) or null
- gender ("male"/"female"/null)
- procedure (string or null)
- location (city or null)
- policy_duration (string or null)
Respond ONLY with valid JSON.
Example:
{"age":46,"gender":"male","procedure":"knee surgery","location":"Pune","policy_duration":"3 months"}
Now parse exactly:
"""${query}"""
`;
    const parseResp = await client.chat.completions.create({
      model: 'llama3-8b-8192',
      messages: [{ role: 'user', content: parsePrompt }],
      temperature: 0,
      max_tokens: 200
    });
    let parsedQuery: any = {};
    try {
      let txt = parseResp.choices[0].message.content || '';
      txt = txt.replace(/```json/g, '').replace(/```/g, '').trim();
      parsedQuery = JSON.parse(txt);
    } catch {
      parsedQuery = {};
    }

    // --- 2) Embedding search ---
    const qEmb = await embedText(query);
    const scored = embeddings.map((e: any) => ({ score: cosine(qEmb, e.embedding), item: e }))
      .sort((a, b) => b.score - a.score);

    let usedClauses = scored
      .filter(s => s.score >= MIN_SIMILARITY)
      .slice(0, TOP_K)
      .map(s => ({
        dataset: s.item.dataset,
        clause_ref: s.item.id,
        excerpt: s.item.text,
        score: s.score
      }))
      .filter(c => c.excerpt && c.excerpt.trim().length > 10);

    if (usedClauses.length === 0) {
      usedClauses = scored.slice(0, TOP_K).map(s => ({
        dataset: s.item.dataset,
        clause_ref: s.item.id,
        excerpt: s.item.text,
        score: s.score
      }));
    }

    // --- 3) Decision making ---
    const clausesText = usedClauses
      .map((c, i) => `CLAUSE_${i + 1} (${c.dataset} / ${c.clause_ref}):\n${c.excerpt}`)
      .join('\n');

    const decisionPrompt = `
You are an insurance adjudicator assistant.
You MUST return ONLY valid JSON in this format:
{
  "parsed_query": { ... },
  "decision": {
    "status": "Approved"|"Rejected"|"Pending",
    "amount": "<amount or null>",
    "justification": "At least 3–5 sentences explaining reasoning, referencing clauses by number."
  },
  "clauses_used": [
    {"dataset":"...","clause_ref":"...","excerpt":"..."},
    ...
  ]
}
Parsed query:
${JSON.stringify(parsedQuery, null, 2)}
Clauses:
${clausesText}
`;
    const decisionResp = await client.chat.completions.create({
      model: 'llama3-8b-8192',
      messages: [{ role: 'user', content: decisionPrompt }],
      temperature: 0,
      max_tokens: 800
    });

    let finalJson: any;
    try {
      let txt = decisionResp.choices[0].message.content || '';
      txt = txt.replace(/```json/g, '').replace(/```/g, '').trim();
      finalJson = JSON.parse(txt.slice(txt.indexOf('{')));
    } catch {
      finalJson = {
        parsed_query: parsedQuery,
        decision: { status: 'Pending', amount: null, justification: 'Could not parse LLM output' },
        clauses_used: usedClauses
      };
    }
    if (!finalJson.clauses_used || finalJson.clauses_used.length === 0) {
      finalJson.clauses_used = usedClauses;
    }

    // --- 4) Save to MongoDB ---
    await QueryLog.create({
      query,
      parsed_query: finalJson.parsed_query,
      decision: finalJson.decision,
      clauses_used: finalJson.clauses_used
    });

    // --- 5) Send response ---
    res.json(finalJson);

  } catch (err: any) {
    console.error('❌ Search error:', err);
    res.status(500).json({ error: err.message || String(err) });
  }
});

// -------------------- FRONTEND SERVE --------------------
const frontendPath = path.join(__dirname, '..', '..', 'frontend', 'project', 'dist');
if (fs.existsSync(frontendPath)) {
  app.use(express.static(frontendPath));
  app.get('*', (req, res) => {
    res.sendFile(path.join(frontendPath, 'index.html'));
  });
}
// -------------------- FULL CLAUSE ROUTE --------------------



app.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`));





