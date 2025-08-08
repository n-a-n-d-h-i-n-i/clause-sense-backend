import 'dotenv/config';
import fs from 'fs';
import path from 'path';
import pdfParse from 'pdf-parse';
import { pipeline } from '@xenova/transformers';

/**
 * Indexer script:
 * - Reads all PDFs in backend/data/pdfs
 * - Splits into chunks
 * - Embeds each chunk locally (MiniLM)
 * - Saves to backend/data/embeddings.json
 */

const PDF_DIR = path.join(__dirname, '..', 'data', 'pdfs');
const OUT_FILE = path.join(__dirname, '..', 'data', 'embeddings.json');

// Load local embedding model
let embedder: any = null;
async function loadEmbedder() {
  if (!embedder) {
    console.log('Loading local embedding model (first call may be slow)...');
    embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    console.log('Embedding model ready.');
  }
  return embedder;
}

async function embedText(text: string): Promise<number[]> {
  const model = await loadEmbedder();
  const out = await model(text, { pooling: 'mean', normalize: true });
  return Array.from(out.data) as number[];
}

// Simple chunker
function chunkText(text: string, chunkSize = 500, overlap = 50): string[] {
  const words = text.split(/\s+/);
  const chunks: string[] = [];
  for (let i = 0; i < words.length; i += chunkSize - overlap) {
    chunks.push(words.slice(i, i + chunkSize).join(' '));
    if (i + chunkSize >= words.length) break;
  }
  return chunks;
}

async function indexPDFs() {
  const files = fs.readdirSync(PDF_DIR).filter(f => f.toLowerCase().endsWith('.pdf'));
  const allEmbeddings: any[] = [];

  for (const file of files) {
    const filePath = path.join(PDF_DIR, file);
    console.log(`Processing ${file}...`);
    const dataBuffer = fs.readFileSync(filePath);
    const pdfData = await pdfParse(dataBuffer);

    // Remove newlines for cleaner chunks
    const cleanText = pdfData.text.replace(/\n+/g, ' ').trim();
    const chunks = chunkText(cleanText, 500, 50);

    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      const embedding = await embedText(chunk);
      allEmbeddings.push({
        dataset: file,
        id: `${file}::${i + 1}`,
        text: chunk,
        embedding
      });
    }
  }

  fs.writeFileSync(OUT_FILE, JSON.stringify(allEmbeddings, null, 2));
  console.log(`Saved ${allEmbeddings.length} embeddings to ${OUT_FILE}`);
}

indexPDFs();



