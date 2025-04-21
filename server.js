// server.js
import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import Replicate from 'replicate';
import AWS from 'aws-sdk';
import { Readable } from 'stream';
import { v4 as uuidv4 } from 'uuid';

const app = express();
app.use(cors());
app.use(express.json({ limit: '200mb' }));
app.use(express.static('public'));

// In-memory store for img2prompt jobs
// { [jobId]: { status: 'pending'|'complete'|'error', prompt?: string, error?: string } }
const promptJobs = {};

// =============== DIGITALOCEAN SPACES ===============
const spaces = new AWS.S3({
  endpoint: new AWS.Endpoint(`${process.env.SPACES_REGION}.digitaloceanspaces.com`),
  accessKeyId: process.env.SPACES_KEY,
  secretAccessKey: process.env.SPACES_SECRET,
});

/** Uploads a base64 data URI to Spaces and returns the public CDN URL. */
async function uploadToSpaces(dataURI) {
  const [, base64] = dataURI.split('base64,');
  const key = `${uuidv4()}.png`;

  await spaces.putObject({
    Bucket: process.env.SPACES_NAME,
    Key: key,
    Body: Buffer.from(base64, 'base64'),
    ACL: 'public-read',
    ContentType: 'image/png',
  }).promise();

  return `https://${process.env.SPACES_NAME}.${process.env.SPACES_REGION}.digitaloceanspaces.com/${key}`;
}

// =============== STABLE DIFFUSION ===============
async function generateStableDiffusion({ prompt, aspect_ratio, numImages }) {
  const replicate = new Replicate({ auth: process.env.REPLICATE_API_TOKEN });
  const modelId = 'stability-ai/stable-diffusion-3.5-medium';

  // 1) run the model
  const outs = [];
  for (let i = 0; i < numImages; i++) {
    const input = { prompt, guidance_scale: 5, steps: 40, output_quality: 90 };
    if (aspect_ratio) input.aspect_ratio = aspect_ratio;
    const result = await replicate.run(modelId, { input });
    outs.push(...(Array.isArray(result) ? result : [result]));
  }

  // 2) convert each to a public Spaces URL
  const finalUrls = [];
  for (const item of outs) {
    if (item && typeof item.getReader === 'function') {
      // stream → buffer → dataURI
      const nodeStream = Readable.fromWeb(item);
      const chunks = [];
      for await (const c of nodeStream) chunks.push(c);
      const buffer = Buffer.concat(chunks);
      const dataURI = `data:image/png;base64,${buffer.toString('base64')}`;
      finalUrls.push(await uploadToSpaces(dataURI));
    } else if (typeof item === 'string') {
      finalUrls.push(item);
    }
  }
  return finalUrls;
}

// =============== IMG2PROMPT (ASYNC) ===============
async function analyzeImageWithImg2Prompt(dataURI) {
  // upload the image to Spaces
  const imageUrl = await uploadToSpaces(dataURI);

  // call the Replicate img2prompt model
  const replicate = new Replicate({ auth: process.env.REPLICATE_API_TOKEN });
  const modelId =
    'methexis-inc/img2prompt:50adaf2d3ad20a6f911a8a9e3ccf777b263b8596fbd2c8fc26e8888f8a0edbb5';
  const output = await replicate.run(modelId, { input: { image: imageUrl } });

  if (!output) throw new Error('img2prompt returned no result');
  return Array.isArray(output) ? output[0] : output;
}

// =============== IN-MEMORY IMAGE STORE ===============
const accountImages = {}; // { [accountId]: [ { id, prompt, src, liked, aspectRatio } ] }

// =============== ROUTES ===============

// List all saved images for an account
app.get('/all-images', (req, res) => {
  const { accountId } = req.query;
  if (!accountId) return res.status(400).json({ error: 'accountId required' });
  res.json({ images: accountImages[accountId] || [] });
});

// Generate new SD images
app.post('/generate-image', async (req, res) => {
  try {
    const { prompt, aspect_ratio, numImages, accountId } = req.body;
    if (!prompt || !accountId) return res.status(400).json({ error: 'prompt & accountId required' });

    const urls = await generateStableDiffusion({
      prompt,
      aspect_ratio,
      numImages: parseInt(numImages) || 1
    });

    const objs = urls.map(src => ({
      id: uuidv4(), prompt, src, liked: false, aspectRatio: aspect_ratio || '', accountId
    }));
    accountImages[accountId] = accountImages[accountId] || [];
    accountImages[accountId].push(...objs);

    res.json({ images: objs });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

// Toggle like
app.post('/like-image', (req, res) => {
  const { id, accountId } = req.body;
  if (!accountId) return res.status(400).json({ error: 'accountId required' });
  const list = accountImages[accountId] || [];
  const img = list.find(i => i.id === id);
  if (!img) return res.status(404).json({ error: 'Not found' });
  img.liked = !img.liked;
  res.json({ success: true, liked: img.liked });
});

// Delete
app.post('/delete-image', (req, res) => {
  const { id, accountId } = req.body;
  if (!accountId || !accountImages[accountId]) return res.status(404).json({ error: 'Not found' });
  accountImages[accountId] = accountImages[accountId].filter(i => i.id !== id);
  res.json({ success: true });
});

// Rotate (new SD call)
app.post('/rotate-image', async (req, res) => {
  const { id, accountId } = req.body;
  if (!accountId || !accountImages[accountId]) return res.status(404).json({ error: 'Not found' });
  const oldImg = accountImages[accountId].find(i => i.id === id);
  if (!oldImg) return res.status(404).json({ error: 'Not found' });

  try {
    const [newUrl] = await generateStableDiffusion({
      prompt: oldImg.prompt,
      aspect_ratio: oldImg.aspectRatio,
      numImages: 1
    });
    const obj = {
      id: uuidv4(),
      prompt: oldImg.prompt,
      src: newUrl,
      liked: false,
      aspectRatio: oldImg.aspectRatio,
      accountId
    };
    accountImages[accountId].push(obj);
    res.json({ success: true, newImage: obj });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

// ===== Async image-to-prompt =====

// Enqueue prompt-generation job
app.post('/image-to-prompt', (req, res) => {
  const { imageBase64 } = req.body;
  if (!imageBase64) return res.status(400).json({ error: 'No image provided' });

  const jobId = uuidv4();
  promptJobs[jobId] = { status: 'pending' };

  // Kick off background work:
  (async () => {
    try {
      const prompt = await analyzeImageWithImg2Prompt(imageBase64);
      promptJobs[jobId] = { status: 'complete', prompt };
    } catch (err) {
      console.error('img2prompt error:', err);
      promptJobs[jobId] = { status: 'error', error: err.message };
    }
  })();

  res.status(202).json({ jobId });
});

// Poll for a prompt result
app.get('/prompt-result', (req, res) => {
  const { jobId } = req.query;
  if (!jobId || !promptJobs[jobId]) {
    return res.status(404).json({ error: 'Unknown jobId' });
  }
  res.json(promptJobs[jobId]);
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server listening on port ${PORT}`));
