import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import Replicate from 'replicate';
import AWS from 'aws-sdk';
import { Readable } from 'stream';
import { v4 as uuidv4 } from 'uuid';

// Initialize DigitalOcean Spaces (S3-compatible)
const spaces = new AWS.S3({
  endpoint: new AWS.Endpoint(`${process.env.SPACES_REGION}.digitaloceanspaces.com`),
  accessKeyId: process.env.SPACES_KEY,
  secretAccessKey: process.env.SPACES_SECRET,
});

const app = express();
app.use(cors());
app.use(express.json({ limit: '200mb' }));
app.use(express.static('public'));

// Store images per account in memory
let accountImages = {};

/**
 * Upload a base64 data URI to DigitalOcean Spaces and return the public URL.
 */
async function uploadToSpaces(dataURI) {
  // strip off "data:image/...;base64,"
  const [, base64] = dataURI.split('base64,');
  const key = `${uuidv4()}.png`;

  await spaces
    .putObject({
      Bucket: process.env.SPACES_NAME,
      Key: key,
      Body: Buffer.from(base64, 'base64'),
      ACL: 'public-read',
      ContentType: 'image/png',
    })
    .promise();

  return `https://${process.env.SPACES_NAME}.${process.env.SPACES_REGION}.digitaloceanspaces.com/${key}`;
}

// =============== STABLE DIFFUSION ===============
async function generateStableDiffusion({ prompt, aspect_ratio, numImages }) {
  const replicate = new Replicate({ auth: process.env.REPLICATE_API_TOKEN });
  const modelIdentifier = 'stability-ai/stable-diffusion-3.5-medium';

  const tempOutputs = [];
  for (let i = 0; i < numImages; i++) {
    const input = { prompt, guidance_scale: 5, steps: 40, output_quality: 90 };
    if (aspect_ratio) input.aspect_ratio = aspect_ratio;
    const result = await replicate.run(modelIdentifier, { input });
    if (Array.isArray(result)) tempOutputs.push(...result);
    else tempOutputs.push(result);
  }

  // Convert streams or URLs into Spaces-hosted URLs
  const finalImages = [];
  for (const item of tempOutputs) {
    if (item && typeof item.getReader === 'function') {
      // ReadableStream => buffer => data URI
      const nodeStream = Readable.fromWeb(item);
      const chunks = [];
      for await (const chunk of nodeStream) chunks.push(chunk);
      const buffer = Buffer.concat(chunks);
      const dataURI = `data:image/png;base64,${buffer.toString('base64')}`;
      const publicUrl = await uploadToSpaces(dataURI);
      finalImages.push(publicUrl);
    } else if (typeof item === 'string') {
      // Replicate returned a URL directly
      finalImages.push(item);
    }
  }

  return finalImages;
}

// =============== IMAGE CAPTIONING (IMG2PROMPT) ===============
async function analyzeImageWithImg2Prompt(base64Image) {
  // Upload to Spaces first
  const imageUrl = await uploadToSpaces(base64Image);

  const replicate = new Replicate({ auth: process.env.REPLICATE_API_TOKEN });
  const modelId = 'methexis-inc/img2prompt:50adaf2d3ad20a6f911a8a9e3ccf777b263b8596fbd2c8fc26e8888f8a0edbb5';
  const output = await replicate.run(modelId, { input: { image: imageUrl } });

  if (!output) throw new Error('img2prompt returned no result');
  return Array.isArray(output) ? output[0] : output;
}

// =============== ENDPOINTS ===============

app.get('/all-images', (req, res) => {
  const { accountId } = req.query;
  if (!accountId) return res.status(400).json({ error: 'accountId is required' });
  res.json({ images: accountImages[accountId] || [] });
});

app.post('/generate-image', async (req, res) => {
  try {
    const { prompt, aspect_ratio, numImages, accountId } = req.body;
    if (!prompt || !accountId) return res.status(400).json({ error: 'prompt & accountId required' });

    const images = await generateStableDiffusion({ prompt, aspect_ratio, numImages: parseInt(numImages) || 1 });
    const newObjects = images.map(src => ({ id: uuidv4(), prompt, src, liked: false, aspectRatio: aspect_ratio || '', accountId }));

    accountImages[accountId] = accountImages[accountId] || [];
    accountImages[accountId].push(...newObjects);

    res.json({ images: newObjects });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

app.post('/like-image', (req, res) => {
  const { id, accountId } = req.body;
  if (!accountId) return res.status(400).json({ error: 'accountId is required' });
  const list = accountImages[accountId] || [];
  const img = list.find(i => i.id === id);
  if (!img) return res.status(404).json({ error: 'Image not found' });
  img.liked = !img.liked;
  res.json({ success: true, liked: img.liked });
});

app.post('/delete-image', (req, res) => {
  const { id, accountId } = req.body;
  if (!accountId || !accountImages[accountId]) return res.status(404).json({ error: 'No images for this account' });
  accountImages[accountId] = accountImages[accountId].filter(i => i.id !== id);
  res.json({ success: true });
});

app.post('/rotate-image', async (req, res) => {
  const { id, accountId } = req.body;
  if (!accountId || !accountImages[accountId]) return res.status(404).json({ error: 'No images for this account' });
  const oldImg = accountImages[accountId].find(i => i.id === id);
  if (!oldImg) return res.status(404).json({ error: 'Image not found' });

  try {
    const [newSrc] = await generateStableDiffusion({ prompt: oldImg.prompt, aspect_ratio: oldImg.aspectRatio, numImages: 1 });
    const newObj = { id: uuidv4(), prompt: oldImg.prompt, src: newSrc, liked: false, aspectRatio: oldImg.aspectRatio, accountId };
    accountImages[accountId].push(newObj);
    res.json({ success: true, newSrc });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

app.post('/image-to-prompt', async (req, res) => {
  try {
    const { imageBase64 } = req.body;
    if (!imageBase64) return res.status(400).json({ error: 'No image data provided' });
    const prompt = await analyzeImageWithImg2Prompt(imageBase64);
    res.json({ prompt });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server listening on port ${PORT}`));