import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';
import OpenAI from 'openai';
import { config, UPLOAD_DIR } from './config.js';
import { nowIso } from './date.js';
import { updateStore } from './store.js';
import type { RemoveMethod, OpenAIRequestRecord, UserRecord } from './types.js';

function createRequestRecord(input: Omit<OpenAIRequestRecord, 'createdAt'>): OpenAIRequestRecord {
  return {
    ...input,
    createdAt: nowIso()
  };
}

function promptFor(method: RemoveMethod) {
  const modeText: Record<RemoveMethod, string> = {
    screenshot: 'Automatically detect the marked watermark or overlay area.',
    doodle: 'Use the submitted user marking as guidance for the region to repair.',
    selection: 'Use the submitted rectangle or selected region as guidance for the area to repair.'
  };

  return [
    'Repair only the user-authorized image area that contains a watermark or unwanted overlay.',
    'Preserve the original composition, subject, lighting, texture, and image dimensions.',
    'Do not add logos, text, signatures, new objects, or stylistic changes.',
    modeText[method]
  ].join(' ');
}

async function saveBase64Image(dataUrl: string, filename: string) {
  const match = dataUrl.match(/^data:image\/[a-zA-Z0-9.+-]+;base64,(.+)$/);
  if (!match) return undefined;
  const outputPath = path.join(UPLOAD_DIR, filename);
  await fsp.writeFile(outputPath, Buffer.from(match[1], 'base64'));
  return outputPath;
}

export async function editImageWithOpenAI(input: {
  user: UserRecord;
  originalPath: string;
  method: RemoveMethod;
  markerDataUrl?: string;
}) {
  const requestId = `oai_${Date.now()}_${Math.random().toString(16).slice(2, 8)}`;

  if (!config.openaiApiKey || !config.enableOpenAIImageEdit) {
    const request = createRequestRecord({
      id: requestId,
      userId: input.user.id,
      openaiUserId: input.user.openaiUserId,
      endpoint: 'images.edit',
      model: config.openaiImageModel,
      status: 'skipped'
    });
    await updateStore((store) => {
      store.openaiRequests.push(request);
    });
    return { provider: 'local-preview' as const, outputPath: input.originalPath, openaiRequestId: request.id };
  }

  const client = new OpenAI({ apiKey: config.openaiApiKey });
  let markerPath: string | undefined;

  try {
    await fsp.mkdir(UPLOAD_DIR, { recursive: true });
    if (input.markerDataUrl) {
      markerPath = await saveBase64Image(input.markerDataUrl, `${requestId}_marker.png`);
    }

    const result = await client.images.edit({
      model: config.openaiImageModel,
      image: fs.createReadStream(input.originalPath) as unknown as File,
      prompt: promptFor(input.method),
      user: input.user.openaiUserId
    } as never);

    const base64 = result.data?.[0]?.b64_json;
    if (!base64) {
      throw new Error('OpenAI image edit did not return b64_json data.');
    }

    const outputPath = path.join(UPLOAD_DIR, `${requestId}_processed.png`);
    await fsp.writeFile(outputPath, Buffer.from(base64, 'base64'));

    const request = createRequestRecord({
      id: requestId,
      userId: input.user.id,
      openaiUserId: input.user.openaiUserId,
      endpoint: 'images.edit',
      model: config.openaiImageModel,
      status: 'completed'
    });
    await updateStore((store) => {
      store.openaiRequests.push(request);
    });

    return { provider: 'openai' as const, outputPath, openaiRequestId: request.id };
  } catch (error) {
    const request = createRequestRecord({
      id: requestId,
      userId: input.user.id,
      openaiUserId: input.user.openaiUserId,
      endpoint: 'images.edit',
      model: config.openaiImageModel,
      status: 'failed',
      error: error instanceof Error ? error.message : 'Unknown OpenAI error'
    });
    await updateStore((store) => {
      store.openaiRequests.push(request);
    });
    throw error;
  } finally {
    if (markerPath) {
      await fsp.rm(markerPath, { force: true });
    }
  }
}
