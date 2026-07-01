import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import cors from 'cors';
import express from 'express';
import multer from 'multer';
import { z } from 'zod';
import { exchangeWechatCode, signAuthToken } from './auth.js';
import { config, UPLOAD_DIR } from './config.js';
import { formatFileSize, nowIso } from './date.js';
import { editImageWithOpenAI } from './openaiImage.js';
import { ensureDataDirs, readStore, updateStore } from './store.js';
import { addRewardedBonus, getUsage, getUser, identifyUser, listUserHistory } from './users.js';
import type { HistoryRecord, ImageHistoryRecord, Md5HistoryRecord, RemoveMethod } from './types.js';

const removeMethods = ['screenshot', 'doodle', 'selection'] as const;
const upload = multer({
  dest: UPLOAD_DIR,
  limits: { fileSize: 40 * 1024 * 1024 }
});

function makeRecordId(prefix: string) {
  return `${prefix}_${crypto.randomUUID().replaceAll('-', '').slice(0, 18)}`;
}

function publicUrlFor(filePath: string) {
  return `${config.publicBaseUrl}/uploads/${path.basename(filePath)}`;
}

function requireFile(file?: Express.Multer.File) {
  if (!file) {
    const error = new Error('FILE_REQUIRED');
    (error as Error & { status?: number }).status = 400;
    throw error;
  }
  return file;
}

function toProcessedFile(record: ImageHistoryRecord) {
  return {
    id: record.id,
    type: 'image' as const,
    originalName: record.originalName,
    thumb: record.originalUrl,
    processedUrl: record.processedUrl,
    processTime: new Date(record.createdAt).toLocaleString('zh-CN', { hour12: false }),
    fileSize: formatFileSize(record.fileSize),
    method: record.method,
    provider: record.provider,
    openaiRequestId: record.openaiRequestId
  };
}

function toMd5Result(record: Md5HistoryRecord) {
  return {
    id: record.id,
    fileName: record.fileName,
    fileSize: formatFileSize(record.fileSize),
    md5: record.md5,
    calcTime: new Date(record.createdAt).toLocaleString('zh-CN', { hour12: false }),
    duplicate: record.duplicate
  };
}

function toHistoryItem(record: HistoryRecord) {
  return record.type === 'image' ? toProcessedFile(record) : toMd5Result(record);
}

export async function createApp() {
  await ensureDataDirs();
  await fs.promises.mkdir(UPLOAD_DIR, { recursive: true });

  const app = express();
  app.use(cors({ origin: true, credentials: true }));
  app.use(express.json({ limit: '2mb' }));
  app.use('/uploads', express.static(UPLOAD_DIR));

  app.get('/health', (_req, res) => {
    res.json({ ok: true, service: 'liquidity-portrait-backend', time: nowIso() });
  });

  app.post('/api/auth/wechat/login', async (req, res, next) => {
    try {
      const body = z
        .object({
          code: z.string().min(1),
          clientId: z.string().min(8),
          platform: z.string().optional(),
          userInfo: z
            .object({
              nickName: z.string().optional(),
              nickname: z.string().optional(),
              avatarUrl: z.string().optional()
            })
            .optional()
        })
        .parse(req.body);

      const wechatSession = await exchangeWechatCode(body.code);
      const user = await identifyUser({
        clientId: body.clientId,
        platform: body.platform ?? 'weapp',
        nickname: body.userInfo?.nickName ?? body.userInfo?.nickname,
        avatarUrl: body.userInfo?.avatarUrl,
        openid: wechatSession.openid,
        unionid: wechatSession.unionid
      });
      const token = signAuthToken({
        userId: user.id,
        openid: wechatSession.openid,
        platform: user.platform
      });

      res.json({
        token,
        user: {
          id: user.id,
          platform: user.platform,
          nickname: user.nickname,
          avatarUrl: user.avatarUrl,
          openid: user.openid,
          unionid: user.unionid,
          openaiUserId: user.openaiUserId,
          createdAt: user.createdAt,
          lastSeenAt: user.lastSeenAt
        },
        usage: await getUsage(user.id)
      });
    } catch (error) {
      next(error);
    }
  });

  app.post('/api/users/identify', async (req, res, next) => {
    try {
      const body = z
        .object({
          clientId: z.string().min(8),
          platform: z.string().optional(),
          nickname: z.string().optional(),
          avatarUrl: z.string().optional(),
          openid: z.string().optional()
        })
        .parse(req.body);
      const user = await identifyUser(body);
      const usage = await getUsage(user.id);
      res.json({
        user: {
          id: user.id,
          platform: user.platform,
          nickname: user.nickname,
          avatarUrl: user.avatarUrl,
          openid: user.openid,
          openaiUserId: user.openaiUserId,
          createdAt: user.createdAt,
          lastSeenAt: user.lastSeenAt
        },
        usage
      });
    } catch (error) {
      next(error);
    }
  });

  app.post('/api/photo/usage-records', async (req, res, next) => {
    try {
      const body = z
        .object({
          id: z.string().min(1),
          userId: z.string().optional(),
          openid: z.string().optional(),
          sizeId: z.string().min(1),
          sizeName: z.string().min(1),
          imagePath: z.string().min(1),
          createdAt: z.string().min(1),
          status: z.literal('completed')
        })
        .parse(req.body);

      if (!body.userId && !body.openid) {
        return res.status(400).json({ error: 'USER_ID_OR_OPENID_REQUIRED' });
      }

      await updateStore((store) => {
        const index = store.photoUsageRecords.findIndex((item) => item.id === body.id);
        if (index >= 0) {
          store.photoUsageRecords[index] = body;
          return;
        }
        store.photoUsageRecords.push(body);
      });

      res.json({ ok: true, record: body });
    } catch (error) {
      next(error);
    }
  });

  app.get('/api/users/:userId/usage', async (req, res, next) => {
    try {
      const user = await getUser(req.params.userId);
      if (!user) return res.status(404).json({ error: 'USER_NOT_FOUND' });
      res.json({ usage: await getUsage(user.id) });
    } catch (error) {
      next(error);
    }
  });

  app.get('/api/users/:userId/history', async (req, res, next) => {
    try {
      const type = req.query.type === 'image' || req.query.type === 'md5' ? req.query.type : undefined;
      const records = await listUserHistory(req.params.userId, type);
      res.json({ records: records.map(toHistoryItem) });
    } catch (error) {
      next(error);
    }
  });

  app.post('/api/ads/reward', async (req, res, next) => {
    try {
      const body = z
        .object({
          userId: z.string().min(1),
          placement: z.enum(['quota', 'download']).default('quota')
        })
        .parse(req.body);
      const usage = body.placement === 'quota' ? await addRewardedBonus(body.userId) : await getUsage(body.userId);
      res.json({ ok: true, usage });
    } catch (error) {
      next(error);
    }
  });

  app.post('/api/process/image', upload.single('image'), async (req, res, next) => {
    try {
      const file = requireFile(req.file);
      const body = z
        .object({
          userId: z.string().min(1),
          method: z.enum(removeMethods).default('screenshot'),
          markerDataUrl: z.string().optional(),
          rightsConfirmed: z.coerce.boolean().default(false)
        })
        .parse(req.body);

      if (!body.rightsConfirmed) {
        return res.status(400).json({ error: 'RIGHTS_CONFIRMATION_REQUIRED' });
      }

      const user = await getUser(body.userId);
      if (!user) return res.status(404).json({ error: 'USER_NOT_FOUND' });

      const usage = await getUsage(user.id);
      if (usage.remaining <= 0) {
        return res.status(429).json({ error: 'QUOTA_EXHAUSTED', usage });
      }

      const ext = path.extname(file.originalname) || '.jpg';
      const originalPath = path.join(UPLOAD_DIR, `${file.filename}${ext}`);
      await fs.promises.rename(file.path, originalPath);

      const edited = await editImageWithOpenAI({
        user,
        originalPath,
        method: body.method as RemoveMethod,
        markerDataUrl: body.markerDataUrl
      });

      const record: ImageHistoryRecord = {
        id: makeRecordId('pf'),
        type: 'image',
        userId: user.id,
        originalName: file.originalname,
        originalUrl: publicUrlFor(originalPath),
        processedUrl: publicUrlFor(edited.outputPath),
        fileSize: file.size,
        method: body.method,
        status: 'completed',
        provider: edited.provider,
        openaiRequestId: edited.openaiRequestId,
        createdAt: nowIso()
      };

      await updateStore((store) => {
        store.history.push(record);
      });

      res.json({ file: toProcessedFile(record), usage: await getUsage(user.id) });
    } catch (error) {
      next(error);
    }
  });

  app.post('/api/tools/md5', upload.single('file'), async (req, res, next) => {
    try {
      const file = requireFile(req.file);
      const body = z.object({ userId: z.string().min(1) }).parse(req.body);
      const user = await getUser(body.userId);
      if (!user) return res.status(404).json({ error: 'USER_NOT_FOUND' });

      const hash = crypto.createHash('md5');
      await new Promise<void>((resolve, reject) => {
        const stream = fs.createReadStream(file.path);
        stream.on('data', (chunk) => hash.update(chunk));
        stream.on('error', reject);
        stream.on('end', resolve);
      });
      const md5 = hash.digest('hex');
      const store = await readStore();
      const duplicate = store.history.some((item) => item.type === 'md5' && item.md5 === md5);

      const record: Md5HistoryRecord = {
        id: makeRecordId('md5'),
        type: 'md5',
        userId: user.id,
        fileName: file.originalname,
        fileSize: file.size,
        md5,
        duplicate,
        createdAt: nowIso()
      };

      await updateStore((nextStore) => {
        nextStore.history.push(record);
      });
      await fs.promises.rm(file.path, { force: true });

      res.json({ result: toMd5Result(record) });
    } catch (error) {
      next(error);
    }
  });

  app.use((error: unknown, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
    const status = (error as Error & { status?: number }).status ?? (error instanceof z.ZodError ? 400 : 500);
    const message = error instanceof Error ? error.message : 'INTERNAL_SERVER_ERROR';
    res.status(status).json({ error: message });
  });

  return app;
}
