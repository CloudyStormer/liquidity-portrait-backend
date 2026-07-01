import path from 'node:path';
import { fileURLToPath } from 'node:url';
import dotenv from 'dotenv';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
export const SRC_DIR = path.dirname(__filename);
export const ROOT_DIR = path.resolve(SRC_DIR, '..');
export const DATA_DIR = path.join(ROOT_DIR, 'data');
export const UPLOAD_DIR = path.join(ROOT_DIR, 'uploads');

function numberFromEnv(name: string, fallback: number) {
  const value = Number(process.env[name]);
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

export const config = {
  port: numberFromEnv('PORT', 8787),
  publicBaseUrl: process.env.PUBLIC_BASE_URL ?? 'http://localhost:8787',
  frontendOrigin: process.env.FRONTEND_ORIGIN ?? 'http://localhost:5173',
  authTokenSecret: process.env.AUTH_TOKEN_SECRET ?? 'liquidity-portrait-dev-secret',
  authTokenTtlSeconds: numberFromEnv('AUTH_TOKEN_TTL_SECONDS', 7 * 24 * 60 * 60),
  wechatAppId: process.env.WECHAT_APP_ID ?? '',
  wechatAppSecret: process.env.WECHAT_APP_SECRET ?? '',
  wechatDevOpenid: process.env.WECHAT_DEV_OPENID ?? '',
  wechatDevUnionid: process.env.WECHAT_DEV_UNIONID ?? '',
  freeDailyQuota: numberFromEnv('FREE_DAILY_QUOTA', 3),
  rewardedAdBonus: numberFromEnv('REWARDED_AD_BONUS', 3),
  openaiApiKey: process.env.OPENAI_API_KEY ?? '',
  openaiImageModel: process.env.OPENAI_IMAGE_MODEL ?? 'gpt-image-1',
  enableOpenAIImageEdit: process.env.ENABLE_OPENAI_IMAGE_EDIT === 'true'
};
