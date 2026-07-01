import crypto from 'node:crypto';
import { config } from './config.js';

interface WechatCodeSession {
  openid?: string;
  session_key?: string;
  unionid?: string;
  errcode?: number;
  errmsg?: string;
}

export interface AuthTokenPayload {
  userId: string;
  openid: string;
  platform: string;
  exp: number;
}

function makeHttpError(message: string, status: number) {
  const error = new Error(message);
  (error as Error & { status?: number }).status = status;
  return error;
}

function base64Url(input: string | Buffer) {
  return Buffer.from(input).toString('base64url');
}

export function signAuthToken(input: Omit<AuthTokenPayload, 'exp'>) {
  const payload: AuthTokenPayload = {
    ...input,
    exp: Math.floor(Date.now() / 1000) + config.authTokenTtlSeconds
  };
  const body = base64Url(JSON.stringify(payload));
  const signature = crypto.createHmac('sha256', config.authTokenSecret).update(body).digest('base64url');
  return `${body}.${signature}`;
}

export function verifyAuthToken(token: string): AuthTokenPayload {
  const [body, signature] = token.split('.');
  if (!body || !signature) {
    throw makeHttpError('INVALID_AUTH_TOKEN', 401);
  }

  const expected = crypto.createHmac('sha256', config.authTokenSecret).update(body).digest('base64url');
  if (!crypto.timingSafeEqual(Buffer.from(signature), Buffer.from(expected))) {
    throw makeHttpError('INVALID_AUTH_TOKEN', 401);
  }

  const payload = JSON.parse(Buffer.from(body, 'base64url').toString('utf8')) as AuthTokenPayload;
  if (!payload.userId || !payload.openid || payload.exp < Math.floor(Date.now() / 1000)) {
    throw makeHttpError('INVALID_AUTH_TOKEN', 401);
  }

  return payload;
}

export async function exchangeWechatCode(code: string) {
  if (!config.wechatAppId || !config.wechatAppSecret) {
    if (config.wechatDevOpenid) {
      return {
        openid: config.wechatDevOpenid,
        unionid: config.wechatDevUnionid || undefined
      };
    }

    throw makeHttpError('WECHAT_CONFIG_MISSING', 500);
  }

  const params = new URLSearchParams({
    appid: config.wechatAppId,
    secret: config.wechatAppSecret,
    js_code: code,
    grant_type: 'authorization_code'
  });

  const response = await fetch(`https://api.weixin.qq.com/sns/jscode2session?${params.toString()}`, {
    method: 'GET'
  });

  if (!response.ok) {
    throw makeHttpError('WECHAT_CODE_EXCHANGE_FAILED', 502);
  }

  const data = (await response.json()) as WechatCodeSession;
  if (data.errcode || !data.openid) {
    const error = makeHttpError(data.errmsg || 'WECHAT_CODE_EXCHANGE_FAILED', 401);
    (error as Error & { status?: number; wechatErrcode?: number }).wechatErrcode = data.errcode;
    throw error;
  }

  return {
    openid: data.openid,
    unionid: data.unionid
  };
}
