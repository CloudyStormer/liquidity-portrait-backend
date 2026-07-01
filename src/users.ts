import crypto from 'node:crypto';
import { config } from './config.js';
import { nowIso, todayKey } from './date.js';
import { readStore, updateStore } from './store.js';
import type { HistoryRecord, UserRecord } from './types.js';

function makeId(prefix: string) {
  return `${prefix}_${crypto.randomUUID().replaceAll('-', '').slice(0, 18)}`;
}

function makeOpenAIUserId(userId: string) {
  return `lp_${crypto.createHash('sha256').update(userId).digest('hex').slice(0, 32)}`;
}

export async function identifyUser(input: {
  clientId: string;
  platform?: string;
  nickname?: string;
  avatarUrl?: string;
  openid?: string;
  unionid?: string;
}) {
  const now = nowIso();
  let user: UserRecord | undefined;

  await updateStore((store) => {
    user = input.openid
      ? store.users.find((item) => item.openid === input.openid)
      : store.users.find((item) => item.clientId === input.clientId);

    if (!user) {
      user = store.users.find((item) => item.clientId === input.clientId);
    }

    if (!user) {
      const id = makeId('usr');
      user = {
        id,
        clientId: input.clientId,
        platform: input.platform ?? 'web',
        nickname: input.nickname,
        avatarUrl: input.avatarUrl,
        openid: input.openid,
        unionid: input.unionid,
        openaiUserId: makeOpenAIUserId(id),
        createdAt: now,
        lastSeenAt: now,
        quotaBonuses: {}
      };
      store.users.push(user);
      return;
    }

    user.platform = input.platform ?? user.platform;
    user.nickname = input.nickname ?? user.nickname;
    user.avatarUrl = input.avatarUrl ?? user.avatarUrl;
    user.openid = input.openid ?? user.openid;
    user.unionid = input.unionid ?? user.unionid;
    user.lastSeenAt = now;
  });

  return user!;
}

export async function getUser(userId: string) {
  const store = await readStore();
  return store.users.find((item) => item.id === userId);
}

export async function getUsage(userId: string) {
  const store = await readStore();
  const date = todayKey();
  const user = store.users.find((item) => item.id === userId);
  const used = store.history.filter(
    (item) => item.userId === userId && item.type === 'image' && item.createdAt.startsWith(date)
  ).length;
  const bonus = user?.quotaBonuses[date] ?? 0;
  const total = config.freeDailyQuota + bonus;

  return {
    date,
    used,
    total,
    remaining: Math.max(0, total - used),
    freeDailyQuota: config.freeDailyQuota,
    bonus
  };
}

export async function addRewardedBonus(userId: string) {
  const date = todayKey();
  await updateStore((store) => {
    const user = store.users.find((item) => item.id === userId);
    if (!user) throw new Error('USER_NOT_FOUND');
    user.quotaBonuses[date] = (user.quotaBonuses[date] ?? 0) + config.rewardedAdBonus;
    user.lastSeenAt = nowIso();
  });
  return getUsage(userId);
}

export async function listUserHistory(userId: string, type?: HistoryRecord['type']) {
  const store = await readStore();
  return store.history
    .filter((item) => item.userId === userId && (!type || item.type === type))
    .sort((a, b) => b.createdAt.localeCompare(a.createdAt));
}
