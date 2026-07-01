export type RemoveMethod = 'screenshot' | 'doodle' | 'selection';
export type HistoryRecordType = 'image' | 'md5';

export interface UserRecord {
  id: string;
  clientId: string;
  platform: string;
  nickname?: string;
  avatarUrl?: string;
  openid?: string;
  unionid?: string;
  openaiUserId: string;
  createdAt: string;
  lastSeenAt: string;
  quotaBonuses: Record<string, number>;
}

export interface ImageHistoryRecord {
  id: string;
  type: 'image';
  userId: string;
  originalName: string;
  originalUrl: string;
  processedUrl: string;
  fileSize: number;
  method: RemoveMethod;
  status: 'completed' | 'failed';
  provider: 'openai' | 'local-preview';
  openaiRequestId?: string;
  createdAt: string;
}

export interface Md5HistoryRecord {
  id: string;
  type: 'md5';
  userId: string;
  fileName: string;
  fileSize: number;
  md5: string;
  duplicate: boolean;
  createdAt: string;
}

export type HistoryRecord = ImageHistoryRecord | Md5HistoryRecord;

export interface PhotoUsageRecord {
  id: string;
  userId?: string;
  openid?: string;
  sizeId: string;
  sizeName: string;
  imagePath: string;
  createdAt: string;
  status: 'completed';
}

export interface OpenAIRequestRecord {
  id: string;
  userId: string;
  openaiUserId: string;
  endpoint: string;
  model: string;
  status: 'skipped' | 'completed' | 'failed';
  createdAt: string;
  error?: string;
}

export interface StoreShape {
  users: UserRecord[];
  history: HistoryRecord[];
  photoUsageRecords: PhotoUsageRecord[];
  openaiRequests: OpenAIRequestRecord[];
}
