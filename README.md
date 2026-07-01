# Liquidity Portrait Backend

Backend service for the image watermark workflow, quota, history, MD5 checks, reward callbacks, and OpenAI-backed processing.

## Run

```bash
npm install
copy .env.example .env
npm run dev
```

Set `OPENAI_API_KEY` and `ENABLE_OPENAI_IMAGE_EDIT=true` to call OpenAI. Without those, image processing records the request and returns the uploaded image as a development preview.

## WeChat OpenID login

Set `WECHAT_APP_ID`, `WECHAT_APP_SECRET`, and `AUTH_TOKEN_SECRET` in `.env`.
The frontend posts the Mini Program login `code` to `POST /api/auth/wechat/login`; the backend calls WeChat `jscode2session`, stores the returned `openid` on the local user, and returns an app token plus the user profile. The WeChat `session_key` is never returned to the frontend.

For local backend-only debugging without WeChat credentials, set `WECHAT_DEV_OPENID` to a test value. The login route will still create the user, store the OpenID, and return a signed app token.

## OpenAI user tracking

Each client receives a stable `userId` from `POST /api/users/identify`. The backend derives a stable `openaiUserId` and passes it to OpenAI as the request `user` value. The same identifier is also stored in `data/store.json` under `openaiRequests` so local records can be reconciled with OpenAI usage/safety logs.

## Main API

- `POST /api/users/identify`
- `POST /api/auth/wechat/login`
- `POST /api/photo/usage-records`
- `GET /api/users/:userId/usage`
- `GET /api/users/:userId/history`
- `POST /api/ads/reward`
- `POST /api/process/image`
- `POST /api/tools/md5`
- `GET /health`
