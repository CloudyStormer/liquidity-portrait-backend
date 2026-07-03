# Liquidity Portrait Backend

Python/FastAPI backend for the ID photo mini-program. It handles WeChat OpenID login, local user records, download/action logs, quota/ad callbacks, history, MD5 checks, and upload records.

## Run

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python run.py
```

For local OpenID debugging without real WeChat credentials:

```env
WECHAT_DEV_OPENID=dev-openid-local
AUTH_TOKEN_SECRET=change-me
```

For real Mini Program login, set:

```env
WECHAT_APP_ID=
WECHAT_APP_SECRET=
AUTH_TOKEN_SECRET=
```

The frontend posts the Mini Program `code` to `POST /api/auth/wechat/login`. The backend calls WeChat `jscode2session`, stores the returned `openid`, returns an app token, and never returns WeChat `session_key` to the frontend.

## Main API

- `GET /health`
- `POST /api/auth/wechat/login`
- `POST /api/users/identify`
- `POST /api/photo/usage-records`
- `POST /api/logs`
- `GET /api/users/:userId/usage`
- `GET /api/users/:userId/history`
- `POST /api/ads/reward`
- `POST /api/process/image`
- `POST /api/tools/md5`
