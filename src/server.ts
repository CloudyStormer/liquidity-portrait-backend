import { config } from './config.js';
import { createApp } from './app.js';

const app = await createApp();

app.listen(config.port, () => {
  console.log(`liquidity-portrait backend listening on http://localhost:${config.port}`);
});
