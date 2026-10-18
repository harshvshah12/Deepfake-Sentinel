const { defineConfig } = require("@playwright/test");

module.exports = defineConfig({
  testDir: "./tests/e2e",
  timeout: 180000,
  use: {
    baseURL: "http://127.0.0.1:8091",
    headless: true,
    channel: "chrome",
  },
  webServer: {
    command: "python -m uvicorn backend.main:app --host 127.0.0.1 --port 8091",
    url: "http://127.0.0.1:8091/health",
    reuseExistingServer: true,
    timeout: 180000,
  },
});
