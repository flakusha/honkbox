import { defineConfig } from "vite";

export default defineConfig({
  appType: "spa",
  build: {
    emptyOutDir: true,
    outDir: "../../web-dist",
  },
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:8080",
      "/v1": "http://localhost:8080",
    },
  },
});
