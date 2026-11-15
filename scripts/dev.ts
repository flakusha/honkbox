export {};

// Runs the Vite dev server (:5173) and the Rust server (:8080) together.
// Ctrl-C kills both; either process exiting lets the other one finish.
const vite = Bun.spawn(["vite", "--port", "5173", "packages/web-ui"], {
  stdio: ["inherit", "inherit", "inherit"],
});
const server = Bun.spawn(["cargo", "watch", "-x", "run -p sd-rip-server -- serve --dev"], {
  stdio: ["inherit", "inherit", "inherit"],
});

function shutdown(): void {
  vite.kill();
  server.kill();
}

process.on("SIGINT", shutdown);
process.on("exit", shutdown);
await Promise.all([vite.exited, server.exited]);
