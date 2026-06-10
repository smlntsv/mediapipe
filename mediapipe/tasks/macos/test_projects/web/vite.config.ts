import { writeFileSync, mkdirSync } from "node:fs";
import { resolve, dirname, basename } from "node:path";
import type { Plugin } from "vite";
import { defineConfig } from "vite";

// Dev-server middleware that lets the page persist its result JSON to
// ../shared/output/<out> via POST /save-web-json?out=<name>, so the parity
// comparison can read it without manual file handling. The filename is
// sanitized to a basename to keep writes inside shared/output.
function saveWebJsonPlugin(): Plugin {
  return {
    name: "save-web-json",
    configureServer(server) {
      server.middlewares.use("/save-web-json", (req, res) => {
        if (req.method !== "POST") {
          res.statusCode = 405;
          res.end("method not allowed");
          return;
        }
        const url = new URL(req.url ?? "", "http://localhost");
        let name = basename(url.searchParams.get("out") || "web.json");
        if (!name.endsWith(".json")) name = "web.json";
        const chunks: Buffer[] = [];
        req.on("data", (c) => chunks.push(c as Buffer));
        req.on("end", () => {
          const out = resolve(__dirname, "../shared/output", name);
          mkdirSync(dirname(out), { recursive: true });
          writeFileSync(out, Buffer.concat(chunks));
          res.statusCode = 200;
          res.setHeader("content-type", "application/json");
          res.end(JSON.stringify({ saved: out }));
        });
      });
    },
  };
}

export default defineConfig({
  plugins: [saveWebJsonPlugin()],
  server: { open: true },
});
