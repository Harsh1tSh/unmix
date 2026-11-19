// Runs Vitest from the canonical-case working directory.
//
// On Windows, a shell whose cwd is "d:\..." (lower-case drive letter, which is
// how VS Code often opens a folder) makes Vitest load its runner twice under
// two different paths, and every suite fails with
//   TypeError: Cannot read properties of undefined (reading 'config')
// realpathSync.native returns the on-disk casing ("D:\..."), which avoids it.
import { spawnSync } from "node:child_process";
import { realpathSync } from "node:fs";
import path from "node:path";

const cwd = realpathSync.native(process.cwd());
const entry = path.join(cwd, "node_modules", "vitest", "vitest.mjs");
const result = spawnSync(process.execPath, [entry, ...process.argv.slice(2)], { cwd, stdio: "inherit" });
process.exit(result.status ?? 1);
