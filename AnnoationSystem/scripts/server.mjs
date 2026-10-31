import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import net from 'node:net';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const runtime = path.join(root, '.dsh');
fs.mkdirSync(runtime, { recursive: true, mode: 0o700 });
const pidFile = path.join(runtime, 'server.pid');
const logFile = path.join(runtime, 'server.log');
const command = process.argv[2] ?? 'status';
function livePid() {
  try {
    const pid = Number(fs.readFileSync(pidFile, 'utf8'));
    if (!Number.isInteger(pid) || pid <= 0) return null;
    process.kill(pid, 0);
    return pid;
  } catch { return null; }
}
function status() {
  const text = fs.existsSync(logFile) ? fs.readFileSync(logFile, 'utf8') : '';
  const urls = text.match(/http:\/\/127\.0\.0\.1:\d+\/\?token=[\w-]+/g);
  return { running: Boolean(livePid()), pid: livePid(), url: livePid() ? urls?.at(-1) ?? null : null, log: logFile };
}
async function portAvailable(port) {
  return new Promise((resolve) => {
    const server = net.createServer();
    server.once('error', () => resolve(false));
    server.listen(port, '127.0.0.1', () => server.close(() => resolve(true)));
  });
}
if (command === 'start') {
  if (!livePid()) {
    const port = Number(process.env.ANNOATION_PORT ?? 3080);
    if (!await portAvailable(port)) throw new Error(`Port ${port} is occupied; set ANNOATION_PORT to a free port.`);
    const log = fs.openSync(logFile, 'w', 0o600);
    const child = spawn(process.execPath, [path.join(root, 'scripts/dsh.mjs'), 'web', '--no-open', '--host', '127.0.0.1', '--port', String(port)], {
      cwd: root, env: process.env, detached: true, stdio: ['ignore', log, log],
    });
    child.unref();
    fs.closeSync(log);
    fs.writeFileSync(pidFile, String(child.pid), { mode: 0o600 });
    for (let attempt = 0; attempt < 120; attempt += 1) {
      await new Promise((resolve) => setTimeout(resolve, 500));
      if (!livePid()) throw new Error(`DSH exited; see ${logFile}`);
      if (status().url) break;
    }
    if (!status().url) throw new Error(`DSH startup timed out; see ${logFile}`);
  }
} else if (command === 'stop') {
  const pid = livePid();
  if (pid) {
    process.kill(pid, 'SIGTERM');
    for (let attempt = 0; attempt < 60 && livePid(); attempt += 1) await new Promise((resolve) => setTimeout(resolve, 500));
    if (livePid()) throw new Error('DSH has not exited yet; PID file preserved.');
  }
  if (fs.existsSync(pidFile)) fs.unlinkSync(pidFile);
} else if (command !== 'status') throw new Error('Use start, stop, or status');
console.log(JSON.stringify(status(), null, 2));
