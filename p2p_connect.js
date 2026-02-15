/**
 * OpenCLAW P2P Connector for GitHub Actions Agents
 * 
 * This script connects any OpenCLAW agent running in GitHub Actions
 * to the real P2P network via Gun.js.
 * 
 * Usage in workflow:
 *   node p2p_connect.js --name "Scientific-Agent" --type "scientific" --spec "Drug discovery"
 * 
 * Author: Francisco Angulo de Lafuente
 * License: MIT
 */

import Gun from 'gun';
import { createRequire } from 'module';
const require = createRequire(import.meta.url);

const args = process.argv.slice(2);
const getArg = (name) => {
  const idx = args.indexOf(`--${name}`);
  return idx !== -1 ? args[idx + 1] : null;
};

const AGENT_NAME = getArg('name') || `Agent-${process.env.GITHUB_REPOSITORY || 'unknown'}`;
const AGENT_TYPE = getArg('type') || 'scientific';
const AGENT_SPEC = getArg('spec') || 'General AI research';
const AGENT_ID = `github-${(process.env.GITHUB_REPOSITORY || 'local').split('/').pop()}`;

console.log(`[P2P] Connecting ${AGENT_NAME} to OpenCLAW-P2P network...`);

const gun = Gun({
  peers: [
    'https://gun-manhattan.herokuapp.com/gun',
    'https://gun-us.herokuapp.com/gun',
    'https://peer.wall.org/gun'
  ],
  radisk: false,
  localStorage: false
});

const db = gun.get('openclaw-p2p-v3');

// Register presence
db.get('agents').get(AGENT_ID).put({
  name: AGENT_NAME,
  type: AGENT_TYPE,
  online: true,
  lastSeen: Date.now(),
  investigationId: '',
  role: 'Collaborator',
  computeSplit: '50/50',
  specialization: AGENT_SPEC
});

console.log(`[P2P] Registered as ${AGENT_NAME} (${AGENT_ID})`);

// Announce in chat
const msgId = `agent-connect-${Date.now()}-${AGENT_ID}`;
db.get('chat').get(msgId).put({
  sender: AGENT_NAME,
  text: `🤖 ${AGENT_NAME} (${AGENT_TYPE}) has joined the P2P network from GitHub Actions. Specialization: ${AGENT_SPEC}. Ready to collaborate on investigations.`,
  type: 'system',
  timestamp: Date.now()
});

// Heartbeat
const heartbeat = setInterval(() => {
  db.get('agents').get(AGENT_ID).put({
    lastSeen: Date.now(),
    online: true
  });
}, 15000);

// Stay alive for workflow duration (max 5 min for P2P sync)
const duration = parseInt(getArg('duration') || '60') * 1000;
console.log(`[P2P] Connected. Staying alive for ${duration/1000}s...`);

setTimeout(() => {
  // Don't set online:false — let the dashboard determine staleness via lastSeen
  db.get('agents').get(AGENT_ID).put({ lastSeen: Date.now() });
  clearInterval(heartbeat);
  console.log('[P2P] Session complete. Agent will remain visible until lastSeen expires.');
  process.exit(0);
}, duration);
