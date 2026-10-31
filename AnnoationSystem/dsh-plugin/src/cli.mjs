#!/usr/bin/env node
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createReviewTask, inspectAsset, rankAssets, saveReview } from './engine.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const defaultDataRoot = path.resolve(here, '../../../a1.0.0_00');
const defaultOutputRoot = path.resolve(here, '../../output');

function options(argv) {
  const result = {};
  for (let i = 0; i < argv.length; i += 1) {
    if (!argv[i].startsWith('--')) continue;
    result[argv[i].slice(2)] = argv[i + 1];
    i += 1;
  }
  return result;
}

function usage() {
  console.log('Usage:\n  node src/cli.mjs rank --data-root PATH [--limit N]\n  node src/cli.mjs inspect --data-root PATH --asset ID\n  node src/cli.mjs task --data-root PATH --output-root PATH --asset ID\n  node src/cli.mjs review --output-root PATH --asset ID --decision VALUE --reviewer NAME [--notes TEXT]');
}

try {
  const [command, ...argv] = process.argv.slice(2);
  const args = options(argv);
  const dataRoot = path.resolve(args['data-root'] ?? process.env.ANNOATION_DATA_ROOT ?? defaultDataRoot);
  const outputRoot = path.resolve(args['output-root'] ?? process.env.ANNOATION_OUTPUT_ROOT ?? defaultOutputRoot);
  let result;
  if (command === 'rank') result = rankAssets(dataRoot, Number(args.limit ?? 50));
  else if (command === 'inspect') result = inspectAsset(dataRoot, args.asset);
  else if (command === 'task') result = createReviewTask(outputRoot, inspectAsset(dataRoot, args.asset), args.notes ?? '');
  else if (command === 'review') result = saveReview(outputRoot, args.asset, args.decision, args.reviewer ?? '', args.notes ?? '');
  else {
    usage();
    process.exitCode = command ? 2 : 0;
  }
  if (result) console.log(JSON.stringify(result, null, 2));
} catch (error) {
  console.error(JSON.stringify({ error: String(error.message ?? error) }, null, 2));
  process.exitCode = 1;
}
