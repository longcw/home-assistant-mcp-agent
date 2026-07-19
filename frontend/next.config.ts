import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  // produce a self-contained server bundle for Docker (.next/standalone/server.js)
  output: 'standalone',
  // this app is the tracing root even though it lives in a monorepo subdirectory
  outputFileTracingRoot: __dirname,
};

export default nextConfig;
