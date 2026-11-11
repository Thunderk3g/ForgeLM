# ForgeLM dashboard (React + Vite)

- `npm install --legacy-peer-deps` (the flag is needed because of an existing peer-dependency conflict).
- `npm run dev` starts the Vite dev server on its default port, 5173, and proxies `/api` to the gateway at `http://localhost:8000` (`vite.config.ts`).
- `npm test` runs `tests/*.test.ts` with Node's built-in test runner and type stripping (Node 22.18 or later).
- `npm run build` runs `tsc -b && vite build`; it currently stops on 75 pre-existing TypeScript errors in older files, none in the Factory page.

The services behind `/api` are described in the root [README.md](../README.md) and in [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) (section 3, "How the existing services use it").
