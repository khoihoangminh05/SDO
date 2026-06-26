FROM node:24-alpine AS base
RUN npm i -g pnpm@11.5.2

FROM base AS pruner
WORKDIR /app
RUN npm i -g turbo
COPY . .
RUN turbo prune orchestrator --docker

FROM base AS installer
WORKDIR /app
COPY --from=pruner /app/out/json/ .
COPY --from=pruner /app/out/pnpm-lock.yaml ./pnpm-lock.yaml
RUN pnpm config set fetch-retries 5 && \
    pnpm config set fetch-retry-maxtimeout 120000 && \
    pnpm install --frozen-lockfile

FROM base AS builder
WORKDIR /app
COPY --from=pruner /app/out/full/ .
COPY --from=installer /app/ .
RUN pnpm turbo run build --filter=orchestrator...

FROM base AS runner
WORKDIR /app
COPY --from=builder /app/ .
EXPOSE 3000
CMD ["pnpm", "--filter", "orchestrator", "start:prod"]
