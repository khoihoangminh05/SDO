import { Injectable, Logger, OnModuleDestroy } from '@nestjs/common';
import Redis from 'ioredis';

@Injectable()
export class CacheService implements OnModuleDestroy {
  private readonly logger = new Logger(CacheService.name);
  private readonly client: Redis;
  private isConnected = false;

  constructor() {
    const host = process.env.REDIS_HOST || 'localhost';
    const port = parseInt(process.env.REDIS_PORT || '6379', 10);

    this.logger.log(`Initializing Redis client connecting to ${host}:${port}`);

    this.client = new Redis({
      host,
      port,
      maxRetriesPerRequest: 1, // Fail fast on connection errors
      retryStrategy: (times) => {
        // Retry connection every 5 seconds, up to a limit
        const delay = Math.min(times * 1000, 5000);
        return delay;
      },
    });

    this.client.on('connect', () => {
      this.isConnected = true;
      this.logger.log('Successfully connected to Redis instance.');
    });

    this.client.on('error', (error) => {
      this.isConnected = false;
      this.logger.warn(`Redis connection error: ${error.message}`);
    });
  }

  generateCacheKey(
    imageId: string,
    zoomLevel: number,
    globalBbox: number[],
  ): string {
    const roundedBbox = globalBbox.map((val) => Math.round(val));
    const bboxStr = roundedBbox.join('_');
    return `inference:${imageId}:lvl_${zoomLevel}:bbox_${bboxStr}`;
  }

  generateParentCacheKey(
    imageId: string,
    className: string,
    globalBbox: number[],
  ): string {
    const roundedBbox = globalBbox.map((val) => Math.round(val));
    const bboxStr = roundedBbox.join('_');
    return `inference:${imageId}:parent:${className.replace(/\s+/g, '_')}:bbox_${bboxStr}`;
  }

  async getInferenceResult(key: string): Promise<any> {
    try {
      if (!this.isConnected) {
        this.logger.verbose('Redis is not connected. Skipping cache read.');
        return null;
      }
      const cachedData = await this.client.get(key);
      if (!cachedData) return null;
      return JSON.parse(cachedData);
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      this.logger.error(`Error reading from Redis cache: ${msg}`);
      return null;
    }
  }

  async setInferenceResult(
    key: string,
    data: any,
    ttlSeconds: number = 86400,
  ): Promise<void> {
    try {
      if (!this.isConnected) {
        this.logger.verbose('Redis is not connected. Skipping cache write.');
        return;
      }
      const serializedData = JSON.stringify(data);
      await this.client.set(key, serializedData, 'EX', ttlSeconds);
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      this.logger.error(`Error writing to Redis cache: ${msg}`);
    }
  }

  onModuleDestroy() {
    this.logger.log('Disconnecting Redis client.');
    this.client.disconnect();
  }
}
