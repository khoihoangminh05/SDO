/* eslint-disable @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-return */
import { Test, TestingModule } from '@nestjs/testing';
import { CacheService } from './cache.service';

jest.mock('ioredis', () => {
  return jest.fn().mockImplementation(() => {
    return {
      on: jest.fn((event, callback) => {
        if (event === 'connect') {
          setTimeout(() => callback(), 0);
        }
      }),
      get: jest.fn(),
      set: jest.fn(),
      disconnect: jest.fn(),
    };
  });
});

describe('CacheService', () => {
  let service: CacheService;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [CacheService],
    }).compile();

    service = module.get<CacheService>(CacheService);
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });

  describe('generateCacheKey', () => {
    it('should correctly round bounding box float coordinates to prevent cache misses', () => {
      const imageId = 'img123';
      const zoomLevel = 2;
      const globalBbox = [100.4, 200.7, 300.1, 400.9];

      const key = service.generateCacheKey(imageId, zoomLevel, globalBbox);

      // 100.4 -> 100
      // 200.7 -> 201
      // 300.1 -> 300
      // 400.9 -> 401
      expect(key).toBe('inference:img123:lvl_2:bbox_100_201_300_401');
    });
  });
});
