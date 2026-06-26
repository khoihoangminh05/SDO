/* eslint-disable @typescript-eslint/no-unsafe-argument, @typescript-eslint/no-unsafe-assignment */
import { Test, TestingModule } from '@nestjs/testing';
import { ExploreController } from './explore.controller';
import { ExploreService } from './explore.service';
import { CacheService } from '../cache/cache.service';
import { StorageService } from '../storage/storage.service';
import { HttpService } from '@nestjs/axios';
import { GlobalExceptionFilter } from '../common/filters/global-exception.filter';
import { INestApplication } from '@nestjs/common';
import request from 'supertest';
import { of, throwError } from 'rxjs';
import * as fs from 'fs';
import axios from 'axios';

jest.mock('fs', () => ({
  existsSync: jest.fn(),
  readdirSync: jest.fn(),
}));

describe('Explore Integration Tests', () => {
  let app: INestApplication;

  const mockCacheService = {
    generateCacheKey: jest.fn(),
    getInferenceResult: jest.fn(),
    setInferenceResult: jest.fn(),
  };

  const mockStorageService = {
    getUploadDir: jest.fn().mockReturnValue('/uploads'),
  };

  const mockHttpService = {
    post: jest.fn(),
  };

  beforeEach(async () => {
    const moduleFixture: TestingModule = await Test.createTestingModule({
      controllers: [ExploreController],
      providers: [
        ExploreService,
        { provide: CacheService, useValue: mockCacheService },
        { provide: StorageService, useValue: mockStorageService },
        { provide: HttpService, useValue: mockHttpService },
      ],
    }).compile();

    app = moduleFixture.createNestApplication();
    app.useGlobalFilters(new GlobalExceptionFilter());
    await app.init();
  });

  afterEach(async () => {
    jest.clearAllMocks();
    await app.close();
  });

  describe('POST /explore', () => {
    it('Scenario A (Cache Hit): should return cached RegionNode and not call Axios', async () => {
      const cachedNode = {
        node_id: 'cache_hit_123',
        image_id: 'img_123',
        zoom_level: 1,
        global_bbox: [100, 200, 300, 400],
        is_leaf: true,
        detections: [],
      };

      mockCacheService.generateCacheKey.mockReturnValue('cache_key_123');
      mockCacheService.getInferenceResult.mockResolvedValue(cachedNode);

      const response = await request(app.getHttpServer())
        .post('/explore')
        .send({
          image_id: 'img_123',
          zoom_level: 1,
          global_bbox: [100, 200, 300, 400],
        });

      expect(response.status).toBe(201);
      expect(response.body).toEqual(cachedNode);
      expect(mockHttpService.post).not.toHaveBeenCalled();
    });

    it('Scenario B (Cache Miss): should call AI worker, cache results, and return them', async () => {
      const workerResponse = {
        node_id: 'worker_node_456',
        image_id: 'img_123',
        zoom_level: 1,
        global_bbox: [100, 200, 300, 400],
        is_leaf: true,
        detections: [],
      };

      mockCacheService.generateCacheKey.mockReturnValue('cache_key_123');
      mockCacheService.getInferenceResult.mockResolvedValue(null);
      (fs.existsSync as jest.Mock).mockReturnValue(true);
      (fs.readdirSync as jest.Mock).mockReturnValue(['img_123.jpg']);

      mockHttpService.post.mockReturnValue(of({ data: workerResponse }));

      const response = await request(app.getHttpServer())
        .post('/explore')
        .send({
          image_id: 'img_123',
          zoom_level: 1,
          global_bbox: [100, 200, 300, 400],
        });

      expect(response.status).toBe(201);
      expect(response.body).toEqual(workerResponse);
      expect(mockHttpService.post).toHaveBeenCalled();
      expect(mockCacheService.setInferenceResult).toHaveBeenCalledWith(
        'cache_key_123',
        workerResponse,
      );
    });

    it('Scenario C (Worker Failure): should propagate Axios error response to client without crashing', async () => {
      mockCacheService.generateCacheKey.mockReturnValue('cache_key_123');
      mockCacheService.getInferenceResult.mockResolvedValue(null);
      (fs.existsSync as jest.Mock).mockReturnValue(true);
      (fs.readdirSync as jest.Mock).mockReturnValue(['img_123.jpg']);

      const axiosError = {
        name: 'AxiosError',
        message: 'Request failed with status code 500',
        isAxiosError: true,
        response: {
          status: 502,
          data: { error: 'Internal Server Error in Python Worker' },
        },
      };

      jest.spyOn(axios, 'isAxiosError').mockReturnValue(true);

      mockHttpService.post.mockReturnValue(throwError(() => axiosError));

      const response = await request(app.getHttpServer())
        .post('/explore')
        .send({
          image_id: 'img_123',
          zoom_level: 1,
          global_bbox: [100, 200, 300, 400],
        });

      expect(response.status).toBe(502);
      expect(response.body).toEqual(
        expect.objectContaining({
          statusCode: 502,
          message: expect.stringContaining('Python Worker error'),
        }),
      );
    });

    it('Scenario C (Worker Offline): should propagate 503 Service Unavailable if worker is down', async () => {
      mockCacheService.generateCacheKey.mockReturnValue('cache_key_123');
      mockCacheService.getInferenceResult.mockResolvedValue(null);
      (fs.existsSync as jest.Mock).mockReturnValue(true);
      (fs.readdirSync as jest.Mock).mockReturnValue(['img_123.jpg']);

      const axiosError = {
        name: 'AxiosError',
        message: 'connect ECONNREFUSED 127.0.0.1:8000',
        isAxiosError: true,
        code: 'ECONNABORTED',
      };

      jest.spyOn(axios, 'isAxiosError').mockReturnValue(true);

      mockHttpService.post.mockReturnValue(throwError(() => axiosError));

      const response = await request(app.getHttpServer())
        .post('/explore')
        .send({
          image_id: 'img_123',
          zoom_level: 1,
          global_bbox: [100, 200, 300, 400],
        });

      expect(response.status).toBe(503);
      expect(response.body).toEqual(
        expect.objectContaining({
          statusCode: 503,
          message: 'Python Worker is unavailable or timed out',
        }),
      );
    });
  });
});
