import { Injectable, Logger, BadRequestException } from '@nestjs/common';
import { HttpService } from '@nestjs/axios';
import { CacheService } from '../cache/cache.service';
import { StorageService } from '../storage/storage.service';
import { firstValueFrom } from 'rxjs';
import * as fs from 'fs';
import * as path from 'path';
import { RegionNode, Detection } from '@recursive-object-detector/types';

@Injectable()
export class ExploreService {
  private readonly logger = new Logger(ExploreService.name);
  private readonly workerUrl =
    process.env.WORKER_URL || 'http://localhost:8000';

  constructor(
    private readonly httpService: HttpService,
    private readonly cacheService: CacheService,
    private readonly storageService: StorageService,
  ) {}

  async exploreRegion(
    imageId: string,
    zoomLevel: number,
    globalBbox: number[],
  ): Promise<unknown> {
    // Step A: Check Redis cache
    const cacheKey = this.cacheService.generateCacheKey(
      imageId,
      zoomLevel,
      globalBbox,
    );
    const cachedResult = (await this.cacheService.getInferenceResult(
      cacheKey,
    )) as unknown;

    if (cachedResult) {
      this.logger.log(`Cache hit for key: ${cacheKey}`);
      return cachedResult;
    }

    this.logger.log(`Cache miss for key: ${cacheKey}. Invoking worker...`);

    // Step B: Resolve absolute image path
    const uploadDir = this.storageService.getUploadDir();
    if (!fs.existsSync(uploadDir)) {
      throw new BadRequestException('Uploads directory does not exist');
    }

    const files = fs.readdirSync(uploadDir);
    const matchedFile = files.find((file) => file.startsWith(imageId));

    if (!matchedFile) {
      throw new BadRequestException(`Image with ID ${imageId} not found`);
    }

    const absolutePath = path.resolve(uploadDir, matchedFile);

    // Step C: Await response from Python worker
    const url = `${this.workerUrl}/api/v1/detect`;
    const payload = {
      image_id: imageId,
      image_path: absolutePath,
      zoom_level: zoomLevel,
      global_bbox: globalBbox,
    };

    const response$ = this.httpService.post(url, payload);
    const response = await firstValueFrom(response$);
    const result = response.data as RegionNode;

    // Step D: Traverse tree, nest sub_detections, and cache parent objects specifically
    await this.processNodeAndCacheParents(result, imageId);

    // Step E: Save exact nested response in Redis cache
    await this.cacheService.setInferenceResult(cacheKey, result);

    // Step F: Return response
    return result;
  }

  private async processNodeAndCacheParents(node: RegionNode, imageId: string): Promise<void> {
    // 1. Nest detections for the current node
    this.nestDetections(node);

    // 2. Cache parent objects in this node
    if (node.detections) {
      for (const det of node.detections) {
        const isParent = det.class_id === 9 || det.class_name.toLowerCase().startsWith('traffic light');
        if (isParent) {
          const parentCacheKey = this.cacheService.generateParentCacheKey(
            imageId,
            det.class_name,
            det.global_bbox,
          );
          this.logger.log(`Caching parent object at: ${parentCacheKey}`);
          await this.cacheService.setInferenceResult(parentCacheKey, det);
        }
      }
    }

    // 3. Recurse for children RegionNodes if any
    if (node.children && node.children.length > 0) {
      for (const childNode of node.children) {
        await this.processNodeAndCacheParents(childNode, imageId);
      }
    }
  }

  private nestDetections(node: RegionNode): void {
    if (!node.detections || node.detections.length === 0) {
      return;
    }

    // Identify parents and children
    const parents: Detection[] = [];
    const children: Detection[] = [];
    const others: Detection[] = [];

    for (const det of node.detections) {
      const isParent = det.class_id === 9 || det.class_name.toLowerCase().startsWith('traffic light');
      const isChild = [91, 92, 93, 94].includes(det.class_id) || 
                      det.class_name.toLowerCase().startsWith('red light') || 
                      det.class_name.toLowerCase().startsWith('yellow light') || 
                      det.class_name.toLowerCase().startsWith('green light') || 
                      det.class_name.toLowerCase().startsWith('digit_');
      
      if (isParent) {
        det.sub_detections = det.sub_detections || [];
        parents.push(det);
      } else if (isChild) {
        children.push(det);
      } else {
        others.push(det);
      }
    }

    // Map children to parents using bounding box containment
    const matchedChildren = new Set<Detection>();
    for (const child of children) {
      let matchedParent: Detection | null = null;
      
      const [cXmin, cYmin, cXmax, cYmax] = child.global_bbox;
      const cCenterX = (cXmin + cXmax) / 2;
      const cCenterY = (cYmin + cYmax) / 2;

      for (const parent of parents) {
        const [pXmin, pYmin, pXmax, pYmax] = parent.global_bbox;
        if (
          cCenterX >= pXmin &&
          cCenterX <= pXmax &&
          cCenterY >= pYmin &&
          cCenterY <= pYmax
        ) {
          matchedParent = parent;
          break;
        }
      }

      if (matchedParent) {
        matchedParent.sub_detections = matchedParent.sub_detections || [];
        matchedParent.sub_detections.push(child);
        matchedChildren.add(child);
      }
    }

    // Reconstruct the node detections list
    const finalDetections: Detection[] = [];
    finalDetections.push(...parents);
    finalDetections.push(...others);
    
    // Fallback: If a child couldn't be matched to any parent, keep it flat
    for (const child of children) {
      if (!matchedChildren.has(child)) {
        finalDetections.push(child);
      }
    }

    node.detections = finalDetections;
  }
}
