import { Controller, Post, Body } from '@nestjs/common';
import { ApiTags, ApiOperation, ApiResponse } from '@nestjs/swagger';
import { ExploreRequest } from '../dtos/explore-request.dto';
import { ExploreService } from './explore.service';

@ApiTags('exploration')
@Controller()
export class ExploreController {
  constructor(private readonly exploreService: ExploreService) {}

  @Post('explore')
  @ApiOperation({
    summary: 'Explore a specific region of an image recursively',
  })
  @ApiResponse({
    status: 200,
    description: 'Returns the region node containing detections and children.',
    schema: {
      type: 'object',
      properties: {
        node_id: { type: 'string', example: 'node_0_0_1_1' },
        image_id: { type: 'string', example: 'img_abc123' },
        zoom_level: { type: 'number', example: 0 },
        global_bbox: {
          type: 'array',
          items: { type: 'number' },
          example: [0, 0, 1, 1],
        },
        is_leaf: { type: 'boolean', example: true },
        detections: { type: 'array', items: { type: 'object' }, example: [] },
      },
    },
  })
  exploreRegion(@Body() exploreDto: ExploreRequest) {
    return this.exploreService.exploreRegion(
      exploreDto.image_id,
      exploreDto.zoom_level,
      exploreDto.global_bbox,
    );
  }
}
