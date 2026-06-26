import { ApiProperty } from '@nestjs/swagger';
import { ExploreRequestDto } from '@recursive-object-detector/types';
import type { BoundingBox } from '@recursive-object-detector/types';

export class ExploreRequest implements ExploreRequestDto {
  @ApiProperty({
    description: 'Unique identifier of the target image',
    example: 'img_abc123',
  })
  image_id!: string;

  @ApiProperty({
    description:
      'Current zoom level of the window (higher means more zoomed in)',
    example: 0,
  })
  zoom_level!: number;

  @ApiProperty({
    description:
      'Global bounding box coordinates representing the region of interest [x_min, y_min, x_max, y_max]',
    type: [Number],
    example: [0.0, 0.0, 1.0, 1.0],
  })
  global_bbox!: BoundingBox;
}
