import {
  Controller,
  Post,
  UploadedFile,
  UseInterceptors,
  ParseFilePipe,
  MaxFileSizeValidator,
  BadRequestException,
} from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import {
  ApiTags,
  ApiOperation,
  ApiConsumes,
  ApiBody,
  ApiResponse,
} from '@nestjs/swagger';
import { diskStorage } from 'multer';
import { extname } from 'path';
import { randomUUID } from 'crypto';

@ApiTags('storage')
@Controller()
export class StorageController {
  @Post('upload')
  @ApiOperation({ summary: 'Upload an image (jpeg, png, webp) up to 50MB' })
  @ApiConsumes('multipart/form-data')
  @ApiBody({
    schema: {
      type: 'object',
      properties: {
        image: {
          type: 'string',
          format: 'binary',
        },
      },
    },
  })
  @ApiResponse({
    status: 201,
    description: 'Image successfully uploaded.',
    schema: {
      type: 'object',
      properties: {
        imageId: {
          type: 'string',
          example: '123e4567-e89b-12d3-a456-426614174000',
        },
        filename: {
          type: 'string',
          example: '123e4567-e89b-12d3-a456-426614174000.jpg',
        },
        message: { type: 'string', example: 'Upload successful' },
      },
    },
  })
  @UseInterceptors(
    FileInterceptor('image', {
      storage: diskStorage({
        destination: './uploads',
        filename: (req, file, callback) => {
          const uniqueId = randomUUID();
          const ext = extname(file.originalname);
          callback(null, `${uniqueId}${ext}`);
        },
      }),
      fileFilter: (req, file, callback) => {
        if (!file.mimetype.match(/\/(jpg|jpeg|png|webp)$/i)) {
          return callback(
            new BadRequestException(
              'Validation failed (expected type is image/(jpeg|png|webp))',
            ),
            false,
          );
        }
        callback(null, true);
      },
    }),
  )
  uploadFile(
    @UploadedFile(
      new ParseFilePipe({
        validators: [
          new MaxFileSizeValidator({ maxSize: 50 * 1024 * 1024 }), // 50MB
        ],
      }),
    )
    file: Express.Multer.File,
  ) {
    const filename = file.filename;
    const imageId = filename.substring(0, filename.lastIndexOf('.'));
    return {
      imageId,
      filename,
      message: 'Upload successful',
    };
  }
}
