import {
  ExceptionFilter,
  Catch,
  ArgumentsHost,
  HttpException,
  HttpStatus,
  Logger,
} from '@nestjs/common';
import { Response } from 'express';
import axios from 'axios';

@Catch()
export class GlobalExceptionFilter implements ExceptionFilter {
  private readonly logger = new Logger(GlobalExceptionFilter.name);

  catch(exception: any, host: ArgumentsHost) {
    const ctx = host.switchToHttp();
    const response = ctx.getResponse<Response>();

    let status = HttpStatus.INTERNAL_SERVER_ERROR;
    let message: string | object = 'Internal server error';

    if (axios.isAxiosError(exception)) {
      this.logger.error(`Axios error: ${exception.message}`, exception.stack);
      if (
        exception.code === 'ECONNABORTED' ||
        exception.code === 'ETIMEDOUT' ||
        !exception.response
      ) {
        status = HttpStatus.SERVICE_UNAVAILABLE;
        message = 'Python Worker is unavailable or timed out';
      } else {
        status = exception.response.status;
        message = `Python Worker error: ${JSON.stringify(exception.response.data)}`;
      }
    } else if (exception instanceof HttpException) {
      status = exception.getStatus();
      message = exception.getResponse();
    } else if (exception instanceof Error) {
      this.logger.error(
        `Unhandled error: ${exception.message}`,
        exception.stack,
      );
      message = exception.message;
    }

    response.status(status).json({
      statusCode: status,
      message,
      timestamp: new Date().toISOString(),
    });
  }
}
