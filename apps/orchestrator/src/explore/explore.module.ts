import { Module } from '@nestjs/common';
import { HttpModule } from '@nestjs/axios';
import { ExploreController } from './explore.controller';
import { ExploreService } from './explore.service';
import { StorageModule } from '../storage/storage.module';

@Module({
  imports: [
    HttpModule.register({
      timeout: 300000, // 300 seconds timeout for large image processing
      maxRedirects: 5,
    }),
    StorageModule,
  ],
  controllers: [ExploreController],
  providers: [ExploreService],
})
export class ExploreModule {}
