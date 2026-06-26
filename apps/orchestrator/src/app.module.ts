import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { StorageModule } from './storage/storage.module';
import { CacheModule } from './cache/cache.module';
import { ExploreModule } from './explore/explore.module';
import { ThrottlerModule, ThrottlerGuard } from '@nestjs/throttler';
import { APP_GUARD } from '@nestjs/core';

@Module({
  imports: [
    StorageModule,
    CacheModule,
    ExploreModule,
    ThrottlerModule.forRoot([
      {
        ttl: 60000, // 1 minute
        limit: 60, // max 60 requests
      },
    ]),
  ],
  controllers: [AppController],
  providers: [
    AppService,
    {
      provide: APP_GUARD,
      useClass: ThrottlerGuard,
    },
  ],
})
export class AppModule {}
