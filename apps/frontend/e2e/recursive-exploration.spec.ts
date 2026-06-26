import { test, expect } from '@playwright/test';

test.describe('Recursive Object Explorer E2E', () => {
  test.beforeEach(async ({ page }) => {
    // Log browser console output and errors
    page.on('console', (msg) => console.log('BROWSER CONSOLE:', msg.text()));
    page.on('pageerror', (err) => console.log('BROWSER ERROR:', err.message));

    // Mock POST /upload
    await page.route('**/upload', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          imageId: 'img_test_123',
          filename: 'img_test_123.jpg',
          message: 'Upload successful',
        }),
      });
    });

    // Mock image request to return a valid 1x1 transparent JPEG/PNG
    const transparentPng = Buffer.from(
      'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=',
      'base64',
    );
    await page.route('**/uploads/img_test_123.jpg', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'image/jpeg',
        body: transparentPng,
      });
    });

    // Mock POST /explore with a slight delay to allow verifying the loading spinner
    await page.route('**/explore', async (route) => {
      const requestBody = route.request().postDataJSON();

      console.log(
        'MOCK RECEIVING EXPLORE REQUEST:',
        JSON.stringify(requestBody),
      );

      // Introduce a 200ms delay to make the loading spinner testable
      await new Promise((resolve) => setTimeout(resolve, 200));

      if (requestBody.zoom_level === 0) {
        // Root detection - coordinates normalized for the 1x1 image size
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            node_id: 'node_root',
            image_id: 'img_test_123',
            zoom_level: 0,
            global_bbox: [0.0, 0.0, 1.0, 1.0],
            is_leaf: false,
            detections: [
              {
                class_id: 0,
                class_name: 'Port',
                confidence: 0.95,
                local_bbox: [0.1, 0.2, 0.3, 0.4],
                global_bbox: [0.1, 0.2, 0.3, 0.4],
              },
            ],
          }),
        });
      } else if (requestBody.zoom_level === 1) {
        // Nested Lvl 1 detection - coordinates nested inside [0.1, 0.2, 0.3, 0.4]
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            node_id: 'node_ship_1',
            image_id: 'img_test_123',
            zoom_level: 1,
            global_bbox: [0.1, 0.2, 0.3, 0.4],
            is_leaf: true,
            detections: [
              {
                class_id: 1,
                class_name: 'Ship',
                confidence: 0.88,
                local_bbox: [0.12, 0.22, 0.22, 0.32],
                global_bbox: [0.12, 0.22, 0.22, 0.32],
              },
            ],
          }),
        });
      } else {
        await route.fulfill({
          status: 404,
          body: 'Not Found',
        });
      }
    });
  });

  test('Happy Path: upload image, show overlays, click-to-explore, check spinner, and navigate sidebar', async ({
    page,
  }) => {
    // 1. Navigate to homepage
    await page.goto('/');

    // Assert homepage layout
    await expect(page.locator('h1')).toContainText('Recursive Object Detector');

    // 2. Select file for upload (mocked)
    await page.locator('input[type="file"]').setInputFiles({
      name: 'test-image.jpg',
      mimeType: 'image/jpeg',
      buffer: Buffer.from('dummy image data'),
    });

    // Submit form
    await page.locator('button[type="submit"]').click();

    // 3. Assert redirect to viewer page
    await expect(page).toHaveURL(/\/viewer\/img_test_123\.jpg/);

    // 4. Assert root bounding box overlay is visible
    const rootOverlay = page.locator('#det-overlay-0');
    await expect(rootOverlay).toBeVisible();
    await expect(rootOverlay).toContainText('Port (95%)');

    // 5. Click the overlay box to zoom in and explore the region
    await rootOverlay.dispatchEvent('click');

    // 6. Assert loading spinner state appears during flight
    const loadingState = page.locator('#det-overlay-0.animate-pulse');
    await expect(loadingState).toBeVisible();

    // 7. Assert new child overlay renders on screen
    const childOverlay = page.locator('#det-overlay-0');
    await expect(childOverlay).toBeVisible();
    await expect(childOverlay).toContainText('Ship (88%)');

    // Assert Tree Explorer sidebar now lists the nested level
    const sidebarTree = page.locator('text=Port (Lvl 1)');
    await expect(sidebarTree).toBeVisible();

    // 8. Test Breadcrumb/Tree Navigation: Click 'Root Image' node in sidebar to zoom out
    const rootTreeNode = page.locator('text=Root Image');
    await rootTreeNode.click();

    // Assert we return to the root state with the Port overlay
    await expect(page.locator('#det-overlay-0')).toContainText('Port (95%)');
  });
});
