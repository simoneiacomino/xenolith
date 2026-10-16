#include "../vision.h"
#include "../format.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: vision_prepare image.png budget output.rgb\n");
        return 2;
    }
    char *end;
    long budget = strtol(argv[2], &end, 10);
    if (*end || budget < 1 || budget > XE_VISUAL_MAX) return 2;
    xe_image im;
    char error[256];
    if (!xe_image_load(&im, argv[1], (int)budget, error, sizeof error)) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    FILE *f = fopen(argv[3], "wb");
    uint8_t header[8];
    format_put_u32le(header, im.resized_width);
    format_put_u32le(header+4, im.resized_height);
    int ok = f && format_write(f, header, sizeof header) &&
             format_write(f, im.resized, (size_t)im.resized_width*im.resized_height*3);
    if (f && fclose(f)) ok = 0;
    printf("width %d height %d rows %d\n", im.resized_width, im.resized_height, im.rows);
    xe_image_free(&im);
    return ok ? 0 : 1;
}
