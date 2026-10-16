#include "../vision.h"
#include "../format.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expr) do { if (!(expr)) { fprintf(stderr,"line %d: %s\n",__LINE__,#expr); return 1; } } while (0)

int main(void) {
    int w,h;
    CHECK(xe_image_size(1920,1344,1120,&w,&h) && w==1920 && h==1344);
    CHECK(xe_image_size(1920,1344,70,&w,&h) && w==480 && h==336);
    CHECK(xe_image_size(1,16384,1120,&w,&h) && w==48 && h==1120*48);
    CHECK(!xe_image_size(1920,1344,512,&w,&h));
    CHECK(!xe_image_size(0,1344,1120,&w,&h));
    float *embeddings=calloc((size_t)XE_VISUAL_MAX*XE_VISUAL_WIDTH,sizeof(float));
    CHECK(embeddings);
    int32_t *tokens=calloc(1200,sizeof(*tokens));CHECK(tokens);
    xe_visual_span spans[2]={{.start=50,.rows=1120,.embeddings=embeddings}};
    xe_input input={tokens,1200,spans,1};
    CHECK(xe_input_valid(&input,65536));
    CHECK(xe_input_boundary(&input,50));
    CHECK(xe_input_boundary(&input,1170));
    CHECK(!xe_input_boundary(&input,512));
    uint8_t before[32],after[32];xe_input_hash(&input,before);
    spans[0].identity[0]=1;xe_input_hash(&input,after);
    CHECK(memcmp(before,after,32));
    spans[0].rows=1121;CHECK(!xe_input_valid(&input,65536));spans[0].rows=1120;
    spans[1]=spans[0];input.visual_count=2;CHECK(!xe_input_valid(&input,65536));input.visual_count=1;
    input.count=512;CHECK(!xe_input_valid(&input,65536));input.count=1200;
    embeddings[0]=NAN;CHECK(!xe_input_valid(&input,65536));embeddings[0]=0;
    CHECK(!xe_input_valid(&input,1000));
    xe_image image={.rows=1120};uint8_t model[32]={0},projector[32]={0};
    xe_visual_identity(&image,model,projector,embeddings,"test/v1",before);
    embeddings[100]=1;xe_visual_identity(&image,model,projector,embeddings,"test/v1",after);
    CHECK(memcmp(before,after,32));embeddings[100]=0;
    projector[0]=1;xe_visual_identity(&image,model,projector,embeddings,"test/v1",after);
    CHECK(memcmp(before,after,32));
    spans[0].media=(const unsigned char *)"original-media";spans[0].media_size=14;spans[0].budget=1120;
    xe_owned_input owned={0};CHECK(xe_input_copy(&owned,&input));
    CHECK(owned.input.visual[0].embeddings!=embeddings);
    xe_input_hash(&owned.input,before);xe_input_hash(&input,after);CHECK(!memcmp(before,after,32));
    embeddings[0]=1;xe_input_hash(&input,after);CHECK(memcmp(before,after,32));embeddings[0]=0;
    FILE *archive=tmpfile();CHECK(archive);CHECK(xe_input_save(archive,&input));
    xe_owned_input loaded={0};CHECK(xe_input_load(archive,65536,&loaded));
    xe_input_hash(&loaded.input,after);CHECK(!memcmp(before,after,32));
    CHECK(fseek(archive,64,SEEK_SET)==0);CHECK(fputc(1,archive)!=EOF);CHECK(fflush(archive)==0);
    xe_owned_input corrupt={0};CHECK(!xe_input_load(archive,65536,&corrupt));
    fclose(archive);xe_input_free(&loaded);xe_input_free(&owned);
    uint8_t record[XE_IMAGE_RECORD_HEADER+8]={0};
    memcpy(record,"XEVIMG1\n",8);format_put_u32le(record+8,33);
    format_put_u32le(record+12,70);format_put_u32le(record+16,70);
    format_put_u32le(record+20,1920);format_put_u32le(record+24,1344);
    format_put_u32le(record+28,480);format_put_u32le(record+32,336);
    format_put_u32le(record+36,XE_IMAGE_RECORD_HEADER);format_put_u64le(record+40,8);
    xe_image meta;int offset;const uint8_t *media;size_t media_size;
    CHECK(xe_image_record_parse(record,sizeof record,&meta,&offset,&media,&media_size));
    CHECK(meta.rows==70&&offset==33&&media_size==8&&media==record+XE_IMAGE_RECORD_HEADER);
    CHECK(!xe_image_record_parse(record,sizeof record-1,&meta,&offset,&media,&media_size));
    record[272]=1;CHECK(!xe_image_record_parse(record,sizeof record,&meta,&offset,&media,&media_size));record[272]=0;
    format_put_u32le(record+12,71);CHECK(!xe_image_record_parse(record,sizeof record,&meta,&offset,&media,&media_size));
    format_put_u32le(record+12,70);format_put_u32le(record+32,335);
    CHECK(!xe_image_record_parse(record,sizeof record,&meta,&offset,&media,&media_size));
    free(tokens);free(embeddings);
    puts("vision input: boundaries, lengths, identity and finite rows PASS");
    return 0;
}
