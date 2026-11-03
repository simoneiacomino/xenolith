#define _POSIX_C_SOURCE 200809L
#include "../conversation.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(void) {
    char root[]="/tmp/xe-steering-history-XXXXXX";assert(mkdtemp(root));
    conversation_store *store=NULL;conversation *c=NULL;conversation_id id;
    assert(conversation_store_open(&store,root)==CONVERSATION_OK);
    assert(conversation_create(store,&c,&id)==CONVERSATION_OK);
    conversation_block block={CONVERSATION_BLOCK_TEXT,"fixture",7};
    int32_t primary[]={2,3},alternate[]={2,4,5},raw[]={10,11,12,13},user[]={20};
    assert(conversation_append_message_variants(c,CONVERSATION_ROLE_SYSTEM,&block,1,
        "a",1,primary,2,"b",1,alternate,3,NULL,0,CONVERSATION_REASONING_NONE)==CONVERSATION_OK);
    uint64_t system=conversation_event_count(c)-1;
    assert(conversation_project(c,1,CONVERSATION_REASONING_DISCARD)==CONVERSATION_OK);
    uint8_t artifact[32]={1};
    assert(conversation_append_steering(c,0,artifact,-0.5f)==CONVERSATION_OK);
    assert(conversation_steering_count(c)==1);
    conversation_generation generation={0}; generation.generation_id=17;
    generation.settings.sampler_abi=CONVERSATION_SAMPLER_ABI;
    generation.settings.top_p=1;generation.settings.max_tokens=64;
    assert(conversation_append_generation_started(c,&generation)==CONVERSATION_OK);
    assert(conversation_append_generation_result_variants(c,17,CONVERSATION_STOP_EOT_SAMPLED,1,
        &block,1,"a",1,primary,2,"b",1,alternate,3,"reason",6,CONVERSATION_REASONING_NATURAL,
        "raw",3,raw,4)==CONVERSATION_OK);
    uint64_t answer=conversation_event_count(c)-1;
    assert(conversation_project(c,1,CONVERSATION_REASONING_DISCARD)==CONVERSATION_OK);
    uint64_t n;const int32_t *tokens=conversation_tokens(c,&n);
    assert(n==7&&!memcmp(tokens,alternate,3*sizeof *tokens)&&!memcmp(tokens+3,raw,4*sizeof *tokens));
    uint8_t off[32]={0};
    assert(conversation_append_steering(c,7,off,0)==CONVERSATION_OK);
    assert(conversation_append_message(c,CONVERSATION_ROLE_USER,&block,1,"u",1,user,1)==CONVERSATION_OK);
    assert(conversation_project(c,0,CONVERSATION_REASONING_DISCARD)==CONVERSATION_OK);
    tokens=conversation_tokens(c,&n);assert(n==8&&tokens[7]==20);
    assert(conversation_commit(c)==CONVERSATION_OK);conversation_close(c);
    assert(conversation_open(store,&id,&c)==CONVERSATION_OK);
    assert(conversation_steering_count(c)==2);
    assert(conversation_project(c,0,CONVERSATION_REASONING_DISCARD)==CONVERSATION_OK);
    tokens=conversation_tokens(c,&n);assert(n==8&&tokens[2]==5&&tokens[6]==13&&tokens[7]==20);
    assert(conversation_append_rewind(c,answer)==CONVERSATION_OK);
    assert(conversation_steering_count(c)==1);
    assert(conversation_project(c,1,CONVERSATION_REASONING_DISCARD)==CONVERSATION_OK);
    conversation_tokens(c,&n);assert(n==7);
    assert(conversation_append_rewind(c,system)==CONVERSATION_OK);assert(conversation_steering_count(c)==0);
    assert(conversation_commit(c)==CONVERSATION_OK);conversation_close(c);
    assert(conversation_open(store,&id,&c)==CONVERSATION_OK);assert(conversation_steering_count(c)==0);
    conversation_close(c);conversation_delete(store,&id);conversation_store_close(store);
    char path[512];snprintf(path,sizeof path,"%s/sessions",root);rmdir(path);rmdir(root);
    puts("steering history: frozen context, reasoning retention, restart and branch rewind PASS");
}
