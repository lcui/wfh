#include "wfh_panel.h"
#include "mongoose/mongoose.h"

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <string.h>
#include <errno.h>

#ifdef _WIN32
#include <windows.h>
#define usleep(a)  Sleep((a)/1000)
#else
#include <unistd.h>
#endif

#define MAX_OPTIONS 100

struct StreamCtx
{
    ImageQueue*  images;
    RequestCb    req_cb;
    void*        user_data;
    struct mg_context* mg_ctx;
};

static char *sdup(const char *str)
{
    char *p;
    if ((p = (char *) malloc(strlen(str) + 1)) != NULL) {
        strcpy(p, str);
    }
    return p;
}


static void set_option(char **options, const char *name, const char *value)
{
    int i = 0;
    for (i = 0; i < MAX_OPTIONS - 3; i++) {
        if (options[i] == NULL) {
            options[i] = sdup(name);
            options[i + 1] = sdup(value);
            options[i + 2] = NULL;
            break;
        } else if (!strcmp(options[i], name)) {
            free(options[i + 1]);
            options[i + 1] = sdup(value);
            break;
        }
    }

    if (i == MAX_OPTIONS - 3) {
    }
}


static int log_message(const struct mg_connection *conn, const char *message) {
    (void) conn;
    printf("%s\n", message);
    return 0;
}

static int request_handler(struct mg_connection *conn)
{
    const struct mg_request_info *request_info = mg_get_request_info(conn);
    StreamCtx *ctx = static_cast<StreamCtx*>(request_info->user_data);
    int processed = 0;
    printf("received request: %s\n", request_info->uri);
    std::string msg;
    if(ctx->req_cb) {
        if (ctx->req_cb(ctx->user_data, request_info->uri, msg)) {
            mg_printf(conn, "%s", "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n");
            mg_write(conn, msg.c_str(), msg.size());
            processed = 1;
        }
    } else {
        processed = 0;
    }
    return processed;
}

static int on_ws_connect(const struct mg_connection *conn)
{
    printf("ws connction request: %p\n", conn);

    return 0;
}

static int on_ws_recv_enc(struct mg_connection *conn, int bits, char *data, size_t data_len)
{
    static int count = -1;
    std::string msg;
    struct mg_request_info *request_info = mg_get_request_info(conn);
    StreamCtx *ctx = static_cast<StreamCtx*>(request_info->user_data);

    data[data_len] = '\0';
    //printf("ws recv: %s\n", data);
    if(strcmp(data, "nextj")==0) {
        while(ctx->images->size() == 0) {
            usleep(100);
        }
        WfhImage& img = ctx->images->back();

        while (img.frame_no <= count) {
            usleep(100);
            img = ctx->images->back();
        }
        count = img.frame_no;

        uint32_t len = img.frame.size();
        const char* buf = (const char*)img.frame.data();

        mg_websocket_write(conn, WEBSOCKET_OPCODE_BINARY, buf, len);
        if (img.meta.size()) {
            mg_websocket_write(conn, WEBSOCKET_OPCODE_TEXT,
                               (const char*)img.meta.data(), img.meta.size());
        }
    } else if(strcmp(data, "heartbeat")==0) {
    } else if(ctx->req_cb && ctx->req_cb(ctx->user_data, data, msg)) {
        mg_websocket_write(conn, WEBSOCKET_OPCODE_TEXT, msg.c_str(), msg.size());
    }

    bits = bits;
    return 1;
}


void* start_panel(ImageQueue* images, RequestCb cb, void* user_data)
{
    StreamCtx* ctx = new StreamCtx;
    ctx->images = images;
    ctx->req_cb = cb;
    ctx->user_data = user_data;

    struct mg_callbacks callbacks;
    char *options[MAX_OPTIONS];
    int i;

    options[0] = NULL;
    set_option(options, "document_root", ".");

    // Setup signal handler: quit on Ctrl-C
    //signal(SIGTERM, signal_handler);
    //signal(SIGINT, signal_handler);

    // Start Mongoose
    memset(&callbacks, 0, sizeof(callbacks));
    callbacks.log_message = &log_message;
    callbacks.begin_request = request_handler;
    callbacks.websocket_connect = on_ws_connect;
    callbacks.websocket_data = on_ws_recv_enc;

    ctx->mg_ctx = mg_start(&callbacks, ctx, (const char **) options);
    for (i = 0; options[i] != NULL; i++) {
        free(options[i]);
    }

    if (ctx->mg_ctx) {
        printf("started on port(s) %s\n", mg_get_option(ctx->mg_ctx, "listening_ports"));
    } else {
        printf("Failed to start streaming\n");
    }

    return ctx;
}

int stop_panel(void* handle)
{
    StreamCtx* ctx = static_cast<StreamCtx*>(handle);

    if (ctx) {
        if (ctx->mg_ctx) {
            printf("Exiting, waiting for all threads to finish...\n");
            mg_stop(ctx->mg_ctx);
            fflush(stdout);
        }

        delete ctx;
        return 0;
    }

    return -1;
}

ImageQueue::ImageQueue()
    : start(0),
      end(0)
{
}

int ImageQueue::size() const
{
    int sz = start - end;
    if (sz < 0) {
        sz += QUEUE_SIZE;
    }

    return sz;
}

bool ImageQueue::full() const
{
    return (size() >= QUEUE_SIZE-1);
}

bool ImageQueue::empty() const
{
    return (start == end);
}

WfhImage& ImageQueue::back()
{
    return frames[start];
}

WfhImage& ImageQueue::front()
{
    return frames[end];
}


bool ImageQueue::push_back(const WfhImage& img)
{
    if (full()) {
        return false;
    }

    int np = start + 1;
    if (np >= QUEUE_SIZE) {
        np = 0;
    }

    frames[np] = img;
    start = np;

    return true;
}

WfhImage* ImageQueue::pop_front()
{
    if (empty()) {
        return NULL;
    }
    WfhImage& img = frames[end];
    int ne = end + 1;
    if (ne >= QUEUE_SIZE) {
        ne = 0;
    }
    end = ne;

    return &img;
}

