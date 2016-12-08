#include "wfh_panel.h"
#include "jpegenc.h"
#if USE_CUDA
#include <cuda_runtime.h>
#endif
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#include <process.h>
#define usleep(a)  Sleep((a)/1000)
#else
#include <pthread.h>
#include <unistd.h>
#endif
using namespace std;

struct RGBA
{
    uint8_t     a;
    uint8_t     r;
    uint8_t     g;
    uint8_t     b;
};

static void genRgba(uint32_t* rgba, const int width, const int height, const int idx)
{
    const struct RGBA BAR_COLOUR[8] = {
        { 0,  255, 255, 255 },  // 100% White
        { 0,  255, 255,   0 },  // Yellow
        { 0,    0, 255, 255 },  // Cyan
        { 0,    0, 255,   0 },  // Green
        { 0,  255,   0, 255 },  // Magenta
        { 0,  255,   0,   0 },  // Red
        { 0,    0,   0, 255 },  // Blue
        { 0,    0,   0,   0 },  // Black
    }; 
    struct RGBA* data = (struct RGBA*)rgba;
    const int columnWidth = width / 8;

    for (int x = 0; x < width; x++) {
        const int col_idx = (x / columnWidth + idx) % 8;
        //fprintf(stdout, "col_idx: %d\n", col_idx);
        for (int y = 0; y < height; y++) {
            data[y * width + x] = BAR_COLOUR[col_idx];
            //data[y * width + x] = BAR_COLOUR[7];
        }
    }
}

struct WfhCtx
{
    int width;
    int height;
    int pitch;
    int state;
    ImageQueue images;
};

static void* supplier(void* user_data)
{
    int i= 0;
    WfhCtx* ctx = static_cast<WfhCtx*>(user_data);
    const int width     = ctx->width;
    const int height    = ctx->height;
    const int nPitch    = ctx->width*4;
    uint32_t *pRgba = NULL;
    uint8_t *apRgba = NULL;
    JpegEncoder *cudaj = NULL;


    // allocate host memory
    pRgba = new uint32_t[width*height];
    cudaj = JpegEncoder::CreateEncoder(2000, 2000);


    while(ctx->state == 0) {
        genRgba(pRgba, width, ctx->height, i++);
        uint32_t len = 0;
#if USE_CUDA
        // allocate rgba surface
        NPP_CHECK_CUDA(cudaMallocPitch(&apRgba, &nPitch, width*4, height));

        cudaMemcpy2D(apRgba, nPitch, pRgba, width*sizeof(uint32_t), width*sizeof(uint32_t), height, cudaMemcpyHostToDevice);
        uint8_t* buf = cudaj->Encode(apRgba, width, nPitch, height, len);
#else
        uint8_t* buf = cudaj->Encode((uint8_t*)pRgba, width, nPitch, height, len);
#endif

        WfhImage img;
        img.frame   = std::vector<uint8_t>(buf, buf+len);
        img.frame_no= i;
        img.meta.resize(100);
        sprintf((char*)img.meta.data(), "Frame %d", i);

        ctx->images.push_back(img);
        //printf("encode one:\n");
        if (ctx->images.full()) {
            ctx->images.pop_front();
        }
        usleep(30*1000);
    }


    if (cudaj) {
        delete cudaj;
    }

    if (pRgba) {
        delete pRgba;
    }

#if USE_CUDA
    if (apRgba) {
        cudaFree(apRgba);
    }
#endif

    ctx->state = 1;
 
    return 0;
}

int on_cmd(void* user_data, const char* cmd, std::string& msg)
{
    WfhCtx* wfh = static_cast<WfhCtx*>(user_data);

    printf("handling request: %s\n", cmd);

    msg = "";
    if (strcmp(cmd, "/") == 0) {
        msg = " <html> <head> <script src=mjpeg.js> </script> </head> <body> <br> <b>Remote Viewer</b> <br><br> <div class='content'> <p> <button id='send_button'>Open Camera</button> </p> <div id='messages'> </div> </div> <canvas id='mycanvas' width=1024 height=1280></canvas> <br><br> </body> </html>";
        return 1;
    } else if (strcmp(cmd, "/mjpeg.js") == 0) {
        msg = " window.onload = function() { var missed_heartbeats = 0; var c=document.getElementById('mycanvas'); var ctx=c.getContext('2d'); var ws = new WebSocket('ws://' + location.host + '/ws'); ws.binaryType = 'arraybuffer'; ws.onopen = function(ev)  { console.log(ev); }; ws.onerror = function(ev) { console.log(ev); }; ws.onclose = function(ev) { console.log(ev); }; ws.onmessage = function(ev) { missed_heartbeats = 0; if(ev.data instanceof ArrayBuffer) { var img = document.getElementById('myimg'); var blob  = new Blob([ev.data],{type: 'image/jpg'}); var img= new Image(); img.src = window.URL.createObjectURL(blob); img.onload = function (e) { ws.send('nextj'); ctx.drawImage(img,10,10); } } else { console.log(ev); document.getElementById('messages').innerHTML = ev.data; } }; document.getElementById('send_button').onclick = function(ev) { ws.send('nextj'); }; missed_heartbeats = 0; heartbeat_interval = setInterval(function() { try { missed_heartbeats++; if (missed_heartbeats >= 10) { throw new Error('Too many missed heartbeats.'); } ws.send('heartbeat'); } catch(e) { clearInterval(heartbeat_interval); heartbeat_interval = null; console.warn('Closing connection. Reason: ' + e.message); ws.close(); } }, 3000); }";
        return 1;
    }
 
    return 0;
}


int main(int argc, char **argv)
{
    WfhCtx ctx;
    char c = 0;
    ctx.width     = 1280;
    ctx.height    = 1024;
    ctx.pitch     = ctx.width * 4;
    ctx.state     = 0;

#ifdef _WIN32
    _beginthread((void (__cdecl *)(void *))supplier, 0, &ctx) == -1L ? -1 : 0;
#else
    pthread_attr_t tattr;
    pthread_t      supplierThread;

    uint32_t     status = pthread_attr_init(&tattr);
    if (!status) {
        status = pthread_attr_setdetachstate(&tattr, PTHREAD_CREATE_DETACHED);
    }
    if (!status) {
        status = pthread_create(&supplierThread, &tattr, supplier, &ctx);
    }
    pthread_attr_destroy(&tattr);
#endif

    void* wfh_ctx = start_panel(&ctx.images, on_cmd, &ctx);
    while(c!='e') {
        c = getchar();
    }
    stop_panel(wfh_ctx);
    ctx.state = -1;
    while(ctx.state != 1) {
        usleep(100*1000);
    }

#if USE_CUDA
    cudaDeviceReset();
#endif
    return EXIT_SUCCESS;
}
