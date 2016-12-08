
#include <stdlib.h>
#include <stdio.h>

#include "jpegenc.h"
#include "jpeglib.h"

class CpuJpeg: public JpegEncoder
{
public:
    CpuJpeg(const int maxw, const int maxh);
    virtual ~CpuJpeg();

    virtual bool        SetQuality(const float q);
    virtual uint8_t*    Encode(const uint8_t* apRgba, const int width, const int pitch, const int height, uint32_t& outlen);

protected:
    unsigned char* outbuffer;
    int            bufsize;
};

CpuJpeg::CpuJpeg(const int maxw, const int maxh)
    : JpegEncoder(), outbuffer(0), bufsize(0)
{
    bufsize = maxw*maxh*2*4;
    outbuffer = new unsigned char[bufsize];
}

CpuJpeg::~CpuJpeg()
{
    if (outbuffer) {
        delete outbuffer;
    }
    outbuffer = 0;
}

bool CpuJpeg::SetQuality(const float q)
{
    return true;
}

uint8_t* CpuJpeg::Encode(const uint8_t* raw_image, const int width, const int pitch, const int height, uint32_t& outlen)
{
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;

    /* this is a pointer to one row of image data */
    JSAMPROW row_pointer[1];

    cinfo.err = jpeg_std_error( &jerr );
    jpeg_create_compress(&cinfo);
    //jpeg_stdio_dest(&cinfo, outfile);
    unsigned long len = bufsize;
    jpeg_mem_dest(&cinfo, &outbuffer, &len);

    /* Setting the parameters of the output file here */
    cinfo.image_width = width;  
    cinfo.image_height = height;
    cinfo.input_components = 3; //bytes_per_pixel;
    cinfo.in_color_space = JCS_RGB;//color_space;
    /* default compression parameters, we shouldn't be worried about these */
    jpeg_set_defaults( &cinfo );
    /* Now do the compression .. */
    jpeg_start_compress( &cinfo, TRUE );
    /* like reading a file, this time write one row at a time */
    while( cinfo.next_scanline < cinfo.image_height ) {
        row_pointer[0] = (JSAMPROW)&raw_image[ cinfo.next_scanline * pitch];
        jpeg_write_scanlines( &cinfo, row_pointer, 1 );
    }
    /* similar to read file, clean up after we're done compressing */
    jpeg_finish_compress( &cinfo );
    jpeg_destroy_compress( &cinfo );

    outlen = len;
    return outbuffer;
}

JpegEncoder* JpegEncoder::CreateEncoder(const int maxw, const int maxh)
{
    return new CpuJpeg(maxw, maxh);
}
struct RGBA
{
    uint8_t     a;
    uint8_t     r;
    uint8_t     g;
    uint8_t     b;
};

#if TEST
static void genRgba(uint32_t* rgba, const int width, const int height, const int idx)
{
    const struct RGBA BAR_COLOUR[8] = {
        { 255, 255, 255 , 0},  // 100% White
        { 255, 255,   0 , 0},  // Yellow
        { 0, 255, 255 , 0},  // Cyan
        { 0, 255,   0 , 0},  // Green
        { 255,   0, 255 , 0},  // Magenta
        { 255,   0,   0 , 0},  // Red
        { 0,   0, 255 , 0},  // Blue
        { 0,   0,   0 , 0},  // Black
    }; 
    struct RGBA* data = (struct RGBA*)rgba;
    const int columnWidth = width / 8;

    for (int x = 0; x < width; x++) {
        const int col_idx = (x / columnWidth + idx) % 8;
        //fprintf(stdout, "col_idx: %d\n", col_idx);
        for (int y = 0; y < height; y++) {
            data[y * width + x] = BAR_COLOUR[col_idx];
        }
    }
}

int main(const int argc, const char* argv[])
{
    JpegEncoder *cj = JpegEncoder::CreateEncoder(2000, 2000);
    uint32_t *raw_image = new uint32_t[1000*1000];

    uint32_t outlen = 0;

    genRgba(raw_image, 1000, 1000, 0);

    uint8_t* outbuffer = cj->Encode((uint8_t*)raw_image, 1000, 1000*4, 1000, outlen);

    FILE *outfile = fopen( "output.jpg", "wb" );
    if ( outfile )
    {
        fwrite(outbuffer, 1, outlen, outfile);
        fclose( outfile );
    }

    printf("done\n");

    delete raw_image;
    delete cj;
    return 0;
}
#endif
