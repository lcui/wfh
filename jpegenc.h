#ifndef WFH_JPEG_H
#define WFH_JPEG_H
#include <stdint.h>

class JpegEncoder
{
public:
    virtual ~JpegEncoder(){};

    virtual bool        SetQuality(const float q) = 0;
    virtual uint8_t*    Encode(const uint8_t* apRgba, const int width, const int pitch, const int height, uint32_t& outlen) = 0;

    static JpegEncoder* CreateEncoder(const int maxw, const int maxh);
};

#endif // WFH_JPEG_H
