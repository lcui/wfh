#include "jpegenc.h"

#include <npp.h>
#include <cuda_runtime.h>

#include <algorithm>

#include "Exceptions.h"
#include <math.h>

#include <string.h>
#include <iostream>
#include <cstdio>
#include <stdint.h>

#include <helper_string.h>
#include <helper_cuda.h>

using namespace std;

struct FrameHeader
{
    uint8_t nSamplePrecision;
    uint16_t nHeight;
    uint16_t nWidth;
    uint8_t nComponents;
    uint8_t aComponentIdentifier[3];
    uint8_t aSamplingFactors[3];
    uint8_t aQuantizationTableSelector[3];
};

struct ScanHeader
{
    uint8_t nComponents;
    uint8_t aComponentSelector[3];
    uint8_t aHuffmanTablesSelector[3];
    uint8_t nSs;
    uint8_t nSe;
    uint8_t nA;
};

struct QuantizationTable
{
    uint8_t nPrecisionAndIdentifier;
    uint8_t aTable[64];
};

struct HuffmanTable
{
    uint8_t nClassAndIdentifier;
    uint8_t aCodes[16];
    uint8_t aTable[256];
};


static inline int DivUp(int x, int d)
{
    return (x + d - 1) / d;
}

template<class T>
static void writeBigEndian(uint8_t *pData, T value)
{
    uint8_t *pValue = reinterpret_cast<uint8_t *>(&value);
    std::reverse_copy(pValue, pValue+sizeof(T), pData);
}

template<typename T>
static void writeAndAdvance(uint8_t *&pData, T nElement)
{
    writeBigEndian<T>(pData, nElement);
    pData += sizeof(T);
}


static void writeMarker(uint8_t nMarker, uint8_t *&pData)
{
    *pData++ = 0x0ff;
    *pData++ = nMarker;
}

static void writeJFIFTag(uint8_t *&pData)
{
    const char JFIF_TAG[] =
    {
        0x4a, 0x46, 0x49, 0x46, 0x00,
        0x01, 0x02,
        0x00,
        0x00, 0x01, 0x00, 0x01,
        0x00, 0x00
    };

    writeMarker(0x0e0, pData);
    writeAndAdvance<uint16_t>(pData, sizeof(JFIF_TAG) + sizeof(uint16_t));
    memcpy(pData, JFIF_TAG, sizeof(JFIF_TAG));
    pData += sizeof(JFIF_TAG);
}

static void writeFrameHeader(const FrameHeader &header, uint8_t *&pData)
{
    uint8_t aTemp[128];
    uint8_t *pTemp = aTemp;

    writeAndAdvance<uint8_t>(pTemp, header.nSamplePrecision);
    writeAndAdvance<uint16_t>(pTemp, header.nHeight);
    writeAndAdvance<uint16_t>(pTemp, header.nWidth);
    writeAndAdvance<uint8_t>(pTemp, header.nComponents);

    for (int c=0; c<header.nComponents; ++c)
    {
        writeAndAdvance<uint8_t>(pTemp,header.aComponentIdentifier[c]);
        writeAndAdvance<uint8_t>(pTemp,header.aSamplingFactors[c]);
        writeAndAdvance<uint8_t>(pTemp,header.aQuantizationTableSelector[c]);
    }

    uint16_t nLength = (uint16_t)(pTemp - aTemp);

    writeMarker(0x0C0, pData);
    writeAndAdvance<uint16_t>(pData, nLength + 2);
    memcpy(pData, aTemp, nLength);
    pData += nLength;
}


static void writeScanHeader(const ScanHeader &header, uint8_t *&pData)
{
    uint8_t aTemp[128];
    uint8_t *pTemp = aTemp;

    writeAndAdvance<uint8_t>(pTemp, header.nComponents);

    for (int c=0; c<header.nComponents; ++c) {
        writeAndAdvance<uint8_t>(pTemp,header.aComponentSelector[c]);
        writeAndAdvance<uint8_t>(pTemp,header.aHuffmanTablesSelector[c]);
    }

    writeAndAdvance<uint8_t>(pTemp,  header.nSs);
    writeAndAdvance<uint8_t>(pTemp,  header.nSe);
    writeAndAdvance<uint8_t>(pTemp,  header.nA);

    uint16_t nLength = (uint16_t)(pTemp - aTemp);

    writeMarker(0x0DA, pData);
    writeAndAdvance<uint16_t>(pData, nLength + 2);
    memcpy(pData, aTemp, nLength);
    pData += nLength;
}


static void writeQuantizationTable(const QuantizationTable &table, uint8_t *&pData)
{
    writeMarker(0x0DB, pData);
    writeAndAdvance<uint16_t>(pData, sizeof(QuantizationTable) + 2);
    memcpy(pData, &table, sizeof(QuantizationTable));
    pData += sizeof(QuantizationTable);
}

static void writeHuffmanTable(const HuffmanTable &table, uint8_t *&pData)
{
    writeMarker(0x0C4, pData);

    // Number of Codes for Bit Lengths [1..16]
    int nCodeCount = 0;

    for (int i = 0; i < 16; ++i) {
        nCodeCount += table.aCodes[i];
    }

    writeAndAdvance<uint16_t>(pData, 17 + nCodeCount + 2);
    memcpy(pData, &table, 17 + nCodeCount);
    pData += 17 + nCodeCount;
}


static bool printfNPPinfo(int cudaVerMajor, int cudaVerMinor)
{
    const NppLibraryVersion *libVer   = nppGetLibVersion();

    printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor, libVer->build);

    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("  CUDA Driver  Version: %d.%d\n", driverVersion/1000, (driverVersion%100)/10);
    printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion/1000, (runtimeVersion%100)/10);

    bool bVal = checkCudaCapabilities(cudaVerMajor, cudaVerMinor);
    return bVal;
}

static const HuffmanTable defaultHuffmanTable[4] = {
    {0,{0,	1,	5,	1,	1,	1,	1,	1,	1,	0,	0,	0,	0,	0,	0,	0}, {0,	1,	2,	3,	4,	5,	6,	7,	8,	9,	10,	11,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0}},
    {1,{0,	3,	1,	1,	1,	1,	1,	1,	1,	1,	1,	0,	0,	0,	0,	0}, {0,	1,	2,	3,	4,	5,	6,	7,	8,	9,	10,	11,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0}},
    {16,{0,	2,	1,	3,	3,	2,	4,	3,	5,	5,	4,	4,	0,	0,	1,	125}, {1,	2,	3,	0,	4,	17,	5,	18,	33,	49,	65,	6,	19,	81,	97,	7,	34,	113,	20,	50,	129,	145,	161,	8,	35,	66,	177,	193,	21,	82,	209,	240,	36,	51,	98,	114,	130,	9,	10,	22,	23,	24,	25,	26,	37,	38,	39,	40,	41,	42,	52,	53,	54,	55,	56,	57,	58,	67,	68,	69,	70,	71,	72,	73,	74,	83,	84,	85,	86,	87,	88,	89,	90,	99,	100,	101,	102,	103,104,	105,	106,	115,	116,	117,	118,	119,	120,	121,	122,	131,	132,	133,	134,	135,	136,	137,	138,	146,	147,	148,	149,	150,	151,	152,153,	154,	162,	163,	164,	165,	166,	167,	168,	169,	170,	178,	179,	180,	181,	182,	183,	184,	185,	186,	194,	195,	196,	197,	198,	199,200,	201,	202,	210,	211,	212,	213,	214,	215,	216,	217,	218,	225,	226,	227,	228,	229,	230,	231,	232,	233,	234,	241,	242,	243,	244,245,	246,	247,	248,	249,	250,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0}},
    {17,{0,	2,	1,	2,	4,	4,	3,	4,	7,	5,	4,	4,	0,	1,	2,	119}, {0,	1,	2,	3,	17,	4,	5,	33,	49,	6,	18,	65,	81,	7,	97,	113,	19,	34,	50,	129,	8,	20,	66,	145,	161,	177,193,	9,	35,	51,	82,	240,	21,	98,	114,	209,	10,	22,	36,	52,	225,	37,	241,	23,	24,	25,	26,	38,	39,	40,	41,	42,	53,	54,	55,	56,	57,	58,	67,	68,	69,	70,	71,	72,	73,	74,	83,	84,	85,	86,	87,	88,	89,	90,	99,	100,	101,	102,103,	104,	105,	106,	115,	116,	117,	118,	119,	120,	121,	122,	130,	131,	132,	133,	134,	135,	136,	137,	138,	146,	147,	148,	149,	150,151,	152,	153,	154,	162,	163,	164,	165,	166,	167,	168,	169,	170,	178,	179,	180,	181,	182,	183,	184,	185,	186,	194,	195,	196,	197,198,	199,	200,	201,	202,	210,	211,	212,	213,	214,	215,	216,	217,	218,	226,	227,	228,	229,	230,	231,	232,	233,	234,	242,	243,	244,245,	246,	247,	248,	249,	250,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0}}
};

static const QuantizationTable defaultQTable[2] = {
    {0,{
           16,  11,  12,  14,  12,  10,  16,  14,
           13,  14,  18,  17,  16,  19,  24,  40,
           26,  24,  22,  22,  24,  49,  35,  37,
           29,  40,  58,  51,  61,  60,  57,  51,
           56,  55,  64,  72,  92,  78,  64,  68,
           87,  69,  55,  56,  80, 109,  81,  87,
           95,  98, 103, 104, 103,  62,  77, 113,
           121, 112, 100, 120,  92, 101, 103,  99}},
    {1,{
           17,  18,  18,  24,  21,  24,  47,  26,
           26,  47,  99,  66,  56,  66,  99,  99,
           99,  99,  99,  99,  99,  99,  99,  99,
           99,  99,  99,  99,  99,  99,  99,  99,
           99,  99,  99,  99,  99,  99,  99,  99,
           99,  99,  99,  99,  99,  99,  99,  99,
           99,  99,  99,  99,  99,  99,  99,  99,
           99,  99,  99,  99,  99,  99,  99,  99}},
};

class CudaJpeg : public JpegEncoder
{
public:
    CudaJpeg(const int maxw, const int maxh);
    virtual ~CudaJpeg();

    virtual bool        SetQuality(const float q);
    virtual uint8_t*    Encode(const uint8_t* apRgba, const int width, const int pitch, const int height, uint32_t& outlen);

protected:
    static const int CUDAJPEG_OUTBUF_SIZE    = (4<<20)*2;

protected:
    float       mQuality;
    uint8_t*    mOutputBuf;
    Npp8u *     mpGpuYCrCbImage[3];
    Npp32s      mGpuYCrCbImagePitch[3];
    NppiDCTState *pDCTState;
    Npp8u *pdQuantizationTables;    // QTable on GPU
    Npp16s *apdDCT[3];
    Npp8u *pdScan;
    Npp8u *pJpegEncoderTemp;
    int         mMaxW;
    int         mMaxH;

    QuantizationTable *aQuantizationTables;
    static const FrameHeader defaultFrameHeader;
    static const ScanHeader defaultScanHeader;
    Npp32s aDCTStep[3];

    NppiEncodeHuffmanSpec *apHuffmanDCTable[3];
    NppiEncodeHuffmanSpec *apHuffmanACTable[3];
};

// default max resolution to 2000x2000
const FrameHeader CudaJpeg::defaultFrameHeader = {8, 2000, 2000, 3, {1, 2, 3}, {34, 17, 17}, {0, 1, 1}};
const ScanHeader CudaJpeg::defaultScanHeader = {3, {1, 2, 3}, {0, 17, 17}, 0, 63, 0};

CudaJpeg::CudaJpeg(const int maxw, const int maxh)
{
    const FrameHeader &oFrameHeader = defaultFrameHeader;;
    const ScanHeader &oScanHeader = defaultScanHeader;

    mMaxW = maxw;
    mMaxH = maxh;

    for(int i=0; i<3; i++) {
        mpGpuYCrCbImage[i] = 0;
        apdDCT[i] = 0;
        apHuffmanDCTable[i] = 0;
        apHuffmanACTable[i] = 0;
    }

    NPP_CHECK_NPP(nppiDCTInitAlloc(&pDCTState));
    cudaMalloc(&pdQuantizationTables, 64 * 2);
    mOutputBuf = new uint8_t[CUDAJPEG_OUTBUF_SIZE];
    NPP_CHECK_CUDA(cudaMalloc(&pdScan, CUDAJPEG_OUTBUF_SIZE));

    Npp32s nTempSize;
    NppiSize SrcSize = {mMaxW, mMaxH};
    NPP_CHECK_NPP(nppiEncodeHuffmanGetSize(SrcSize, 3, &nTempSize));
    NPP_CHECK_CUDA(cudaMalloc(&pJpegEncoderTemp, nTempSize));

    aQuantizationTables = new QuantizationTable[2];

    NppiSize aSrcSize[3];

    int nMCUBlocksH = 2;
    int nMCUBlocksV = 2;

    for (int i=0; i < oFrameHeader.nComponents; ++i) {
        NppiSize oBlocks;
        NppiSize oBlocksPerMCU = { oFrameHeader.aSamplingFactors[i]  >> 4, oFrameHeader.aSamplingFactors[i] & 0x0f};

        oBlocks.width = (int)ceil((mMaxW + 7)/8  * static_cast<float>(oBlocksPerMCU.width)/nMCUBlocksH);
        oBlocks.width = DivUp(oBlocks.width, oBlocksPerMCU.width) * oBlocksPerMCU.width;

        oBlocks.height = (int)ceil((mMaxH+7)/8 * static_cast<float>(oBlocksPerMCU.height)/nMCUBlocksV);
        oBlocks.height = DivUp(oBlocks.height, oBlocksPerMCU.height) * oBlocksPerMCU.height;

        aSrcSize[i].width = oBlocks.width * 8;
        aSrcSize[i].height = oBlocks.height * 8;

        // Allocate Memory
        size_t nPitch;
        NPP_CHECK_CUDA(cudaMallocPitch(&apdDCT[i], &nPitch, oBlocks.width * 64 * sizeof(Npp16s), oBlocks.height));
        aDCTStep[i] = static_cast<Npp32s>(nPitch);

        NPP_CHECK_CUDA(cudaMallocPitch(&mpGpuYCrCbImage[i], &nPitch, aSrcSize[i].width, aSrcSize[i].height));
        mGpuYCrCbImagePitch[i] = static_cast<Npp32s>(nPitch);
        cudaMemset(mpGpuYCrCbImage[i], 128, nPitch*aSrcSize[i].height);
    }

    // Allocate Huffman Table
    const HuffmanTable *pHuffmanDCTables = &defaultHuffmanTable[0];
    const HuffmanTable *pHuffmanACTables = &defaultHuffmanTable[2];
    for (int i = 0; i < 3; ++i) {
        nppiEncodeHuffmanSpecInitAlloc_JPEG(pHuffmanDCTables[(oScanHeader.aHuffmanTablesSelector[i] >> 4)].aCodes, nppiDCTable, &apHuffmanDCTable[i]);
        nppiEncodeHuffmanSpecInitAlloc_JPEG(pHuffmanACTables[(oScanHeader.aHuffmanTablesSelector[i] & 0x0f)].aCodes, nppiACTable, &apHuffmanACTable[i]);
    }

    SetQuality(0.8);
}

CudaJpeg::~CudaJpeg()
{
    nppiDCTFree(pDCTState);
    cudaFree(pdQuantizationTables);

    if (aQuantizationTables) {
        delete aQuantizationTables;
        aQuantizationTables = NULL;
    }

    for (int i = 0; i < 3; ++i) {
        cudaFree(apdDCT[i]);
        cudaFree(mpGpuYCrCbImage[i]);
        if (apHuffmanDCTable[i]) {
            nppiEncodeHuffmanSpecFree_JPEG(apHuffmanDCTable[i]);
        }
        if (apHuffmanACTable[i]) {
            nppiEncodeHuffmanSpecFree_JPEG(apHuffmanACTable[i]);
        }
    }

    cudaFree(pJpegEncoderTemp);
    cudaFree(pdScan);

    delete mOutputBuf;
    mOutputBuf = NULL;
}

bool CudaJpeg::SetQuality(const float q)
{
    if (q <= 1.f && q > 0.f) {
        float old = mQuality;
        mQuality = q;

        const float quality = 100 * q;
        const float s = (quality < 50) ? (5000 / quality) : (200 - (2 * quality));
        for (int j=0; j<2; j++) {
            aQuantizationTables[j] = defaultQTable[j];
            uint8_t* table_raw = &aQuantizationTables[j].aTable[0];
            for ( int i = 0; i < 64; i++ ) {
                int value = (s * (int)table_raw[i] + 50) / 100;
                if ( value == 0 ) {
                    value = 1;
                }
                if ( value > 255 ) {
                    value = 255;
                }
                table_raw[i] = (uint8_t)value;
            }

            // Copy DCT coefficients and Quantization Tables from host to device
            NPP_CHECK_CUDA(cudaMemcpyAsync(pdQuantizationTables + j * 64, table_raw, 64, cudaMemcpyHostToDevice));
        }

        return true;
    } else {
        return false;
    }
}

uint8_t* CudaJpeg::Encode(const uint8_t* apRgba, const int width, const int pitch, const int height, uint32_t& outlen)
{
    if (width > mMaxW || height > mMaxH) {
        fprintf(stderr, "Error: input surface(%dx%d) is beyond maximum resolution(%dx%d).\n", width, height, mMaxW, mMaxH);
        return NULL;
    }

    FrameHeader oFrameHeader = defaultFrameHeader;;
    ScanHeader oScanHeader = defaultScanHeader;
    NppiSize aSrcSize[3];
    const HuffmanTable *pHuffmanDCTables = &defaultHuffmanTable[0];
    const HuffmanTable *pHuffmanACTables = &defaultHuffmanTable[2];

    oFrameHeader.nWidth = width;
    oFrameHeader.nHeight = height;

    // based on oFrameHeader
    const int nMCUBlocksH = 2;
    const int nMCUBlocksV = 2;

    for (int i=0; i < oFrameHeader.nComponents; ++i) {
        NppiSize oBlocks;
        NppiSize oBlocksPerMCU = { oFrameHeader.aSamplingFactors[i]  >> 4, oFrameHeader.aSamplingFactors[i] & 0x0f};

        oBlocks.width = (int)ceil((oFrameHeader.nWidth + 7)/8  *
                                  static_cast<float>(oBlocksPerMCU.width)/nMCUBlocksH);
        oBlocks.width = DivUp(oBlocks.width, oBlocksPerMCU.width) * oBlocksPerMCU.width;

        oBlocks.height = (int)ceil((oFrameHeader.nHeight+7)/8 *
                                   static_cast<float>(oBlocksPerMCU.height)/nMCUBlocksV);
        oBlocks.height = DivUp(oBlocks.height, oBlocksPerMCU.height) * oBlocksPerMCU.height;

        aSrcSize[i].width = oBlocks.width * 8;
        aSrcSize[i].height = oBlocks.height * 8;

        //printf("aSrcSize[%d]: %dx%d\n", i, aSrcSize[i].width, aSrcSize[i].height);
    }

    // Convert RGBA to YCrCb
    nppiBGRToYCrCb420_8u_AC4P3R(apRgba, pitch, mpGpuYCrCbImage, mGpuYCrCbImagePitch, aSrcSize[0]);


    // Forward DCT
    for (int i = 0; i < 3; ++i) {
        NPP_CHECK_NPP(nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R_NEW(mpGpuYCrCbImage[i], mGpuYCrCbImagePitch[i],
                                                              apdDCT[i], aDCTStep[i],
                                                              pdQuantizationTables + oFrameHeader.aQuantizationTableSelector[i] * 64,
                                                              aSrcSize[i],
                                                              pDCTState));
    }

    // Huffman Encoding
    Npp32s nScanLength;
    NPP_CHECK_NPP(nppiEncodeHuffmanScan_JPEG_8u16s_P3R(apdDCT, aDCTStep,
                                                       0, oScanHeader.nSs, oScanHeader.nSe, oScanHeader.nA >> 4, oScanHeader.nA & 0x0f,
                                                       pdScan, &nScanLength,
                                                       apHuffmanDCTable,
                                                       apHuffmanACTable,
                                                       aSrcSize,
                                                       pJpegEncoderTemp));

    // Write Simple JPEG
    uint8_t *pDstOutput = mOutputBuf;

    writeMarker(0x0D8, pDstOutput);
    writeJFIFTag(pDstOutput);
    writeQuantizationTable(aQuantizationTables[0], pDstOutput);
    writeQuantizationTable(aQuantizationTables[1], pDstOutput);
    writeFrameHeader(oFrameHeader, pDstOutput);
    writeHuffmanTable(pHuffmanDCTables[0], pDstOutput);
    writeHuffmanTable(pHuffmanACTables[0], pDstOutput);
    writeHuffmanTable(pHuffmanDCTables[1], pDstOutput);
    writeHuffmanTable(pHuffmanACTables[1], pDstOutput);
    writeScanHeader(oScanHeader, pDstOutput);
    NPP_CHECK_CUDA(cudaMemcpy(pDstOutput, pdScan, nScanLength, cudaMemcpyDeviceToHost));
    pDstOutput += nScanLength;
    writeMarker(0x0D9, pDstOutput);

    outlen = size_t(pDstOutput - mOutputBuf);
    return mOutputBuf;
}

JpegEncoder* JpegEncoder::CreateEncoder(const int maxw, const int maxh)
{
    return new CudaJpeg(maxw, maxh);
}
