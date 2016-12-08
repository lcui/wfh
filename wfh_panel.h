#ifndef WFH_PANEL_H_
#define WFH_PANEL_H_
#include <vector>
#include <list>
#include <string>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef int (*RequestCb)(void* user_data, const char* request, std::string& msg);
struct WfhImage {
    std::vector<uint8_t> frame;
    int         frame_no;
    std::vector<uint8_t> meta;
};
class ImageQueue {
public:
    ImageQueue();
    int         size() const;
    bool        full() const;
    bool        empty() const;
    bool        push_back(const WfhImage& img); // push to start
    WfhImage&   back();                         // get start frame
    WfhImage&   front();                        // get end frame
    WfhImage*   pop_front();                    // pop from end

protected:
    enum {QUEUE_SIZE = 100};
    WfhImage    frames[QUEUE_SIZE];
    int         start;
    int         end;
};

void* start_panel(ImageQueue* images,
                  RequestCb cb, void* user_data);
int stop_panel(void* ctx);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif
