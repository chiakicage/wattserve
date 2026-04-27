#include <cuda.h>
#include <cupti.h>
#include <cupti_callbacks.h>
#include <cupti_events.h>
#include <cupti_result.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

struct EventGroupState {
    CUpti_EventGroup group = nullptr;
    CUpti_EventID event_id = 0;
    std::string name;
    uint32_t instance_count = 0;
};

CUpti_SubscriberHandle g_subscriber = nullptr;
CUcontext g_context = nullptr;
CUdevice g_device = 0;
std::vector<std::string> g_event_names;
std::vector<EventGroupState> g_event_groups;
std::atomic<bool> g_running{false};
std::thread g_thread;
std::mutex g_mutex;
std::ofstream g_csv;
std::chrono::steady_clock::time_point g_start_time;
int g_interval_ms = 10;

const char *result_name(CUptiResult result) {
    const char *text = "unknown";
    cuptiGetResultString(result, &text);
    return text;
}

bool check_cupti(CUptiResult result, const char *call) {
    if (result == CUPTI_SUCCESS) {
        return true;
    }
    std::fprintf(
        stderr,
        "CUPTI event monitor: %s failed: %s\n",
        call,
        result_name(result));
    return false;
}

bool check_cuda(CUresult result, const char *call) {
    if (result == CUDA_SUCCESS) {
        return true;
    }
    const char *name = "unknown";
    cuGetErrorName(result, &name);
    std::fprintf(
        stderr,
        "CUPTI event monitor: %s failed: %s\n",
        call,
        name);
    return false;
}

std::vector<std::string> split_events(const char *value) {
    std::vector<std::string> events;
    std::stringstream stream(value ? value : "inst_executed");
    std::string item;
    while (std::getline(stream, item, ',')) {
        size_t begin = item.find_first_not_of(" \t\n\r");
        size_t end = item.find_last_not_of(" \t\n\r");
        if (begin != std::string::npos && end != std::string::npos) {
            events.push_back(item.substr(begin, end - begin + 1));
        }
    }
    if (events.empty()) {
        events.push_back("inst_executed");
    }
    return events;
}

std::string output_path() {
    const char *path = std::getenv("CUPTI_EVENT_MONITOR_CSV");
    return path && path[0] ? path : "cupti_active_event_monitor.csv";
}

int interval_ms() {
    const char *value = std::getenv("CUPTI_EVENT_MONITOR_INTERVAL_MS");
    if (!value || !value[0]) {
        return 10;
    }
    int parsed = std::atoi(value);
    return parsed > 0 ? parsed : 10;
}

bool make_context_current(CUcontext context, CUcontext *previous) {
    if (!context) {
        return false;
    }
    if (!check_cuda(cuCtxGetCurrent(previous), "cuCtxGetCurrent")) {
        return false;
    }
    if (*previous == context) {
        return true;
    }
    return check_cuda(cuCtxSetCurrent(context), "cuCtxSetCurrent");
}

void restore_context(CUcontext previous) {
    check_cuda(cuCtxSetCurrent(previous), "cuCtxSetCurrent(previous)");
}

void destroy_event_groups() {
    for (EventGroupState &state : g_event_groups) {
        if (state.group) {
            check_cupti(cuptiEventGroupDisable(state.group), "cuptiEventGroupDisable");
            check_cupti(cuptiEventGroupDestroy(state.group), "cuptiEventGroupDestroy");
        }
    }
    g_event_groups.clear();
}

bool create_event_groups() {
    uint32_t profile_all = 1;
    destroy_event_groups();
    if (!check_cupti(
            cuptiSetEventCollectionMode(g_context, CUPTI_EVENT_COLLECTION_MODE_CONTINUOUS),
            "cuptiSetEventCollectionMode")) {
        return false;
    }

    for (const std::string &name : g_event_names) {
        EventGroupState state;
        state.name = name;
        if (!check_cupti(
                cuptiEventGetIdFromName(g_device, name.c_str(), &state.event_id),
                "cuptiEventGetIdFromName")) {
            continue;
        }
        if (!check_cupti(
                cuptiEventGroupCreate(g_context, &state.group, 0),
                "cuptiEventGroupCreate")) {
            continue;
        }
        if (!check_cupti(
                cuptiEventGroupAddEvent(state.group, state.event_id),
                "cuptiEventGroupAddEvent")) {
            check_cupti(cuptiEventGroupDestroy(state.group), "cuptiEventGroupDestroy");
            continue;
        }
        check_cupti(
            cuptiEventGroupSetAttribute(
                state.group,
                CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES,
                sizeof(profile_all),
                &profile_all),
            "cuptiEventGroupSetAttribute(PROFILE_ALL_DOMAIN_INSTANCES)");
        size_t size = sizeof(state.instance_count);
        check_cupti(
            cuptiEventGroupGetAttribute(
                state.group,
                CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT,
                &size,
                &state.instance_count),
            "cuptiEventGroupGetAttribute(INSTANCE_COUNT)");
        if (!check_cupti(cuptiEventGroupEnable(state.group), "cuptiEventGroupEnable")) {
            check_cupti(cuptiEventGroupDestroy(state.group), "cuptiEventGroupDestroy");
            continue;
        }
        g_event_groups.push_back(state);
    }

    if (g_event_groups.empty()) {
        std::fprintf(stderr, "CUPTI event monitor: no event groups enabled\n");
        return false;
    }
    return true;
}

void sample_loop() {
    CUcontext previous = nullptr;
    if (!make_context_current(g_context, &previous)) {
        return;
    }

    while (g_running.load()) {
        auto now = std::chrono::steady_clock::now();
        double elapsed_seconds =
            std::chrono::duration<double>(now - g_start_time).count();
        for (EventGroupState &state : g_event_groups) {
            size_t bytes = sizeof(uint64_t) * state.instance_count;
            std::vector<uint64_t> values(state.instance_count, 0);
            CUptiResult result = cuptiEventGroupReadEvent(
                state.group,
                CUPTI_EVENT_READ_FLAG_NONE,
                state.event_id,
                &bytes,
                values.data());
            if (result != CUPTI_SUCCESS) {
                check_cupti(result, "cuptiEventGroupReadEvent");
                continue;
            }
            uint64_t total = 0;
            for (uint64_t value : values) {
                total += value;
            }
            g_csv << elapsed_seconds << ','
                  << state.name << ','
                  << total << ','
                  << state.instance_count << '\n';
        }
        g_csv.flush();
        std::this_thread::sleep_for(std::chrono::milliseconds(g_interval_ms));
    }

    restore_context(previous);
}

void remember_context(CUcontext context) {
    if (!context || g_context) {
        return;
    }
    CUcontext previous = nullptr;
    if (!make_context_current(context, &previous)) {
        return;
    }
    CUdevice device = 0;
    if (check_cuda(cuCtxGetDevice(&device), "cuCtxGetDevice")) {
        g_context = context;
        g_device = device;
    }
    restore_context(previous);
}

void start_monitor(CUcontext context) {
    std::lock_guard<std::mutex> guard(g_mutex);
    if (g_running.load()) {
        return;
    }
    remember_context(context);
    if (!g_context) {
        std::fprintf(stderr, "CUPTI event monitor: no CUDA context available\n");
        return;
    }
    CUcontext previous = nullptr;
    if (!make_context_current(g_context, &previous)) {
        return;
    }
    bool created = create_event_groups();
    restore_context(previous);
    if (!created) {
        return;
    }

    g_csv.open(output_path(), std::ios::out | std::ios::trunc);
    if (!g_csv) {
        std::fprintf(stderr, "CUPTI event monitor: failed to open output CSV\n");
        destroy_event_groups();
        return;
    }
    g_csv << "elapsed_seconds,event_name,value_delta,instance_count\n";
    g_start_time = std::chrono::steady_clock::now();
    g_running.store(true);
    g_thread = std::thread(sample_loop);
}

void stop_monitor() {
    std::lock_guard<std::mutex> guard(g_mutex);
    if (!g_running.load()) {
        return;
    }
    g_running.store(false);
    if (g_thread.joinable()) {
        g_thread.join();
    }
    destroy_event_groups();
    if (g_csv.is_open()) {
        g_csv.close();
    }
}

void CUPTIAPI callback(
    void *,
    CUpti_CallbackDomain domain,
    CUpti_CallbackId callback_id,
    const void *callback_data) {
    if (domain == CUPTI_CB_DOMAIN_RESOURCE) {
        if (callback_id == CUPTI_CBID_RESOURCE_CONTEXT_CREATED) {
            auto *resource =
                reinterpret_cast<const CUpti_ResourceData *>(callback_data);
            remember_context(resource->context);
        }
        return;
    }

    auto *info = reinterpret_cast<const CUpti_CallbackData *>(callback_data);
    if (domain == CUPTI_CB_DOMAIN_DRIVER_API) {
        if (callback_id == CUPTI_DRIVER_TRACE_CBID_cuProfilerStart &&
            info->callbackSite == CUPTI_API_EXIT) {
            start_monitor(info->context);
        } else if (callback_id == CUPTI_DRIVER_TRACE_CBID_cuProfilerStop &&
                   info->callbackSite == CUPTI_API_ENTER) {
            stop_monitor();
        }
        return;
    }

    if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
        if (callback_id == CUPTI_RUNTIME_TRACE_CBID_cudaProfilerStart_v4000 &&
            info->callbackSite == CUPTI_API_EXIT) {
            start_monitor(info->context);
        } else if (
            callback_id == CUPTI_RUNTIME_TRACE_CBID_cudaProfilerStop_v4000 &&
            info->callbackSite == CUPTI_API_ENTER) {
            stop_monitor();
        }
    }
}

void at_exit() {
    stop_monitor();
    if (g_subscriber) {
        check_cupti(cuptiUnsubscribe(g_subscriber), "cuptiUnsubscribe");
        g_subscriber = nullptr;
    }
}

}  // namespace

extern "C" int InitializeInjection() {
    static std::atomic<bool> initialized{false};
    bool expected = false;
    if (!initialized.compare_exchange_strong(expected, true)) {
        return 1;
    }

    g_event_names = split_events(std::getenv("CUPTI_EVENT_MONITOR_EVENTS"));
    g_interval_ms = interval_ms();

    if (!check_cupti(cuptiSubscribe(&g_subscriber, callback, nullptr), "cuptiSubscribe")) {
        return 0;
    }
    check_cupti(
        cuptiEnableCallback(
            1,
            g_subscriber,
            CUPTI_CB_DOMAIN_RESOURCE,
            CUPTI_CBID_RESOURCE_CONTEXT_CREATED),
        "cuptiEnableCallback(CONTEXT_CREATED)");
    check_cupti(
        cuptiEnableCallback(
            1,
            g_subscriber,
            CUPTI_CB_DOMAIN_DRIVER_API,
            CUPTI_DRIVER_TRACE_CBID_cuProfilerStart),
        "cuptiEnableCallback(cuProfilerStart)");
    check_cupti(
        cuptiEnableCallback(
            1,
            g_subscriber,
            CUPTI_CB_DOMAIN_DRIVER_API,
            CUPTI_DRIVER_TRACE_CBID_cuProfilerStop),
        "cuptiEnableCallback(cuProfilerStop)");
    check_cupti(
        cuptiEnableCallback(
            1,
            g_subscriber,
            CUPTI_CB_DOMAIN_RUNTIME_API,
            CUPTI_RUNTIME_TRACE_CBID_cudaProfilerStart_v4000),
        "cuptiEnableCallback(cudaProfilerStart)");
    check_cupti(
        cuptiEnableCallback(
            1,
            g_subscriber,
            CUPTI_CB_DOMAIN_RUNTIME_API,
            CUPTI_RUNTIME_TRACE_CBID_cudaProfilerStop_v4000),
        "cuptiEnableCallback(cudaProfilerStop)");

    std::atexit(at_exit);
    std::fprintf(
        stderr,
        "CUPTI event monitor: initialized, interval=%d ms, output=%s\n",
        g_interval_ms,
        output_path().c_str());
    return 1;
}
