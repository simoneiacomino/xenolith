/* Host-only loader fixture for check-device. Never opens real hardware. */
#include <level_zero/ze_api.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int scenario(const char *name) {
    const char *value = getenv("XE_PROBE_TEST_CASE");
    return value && !strcmp(value, name);
}

static void record(const char *name) {
    FILE *f = fopen(getenv("XE_PROBE_TEST_TRACE"), "a");
    if (!f) abort();
    fprintf(f, "%s\n", name);
    fclose(f);
}

ze_result_t zeInit(ze_init_flags_t flags) {
    record("init");
    if (flags != ZE_INIT_FLAG_GPU_ONLY) abort();
    return scenario("unavailable") ? ZE_RESULT_ERROR_UNINITIALIZED : ZE_RESULT_SUCCESS;
}

ze_result_t zeDriverGet(uint32_t *count, ze_driver_handle_t *drivers) {
    record(drivers ? "driver" : "driver-count");
    if (scenario("driver-error")) return ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE;
    if (!drivers) *count = scenario("no-drivers") ? 0 : scenario("multiple") ? 2 : 1;
    else {
        if (*count != 1) abort();
        drivers[0] = (ze_driver_handle_t)(uintptr_t)1;
    }
    return ZE_RESULT_SUCCESS;
}

ze_result_t zeDeviceGet(ze_driver_handle_t driver, uint32_t *count, ze_device_handle_t *devices) {
    record(devices ? "device" : "device-count");
    if ((uintptr_t)driver != 1) abort();
    if (scenario("device-error")) return ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS;
    if (!devices) *count = scenario("no-devices") ? 0 : scenario("multiple") ? 2 : 1;
    else {
        if (*count != 1) abort();
        devices[0] = (ze_device_handle_t)(uintptr_t)2;
    }
    return ZE_RESULT_SUCCESS;
}

ze_result_t zeDeviceGetProperties(ze_device_handle_t device, ze_device_properties_t *properties) {
    record("properties");
    if ((uintptr_t)device != 2 || properties->stype != ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES) abort();
    if (scenario("properties-error")) return ZE_RESULT_ERROR_DEVICE_LOST;
    properties->vendorId = scenario("wrong-vendor") ? 0x1234 : 0x8086;
    properties->deviceId = scenario("untested") ? 0x46a0 : scenario("unknown") || scenario("multiple") ? 0xffff : 0xa7a0;
    snprintf(properties->name, sizeof properties->name, "Fixture Intel GPU");
    return ZE_RESULT_SUCCESS;
}

ze_result_t zeDeviceGetStatus(ze_device_handle_t device) {
    record("status");
    if ((uintptr_t)device != 2) abort();
    if (scenario("permissions")) return ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS;
    if (scenario("lost")) return ZE_RESULT_ERROR_DEVICE_LOST;
    return ZE_RESULT_SUCCESS;
}

/* Any context, command list or inference allocation is a test failure. */
ze_result_t zeContextCreate(ze_driver_handle_t driver, const ze_context_desc_t *desc, ze_context_handle_t *context) {
    (void)driver; (void)desc; (void)context; record("FORBIDDEN-context"); abort();
}

ze_result_t zeMemAllocShared(ze_context_handle_t context, const ze_device_mem_alloc_desc_t *device_desc,
                           const ze_host_mem_alloc_desc_t *host_desc, size_t size, size_t alignment,
                           ze_device_handle_t device, void **pointer) {
    (void)context; (void)device_desc; (void)host_desc; (void)size; (void)alignment; (void)device; (void)pointer;
    record("FORBIDDEN-allocation"); abort();
}
