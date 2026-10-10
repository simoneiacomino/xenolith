/* Run the real allocator against simulated Level Zero driver limits. */
#define zeMemAllocShared test_zeMemAllocShared
#define zeMemFree test_zeMemFree
#define zeDriverGetExtensionProperties test_zeDriverGetExtensionProperties
#include "../xenolith.c"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <sys/wait.h>

static ze_driver_extension_properties_t extensions[2];
static uint32_t extension_count;
static uint64_t limit;
static ze_result_t allocation_result;
static size_t allocated_size;
static ze_structure_type_t allocated_stype;
static unsigned calls, frees;
static unsigned char storage;

ze_result_t ZE_APICALL test_zeDriverGetExtensionProperties(
    ze_driver_handle_t driver, uint32_t *count, ze_driver_extension_properties_t *out) {
    assert(driver);
    if (out) {
        assert(*count >= extension_count);
        memcpy(out, extensions, extension_count * sizeof(*out));
    }
    *count = extension_count;
    return ZE_RESULT_SUCCESS;
}

ze_result_t ZE_APICALL test_zeMemAllocShared(
    ze_context_handle_t context, const ze_device_mem_alloc_desc_t *dd,
    const ze_host_mem_alloc_desc_t *hd, size_t size, size_t alignment,
    ze_device_handle_t device, void **out) {
    assert(context && device && alignment == 64 && size % 64 == 0);
    assert(dd->stype == ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC);
    assert(hd->stype == ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC && hd->pNext == NULL);
    calls++;
    allocated_size = size;
    allocated_stype = 0;
    if (dd->pNext) {
        memcpy(&allocated_stype, dd->pNext, sizeof allocated_stype);
        if (allocated_stype == ZE_STRUCTURE_TYPE_RELAXED_ALLOCATION_LIMITS_EXP_DESC) {
            const ze_relaxed_allocation_limits_exp_desc_t *relaxed = dd->pNext;
            assert(relaxed->pNext == NULL);
            assert(relaxed->flags == ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_MAX_SIZE);
#ifdef ZE_RELAXED_ALLOCATION_LIMITS_EXT_NAME
        } else if (allocated_stype == ZE_STRUCTURE_TYPE_RELAXED_ALLOCATION_LIMITS_EXT_DESC) {
            const ze_relaxed_allocation_limits_ext_desc_t *relaxed = dd->pNext;
            assert(relaxed->pNext == NULL);
            assert(relaxed->flags == ZE_RELAXED_ALLOCATION_LIMITS_EXT_FLAG_MAX_SIZE);
#endif
        } else {
            assert(0 && "unexpected allocation extension");
        }
    }
    if (size > limit && !dd->pNext) return ZE_RESULT_ERROR_UNSUPPORTED_SIZE;
    if (allocation_result != ZE_RESULT_SUCCESS) return allocation_result;
    *out = &storage;
    return ZE_RESULT_SUCCESS;
}

ze_result_t ZE_APICALL test_zeMemFree(ze_context_handle_t context, void *p) {
    assert(context && p == &storage);
    frees++;
    return ZE_RESULT_SUCCESS;
}

static void extension(uint32_t index, const char *name) {
    snprintf(extensions[index].name, sizeof extensions[index].name, "%s", name);
    extensions[index].version = ZE_MAKE_VERSION(1, 0);
}

static void allocate(xe_engine *e, size_t size, ze_structure_type_t stype) {
    unsigned before = calls, before_free = frees;
    void *p = xe_alloc(e, size, XE_MEM_SHARED);
    assert(p == &storage && calls == before + 1);
    assert(allocated_size == (size + 63) / 64 * 64 && allocated_stype == stype);
    xe_free(e, p, XE_MEM_SHARED);
    assert(frees == before_free + 1);
}

static void failure(xe_engine *e, size_t size, const char *message) {
    int output[2];
    assert(pipe(output) == 0);
    pid_t pid = fork();
    assert(pid >= 0);
    if (!pid) {
        close(output[0]);
        assert(dup2(output[1], STDERR_FILENO) >= 0);
        close(output[1]);
        xe_alloc(e, size, XE_MEM_SHARED);
        _exit(0);
    }
    close(output[1]);
    char error[1024];
    size_t length = 0;
    ssize_t n;
    while ((n = read(output[0], error + length, sizeof error - 1 - length)) > 0)
        length += (size_t)n;
    error[length] = 0;
    close(output[0]);
    int status;
    assert(waitpid(pid, &status, 0) == pid);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 1);
    assert(strstr(error, message));
}

int main(void) {
    const size_t gib = (size_t)1024 * 1024 * 1024, cache = 5 * gib / 2;
    ze_driver_handle_t driver = (ze_driver_handle_t)(uintptr_t)1;
    assert(xe_gpu_get_relaxed_limits(driver) == XE_RELAXED_NONE);
    extension_count = 1;
    extension(0, "unrelated_extension");
    assert(xe_gpu_get_relaxed_limits(driver) == XE_RELAXED_NONE);
    extension(0, ZE_RELAXED_ALLOCATION_LIMITS_EXP_NAME);
    assert(xe_gpu_get_relaxed_limits(driver) == XE_RELAXED_EXP);
    extensions[0].version = 0;
    assert(xe_gpu_get_relaxed_limits(driver) == XE_RELAXED_NONE);
#ifdef ZE_RELAXED_ALLOCATION_LIMITS_EXT_NAME
    extension(0, ZE_RELAXED_ALLOCATION_LIMITS_EXT_NAME);
    assert(xe_gpu_get_relaxed_limits(driver) == XE_RELAXED_EXT);
    extension_count = 2;
    extension(0, ZE_RELAXED_ALLOCATION_LIMITS_EXP_NAME);
    extension(1, ZE_RELAXED_ALLOCATION_LIMITS_EXT_NAME);
    assert(xe_gpu_get_relaxed_limits(driver) == XE_RELAXED_EXT);
    extension(0, ZE_RELAXED_ALLOCATION_LIMITS_EXT_NAME);
    extension(1, ZE_RELAXED_ALLOCATION_LIMITS_EXP_NAME);
    assert(xe_gpu_get_relaxed_limits(driver) == XE_RELAXED_EXT);
    extensions[0].version = 0;
    assert(xe_gpu_get_relaxed_limits(driver) == XE_RELAXED_EXP);
#endif
    xe_engine e = {0};
    e.gpu.context = (ze_context_handle_t)(uintptr_t)1;
    e.gpu.device = (ze_device_handle_t)(uintptr_t)1;
    e.gpu.max_mem_alloc_size = limit = gib;
    e.gpu.relaxed_limits = XE_RELAXED_EXP;
    allocate(&e, 100 * 1024 * 1024, 0);
    allocate(&e, gib - 1, 0);
    allocate(&e, gib + 1, ZE_STRUCTURE_TYPE_RELAXED_ALLOCATION_LIMITS_EXP_DESC);
    allocate(&e, cache, ZE_STRUCTURE_TYPE_RELAXED_ALLOCATION_LIMITS_EXP_DESC);
#ifdef ZE_RELAXED_ALLOCATION_LIMITS_EXT_NAME
    e.gpu.relaxed_limits = XE_RELAXED_EXT;
    allocate(&e, cache, ZE_STRUCTURE_TYPE_RELAXED_ALLOCATION_LIMITS_EXT_DESC);
#endif
    e.gpu.relaxed_limits = XE_RELAXED_NONE;
    e.gpu.max_mem_alloc_size = limit = 16 * gib;
    allocate(&e, cache, 0);
    e.gpu.max_mem_alloc_size = limit = gib;
    allocate(&e, 100 * 1024 * 1024, 0);
    failure(&e, cache, "driver does not support relaxed allocation limits");
    e.gpu.relaxed_limits = XE_RELAXED_EXP;
    allocation_result = ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY;
    failure(&e, cache, "0x70000003 allocating 2684354560 bytes "
            "(device limit 1073741824, relaxed limits enabled)");
    failure(&e, 100 * 1024 * 1024, "0x70000003 allocating 104857600 bytes "
            "(device limit 1073741824, relaxed limits disabled)");
    puts("GPU allocation limits: PASS");
    return 0;
}
